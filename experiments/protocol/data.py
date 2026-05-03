from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset


class ListGraphDataset(Dataset):
    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


from get.data.synthetic import sample_from_adj


def _limit_split(items: List[Dict[str, torch.Tensor]], max_graphs: int) -> List[Dict[str, torch.Tensor]]:
    if max_graphs > 0:
        return items[:max_graphs]
    return items


def _infer_output_dim(splits: Dict[str, List[Dict[str, torch.Tensor]]]) -> int:
    for items in splits.values():
        if items:
            return int(items[0]["y"].numel())
    return 1


def _scalar_labels(items: List[Dict[str, torch.Tensor]]) -> List[int] | None:
    labels: List[int] = []
    for sample in items:
        y = sample["y"].reshape(-1)
        if y.numel() == 0:
            return None
        labels.append(int(round(float(y[0].item()))))
    return labels


def summarize_split_items(items: List[Dict[str, torch.Tensor]], task_type: str | None = None) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_graphs": len(items)}
    if not items:
        summary["single_class"] = True
        return summary

    labels = _scalar_labels(items) if task_type in {"binary", "multiclass"} else None
    if labels is None:
        summary["target_dim"] = int(items[0]["y"].numel())
        return summary

    counts = Counter(labels)
    summary["class_counts"] = {str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[0])}
    summary["single_class"] = len(counts) <= 1
    if set(counts.keys()).issubset({0, 1}):
        pos = int(counts.get(1, 0))
        neg = int(counts.get(0, 0))
        summary["positive"] = pos
        summary["negative"] = neg
        summary["positive_rate"] = float(pos / len(items)) if items else 0.0
    return summary


def summarize_splits(splits: Dict[str, List[Dict[str, torch.Tensor]]], task_type: str | None = None) -> Dict[str, Dict[str, object]]:
    return {split_name: summarize_split_items(split_items, task_type=task_type) for split_name, split_items in splits.items()}


def graph_to_sample(data, in_dim: int, max_motifs_per_anchor: int, y_mode: str = "binary") -> Dict[str, torch.Tensor]:
    n = int(data.num_nodes)
    edge_index = data.edge_index.long()
    x = data.x
    if x is None:
        x = torch.ones((n, 1), dtype=torch.float32)
    x = x.float()
    if x.size(1) < in_dim:
        x = torch.cat([x, torch.zeros((n, in_dim - x.size(1)), dtype=x.dtype)], dim=1)
    elif x.size(1) > in_dim:
        x = x[:, :in_dim]

    adj = torch.zeros((n, n), dtype=torch.bool)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = True
    adj.fill_diagonal_(False)

    yv = data.y
    if yv is None:
        y = torch.tensor([0.0])
    else:
        y = yv.view(-1).float()

    if y_mode == "binary":
        yy = torch.tensor([1.0 if float(y[0].item()) > 0 else 0.0])
    elif y_mode == "multiclass":
        yy = torch.tensor([float(int(y[0].item()))])
    elif y_mode == "regression":
        yy = torch.tensor([float(y[0].item())])
    elif y_mode in {"multilabel", "vector"}:
        yy = y.float().view(-1)
    else:
        raise ValueError(y_mode)

    return sample_from_adj(adj=adj, x=x, y=yy, max_motifs_per_anchor=max_motifs_per_anchor)


def _random_split_items(items: List[Dict[str, torch.Tensor]], seed: int, train_ratio: float, val_ratio: float):
    n = len(items)
    if n == 0:
        return [], [], []

    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    items = [items[i] for i in idx]

    if n == 1:
        return items[:1], [], []
    if n == 2:
        return items[:1], items[1:], []

    ntr = max(1, int(round(n * train_ratio)))
    ntr = min(ntr, n - 2)
    nval = max(1, int(round(n * val_ratio)))
    if train_ratio + val_ratio >= 1.0:
        nval = n - ntr
    else:
        nval = min(nval, n - ntr - 1)

    tr = items[:ntr]
    va = items[ntr:ntr + nval]
    te = items[ntr + nval:]

    if not te and train_ratio + val_ratio < 1.0 and len(items) > 2:
        te = [va.pop()] if va else [tr.pop()]
    if not va and len(items) > 1:
        va = [tr.pop()]
    return tr, va, te


def _stratified_split_items(items: List[Dict[str, torch.Tensor]], seed: int, train_ratio: float, val_ratio: float):
    labels = _scalar_labels(items)
    if labels is None or len(set(labels)) <= 1 or len(items) < 3:
        return _random_split_items(items, seed, train_ratio, val_ratio)

    idx = np.arange(len(items))
    try:
        outer = StratifiedShuffleSplit(n_splits=1, test_size=max(1e-9, 1.0 - train_ratio), random_state=seed)
        train_idx, temp_idx = next(outer.split(idx, labels))
        train_items = [items[int(i)] for i in train_idx.tolist()]

        if train_ratio + val_ratio >= 1.0:
            val_items = [items[int(i)] for i in temp_idx.tolist()]
            test_items: List[Dict[str, torch.Tensor]] = []
        else:
            temp_labels = [labels[int(i)] for i in temp_idx.tolist()]
            val_fraction = val_ratio / max(1e-12, 1.0 - train_ratio)
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=max(1e-9, 1.0 - val_fraction),
                random_state=seed + 1,
            )
            val_rel_idx, test_rel_idx = next(inner.split(np.arange(len(temp_idx)), temp_labels))
            val_items = [items[int(temp_idx[i])] for i in val_rel_idx.tolist()]
            test_items = [items[int(temp_idx[i])] for i in test_rel_idx.tolist()]
        return train_items, val_items, test_items
    except ValueError:
        return _random_split_items(items, seed, train_ratio, val_ratio)


def split_items(
    items: List[Dict[str, torch.Tensor]],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    task_type: str | None = None,
):
    if task_type in {"binary", "multiclass"}:
        return _stratified_split_items(items, seed, train_ratio, val_ratio)
    return _random_split_items(items, seed, train_ratio, val_ratio)


def _make_stage1_wedge_triangle(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    for _ in range(args.max_graphs if args.max_graphs > 0 else 256):
        n = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        p = float(args.edge_prob)
        adj = torch.from_numpy((rng.random((n, n)) < p)).bool()
        adj = torch.triu(adj, diagonal=1)
        adj = adj | adj.t()
        tri = torch.trace((adj.float() @ adj.float() @ adj.float())) / 6.0
        wedges = 0.5 * ((adj.sum(1) * (adj.sum(1) - 1)).sum() - 6 * tri)
        y = torch.tensor([1.0 if tri > wedges * 0.08 else 0.0])
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _make_stage1_triangle_regression(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    for _ in range(args.max_graphs if args.max_graphs > 0 else 256):
        n = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        p = float(args.edge_prob)
        adj = torch.from_numpy((rng.random((n, n)) < p)).bool()
        adj = torch.triu(adj, diagonal=1)
        adj = adj | adj.t()
        tri = float((torch.trace((adj.float() @ adj.float() @ adj.float())) / 6.0).item())
        y = torch.tensor([tri / max(1.0, n)])
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _make_stage1_cycle_parity(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    m = args.max_graphs if args.max_graphs > 0 else 256
    for _ in range(m):
        n = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        cycle_len = int(rng.integers(5, 12))
        g = nx.cycle_graph(cycle_len)
        extra = nx.gnp_random_graph(max(n - cycle_len, 1), 0.2, seed=int(rng.integers(1, 1_000_000)))
        extra = nx.relabel_nodes(extra, lambda u: u + cycle_len)
        g = nx.disjoint_union(g, extra)
        n = g.number_of_nodes()
        adj = torch.zeros((n, n), dtype=torch.bool)
        for u, v in g.edges():
            adj[u, v] = True
            adj[v, u] = True
        y = torch.tensor([1.0 if cycle_len % 2 == 1 else 0.0])
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _is_sat_bruteforce(clauses: List[Tuple[int, int, int]], signs: List[Tuple[int, int, int]], n_vars: int) -> bool:
    for mask in range(1 << n_vars):
        ok = True
        for (a, b, c), (sa, sb, sc) in zip(clauses, signs):
            va = ((mask >> a) & 1)
            vb = ((mask >> b) & 1)
            vc = ((mask >> c) & 1)
            la = va if sa > 0 else (1 - va)
            lb = vb if sb > 0 else (1 - vb)
            lc = vc if sc > 0 else (1 - vc)
            if (la | lb | lc) == 0:
                ok = False
                break
        if ok:
            return True
    return False


def _make_stage1_max3sat_formula(
    rng: np.random.Generator,
    n_vars: int,
    n_clauses: int,
    satisfiable: bool,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    if satisfiable:
        while True:
            clauses: List[Tuple[int, int, int]] = []
            signs: List[Tuple[int, int, int]] = []
            for _ in range(n_clauses):
                vars3 = rng.choice(n_vars, size=3, replace=False)
                s3 = rng.choice([-1, 1], size=3, replace=True)
                clauses.append((int(vars3[0]), int(vars3[1]), int(vars3[2])))
                signs.append((int(s3[0]), int(s3[1]), int(s3[2])))
            if _is_sat_bruteforce(clauses, signs, n_vars=n_vars):
                return clauses, signs

    core_vars = rng.choice(n_vars, size=3, replace=False)
    clauses = []
    signs = []
    for pattern in range(8):
        signs.append(tuple(1 if (pattern >> bit) & 1 else -1 for bit in range(3)))
        clauses.append((int(core_vars[0]), int(core_vars[1]), int(core_vars[2])))
    for _ in range(n_clauses - 8):
        vars3 = rng.choice(n_vars, size=3, replace=False)
        s3 = rng.choice([-1, 1], size=3, replace=True)
        clauses.append((int(vars3[0]), int(vars3[1]), int(vars3[2])))
        signs.append((int(s3[0]), int(s3[1]), int(s3[2])))
    return clauses, signs


def _make_stage1_max3sat(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    m = args.max_graphs if args.max_graphs > 0 else 128
    target_pos = m // 2
    target_neg = m - target_pos
    pos_count = 0
    neg_count = 0
    n_vars = 10
    n_clauses = 18
    while len(out) < m:
        want_sat = pos_count < target_pos
        want_unsat = neg_count < target_neg
        if want_sat:
            clauses, signs = _make_stage1_max3sat_formula(rng, n_vars, n_clauses, satisfiable=True)
            sat = True
        elif want_unsat:
            clauses, signs = _make_stage1_max3sat_formula(rng, n_vars, n_clauses, satisfiable=False)
            sat = False
        else:
            break
        n = n_vars + n_clauses
        adj = torch.zeros((n, n), dtype=torch.bool)
        for cidx, vars3 in enumerate(clauses):
            cnode = n_vars + cidx
            for v in vars3:
                adj[cnode, v] = True
                adj[v, cnode] = True
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        for cidx, s3 in enumerate(signs):
            x[n_vars + cidx, 0] = float(s3[0])
            x[n_vars + cidx, 1] = float(s3[1])
            x[n_vars + cidx, 2] = float(s3[2])
        y = torch.tensor([1.0 if sat else 0.0])
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
        if sat:
            pos_count += 1
        else:
            neg_count += 1
    perm = rng.permutation(len(out))
    out = [out[i] for i in perm.tolist()]
    return out


def _is_xorsat_bruteforce(clauses: List[Tuple[int, int, int]], rhs: List[int], n_vars: int) -> bool:
    for mask in range(1 << n_vars):
        ok = True
        for (a, b, c), r in zip(clauses, rhs):
            val = ((mask >> a) & 1) ^ ((mask >> b) & 1) ^ ((mask >> c) & 1)
            if val != r:
                ok = False
                break
        if ok:
            return True
    return False


def _make_stage1_xorsat(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed + 17)
    m = args.max_graphs if args.max_graphs > 0 else 128
    target_pos = m // 2
    target_neg = m - target_pos
    pos_count = 0
    neg_count = 0
    n_vars = 12
    n_clauses = 11
    while len(out) < m:
        clauses = []
        rhs = []
        for _c in range(n_clauses):
            vars3 = rng.choice(n_vars, size=3, replace=False)
            clauses.append((int(vars3[0]), int(vars3[1]), int(vars3[2])))
            rhs.append(int(rng.integers(0, 2)))
        sat = _is_xorsat_bruteforce(clauses, rhs, n_vars=n_vars)
        if sat and pos_count >= target_pos:
            continue
        if not sat and neg_count >= target_neg:
            continue

        n = n_vars + n_clauses
        adj = torch.zeros((n, n), dtype=torch.bool)
        for cidx, vars3 in enumerate(clauses):
            cnode = n_vars + cidx
            for v in vars3:
                adj[cnode, v] = True
                adj[v, cnode] = True
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        for cidx, r in enumerate(rhs):
            x[n_vars + cidx, 0] = float(r)
        y = torch.tensor([1.0 if sat else 0.0])
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
        if sat:
            pos_count += 1
        else:
            neg_count += 1
    return out


def _make_rook_4x4() -> nx.Graph:
    g = nx.Graph()
    nodes = [(i, j) for i in range(4) for j in range(4)]
    g.add_nodes_from(nodes)
    for i1, j1 in nodes:
        for i2, j2 in nodes:
            if (i1, j1) == (i2, j2):
                continue
            if i1 == i2 or j1 == j2:
                g.add_edge((i1, j1), (i2, j2))
    return nx.convert_node_labels_to_integers(g)


def _make_shrikhande() -> nx.Graph:
    g = nx.Graph()
    nodes = [(a, b) for a in range(4) for b in range(4)]
    g.add_nodes_from(nodes)
    s = [(0, 1), (0, 3), (1, 0), (3, 0), (1, 1), (3, 3)]
    for a, b in nodes:
        for da, db in s:
            u = ((a + da) % 4, (b + db) % 4)
            g.add_edge((a, b), u)
    return nx.convert_node_labels_to_integers(g)


def _make_stage1_srg_discrimination(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rook = _make_rook_4x4()
    shri = _make_shrikhande()
    base = [rook, shri]
    rng = np.random.default_rng(args.seed + 23)
    m = args.max_graphs if args.max_graphs > 0 else 256
    for i in range(m):
        yv = i % 2
        g = base[yv].copy()
        perm = rng.permutation(g.number_of_nodes())
        mp = {old: int(perm[old]) for old in range(g.number_of_nodes())}
        gp = nx.relabel_nodes(g, mp)
        n = gp.number_of_nodes()
        adj = torch.zeros((n, n), dtype=torch.bool)
        for u, v in gp.edges():
            adj[u, v] = True
            adj[v, u] = True
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        y = torch.tensor([float(yv)])
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _load_stage2_csl(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import GNNBenchmarkDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    parts = {
        "train": GNNBenchmarkDataset(root=str(root), name="CSL", split="train"),
        "val": GNNBenchmarkDataset(root=str(root), name="CSL", split="val"),
        "test": GNNBenchmarkDataset(root=str(root), name="CSL", split="test"),
    }
    splits: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for split, dataset in parts.items():
        items = _limit_split(list(dataset), args.max_graphs)
        splits[split] = [graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="multiclass") for d in items]
    nclass = int(max(int(s["y"].item()) for split in splits.values() for s in split) + 1)
    return splits, nclass


def _load_stage2_brec(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    if not args.brec_file:
        raise ValueError("--brec_file required for stage2_brec")
    data_list = torch.load(Path(args.brec_file).expanduser(), map_location="cpu")
    if args.max_graphs > 0:
        data_list = data_list[: args.max_graphs]
    samples = [graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="binary") for d in data_list]
    return samples, 2


def _load_stage3_zinc(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import ZINC

    root = Path(args.dataset_root).expanduser() / "pyg"
    parts = {s: ZINC(root=str(root), subset=True, split=s) for s in ["train", "val", "test"]}
    splits: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for split, dataset in parts.items():
        items = _limit_split(list(dataset), args.max_graphs)
        splits[split] = [graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="regression") for d in items]
    return splits, 1


def _load_stage3_molhiv(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from ogb.graphproppred import PygGraphPropPredDataset

    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(Path(args.dataset_root).expanduser() / "ogb"))
    idx = ds.get_idx_split()
    splits: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for split_name, split_idx in {"train": idx["train"], "val": idx["valid"], "test": idx["test"]}.items():
        order = split_idx.tolist()
        order = order[: args.max_graphs] if args.max_graphs > 0 else order
        splits[split_name] = [graph_to_sample(ds[i], args.in_dim, args.max_motifs_per_anchor, y_mode="binary") for i in order]
    return splits, 2


def _load_stage3_peptides(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import LRGBDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    parts = {s: LRGBDataset(root=str(root), name="Peptides-struct", split=s) for s in ["train", "val", "test"]}
    splits: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for split, dataset in parts.items():
        items = _limit_split(list(dataset), args.max_graphs)
        splits[split] = [graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="vector") for d in items]
    return splits, _infer_output_dim(splits)


def _load_stage3_peptides_func(args) -> Tuple[Dict[str, List[Dict[str, torch.Tensor]]], int]:
    from torch_geometric.datasets import LRGBDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    out = {}
    for s in ["train", "val", "test"]:
        part = _limit_split(list(LRGBDataset(root=str(root), name="Peptides-func", split=s)), args.max_graphs)
        out[s] = [graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="multilabel") for d in part]
    return out, _infer_output_dim(out)


def _load_stage4_tu(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import TUDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    ds = TUDataset(root=str(root), name=args.tu_name)
    items = list(ds)
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    samples = [graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="multiclass") for d in items]
    nclass = int(max(int(s["y"].item()) for s in samples) + 1)
    return samples, nclass


def _load_stage4_anomaly(args, name: str) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from pygod.utils import load_data
    from torch_geometric.utils import k_hop_subgraph

    alias_map = {
        "amazon": ["inj_amazon", "amazon"],
        "yelpchi": ["yelpchi"],
        "tfinance": ["tfinance"],
        "tsocial": ["tsocial"],
    }
    candidates = alias_map.get(name.lower(), [name, name.lower(), name.upper()])
    data = None
    for nm in candidates:
        try:
            data = load_data(nm, cache_dir=str(Path(args.dataset_root).expanduser() / "pygod"))
            break
        except Exception:
            pass
    if data is None:
        raise RuntimeError(f"Unable to load anomaly dataset {name} via PyGOD load_data")

    y = data.y.view(-1).long()
    nodes = torch.arange(data.num_nodes)
    if args.max_graphs > 0:
        nodes = nodes[: args.max_graphs]
    samples = []
    for nid in nodes.tolist():
        subset, edge_index, _, _ = k_hop_subgraph(nid, args.ego_hops, data.edge_index, relabel_nodes=True)
        x = data.x[subset].float()
        n = x.size(0)
        adj = torch.zeros((n, n), dtype=torch.bool)
        if edge_index.numel() > 0:
            adj[edge_index[0], edge_index[1]] = True
        yy = torch.tensor([1.0 if int(y[nid].item()) > 0 else 0.0])
        feat = x[:, : args.in_dim] if x.size(1) >= args.in_dim else torch.cat([x, torch.zeros((n, args.in_dim - x.size(1)))], dim=1)
        samples.append(sample_from_adj(adj, feat, yy, args.max_motifs_per_anchor))
    return samples, 2


def build_dataset(task: str, args) -> Tuple[Union[List[Dict[str, torch.Tensor]], Dict[str, List[Dict[str, torch.Tensor]]]], int]:
    import hashlib
    import pickle
    
    cache_version = "v2"
    key_str = f"{cache_version}_{task}_{args.seed}_{args.max_graphs}_{args.in_dim}_{args.max_motifs_per_anchor}_{getattr(args, 'ego_hops', 1)}"
    key = hashlib.md5(key_str.encode()).hexdigest()
    cache_dir = Path(args.dataset_root).expanduser() / "protocol_cache"
    cache_path = cache_dir / f"{key}.pkl"
    
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if task == "stage1_wedge_triangle":
        res = _make_stage1_wedge_triangle(args), 2
    elif task == "stage1_triangle_regression":
        res = _make_stage1_triangle_regression(args), 1
    elif task == "stage1_cycle_parity":
        res = _make_stage1_cycle_parity(args), 2
    elif task == "stage1_max3sat":
        res = _make_stage1_max3sat(args), 2
    elif task == "stage1_xorsat":
        res = _make_stage1_xorsat(args), 2
    elif task == "stage1_srg_discrimination":
        res = _make_stage1_srg_discrimination(args), 2
    elif task == "stage2_csl":
        res = _load_stage2_csl(args)
    elif task == "stage2_brec":
        res = _load_stage2_brec(args)
    elif task == "stage3_zinc":
        res = _load_stage3_zinc(args)
    elif task == "stage3_molhiv":
        res = _load_stage3_molhiv(args)
    elif task == "stage3_peptides_struct_probe":
        res = _load_stage3_peptides(args)
    elif task == "stage3_peptides_func_probe":
        res = _load_stage3_peptides_func(args)
    elif task == "stage4_tu_classification":
        res = _load_stage4_tu(args)
    elif task == "stage4_yelpchi_anomaly":
        res = _load_stage4_anomaly(args, "yelpchi")
    elif task == "stage4_amazon_anomaly":
        res = _load_stage4_anomaly(args, "amazon")
    elif task == "stage4_tfinance_anomaly":
        res = _load_stage4_anomaly(args, "tfinance")
    elif task == "stage4_tsocial_anomaly":
        res = _load_stage4_anomaly(args, "tsocial")
    else:
        raise ValueError(f"Unknown task: {task}")
        
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(res, f)
    return res
