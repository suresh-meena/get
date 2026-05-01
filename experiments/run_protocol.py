from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.regression import MeanAbsoluteError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.graph_baselines.torch_baselines import (
    ExternalGraphBaseline,
    GATGraphBaseline,
    GCNGraphBaseline,
    GINGraphBaseline,
)
from get.data.synthetic import collate_graph_samples
from get.models import EnergyGraphClassifier
from get.utils.seed import seed_everything


@dataclass
class TaskSpec:
    task_type: str  # binary | multiclass | regression
    stage: str


TASK_SPECS: Dict[str, TaskSpec] = {
    "stage1_wedge_triangle": TaskSpec(task_type="binary", stage="1"),
    "stage1_triangle_regression": TaskSpec(task_type="regression", stage="1"),
    "stage1_cycle_parity": TaskSpec(task_type="binary", stage="1"),
    "stage1_max3sat": TaskSpec(task_type="binary", stage="1"),
    "stage1_xorsat": TaskSpec(task_type="binary", stage="1"),
    "stage1_srg_discrimination": TaskSpec(task_type="binary", stage="1"),
    "stage2_csl": TaskSpec(task_type="multiclass", stage="2"),
    "stage2_brec": TaskSpec(task_type="binary", stage="2"),
    "stage3_zinc": TaskSpec(task_type="regression", stage="3"),
    "stage3_molhiv": TaskSpec(task_type="binary", stage="3"),
    "stage3_peptides": TaskSpec(task_type="regression", stage="3"),
    "stage3_peptides_func": TaskSpec(task_type="binary", stage="3"),
    "stage4_tu_classification": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_yelpchi_anomaly": TaskSpec(task_type="binary", stage="4"),
    "stage4_amazon_anomaly": TaskSpec(task_type="binary", stage="4"),
    "stage4_tfinance_anomaly": TaskSpec(task_type="binary", stage="4"),
    "stage4_tsocial_anomaly": TaskSpec(task_type="binary", stage="4"),
}


class ListGraphDataset(Dataset):
    def __init__(self, samples: List[Dict[str, torch.Tensor]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]



def _extract_motifs_from_adj(adj: torch.Tensor, max_motifs_per_anchor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = adj.size(0)
    c3, u3, v3, tt = [], [], [], []
    for c in range(n):
        neigh = torch.nonzero(adj[c], as_tuple=False).flatten()
        if neigh.numel() < 2:
            continue
        budget = 0
        for i in range(neigh.numel()):
            if budget >= max_motifs_per_anchor:
                break
            for j in range(i + 1, neigh.numel()):
                if budget >= max_motifs_per_anchor:
                    break
                u, v = int(neigh[i]), int(neigh[j])
                c3.append(c)
                u3.append(u)
                v3.append(v)
                tt.append(1 if bool(adj[u, v]) else 0)
                budget += 1
    if not c3:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty
    return torch.tensor(c3), torch.tensor(u3), torch.tensor(v3), torch.tensor(tt)


def _sample_from_adj(adj: torch.Tensor, x: torch.Tensor, y: torch.Tensor, max_motifs_per_anchor: int) -> Dict[str, torch.Tensor]:
    edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
    c2 = edge_index[0] if edge_index.numel() > 0 else torch.empty(0, dtype=torch.long)
    u2 = edge_index[1] if edge_index.numel() > 0 else torch.empty(0, dtype=torch.long)
    c3, u3, v3, tt = _extract_motifs_from_adj(adj, max_motifs_per_anchor=max_motifs_per_anchor)
    return {
        "x": x.float(),
        "y": y.float().view(1),
        "c_2": c2.long(),
        "u_2": u2.long(),
        "c_3": c3.long(),
        "u_3": u3.long(),
        "v_3": v3.long(),
        "t_tau": tt.long(),
    }


def _graph_to_sample(data, in_dim: int, max_motifs_per_anchor: int, y_mode: str = "binary") -> Dict[str, torch.Tensor]:
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
    else:
        raise ValueError(y_mode)

    return _sample_from_adj(adj=adj, x=x, y=yy, max_motifs_per_anchor=max_motifs_per_anchor)


def _split(items: List[Dict[str, torch.Tensor]], seed: int, train_ratio: float = 0.7, val_ratio: float = 0.15):
    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    items = [items[i] for i in idx]
    n = len(items)
    ntr = max(1, int(n * train_ratio))
    nval = max(1, int(n * val_ratio))
    if ntr + nval >= n:
        nval = max(1, n - ntr - 1)
    tr = items[:ntr]
    va = items[ntr:ntr + nval]
    te = items[ntr + nval:]
    if not te:
        te = va[-1:]
    return tr, va, te


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
        out.append(_sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
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
        out.append(_sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _make_stage1_cycle_parity(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    m = args.max_graphs if args.max_graphs > 0 else 256
    for _ in range(m):
        n = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        cycle_len = int(rng.integers(5, 12))
        G = nx.cycle_graph(cycle_len)
        extra = nx.gnp_random_graph(max(n - cycle_len, 1), 0.2, seed=int(rng.integers(1, 1_000_000)))
        extra = nx.relabel_nodes(extra, lambda u: u + cycle_len)
        G = nx.disjoint_union(G, extra)
        if G.number_of_nodes() < n:
            for k in range(G.number_of_nodes(), n):
                G.add_node(k)
        adj = torch.zeros((n, n), dtype=torch.bool)
        for u, v in G.edges():
            adj[u, v] = True
            adj[v, u] = True
        y = torch.tensor([1.0 if cycle_len % 2 == 1 else 0.0])
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        out.append(_sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
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


def _make_stage1_max3sat(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    m = args.max_graphs if args.max_graphs > 0 else 128
    n_vars = 10
    n_clauses = 18
    for _ in range(m):
        clauses = []
        signs = []
        for _c in range(n_clauses):
            vars3 = rng.choice(n_vars, size=3, replace=False)
            s3 = rng.choice([-1, 1], size=3, replace=True)
            clauses.append((int(vars3[0]), int(vars3[1]), int(vars3[2])))
            signs.append((int(s3[0]), int(s3[1]), int(s3[2])))
        sat = _is_sat_bruteforce(clauses, signs, n_vars=n_vars)
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
        out.append(_sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
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
    n_vars = 12
    n_clauses = 22
    for _ in range(m):
        clauses = []
        rhs = []
        for _c in range(n_clauses):
            vars3 = rng.choice(n_vars, size=3, replace=False)
            clauses.append((int(vars3[0]), int(vars3[1]), int(vars3[2])))
            rhs.append(int(rng.integers(0, 2)))
        sat = _is_xorsat_bruteforce(clauses, rhs, n_vars=n_vars)
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
        out.append(_sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
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
    # Cayley graph on Z4 x Z4 with connection set yielding (16,6,2,2) SRG.
    g = nx.Graph()
    nodes = [(a, b) for a in range(4) for b in range(4)]
    g.add_nodes_from(nodes)
    S = [(0, 1), (0, 3), (1, 0), (3, 0), (1, 1), (3, 3)]
    for a, b in nodes:
        for da, db in S:
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
        out.append(_sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _load_stage2_csl(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import GNNBenchmarkDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    parts = [
        GNNBenchmarkDataset(root=str(root), name="CSL", split="train"),
        GNNBenchmarkDataset(root=str(root), name="CSL", split="val"),
        GNNBenchmarkDataset(root=str(root), name="CSL", split="test"),
    ]
    items = []
    for p in parts:
        items.extend(list(p))
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    samples = [_graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="multiclass") for d in items]
    nclass = int(max(int(s["y"].item()) for s in samples) + 1)
    return samples, nclass


def _load_stage2_brec(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    if not args.brec_file:
        raise ValueError("--brec_file required for stage2_brec")
    data_list = torch.load(Path(args.brec_file).expanduser(), map_location="cpu")
    if args.max_graphs > 0:
        data_list = data_list[: args.max_graphs]
    samples = [_graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="binary") for d in data_list]
    return samples, 2


def _load_stage3_zinc(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import ZINC

    root = Path(args.dataset_root).expanduser() / "pyg"
    parts = [ZINC(root=str(root), subset=True, split=s) for s in ["train", "val", "test"]]
    items = []
    for p in parts:
        items.extend(list(p))
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    samples = [_graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="regression") for d in items]
    return samples, 1


def _load_stage3_molhiv(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from ogb.graphproppred import PygGraphPropPredDataset

    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(Path(args.dataset_root).expanduser() / "ogb"))
    idx = ds.get_idx_split()
    order = torch.cat([idx["train"], idx["valid"], idx["test"]], dim=0).tolist()
    if args.max_graphs > 0:
        order = order[: args.max_graphs]
    samples = [_graph_to_sample(ds[i], args.in_dim, args.max_motifs_per_anchor, y_mode="binary") for i in order]
    return samples, 2


def _load_stage3_peptides(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import LRGBDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    # Use Peptides-struct (regression); take first target for compact protocol.
    parts = [LRGBDataset(root=str(root), name="Peptides-struct", split=s) for s in ["train", "val", "test"]]
    items = []
    for p in parts:
        items.extend(list(p))
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    samples = []
    for d in items:
        if d.y is not None and d.y.numel() > 1:
            d.y = d.y.view(-1)[:1]
        samples.append(_graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="regression"))
    return samples, 1


def _load_stage3_peptides_func(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import LRGBDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    parts = [LRGBDataset(root=str(root), name="Peptides-func", split=s) for s in ["train", "val", "test"]]
    items = []
    for p in parts:
        items.extend(list(p))
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    samples = []
    for d in items:
        # Convert multi-task target to a compact binary proxy: first task bit.
        if d.y is not None and d.y.numel() > 1:
            d.y = d.y.view(-1)[:1]
        samples.append(_graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="binary"))
    return samples, 2


def _load_stage4_tu(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import TUDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    ds = TUDataset(root=str(root), name=args.tu_name)
    items = list(ds)
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    samples = [_graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="multiclass") for d in items]
    nclass = int(max(int(s["y"].item()) for s in samples) + 1)
    return samples, nclass


def _load_stage4_anomaly(args, name: str) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from pygod.utils import load_data
    from torch_geometric.utils import k_hop_subgraph

    alias_map = {
        "amazon": ["inj_amazon", "amazon", "gen_10000"],
        "yelpchi": ["reddit", "weibo", "gen_10000"],
        "tfinance": ["weibo", "gen_10000"],
        "tsocial": ["reddit", "gen_10000"],
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
        samples.append(_sample_from_adj(adj, x[:, : args.in_dim] if x.size(1) >= args.in_dim else torch.cat([x, torch.zeros((n, args.in_dim - x.size(1)))], dim=1), yy, args.max_motifs_per_anchor))
    return samples, 2


def build_dataset(task: str, args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    if task == "stage1_wedge_triangle":
        return _make_stage1_wedge_triangle(args), 2
    if task == "stage1_triangle_regression":
        return _make_stage1_triangle_regression(args), 1
    if task == "stage1_cycle_parity":
        return _make_stage1_cycle_parity(args), 2
    if task == "stage1_max3sat":
        return _make_stage1_max3sat(args), 2
    if task == "stage1_xorsat":
        return _make_stage1_xorsat(args), 2
    if task == "stage1_srg_discrimination":
        return _make_stage1_srg_discrimination(args), 2
    if task == "stage2_csl":
        return _load_stage2_csl(args)
    if task == "stage2_brec":
        return _load_stage2_brec(args)
    if task == "stage3_zinc":
        return _load_stage3_zinc(args)
    if task == "stage3_molhiv":
        return _load_stage3_molhiv(args)
    if task == "stage3_peptides":
        return _load_stage3_peptides(args)
    if task == "stage3_peptides_func":
        return _load_stage3_peptides_func(args)
    if task == "stage4_tu_classification":
        return _load_stage4_tu(args)
    if task == "stage4_yelpchi_anomaly":
        return _load_stage4_anomaly(args, "yelpchi")
    if task == "stage4_amazon_anomaly":
        return _load_stage4_anomaly(args, "amazon")
    if task == "stage4_tfinance_anomaly":
        return _load_stage4_anomaly(args, "tfinance")
    if task == "stage4_tsocial_anomaly":
        return _load_stage4_anomaly(args, "tsocial")
    raise ValueError(task)


def build_model(args, task_type: str, num_classes: int):
    out_dim = 1 if task_type in {"binary", "regression"} else num_classes
    if args.model_name == "fullget":
        return EnergyGraphClassifier(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            num_steps=args.num_steps,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            R=args.R,
            K=args.K,
            num_motif_types=2,
            lambda_2=args.lambda_2,
            lambda_3=args.lambda_3,
            lambda_m=args.lambda_m,
            beta_2=args.beta_2,
            beta_3=args.beta_3,
            beta_m=args.beta_m,
            update_damping=args.update_damping,
            fixed_step_size=args.fixed_step_size,
            armijo_eta0=args.armijo_eta0,
            armijo_gamma=args.armijo_gamma,
            armijo_c=args.armijo_c,
            armijo_max_backtracks=args.armijo_max_backtracks,
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
            energy_name="get_full",
        )
    if args.model_name == "pairwiseget":
        return EnergyGraphClassifier(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            num_steps=args.num_steps,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            R=args.R,
            K=args.K,
            num_motif_types=2,
            lambda_2=args.lambda_2,
            lambda_3=0.0,
            lambda_m=args.lambda_m,
            beta_2=args.beta_2,
            beta_3=args.beta_3,
            beta_m=args.beta_m,
            update_damping=args.update_damping,
            fixed_step_size=args.fixed_step_size,
            armijo_eta0=args.armijo_eta0,
            armijo_gamma=args.armijo_gamma,
            armijo_c=args.armijo_c,
            armijo_max_backtracks=args.armijo_max_backtracks,
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
            energy_name="pairwise_only",
        )
    if args.model_name == "external_baseline":
        return ExternalGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    if args.model_name == "gin":
        return GINGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    if args.model_name == "gcn":
        return GCNGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    if args.model_name == "gat":
        return GATGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    raise ValueError(args.model_name)


def run_epoch(model, loader, device, task_type: str, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    bacc = BinaryAccuracy().to(device) if task_type == "binary" else None
    macc = MulticlassAccuracy(num_classes=loader.dataset.samples[0].get("num_classes", 2)).to(device) if task_type == "multiclass" else None
    mae = MeanAbsoluteError().to(device) if task_type == "regression" else None

    losses = []
    y_true = []
    y_score = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        y = batch["y"].view(-1)
        if train:
            optimizer.zero_grad(set_to_none=True)
        # GET solver-based forward uses autograd even in eval mode.
        with torch.set_grad_enabled(True):
            out = model(batch)
            if task_type == "binary":
                out = out.view(-1)
                loss = F.binary_cross_entropy_with_logits(out, y)
                prob = torch.sigmoid(out)
                bacc.update(prob, y.long())
                y_true.extend(y.detach().cpu().tolist())
                y_score.extend(prob.detach().cpu().tolist())
            elif task_type == "multiclass":
                target = y.long()
                loss = F.cross_entropy(out, target)
                pred = torch.softmax(out, -1)
                acc = (pred.argmax(-1) == target).float().mean()
                losses.append(float(loss.detach().cpu().item()))
                y_true.extend(target.detach().cpu().tolist())
                y_score.extend(pred.argmax(-1).detach().cpu().tolist())
            else:
                out = out.view(-1)
                loss = F.mse_loss(out, y)
                mae.update(out, y)
                y_true.extend(y.detach().cpu().tolist())
                y_score.extend(out.detach().cpu().tolist())
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    metrics = {"loss": float(np.mean(losses)) if losses else 0.0}
    if task_type == "binary":
        metrics["acc"] = float(bacc.compute().item())
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.5
        except Exception:
            metrics["auc"] = 0.5
    elif task_type == "multiclass":
        yt = np.array(y_true)
        yp = np.array(y_score)
        metrics["acc"] = float((yt == yp).mean()) if yt.size > 0 else 0.0
    else:
        metrics["mae"] = float(mae.compute().item())
    return metrics


def fit_once(args, task_type: str, num_classes: int, tr, va, te, device):
    model = build_model(args, task_type=task_type, num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 1e18
    best_state = None
    final_train = final_val = None
    for _ in range(args.epochs):
        final_train = run_epoch(model, tr, device, task_type=task_type, optimizer=optim)
        final_val = run_epoch(model, va, device, task_type=task_type, optimizer=None)
        key = "loss"
        if final_val[key] < best:
            best = final_val[key]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test = run_epoch(model, te, device, task_type=task_type, optimizer=None)
    return {"best_val_loss": float(best), "final_train": final_train, "final_val": final_val, "test": test}


def make_loaders(items, args):
    tr_i, va_i, te_i = _split(items, seed=args.seed)
    tr = DataLoader(ListGraphDataset(tr_i), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_samples)
    va = DataLoader(ListGraphDataset(va_i), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples)
    te = DataLoader(ListGraphDataset(te_i), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples)
    return tr, va, te


def main():
    p = argparse.ArgumentParser(description="Full protocol runner (Stage 1-4)")
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS.keys()))
    p.add_argument("--model_name", type=str, default="fullget", choices=["fullget", "pairwiseget", "external_baseline", "gin", "gcn", "gat"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dataset_root", type=str, default="data")
    p.add_argument("--brec_file", type=str, default="")
    p.add_argument("--tu_name", type=str, default="MUTAG")
    p.add_argument("--cv_folds", type=int, default=1)
    p.add_argument("--max_graphs", type=int, default=0)
    p.add_argument("--ego_hops", type=int, default=1)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--min_nodes", type=int, default=10)
    p.add_argument("--max_nodes", type=int, default=20)
    p.add_argument("--edge_prob", type=float, default=0.2)
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--max_motifs_per_anchor", type=int, default=8)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--R", type=int, default=2)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--lambda_2", type=float, default=1.0)
    p.add_argument("--lambda_3", type=float, default=1.0)
    p.add_argument("--lambda_m", type=float, default=0.0)
    p.add_argument("--beta_2", type=float, default=1.0)
    p.add_argument("--beta_3", type=float, default=1.0)
    p.add_argument("--beta_m", type=float, default=1.0)
    p.add_argument("--update_damping", type=float, default=0.0)
    p.add_argument("--fixed_step_size", type=float, default=0.1)
    p.add_argument("--armijo_eta0", type=float, default=0.2)
    p.add_argument("--armijo_gamma", type=float, default=0.5)
    p.add_argument("--armijo_c", type=float, default=1e-4)
    p.add_argument("--armijo_max_backtracks", type=int, default=20)
    p.add_argument("--inference_mode_train", type=str, default="fixed", choices=["fixed", "armijo"])
    p.add_argument("--inference_mode_eval", type=str, default="armijo", choices=["fixed", "armijo"])

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--output", type=str, default="outputs/protocol/last_metrics.json")

    args = p.parse_args()
    seed_everything(args.seed)
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    spec = TASK_SPECS[args.task]
    items, num_classes = build_dataset(args.task, args)
    if len(items) < 4:
        raise RuntimeError("Dataset too small after loading; increase --max_graphs or fix dataset path")

    if args.cv_folds > 1 and args.task == "stage2_csl":
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold = []
        idx = np.arange(len(items))
        for i, (trainval, test) in enumerate(kf.split(idx), start=1):
            trainval_items = [items[j] for j in trainval.tolist()]
            test_items = [items[j] for j in test.tolist()]
            tr_i, va_i, _ = _split(trainval_items, seed=args.seed + i)
            tr = DataLoader(ListGraphDataset(tr_i), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_samples)
            va = DataLoader(ListGraphDataset(va_i), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples)
            te = DataLoader(ListGraphDataset(test_items), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples)
            m = fit_once(args, spec.task_type, num_classes, tr, va, te, device)
            m["fold"] = i
            fold.append(m)
        accs = [f["test"].get("acc", 0.0) for f in fold]
        losses = [f["test"].get("loss", 0.0) for f in fold]
        result = {
            "task": args.task,
            "cv_folds": args.cv_folds,
            "fold_metrics": fold,
            "summary": {
                "test_metric_mean": float(np.mean(accs) if accs else 0.0),
                "test_metric_std": float(np.std(accs) if accs else 0.0),
                "test_loss_mean": float(np.mean(losses) if losses else 0.0),
                "test_loss_std": float(np.std(losses) if losses else 0.0),
            },
        }
    else:
        tr, va, te = make_loaders(items, args)
        m = fit_once(args, spec.task_type, num_classes, tr, va, te, device)
        result = {"task": args.task, "stage": spec.stage, "metrics": m}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
