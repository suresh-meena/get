from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold
from torch.utils.data import Dataset
from get.data.synthetic import sample_from_adj, sample_from_edge_index
from get.data.leakage_control import (
    create_matched_pairs,
    similarity_aggregate,
    similarity_euclidean_degree,
)


@dataclass
class TaskSpec:
    task_type: str  # binary | multiclass | multilabel | regression | node_binary
    stage: str


@dataclass(frozen=True)
class CachedSampleRef:
    path: str
    label0: float | None = None


TASK_SPECS: Dict[str, TaskSpec] = {
    # Original Stage 1 tasks
    "stage1_wedge_triangle": TaskSpec(task_type="binary", stage="1"),
    "stage1_triangle_regression": TaskSpec(task_type="regression", stage="1"),
    "stage1_cycle_parity": TaskSpec(task_type="binary", stage="1"),
    "stage1_max3sat": TaskSpec(task_type="binary", stage="1"),
    "stage1_xorsat": TaskSpec(task_type="binary", stage="1"),
    "stage1_srg_discrimination": TaskSpec(task_type="binary", stage="1"),
    # Leakage-controlled Stage 1 tasks
    "stage1_wedge_triangle_matched": TaskSpec(task_type="binary", stage="1"),
    "stage1_cycle_parity_matched": TaskSpec(task_type="binary", stage="1"),
    "stage1_max3sat_matched": TaskSpec(task_type="binary", stage="1"),
    # Degree-only matching baselines
    "stage1_wedge_triangle_degree_only": TaskSpec(task_type="binary", stage="1"),
    "stage1_cycle_parity_degree_only": TaskSpec(task_type="binary", stage="1"),
    "stage1_max3sat_degree_only": TaskSpec(task_type="binary", stage="1"),
    # Edge-count-only matching baselines
    "stage1_wedge_triangle_edge_only": TaskSpec(task_type="binary", stage="1"),
    "stage1_cycle_parity_edge_only": TaskSpec(task_type="binary", stage="1"),
    "stage1_max3sat_edge_only": TaskSpec(task_type="binary", stage="1"),
    # Two-hop-only matching baselines
    "stage1_wedge_triangle_twohop_only": TaskSpec(task_type="binary", stage="1"),
    # Stage 2-4 tasks
    "stage2_csl": TaskSpec(task_type="multiclass", stage="2"),
    "stage2_brec": TaskSpec(task_type="binary", stage="2"),
    "stage3_zinc": TaskSpec(task_type="regression", stage="3"),
    "stage3_molhiv": TaskSpec(task_type="binary", stage="3"),
    "stage3_molpcba": TaskSpec(task_type="multilabel", stage="3"),
    "stage3_peptides_struct_probe": TaskSpec(task_type="regression", stage="3"),
    "stage3_peptides_func_probe": TaskSpec(task_type="multilabel", stage="3"),
    "stage4_tu_classification": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_tu_proteins": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_tu_nci1": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_tu_nci109": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_tu_dd": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_tu_enzymes": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_tu_mutagenicity": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_tu_frankenstein": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_yelpchi_anomaly": TaskSpec(task_type="node_binary", stage="4"),
    "stage4_amazon_anomaly": TaskSpec(task_type="node_binary", stage="4"),
    "stage4_tfinance_anomaly": TaskSpec(task_type="node_binary", stage="4"),
    "stage4_tsocial_anomaly": TaskSpec(task_type="node_binary", stage="4"),
}


class ListGraphDataset(Dataset):
    def __init__(self, samples: List[Any]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return _load_sample_item(self.samples[idx])


def _normalize_edge_attr(edge_attr: Any) -> Any:
    if edge_attr is None or not torch.is_tensor(edge_attr) or edge_attr.numel() == 0:
        return edge_attr
    edge_attr = edge_attr.float()
    if edge_attr.dim() == 1:
        return edge_attr.unsqueeze(-1)
    if edge_attr.dim() > 2:
        return edge_attr.reshape(edge_attr.size(0), -1)
    return edge_attr


def _load_sample_item(item: Any) -> Dict[str, torch.Tensor]:
    if isinstance(item, CachedSampleRef):
        item = torch.load(item.path, map_location="cpu", weights_only=False)
    elif isinstance(item, Path):
        item = torch.load(str(item), map_location="cpu", weights_only=False)
    elif isinstance(item, str):
        item = torch.load(item, map_location="cpu", weights_only=False)
    elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], (str, Path)):
        item = torch.load(str(item[0]), map_location="cpu", weights_only=False)

    if isinstance(item, dict) and "edge_attr" in item:
        item["edge_attr"] = _normalize_edge_attr(item["edge_attr"])
    elif hasattr(item, "edge_attr"):
        edge_attr = getattr(item, "edge_attr", None)
        if edge_attr is not None:
            setattr(item, "edge_attr", _normalize_edge_attr(edge_attr))
    return item


def _limit_split(items: List[Any], max_graphs: int) -> List[Any]:
    if max_graphs > 0:
        return items[:max_graphs]
    return items


def _sample_label0(sample: Any) -> float | None:
    if isinstance(sample, CachedSampleRef):
        return sample.label0
    if isinstance(sample, tuple) and len(sample) > 1 and isinstance(sample[0], (str, Path)):
        try:
            return float(sample[1])
        except Exception:
            return None
    if isinstance(sample, dict) and "y" in sample:
        y = sample["y"].reshape(-1)
        if y.numel() == 0:
            return None
        return float(y[0].item())
    if hasattr(sample, "y"):
        y = getattr(sample, "y")
        if y is None:
            return None
        y = y.reshape(-1)
        if y.numel() == 0:
            return None
        return float(y[0].item())
    return None



def _iter_sample_items(items: Any):
    if isinstance(items, dict):
        for value in items.values():
            yield from _iter_sample_items(value)
        return
    if hasattr(items, "samples") and not isinstance(items, (CachedSampleRef, Path, str)):
        yield from _iter_sample_items(getattr(items, "samples"))
        return
    if isinstance(items, tuple) and len(items) > 0 and isinstance(items[0], (str, Path)):
        yield items
        return
    if isinstance(items, (list, tuple)):
        for item in items:
            yield from _iter_sample_items(item)
        return
    yield items


def _edge_attr_feature_dim(sample: Any) -> int:
    edge_attr = sample.get("edge_attr") if isinstance(sample, dict) else getattr(sample, "edge_attr", None)
    if edge_attr is None:
        return 0
    if not torch.is_tensor(edge_attr):
        try:
            edge_attr = torch.as_tensor(edge_attr)
        except Exception:
            return 0
    if edge_attr.numel() == 0:
        return 0
    if edge_attr.dim() <= 1:
        return 1
    return int(edge_attr.size(-1))


def infer_edge_attr_dim(items: Any) -> int:
    for item in _iter_sample_items(items):
        sample = _load_sample_item(item)
        dim = _edge_attr_feature_dim(sample)
        if dim > 0:
            return dim
    return 0


def _scalar_labels(items: List[Any]) -> List[int] | None:
    labels: List[int] = []
    for sample in items:
        y0 = _sample_label0(sample)
        if y0 is None:
            return None
        labels.append(int(round(float(y0))))
    return labels


def summarize_split_items(items: List[Any], task_type: str | None = None) -> Dict[str, object]:
    summary: Dict[str, object] = {"num_graphs": len(items)}
    if not items:
        summary["single_class"] = True
        return summary

    labels = _scalar_labels(items) if task_type in {"binary", "multiclass"} else None
    if labels is None:
        first = items[0]
        if isinstance(first, dict):
            summary["target_dim"] = int(first["y"].numel())
        else:
            summary["target_dim"] = "unknown"
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


def summarize_splits(splits: Dict[str, List[Any]], task_type: str | None = None) -> Dict[str, Dict[str, object]]:
    return {split_name: summarize_split_items(split_items, task_type=task_type) for split_name, split_items in splits.items()}


def graph_to_sample(
    data,
    in_dim: int,
    max_motifs_per_anchor: int,
    y_mode: str = "binary",
    pos_k: int = 0,
    preserve_nan_mask: bool = False,
) -> Dict[str, torch.Tensor]:
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

    edge_attr = getattr(data, "edge_attr", None)

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
        yv = y.float().view(-1)
        if preserve_nan_mask:
            valid_mask = torch.isfinite(yv)
            yy = torch.where(valid_mask, yv, torch.zeros_like(yv))
        else:
            yy = torch.nan_to_num(yv, nan=0.0)
    else:
        raise ValueError(y_mode)

    sample = sample_from_edge_index(
        edge_index=edge_index,
        num_nodes=n,
        x=x,
        y=yy,
        max_motifs_per_anchor=max_motifs_per_anchor,
        pos_k=pos_k,
        edge_attr=edge_attr,
    )
    if y_mode in {"multilabel", "vector"} and preserve_nan_mask:
        sample.y_mask = valid_mask.bool()
    return sample


def _processed_cache_dir(args, cache_tag: str) -> Path:
    root = Path(getattr(args, "dataset_root", "data")).expanduser()
    key = "|".join(
        [
            "processed_v1",
            cache_tag,
            str(int(getattr(args, "in_dim", 32))),
            str(int(getattr(args, "max_motifs_per_anchor", 8))),
            str(int(getattr(args, "pos_k", 0))),
        ]
    )
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return root / "protocol_cache" / "processed" / cache_tag / digest


def _cached_transform_split(
    args,
    cache_tag: str,
    split_name: str,
    dataset,
    indices: List[int],
    *,
    y_mode: str,
    preserve_nan_mask: bool = False,
) -> List[CachedSampleRef]:
    split_dir = _processed_cache_dir(args, cache_tag) / split_name
    meta_path = split_dir / "meta.pt"
    data_files = [split_dir / f"s_{i}.pt" for i in range(len(indices))]
    if meta_path.exists() and all(p.exists() for p in data_files):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        labels = meta.get("labels0", [None] * len(indices))
        return [CachedSampleRef(path=str(path), label0=(None if labels[i] is None else float(labels[i]))) for i, path in enumerate(data_files)]

    split_dir.mkdir(parents=True, exist_ok=True)
    refs: List[CachedSampleRef] = []
    labels0: List[float | None] = []
    for local_i, idx in enumerate(indices):
        sample = graph_to_sample(
            dataset[idx],
            args.in_dim,
            args.max_motifs_per_anchor,
            y_mode=y_mode,
            pos_k=getattr(args, "pos_k", 0),
            preserve_nan_mask=preserve_nan_mask,
        )
        path = split_dir / f"s_{local_i}.pt"
        torch.save(sample, path)
        y0 = _sample_label0(sample)
        labels0.append(y0)
        refs.append(CachedSampleRef(path=str(path), label0=y0))
    torch.save({"num_samples": len(indices), "labels0": labels0}, meta_path)
    return refs


def _canonical_brec_category(category: Any) -> str:
    if category is None:
        return "unknown"
    raw = str(category).strip().lower().replace("-", "_").replace(" ", "_")
    alias = {
        "basic": "Basic",
        "regular": "Regular",
        "extension": "Extension",
        "cfi": "CFI",
        "4_vertex_condition": "4-Vertex_Condition",
        "4vertex_condition": "4-Vertex_Condition",
        "distance_regular": "Distance_Regular",
    }
    return alias.get(raw, str(category))


def _coerce_brec_entries(payload: Any) -> List[Tuple[Any, str]]:
    entries: List[Tuple[Any, str]] = []
    if isinstance(payload, dict):
        data_list = payload.get("data_list")
        categories = payload.get("categories") or payload.get("brec_categories")
        if data_list is None:
            raise ValueError("BREC payload dict must contain `data_list`.")
        if categories is not None and len(categories) == len(data_list):
            for d, c in zip(data_list, categories):
                entries.append((d, _canonical_brec_category(c)))
            return entries
        for d in data_list:
            cat = getattr(d, "brec_category", None) or getattr(d, "category", None)
            entries.append((d, _canonical_brec_category(cat)))
        return entries

    if not isinstance(payload, list):
        raise ValueError("BREC file must contain a list or dict payload.")

    for item in payload:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            entries.append((item[0], _canonical_brec_category(item[1])))
            continue
        if isinstance(item, dict) and "data" in item:
            entries.append((item["data"], _canonical_brec_category(item.get("category"))))
            continue
        cat = getattr(item, "brec_category", None) or getattr(item, "category", None)
        entries.append((item, _canonical_brec_category(cat)))
    return entries


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


def get_k_fold_splits(
    items: List[Dict[str, torch.Tensor]],
    num_folds: int,
    seed: int,
    task_type: str | None = None,
) -> List[Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]]:
    labels = _scalar_labels(items)
    idx = np.arange(len(items))
    
    if task_type in {"binary", "multiclass"} and labels is not None and len(set(labels)) > 1:
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = list(kf.split(idx, labels))
    else:
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
        folds = list(kf.split(idx))
        
    res = []
    for train_val_idx, test_idx in folds:
        tv_items = [items[int(i)] for i in train_val_idx]
        test_items = [items[int(i)] for i in test_idx]
        
        # Split train_val into train and val (roughly 80/10/10 split if num_folds=10)
        tr, va, _ = split_items(tv_items, seed=seed, train_ratio=0.88, val_ratio=0.12, task_type=task_type)
        res.append((tr, va, test_items))
    return res


def _make_stage1_wedge_triangle(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    for _ in range(getattr(args, "max_graphs", 256) if getattr(args, "max_graphs", 0) > 0 else 256):
        n = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        p = float(args.edge_prob)
        adj = torch.from_numpy((rng.random((n, n)) < p)).bool()
        adj = torch.triu(adj, diagonal=1)
        adj = adj | adj.t()
        adj_f = adj.float()
        tri = ((adj_f @ adj_f) * adj_f).sum() / 6.0
        wedges = 0.5 * ((adj.sum(1) * (adj.sum(1) - 1)).sum() - 6 * tri)
        y = torch.tensor([1.0 if tri > wedges * 0.08 else 0.0])
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _make_stage1_triangle_regression(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    for _ in range(getattr(args, "max_graphs", 256) if getattr(args, "max_graphs", 0) > 0 else 256):
        n = int(rng.integers(args.min_nodes, args.max_nodes + 1))
        p = float(args.edge_prob)
        adj = torch.from_numpy((rng.random((n, n)) < p)).bool()
        adj = torch.triu(adj, diagonal=1)
        adj = adj | adj.t()
        adj_f = adj.float()
        tri = float((((adj_f @ adj_f) * adj_f).sum() / 6.0).item())
        y = torch.tensor([tri / max(1.0, n)])
        x = torch.randn((n, args.in_dim), dtype=torch.float32)
        out.append(sample_from_adj(adj, x, y, args.max_motifs_per_anchor))
    return out


def _make_stage1_cycle_parity(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed)
    m = getattr(args, "max_graphs", 256) if getattr(args, "max_graphs", 0) > 0 else 256
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
    assign_bits = _assignment_bits(n_vars)
    sat = np.ones(assign_bits.shape[0], dtype=np.bool_)
    for (a, b, c), (sa, sb, sc) in zip(clauses, signs):
        la = assign_bits[:, a] if sa > 0 else (1 - assign_bits[:, a])
        lb = assign_bits[:, b] if sb > 0 else (1 - assign_bits[:, b])
        lc = assign_bits[:, c] if sc > 0 else (1 - assign_bits[:, c])
        sat &= (la | lb | lc).astype(np.bool_)
        if not sat.any():
            return False
    return bool(sat.any())


@lru_cache(maxsize=None)
def _assignment_bits(n_vars: int) -> np.ndarray:
    masks = np.arange(1 << n_vars, dtype=np.uint32)
    shifts = np.arange(n_vars, dtype=np.uint32)
    return ((masks[:, None] >> shifts[None, :]) & 1).astype(np.uint8)


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
    m = getattr(args, "max_graphs", 128) if getattr(args, "max_graphs", 0) > 0 else 128
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
    assign_bits = _assignment_bits(n_vars)
    sat = np.ones(assign_bits.shape[0], dtype=np.bool_)
    for (a, b, c), r in zip(clauses, rhs):
        val = assign_bits[:, a] ^ assign_bits[:, b] ^ assign_bits[:, c]
        sat &= (val == r)
        if not sat.any():
            return False
    return bool(sat.any())


def _make_stage1_xorsat(args) -> List[Dict[str, torch.Tensor]]:
    out = []
    rng = np.random.default_rng(args.seed + 17)
    m = getattr(args, "max_graphs", 128) if getattr(args, "max_graphs", 0) > 0 else 128
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
    m = getattr(args, "max_graphs", 256) if getattr(args, "max_graphs", 0) > 0 else 256
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

    root = Path(getattr(args, "dataset_root", "data")).expanduser() / "pyg"
    parts = {
        "train": GNNBenchmarkDataset(root=str(root), name="CSL", split="train"),
        "val": GNNBenchmarkDataset(root=str(root), name="CSL", split="val"),
        "test": GNNBenchmarkDataset(root=str(root), name="CSL", split="test"),
    }
    splits: Dict[str, List[Dict[str, torch.Tensor]]] = {}
    for split, dataset in parts.items():
        items = _limit_split(list(dataset), getattr(args, "max_graphs", 0))
        splits[split] = [graph_to_sample(d, args.in_dim, args.max_motifs_per_anchor, y_mode="multiclass", pos_k=getattr(args, "pos_k", 0)) for d in items]
    nclass = int(max(int(s["y"].item()) for split in splits.values() for s in split) + 1)
    return splits, nclass


def _load_stage2_brec(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    brec_file = getattr(args, "brec_file", "")
    if not brec_file:
        raise ValueError("--brec_file required for stage2_brec")

    payload = torch.load(Path(brec_file).expanduser(), map_location="cpu", weights_only=False)
    entries = _coerce_brec_entries(payload)
    if getattr(args, "max_graphs", 0) > 0:
        entries = entries[: args.max_graphs]

    categories = sorted({cat for _, cat in entries})
    cat_to_id = {cat: idx for idx, cat in enumerate(categories)}
    setattr(args, "_brec_category_names", {idx: cat for cat, idx in cat_to_id.items()})

    samples = []
    for data_obj, cat_name in entries:
        sample = graph_to_sample(
            data_obj,
            args.in_dim,
            args.max_motifs_per_anchor,
            y_mode="binary",
            pos_k=getattr(args, "pos_k", 0),
        )
        sample.brec_category_id = torch.tensor([cat_to_id[cat_name]], dtype=torch.long)
        samples.append(sample)
    return samples, 2


def _load_stage3_zinc(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import ZINC

    root = Path(getattr(args, "dataset_root", "data")).expanduser() / "pyg"
    parts = {s: ZINC(root=str(root), subset=True, split=s) for s in ["train", "val", "test"]}
    splits: Dict[str, List[CachedSampleRef]] = {}
    for split, dataset in parts.items():
        idxs = list(range(len(dataset)))
        idxs = _limit_split(idxs, getattr(args, "max_graphs", 0))
        splits[split] = _cached_transform_split(
            args,
            cache_tag="stage3_zinc",
            split_name=split,
            dataset=dataset,
            indices=idxs,
            y_mode="regression",
        )
    return splits, 1


def _load_stage3_molhiv(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from ogb.graphproppred import PygGraphPropPredDataset

    ds = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(Path(getattr(args, "dataset_root", "data")).expanduser() / "ogb"))
    idx = ds.get_idx_split()
    splits: Dict[str, List[CachedSampleRef]] = {}
    for split_name, split_idx in {"train": idx["train"], "val": idx["valid"], "test": idx["test"]}.items():
        order = split_idx.tolist()
        order = order[: args.max_graphs] if getattr(args, "max_graphs", 0) > 0 else order
        splits[split_name] = _cached_transform_split(
            args,
            cache_tag="stage3_molhiv",
            split_name=split_name,
            dataset=ds,
            indices=order,
            y_mode="binary",
        )
    return splits, 2


def _load_stage3_molpcba(args) -> Tuple[Dict[str, List[CachedSampleRef]], int]:
    from ogb.graphproppred import PygGraphPropPredDataset

    ds = PygGraphPropPredDataset(name="ogbg-molpcba", root=str(Path(getattr(args, "dataset_root", "data")).expanduser() / "ogb"))
    idx = ds.get_idx_split()
    splits: Dict[str, List[CachedSampleRef]] = {}
    num_targets = int(ds.num_tasks)
    for split_name, split_idx in {"train": idx["train"], "val": idx["valid"], "test": idx["test"]}.items():
        order = split_idx.tolist()
        order = order[: args.max_graphs] if getattr(args, "max_graphs", 0) > 0 else order
        splits[split_name] = _cached_transform_split(
            args,
            cache_tag="stage3_molpcba",
            split_name=split_name,
            dataset=ds,
            indices=order,
            y_mode="multilabel",
            preserve_nan_mask=True,
        )
    return splits, num_targets


def _load_stage3_peptides(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import LRGBDataset

    root = Path(getattr(args, "dataset_root", "data")).expanduser() / "pyg"
    parts = {s: LRGBDataset(root=str(root), name="Peptides-struct", split=s) for s in ["train", "val", "test"]}
    num_targets = int(parts["train"][0].y.view(-1).numel()) if len(parts["train"]) > 0 else 1
    splits: Dict[str, List[CachedSampleRef]] = {}
    for split, dataset in parts.items():
        idxs = list(range(len(dataset)))
        idxs = _limit_split(idxs, getattr(args, "max_graphs", 0))
        splits[split] = _cached_transform_split(
            args,
            cache_tag="stage3_peptides_struct",
            split_name=split,
            dataset=dataset,
            indices=idxs,
            y_mode="vector",
        )
    return splits, num_targets


def _load_stage3_peptides_func(args) -> Tuple[Dict[str, List[CachedSampleRef]], int]:
    from torch_geometric.datasets import LRGBDataset

    root = Path(getattr(args, "dataset_root", "data")).expanduser() / "pyg"
    out = {}
    train_ds = LRGBDataset(root=str(root), name="Peptides-func", split="train")
    num_targets = int(train_ds[0].y.view(-1).numel()) if len(train_ds) > 0 else 1
    split_datasets = {"train": train_ds}
    split_datasets["val"] = LRGBDataset(root=str(root), name="Peptides-func", split="val")
    split_datasets["test"] = LRGBDataset(root=str(root), name="Peptides-func", split="test")

    for s, dataset in split_datasets.items():
        idxs = list(range(len(dataset)))
        idxs = _limit_split(idxs, getattr(args, "max_graphs", 0))
        out[s] = _cached_transform_split(
            args,
            cache_tag="stage3_peptides_func",
            split_name=s,
            dataset=dataset,
            indices=idxs,
            y_mode="multilabel",
            preserve_nan_mask=True,
        )
    return out, num_targets


def _load_stage4_tu(args) -> Tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import TUDataset

    root = Path(getattr(args, "dataset_root", "data")).expanduser() / "pyg"
    ds = TUDataset(root=str(root), name=getattr(args, "tu_name", "MUTAG"), use_node_attr=True)
    idxs = list(range(len(ds)))
    idxs = _limit_split(idxs, getattr(args, "max_graphs", 0))
    samples = _cached_transform_split(
        args,
        cache_tag=f"stage4_tu_{getattr(args, 'tu_name', 'MUTAG').lower()}",
        split_name="all",
        dataset=ds,
        indices=idxs,
        y_mode="multiclass",
    )
    labels = [ref.label0 for ref in samples if ref.label0 is not None]
    nclass = int(max(int(round(float(y))) for y in labels) + 1) if labels else 1
    return samples, nclass


def _load_stage4_anomaly(args, name: str) -> Tuple[Dict[str, List[Dict[str, torch.Tensor]]], int]:
    from pygod.utils import load_data
    from torch_geometric.utils import remove_self_loops
    from get.data.synthetic import sample_from_edge_index
    from sklearn.model_selection import StratifiedShuffleSplit

    cache_version = "v2"
    root = getattr(args, "dataset_root", "data")
    cache_key = "|".join(
        [
            cache_version,
            name.lower(),
            str(Path(root).expanduser().resolve()),
            str(int(getattr(args, "pos_k", 0))),
            str(int(args.in_dim)),
            str(int(args.max_motifs_per_anchor)),
            str(int(getattr(args, "seed", 42))),
        ]
    )
    cache_digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    cache_dir = Path(root).expanduser() / "protocol_cache" / "anomaly"
    cache_path = cache_dir / f"{cache_digest}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get("splits"), dict):
            return payload["splits"], int(payload.get("num_classes", 2))

    alias_map = {
        "amazon": ["amazon", "inj_amazon"],
        "yelpchi": ["yelpchi", "weibo"],
        "tfinance": ["tfinance", "weibo"],
        "tsocial": ["tsocial", "reddit"],
    }
    candidates = alias_map.get(name.lower(), [name, name.lower(), name.upper()])
    data = None
    for nm in candidates:
        try:
            data = load_data(nm, cache_dir=str(Path(root).expanduser() / "pygod"))
            break
        except Exception:
            pass
    if data is None:
        raise RuntimeError(f"Unable to load anomaly dataset {name} via PyGOD load_data")

    y = data.y.view(-1).long()
    
    y_np = y.numpy()
    seed = getattr(args, "seed", 42)
    train_ratio = getattr(args, "train_ratio", 0.7)
    val_ratio = getattr(args, "val_ratio", 0.15)
    
    try:
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=max(0.001, 1.0 - train_ratio), random_state=seed)
        train_idx, temp_idx = next(sss1.split(np.zeros(len(y_np)), y_np))
        
        rel_val_size = val_ratio / max(1e-7, (1.0 - train_ratio))
        rel_val_size = min(0.99, max(0.01, rel_val_size))
        
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=max(0.001, 1.0 - rel_val_size), random_state=seed)
        val_rel_idx, test_rel_idx = next(sss2.split(np.zeros(len(temp_idx)), y_np[temp_idx]))
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]
    except ValueError:
        from sklearn.model_selection import ShuffleSplit
        ss1 = ShuffleSplit(n_splits=1, test_size=max(0.001, 1.0 - train_ratio), random_state=seed)
        train_idx, temp_idx = next(ss1.split(np.zeros(len(y_np))))
        
        rel_val_size = val_ratio / max(1e-7, (1.0 - train_ratio))
        rel_val_size = min(0.99, max(0.01, rel_val_size))
        
        ss2 = ShuffleSplit(n_splits=1, test_size=max(0.001, 1.0 - rel_val_size), random_state=seed)
        val_rel_idx, test_rel_idx = next(ss2.split(np.zeros(len(temp_idx))))
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True

    edge_index, _ = remove_self_loops(data.edge_index.long())
    x = data.x.float()
    n = x.size(0)
    feat = x[:, : args.in_dim] if x.size(1) >= args.in_dim else torch.cat([x, torch.zeros((n, args.in_dim - x.size(1)))], dim=1)
    yy = torch.tensor([1.0 if int(yi) > 0 else 0.0 for yi in y.tolist()])
    edge_attr = getattr(data, "edge_attr", None)

    sample = sample_from_edge_index(edge_index, n, feat, yy, args.max_motifs_per_anchor, edge_attr=edge_attr)
    
    train_sample = sample.clone()
    train_sample.mask = train_mask
    val_sample = sample.clone()
    val_sample.mask = val_mask
    test_sample = sample.clone()
    test_sample.mask = test_mask
    
    splits = {"train": [train_sample], "val": [val_sample], "test": [test_sample]}
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"cache_version": cache_version, "splits": splits, "num_classes": 2}, cache_path)
    return splits, 2


def _apply_leakage_control(
    graphs,  # List of graph samples (dict or GraphSampleData objects)
    similarity_fn,
):
    """Apply matched pairing to graphs based on similarity function.
    
    Creates balanced pairs of positive and negative graphs with similar statistics,
    keeping only the best-matched pairs.
    """
    graphs_with_idx_label = []
    for i, g in enumerate(graphs):
        # Handle both dict and object attribute access
        if isinstance(g, dict):
            label = float(g["y"][0].item())
            num_nodes = g["x"].size(0)
            # Convert sample dict to adjacency for statistics
            if "edge_index" in g:
                ei = g["edge_index"].long()
                adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
                if ei.size(1) > 0:
                    adj[ei[0], ei[1]] = True
            else:
                adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        else:
            # GraphSampleData or similar object
            label = float(g.y[0].item())
            num_nodes = g.x.size(0)
            # Check for edge_index or c_2 (motif membership)
            if hasattr(g, "edge_index") and g.edge_index is not None:
                ei = g.edge_index.long()
                adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
                if ei.numel() > 0 and ei.size(1) > 0:
                    adj[ei[0], ei[1]] = True
            else:
                # For motif-based representation, reconstruct from c_2, u_2, etc.
                adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
                if hasattr(g, "c_2") and hasattr(g, "u_2"):
                    c_2, u_2 = g.c_2.long(), g.u_2.long()
                    valid = (c_2 < num_nodes) & (u_2 < num_nodes)
                    if valid.any():
                        adj[c_2[valid], u_2[valid]] = True
        
        graphs_with_idx_label.append((i, adj, label))
    
    # Create matched pairs
    pairs = create_matched_pairs(graphs_with_idx_label, similarity_fn=similarity_fn)
    
    # Keep only matched samples in order
    result = []
    seen = set()
    for pos_idx, neg_idx, _, _ in pairs:
        if pos_idx not in seen:
            result.append(graphs[pos_idx])
            seen.add(pos_idx)
        if neg_idx not in seen:
            result.append(graphs[neg_idx])
            seen.add(neg_idx)
    
    return result


def _make_stage1_wedge_triangle_matched(args) -> List[Dict[str, torch.Tensor]]:
    """Wedge/triangle with full leakage control (matched degree, edge, and two-hop counts)."""
    graphs = _make_stage1_wedge_triangle(args)
    return _apply_leakage_control(graphs, similarity_fn=similarity_aggregate)


def _make_stage1_wedge_triangle_degree_only(args) -> List[Dict[str, torch.Tensor]]:
    """Wedge/triangle matched on degree histogram only (leakage baseline)."""
    graphs = _make_stage1_wedge_triangle(args)
    return _apply_leakage_control(graphs, similarity_fn=similarity_euclidean_degree)


def _make_stage1_wedge_triangle_edge_only(args) -> List[Dict[str, torch.Tensor]]:
    """Wedge/triangle matched on edge count only (leakage baseline)."""
    graphs = _make_stage1_wedge_triangle(args)
    
    # Filter by edge count only
    def edge_count_similarity(adj1, adj2):
        from get.data.leakage_control import compute_edge_count
        e1 = float(compute_edge_count(adj1))
        e2 = float(compute_edge_count(adj2))
        return abs(e1 - e2) / max(e1, e2, 1.0)
    
    return _apply_leakage_control(graphs, similarity_fn=edge_count_similarity)


def _make_stage1_wedge_triangle_twohop_only(args) -> List[Dict[str, torch.Tensor]]:
    """Wedge/triangle matched on two-hop count only (leakage baseline)."""
    graphs = _make_stage1_wedge_triangle(args)
    
    # Filter by two-hop count only
    def twohop_similarity(adj1, adj2):
        from get.data.leakage_control import compute_two_hop_count
        t1 = compute_two_hop_count(adj1)
        t2 = compute_two_hop_count(adj2)
        return abs(t1 - t2) / max(t1, t2, 1e-6)
    
    return _apply_leakage_control(graphs, similarity_fn=twohop_similarity)


def _make_stage1_cycle_parity_matched(args) -> List[Dict[str, torch.Tensor]]:
    """Cycle parity with full leakage control."""
    graphs = _make_stage1_cycle_parity(args)
    return _apply_leakage_control(graphs, similarity_fn=similarity_aggregate)


def _make_stage1_cycle_parity_degree_only(args) -> List[Dict[str, torch.Tensor]]:
    """Cycle parity matched on degree only."""
    graphs = _make_stage1_cycle_parity(args)
    return _apply_leakage_control(graphs, similarity_fn=similarity_euclidean_degree)


def _make_stage1_cycle_parity_edge_only(args) -> List[Dict[str, torch.Tensor]]:
    """Cycle parity matched on edge count only."""
    graphs = _make_stage1_cycle_parity(args)
    
    def edge_count_similarity(adj1, adj2):
        from get.data.leakage_control import compute_edge_count
        e1 = float(compute_edge_count(adj1))
        e2 = float(compute_edge_count(adj2))
        return abs(e1 - e2) / max(e1, e2, 1.0)
    
    return _apply_leakage_control(graphs, similarity_fn=edge_count_similarity)


def _make_stage1_max3sat_matched(args) -> List[Dict[str, torch.Tensor]]:
    """Max-3-SAT with full leakage control."""
    graphs = _make_stage1_max3sat(args)
    return _apply_leakage_control(graphs, similarity_fn=similarity_aggregate)


def _make_stage1_max3sat_degree_only(args) -> List[Dict[str, torch.Tensor]]:
    """Max-3-SAT matched on degree only."""
    graphs = _make_stage1_max3sat(args)
    return _apply_leakage_control(graphs, similarity_fn=similarity_euclidean_degree)


def _make_stage1_max3sat_edge_only(args) -> List[Dict[str, torch.Tensor]]:
    """Max-3-SAT matched on edge count only."""
    graphs = _make_stage1_max3sat(args)
    
    def edge_count_similarity(adj1, adj2):
        from get.data.leakage_control import compute_edge_count
        e1 = float(compute_edge_count(adj1))
        e2 = float(compute_edge_count(adj2))
        return abs(e1 - e2) / max(e1, e2, 1.0)
    
    return _apply_leakage_control(graphs, similarity_fn=edge_count_similarity)


def build_dataset(task: str, args) -> Tuple[Union[List[Dict[str, torch.Tensor]], Dict[str, List[Dict[str, torch.Tensor]]]], int]:
    if task == "stage1_wedge_triangle":
        res = _make_stage1_wedge_triangle(args), 2
    elif task == "stage1_wedge_triangle_matched":
        res = _make_stage1_wedge_triangle_matched(args), 2
    elif task == "stage1_wedge_triangle_degree_only":
        res = _make_stage1_wedge_triangle_degree_only(args), 2
    elif task == "stage1_wedge_triangle_edge_only":
        res = _make_stage1_wedge_triangle_edge_only(args), 2
    elif task == "stage1_wedge_triangle_twohop_only":
        res = _make_stage1_wedge_triangle_twohop_only(args), 2
    elif task == "stage1_triangle_regression":
        res = _make_stage1_triangle_regression(args), 1
    elif task == "stage1_cycle_parity":
        res = _make_stage1_cycle_parity(args), 2
    elif task == "stage1_cycle_parity_matched":
        res = _make_stage1_cycle_parity_matched(args), 2
    elif task == "stage1_cycle_parity_degree_only":
        res = _make_stage1_cycle_parity_degree_only(args), 2
    elif task == "stage1_cycle_parity_edge_only":
        res = _make_stage1_cycle_parity_edge_only(args), 2
    elif task == "stage1_max3sat":
        res = _make_stage1_max3sat(args), 2
    elif task == "stage1_max3sat_matched":
        res = _make_stage1_max3sat_matched(args), 2
    elif task == "stage1_max3sat_degree_only":
        res = _make_stage1_max3sat_degree_only(args), 2
    elif task == "stage1_max3sat_edge_only":
        res = _make_stage1_max3sat_edge_only(args), 2
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
    elif task == "stage3_molpcba":
        res = _load_stage3_molpcba(args)
    elif task == "stage3_peptides_struct_probe":
        res = _load_stage3_peptides(args)
    elif task == "stage3_peptides_func_probe":
        res = _load_stage3_peptides_func(args)
    elif task in {"stage4_tu_classification", "tu"}:
        res = _load_stage4_tu(args)
    elif task == "stage4_tu_proteins":
        args.tu_name = "PROTEINS"
        res = _load_stage4_tu(args)
    elif task == "stage4_tu_nci1":
        args.tu_name = "NCI1"
        res = _load_stage4_tu(args)
    elif task == "stage4_tu_nci109":
        args.tu_name = "NCI109"
        res = _load_stage4_tu(args)
    elif task == "stage4_tu_dd":
        args.tu_name = "DD"
        res = _load_stage4_tu(args)
    elif task == "stage4_tu_enzymes":
        args.tu_name = "ENZYMES"
        res = _load_stage4_tu(args)
    elif task == "stage4_tu_mutagenicity":
        args.tu_name = "Mutagenicity"
        res = _load_stage4_tu(args)
    elif task == "stage4_tu_frankenstein":
        args.tu_name = "FRANKENSTEIN"
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
        # Try to treat as a TU dataset name directly
        try:
            args.tu_name = task.upper()
            res = _load_stage4_tu(args)
        except Exception:
            raise ValueError(f"Unknown task: {task}")
        
    return res
