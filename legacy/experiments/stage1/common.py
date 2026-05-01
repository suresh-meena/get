"""Shared helpers for Stage 1 experiments."""
from __future__ import annotations

import random
import networkx as nx
import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter

from get import FullGET, PairwiseGET
from get.data import add_structural_node_features
from get.data.motif_extraction import _numba_edges_to_csr, get_incidence_matrices
from get.data.positional import get_rwse
from experiments.shared.common import GETTrainer


def count_trainable_params(model) -> int:
    total = 0
    for param in model.parameters():
        if not param.requires_grad or isinstance(param, UninitializedParameter):
            continue
        total += int(param.numel())
    return total


def compute_rwse_features(graph: nx.Graph, rwse_k: int) -> torch.Tensor | None:
    if int(rwse_k) <= 0:
        return None

    adjacency = nx.adjacency_matrix(graph).toarray().astype(float)
    degrees = np.sum(adjacency, axis=1)
    inv_degrees = np.zeros_like(degrees)
    nonzero = degrees > 0
    inv_degrees[nonzero] = 1.0 / degrees[nonzero]

    transition = adjacency @ np.diag(inv_degrees)
    current = np.eye(graph.number_of_nodes())
    traces = []
    for _ in range(int(rwse_k)):
        current = current @ transition
        traces.append(np.diag(current))

    return torch.from_numpy(np.stack(traces, axis=1)).float()


def _graph_to_csr(graph_item: dict) -> tuple[int, np.ndarray, np.ndarray]:
    num_nodes = int(graph_item["x"].size(0))
    edges = graph_item.get("edges", [])
    if len(edges) == 0:
        edge_arr = np.empty((0, 2), dtype=np.int64)
    else:
        edge_arr = np.ascontiguousarray(np.array(edges, dtype=np.int64).reshape(-1, 2))
    indptr, indices = _numba_edges_to_csr(num_nodes, edge_arr)
    return num_nodes, indptr, indices


def _canonical_motif_mask(c_3: torch.Tensor, u_3: torch.Tensor, v_3: torch.Tensor) -> torch.Tensor:
    # Stage-1 tests should match the model definition: every node can anchor
    # its own neighbor-pair motifs. Do not drop triangle copies anchored at
    # other vertices, because that changes the intended order-3 support.
    return torch.ones_like(c_3, dtype=torch.bool)


def _select_support_indices(
    c_3: torch.Tensor,
    t_tau: torch.Tensor,
    *,
    support_mode: str,
    max_motifs: int | None,
    seed: int,
) -> torch.Tensor:
    if c_3.numel() == 0:
        return torch.zeros(0, dtype=torch.long)

    mode = str(support_mode)
    unlimited = max_motifs is None or int(max_motifs) < 0
    if mode in {"full", "oracle"} or unlimited:
        return torch.arange(c_3.numel(), dtype=torch.long)

    budget = int(max_motifs)
    if budget <= 0:
        return torch.zeros(0, dtype=torch.long)

    c_np = c_3.detach().cpu().numpy()
    t_np = t_tau.detach().cpu().numpy().astype(np.float32)
    keep_parts: list[torch.Tensor] = []

    for center in np.unique(c_np):
        center_idx = np.flatnonzero(c_np == center)
        if center_idx.size <= budget:
            chosen = center_idx
        else:
            if mode in {"exact", "topB_closed_first", "random"}:
                scores = t_np[center_idx]
            elif mode == "topB_open_first":
                scores = 1.0 - t_np[center_idx]
            else:
                raise ValueError(f"Unsupported support_mode='{support_mode}'.")

            kth = np.partition(scores, -budget)[-budget]
            chosen = center_idx[scores >= kth]

        if mode == "random" and chosen.size > 1:
            # Seeded cyclic shift gives deterministic but different orderings across seeds.
            shift = int(seed) % chosen.size
            chosen = np.roll(chosen, shift)

        keep_parts.append(torch.as_tensor(chosen, dtype=torch.long))

    if not keep_parts:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat(keep_parts, dim=0)


def prepare_stage1_graph(
    graph: dict,
    *,
    feature_mode: str = "core",
    support_mode: str = "exact",
    max_motifs: int | None = None,
    rwse_k: int = 0,
    seed: int = 0,
) -> dict:
    item = dict(graph)
    if "x" not in item:
        raise ValueError("Graph item must include 'x'.")

    x = item["x"]
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    if x.dtype != torch.float32:
        x = x.float()
    item["x"] = x.clone()

    num_nodes, indptr, indices = _graph_to_csr(item)
    c_2, u_2, c_3_raw, u_3_raw, v_3_raw, t_tau_raw = get_incidence_matrices(
        num_nodes,
        indptr,
        indices,
        max_motifs_per_node=None,
    )

    canonical_mask = _canonical_motif_mask(c_3_raw, u_3_raw, v_3_raw)
    c_3 = c_3_raw[canonical_mask]
    u_3 = u_3_raw[canonical_mask]
    v_3 = v_3_raw[canonical_mask]
    t_tau = t_tau_raw[canonical_mask]

    keep_idx = _select_support_indices(
        c_3,
        t_tau,
        support_mode=support_mode,
        max_motifs=max_motifs,
        seed=seed,
    )
    c_3_keep = c_3[keep_idx]
    u_3_keep = u_3[keep_idx]
    v_3_keep = v_3[keep_idx]
    t_tau_keep = t_tau[keep_idx]

    item["c_2"] = c_2
    item["u_2"] = u_2
    item["c_3"] = c_3_keep
    item["u_3"] = u_3_keep
    item["v_3"] = v_3_keep
    item["t_tau"] = t_tau_keep

    candidate_closed = int((t_tau == 1).sum().item())
    candidate_open = int((t_tau == 0).sum().item())
    retained_closed = int((t_tau_keep == 1).sum().item())
    retained_open = int((t_tau_keep == 0).sum().item())
    candidate_total = int(t_tau.numel())
    retained_total = int(t_tau_keep.numel())

    item["candidate_motif_count"] = candidate_total
    item["retained_motif_count"] = retained_total
    item["candidate_closed"] = candidate_closed
    item["candidate_open"] = candidate_open
    item["retained_closed"] = retained_closed
    item["retained_open"] = retained_open
    item["retained_fraction"] = float(retained_total / candidate_total) if candidate_total > 0 else 1.0

    mode = str(feature_mode)
    if mode == "static_motif":
        degree = torch.as_tensor(np.diff(indptr), dtype=item["x"].dtype, device=item["x"].device)
        open_counts = torch.zeros(num_nodes, dtype=item["x"].dtype, device=item["x"].device)
        closed_counts = torch.zeros(num_nodes, dtype=item["x"].dtype, device=item["x"].device)
        if c_3.numel() > 0:
            open_counts.index_add_(0, c_3, (t_tau == 0).to(dtype=item["x"].dtype))
            closed_counts.index_add_(0, c_3, (t_tau == 1).to(dtype=item["x"].dtype))
        static_feat = torch.stack([degree, open_counts, closed_counts], dim=-1)
        item["x"] = torch.cat([item["x"], static_feat], dim=-1)
    elif mode == "rwse":
        if int(rwse_k) <= 0:
            raise ValueError("feature_mode='rwse' requires rwse_k > 0.")
        rwse = get_rwse(num_nodes, indptr, indices, int(rwse_k)).to(dtype=item["x"].dtype, device=item["x"].device)
        item["rwse"] = rwse
        # Keep PE field populated for downstream code that expects positional tensors.
        item["pe"] = rwse
        item["x"] = torch.cat([item["x"], rwse], dim=-1)
    elif mode != "core":
        raise ValueError(f"Unsupported feature_mode='{feature_mode}'.")

    return item


def summarize_stage1_support(dataset: list[dict]) -> dict:
    summary = {
        "num_graphs": float(len(dataset)),
        "candidate_motif_count": 0.0,
        "retained_motif_count": 0.0,
        "candidate_closed": 0.0,
        "candidate_open": 0.0,
        "retained_closed": 0.0,
        "retained_open": 0.0,
        "retained_fraction": 1.0,
    }
    if not dataset:
        return summary

    for item in dataset:
        summary["candidate_motif_count"] += float(item.get("candidate_motif_count", 0.0))
        summary["retained_motif_count"] += float(item.get("retained_motif_count", 0.0))
        summary["candidate_closed"] += float(item.get("candidate_closed", 0.0))
        summary["candidate_open"] += float(item.get("candidate_open", 0.0))
        summary["retained_closed"] += float(item.get("retained_closed", 0.0))
        summary["retained_open"] += float(item.get("retained_open", 0.0))

    cand = summary["candidate_motif_count"]
    summary["retained_fraction"] = float(summary["retained_motif_count"] / cand) if cand > 0.0 else 1.0
    return summary


def build_stage1_graph_item(
    graph: nx.Graph,
    y: float,
    pair_id: int,
    *,
    rwse_k: int = 0,
    include_degree: bool = False,
    include_motif_counts: bool = False,
    base_x: torch.Tensor | None = None,
) -> dict:
    x = base_x if base_x is not None else torch.ones(graph.number_of_nodes(), 1, dtype=torch.float32)
    item = {
        "x": x,
        "edges": list(graph.edges()),
        "y": torch.tensor([float(y)], dtype=torch.float32),
        "graph_id": int(pair_id),
        "pair_id": int(pair_id),
    }

    if include_degree or include_motif_counts:
        item = add_structural_node_features(
            item,
            include_degree=include_degree,
            include_motif_counts=include_motif_counts,
        )

    rwse = compute_rwse_features(graph, rwse_k)
    if rwse is not None:
        rwse = rwse.to(dtype=item["x"].dtype, device=item["x"].device)
        item["x"] = torch.cat([item["x"], rwse], dim=-1)

    return item


def match_pairwise_width(
    in_dim: int,
    num_classes: int,
    full_d: int,
    *,
    full_kwargs: dict | None = None,
    pairwise_kwargs: dict | None = None,
    min_width: int = 8,
    lower_factor: float = 0.6,
    upper_factor: float = 1.4,
) -> dict:
    full_kwargs = dict(full_kwargs or {})
    pairwise_kwargs = dict(pairwise_kwargs or {})

    target_params = count_trainable_params(FullGET(in_dim, full_d, num_classes, **full_kwargs))
    heuristic = max(min_width, int(round(full_d * 1.73)))
    search_low = max(min_width, int(round(heuristic * lower_factor)))
    search_high = max(search_low, int(round(heuristic * upper_factor)))

    best_width = heuristic
    best_params = None
    best_error = None

    for width in range(search_low, search_high + 1):
        candidate = PairwiseGET(in_dim, width, num_classes, **pairwise_kwargs)
        candidate_params = count_trainable_params(candidate)
        error = abs(candidate_params - target_params)
        if best_error is None or error < best_error:
            best_width = width
            best_params = candidate_params
            best_error = error
            if error == 0:
                break

    return {
        "pairwise_width": int(best_width),
        "pairwise_params": int(best_params if best_params is not None else 0),
        "full_params": int(target_params),
        "relative_error": float((best_error or 0) / max(target_params, 1)),
    }


def match_pairwise_hidden_dim(
    *,
    in_dim: int,
    num_classes: int,
    full_hidden_dim: int,
    common_kwargs: dict | None = None,
    full_kwargs: dict | None = None,
    pairwise_kwargs: dict | None = None,
) -> tuple[int, dict]:
    common = dict(common_kwargs or {})
    full_cfg = dict(common)
    pair_cfg = dict(common)
    if full_kwargs:
        full_cfg.update(full_kwargs)
    if pairwise_kwargs:
        pair_cfg.update(pairwise_kwargs)

    match = match_pairwise_width(
        in_dim=in_dim,
        num_classes=num_classes,
        full_d=full_hidden_dim,
        full_kwargs=full_cfg,
        pairwise_kwargs=pair_cfg,
    )
    return match["pairwise_width"], {
        "target_params": match["full_params"],
        "matched_params": match["pairwise_params"],
        "relative_error": match["relative_error"],
    }


def generate_true_triangle_regression_dataset(num_graphs=2000, n_nodes=24, degree_range=(2, 6), seed=0):
    dataset = []
    rng = random.Random(seed)

    for i in range(num_graphs):
        degree = rng.randint(degree_range[0], degree_range[1])
        base_seed = rng.randint(0, 10**9)

        try:
            graph = nx.random_regular_graph(degree, n_nodes, seed=base_seed)
        except nx.NetworkXError:
            graph = nx.fast_gnp_random_graph(n_nodes, degree / n_nodes, seed=base_seed)

        nswap = rng.randint(4, degree * n_nodes)
        try:
            nx.double_edge_swap(graph, nswap=nswap, max_tries=nswap * 10, seed=rng.randint(0, 10**9))
        except nx.NetworkXAlgorithmError:
            pass

        tri_count = sum(nx.triangles(graph).values()) // 3
        dataset.append(
            {
                "x": torch.ones(n_nodes, 1),
                "edges": list(graph.edges()),
                "y": torch.tensor([float(tri_count)], dtype=torch.float32),
                "degree": float(degree),
                "graph_id": i,
                "tri_count": float(tri_count),
            }
        )

    counts = np.array([g["tri_count"] for g in dataset], dtype=np.float64)
    print(f"Triangle range: [{counts.min():.0f}, {counts.max():.0f}], mean: {counts.mean():.2f}")
    return dataset


def generate_degree_controlled_triangle_dataset(
    num_graphs=2000,
    n_nodes=24,
    degree=4,
    seed=0,
    rwse_k=0,
    include_degree=False,
    include_motif_counts=False,
    support_mode="exact",
    max_motifs=None,
    feature_mode="core",
):
    rows = []
    rng = random.Random(seed)
    while len(rows) < num_graphs:
        base_seed = rng.randint(0, 10**9)
        graph = nx.random_regular_graph(degree, n_nodes, seed=base_seed)
        nswap = rng.randint(max(4, degree * 2), max(8, degree * n_nodes))
        try:
            nx.double_edge_swap(graph, nswap=nswap, max_tries=max(100, nswap * 20), seed=rng.randint(0, 10**9))
        except nx.NetworkXException:
            continue
        tri_count = sum(nx.triangles(graph).values()) // 3
        rows.append({"graph": graph, "tri_count": tri_count})

    counts = np.array([r["tri_count"] for r in rows], dtype=np.float64)

    dataset = []
    for i, r in enumerate(rows):
        base_item = build_stage1_graph_item(
            r["graph"],
            float(r["tri_count"]),
            i,
            rwse_k=0,
            include_degree=include_degree,
            include_motif_counts=include_motif_counts,
        )
        item = prepare_stage1_graph(
            base_item,
            feature_mode=feature_mode,
            support_mode=support_mode,
            max_motifs=max_motifs,
            rwse_k=rwse_k,
            seed=seed + i,
        )
        item["degree"] = float(degree)
        item["tri_count"] = float(r["tri_count"])
        dataset.append(item)

    print(
        f"count range: [{counts.min():.0f}, {counts.max():.0f}], "
        f"mean: {counts.mean():.2f}, std: {counts.std():.2f}"
    )
    return dataset


def generate_srg_dataset(num_pairs=500, seed=0):
    nodes = [(x, y) for x in range(4) for y in range(4)]
    node_to_idx = {p: i for i, p in enumerate(nodes)}

    gens0 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
    edges0_base = []
    for x, y in nodes:
        for dx, dy in gens0:
            nx_, ny_ = (x + dx) % 4, (y + dy) % 4
            u, v = node_to_idx[(x, y)], node_to_idx[(nx_, ny_)]
            if u < v:
                edges0_base.append((u, v))

    edges1_base = []
    for x, y in nodes:
        for i in range(1, 4):
            nx_, ny_ = (x + i) % 4, y
            u, v = node_to_idx[(x, y)], node_to_idx[(nx_, ny_)]
            if u < v:
                edges1_base.append((u, v))
            nx_, ny_ = x, (y + i) % 4
            u, v = node_to_idx[(x, y)], node_to_idx[(nx_, ny_)]
            if u < v:
                edges1_base.append((u, v))
    edges1_base = list(set(tuple(sorted(e)) for e in edges1_base))

    rng = np.random.default_rng(seed)
    dataset = []
    for pair_id in range(num_pairs):
        perm = rng.permutation(16)
        mapping = {i: int(perm[i]) for i in range(16)}
        edges0 = [(mapping[u], mapping[v]) for u, v in edges0_base]
        dataset.append({"x": torch.ones(16, 1), "edges": edges0, "y": torch.tensor([0.0]), "graph_id": f"shrikhande_{pair_id}"})

        perm = rng.permutation(16)
        mapping = {i: int(perm[i]) for i in range(16)}
        edges1 = [(mapping[u], mapping[v]) for u, v in edges1_base]
        dataset.append({"x": torch.ones(16, 1), "edges": edges1, "y": torch.tensor([1.0]), "graph_id": f"rook_{pair_id}"})
    return dataset


def generate_cycle_parity_dataset(num_pairs=500, n=20, seed=0):
    rng = np.random.default_rng(seed)
    dataset = []
    for pair_id in range(num_pairs):
        g0 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        while not nx.is_bipartite(g0):
            g0 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        dataset.append({"x": torch.ones(n, 1), "edges": list(g0.edges()), "y": torch.tensor([0.0]), "graph_id": f"bipartite_{pair_id}"})

        g1 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        while nx.is_bipartite(g1):
            g1 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        dataset.append({"x": torch.ones(n, 1), "edges": list(g1.edges()), "y": torch.tensor([1.0]), "graph_id": f"nonbipartite_{pair_id}"})
    return dataset


def stratified_split(dataset, seed):
    from sklearn.model_selection import train_test_split

    labels = np.array([int(float(item["y"].view(-1)[0].item()) >= 0.5) for item in dataset])
    idx = np.arange(len(dataset))
    train_idx, tmp_idx, y_train, y_tmp = train_test_split(
        idx,
        labels,
        train_size=0.70,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        tmp_idx,
        train_size=0.50,
        random_state=seed + 1,
        stratify=y_tmp,
    )
    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx]


def run_model_suite(
    train_ds,
    val_ds,
    test_ds,
    model_specs,
    *,
    device,
    epochs,
    batch_size,
    task_type,
):
    results = {}
    for spec in model_specs:
        name = spec["name"]
        model_obj = spec["model"]
        model = model_obj() if callable(model_obj) and not isinstance(model_obj, torch.nn.Module) else model_obj
        trainer_kwargs = dict(spec.get("trainer_kwargs", {}))
        trainer = GETTrainer(model, task_type=task_type, device=device, **trainer_kwargs)
        res = trainer.run(train_ds, val_ds, test_ds, epochs, batch_size)
        results[name] = res
        metric_name = spec.get("metric_name", "metric")
        print(f"{name} {spec.get('report_label', metric_name)}: {res[metric_name]:.4f}")
    return results
