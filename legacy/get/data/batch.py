"""Batching and graph preprocessing helpers for GET."""

from __future__ import annotations

import hashlib
import numpy as np
import torch
from numba import njit


def _as_long_tensor(values, device=None):
    tensor = torch.as_tensor(values, dtype=torch.long, device=device)
    return tensor.contiguous()


def _edge_pairs_array(edges):
    if edges is None:
        return np.empty((0, 2), dtype=np.int32)
    if torch.is_tensor(edges):
        edge_index = edges.detach().cpu()
        if edge_index.dim() != 2:
            raise ValueError("edge_index must be 2D")
        if edge_index.size(0) == 2:
            pairs = edge_index.t().contiguous().numpy()
        elif edge_index.size(1) == 2:
            pairs = edge_index.contiguous().numpy()
        else:
            raise ValueError("edge_index must have shape [2, E] or [E, 2]")
        return np.ascontiguousarray(pairs, dtype=np.int32)

    edge_pairs = list(edges)
    if len(edge_pairs) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.ascontiguousarray(np.asarray(edge_pairs, dtype=np.int32).reshape(-1, 2))


def _canonicalize_undirected_edge_pairs(edge_pairs, num_nodes=None):
    if edge_pairs.size == 0:
        return edge_pairs.reshape(0, 2)

    u = edge_pairs[:, 0]
    v = edge_pairs[:, 1]
    valid = u != v
    if num_nodes is not None:
        n = int(num_nodes)
        valid &= (u >= 0) & (v >= 0) & (u < n) & (v < n)

    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)

    lo = np.minimum(u[valid], v[valid])
    hi = np.maximum(u[valid], v[valid])
    canonical = np.stack((lo, hi), axis=1).astype(np.int32, copy=False)
    order = np.lexsort((canonical[:, 1], canonical[:, 0]))
    canonical = canonical[order]

    if canonical.shape[0] > 1:
        keep = np.ones(canonical.shape[0], dtype=np.bool_)
        keep[1:] = np.any(canonical[1:] != canonical[:-1], axis=1)
        canonical = canonical[keep]

    return np.ascontiguousarray(canonical, dtype=np.int32)


def _normalize_edge_index(edges):
    canonical = _canonicalize_undirected_edge_pairs(_edge_pairs_array(edges))
    return [tuple(int(x) for x in row) for row in canonical]


def _edges_to_attr_map(edges, edge_attr):
    edge_pairs = _edge_pairs_array(edges)
    normalized = [tuple(int(x) for x in row) for row in _canonicalize_undirected_edge_pairs(edge_pairs)]
    if edge_attr is None:
        return normalized, None

    attr = edge_attr.detach().clone() if torch.is_tensor(edge_attr) else torch.as_tensor(edge_attr)
    if attr.dim() == 1:
        attr = attr.view(-1, 1)

    attr_map: dict[tuple[int, int], torch.Tensor] = {}
    for idx in range(edge_pairs.shape[0]):
        ui = int(edge_pairs[idx, 0])
        vi = int(edge_pairs[idx, 1])
        if ui == vi:
            continue
        key = (ui, vi) if ui < vi else (vi, ui)
        if key not in attr_map:
            attr_map[key] = attr[idx]

    return normalized, attr_map


def _pack_undirected_edge_keys(u, v):
    lo = np.minimum(np.asarray(u, dtype=np.uint64), np.asarray(v, dtype=np.uint64))
    hi = np.maximum(np.asarray(u, dtype=np.uint64), np.asarray(v, dtype=np.uint64))
    return (lo << np.uint64(32)) | hi


def _numba_edges_to_csr(num_nodes, edges_arr):
    n = int(num_nodes)
    edge_pairs = _canonicalize_undirected_edge_pairs(_edge_pairs_array(edges_arr), num_nodes=n)
    if edge_pairs.size == 0:
        return np.zeros(n + 1, dtype=np.int32), np.empty(0, dtype=np.int32)

    directed = np.empty((edge_pairs.shape[0] * 2, 2), dtype=np.int32)
    directed[0::2] = edge_pairs
    directed[1::2, 0] = edge_pairs[:, 1]
    directed[1::2, 1] = edge_pairs[:, 0]

    order = np.lexsort((directed[:, 1], directed[:, 0]))
    directed = directed[order]

    if directed.shape[0] > 1:
        keep = np.ones(directed.shape[0], dtype=np.bool_)
        keep[1:] = np.any(directed[1:] != directed[:-1], axis=1)
        directed = directed[keep]

    rows = directed[:, 0]
    cols = directed[:, 1]

    indptr = np.zeros(n + 1, dtype=np.int32)
    row_counts = np.asarray(np.bincount(rows, minlength=n), dtype=np.int32)
    indptr[1:] = np.cumsum(row_counts, dtype=np.int32)

    return indptr, np.ascontiguousarray(cols, dtype=np.int32)


@njit
def _numba_row_contains(indices, start, end, value):
    left = 0
    right = end - start - 1
    while left <= right:
        mid = (left + right) // 2
        current = indices[start + mid]
        if current == value:
            return True
        if current < value:
            left = mid + 1
        else:
            right = mid - 1
    return False


@njit
def _numba_count_incidence_sizes(indptr, indices, motif_budget):
    num_nodes = len(indptr) - 1
    edge_count = 0
    motif_count = 0
    closed_counts = np.zeros(num_nodes, dtype=np.int32)

    for center in range(num_nodes):
        start = indptr[center]
        end = indptr[center + 1]
        degree_center = end - start
        closed_count = 0
        total_count = 0

        for pos in range(degree_center):
            nbr = indices[start + pos]
            if center < nbr:
                edge_count += 2

            nbr_start = indptr[nbr]
            nbr_end = indptr[nbr + 1]
            degree_nbr = nbr_end - nbr_start

            for next_pos in range(pos + 1, degree_center):
                other = indices[start + next_pos]
                other_start = indptr[other]
                other_end = indptr[other + 1]
                degree_other = other_end - other_start
                if degree_center >= degree_nbr and degree_center >= degree_other:
                    total_count += 1
                    if _numba_row_contains(indices, nbr_start, nbr_end, other):
                        closed_count += 1

        closed_counts[center] = closed_count
        if motif_budget < 0:
            motif_count += total_count
        elif motif_budget == 0:
            continue
        elif closed_count >= motif_budget:
            motif_count += closed_count
        else:
            motif_count += total_count

    return edge_count, motif_count, closed_counts


@njit
def _numba_fill_incidence_arrays(indptr, indices, motif_budget, closed_counts, c_2, u_2, c_3, u_3, v_3, t_tau):
    num_nodes = len(indptr) - 1
    edge_cursor = 0
    motif_cursor = 0

    for center in range(num_nodes):
        start = indptr[center]
        end = indptr[center + 1]
        degree_center = end - start

        for pos in range(degree_center):
            nbr = indices[start + pos]
            if center < nbr:
                c_2[edge_cursor] = center
                u_2[edge_cursor] = nbr
                edge_cursor += 1
                c_2[edge_cursor] = nbr
                u_2[edge_cursor] = center
                edge_cursor += 1

        if motif_budget == 0:
            continue

        emit_closed_only = motif_budget > 0 and closed_counts[center] >= motif_budget

        if emit_closed_only:
            for pos in range(degree_center):
                u = indices[start + pos]
                u_start = indptr[u]
                u_end = indptr[u + 1]
                degree_u = u_end - u_start
                for next_pos in range(pos + 1, degree_center):
                    v = indices[start + next_pos]
                    v_start = indptr[v]
                    v_end = indptr[v + 1]
                    degree_v = v_end - v_start
                    if degree_center >= degree_u and degree_center >= degree_v and _numba_row_contains(indices, u_start, u_end, v):
                        c_3[motif_cursor] = center
                        u_3[motif_cursor] = u
                        v_3[motif_cursor] = v
                        t_tau[motif_cursor] = 1
                        motif_cursor += 1
        else:
            for pos in range(degree_center):
                u = indices[start + pos]
                u_start = indptr[u]
                u_end = indptr[u + 1]
                degree_u = u_end - u_start
                for next_pos in range(pos + 1, degree_center):
                    v = indices[start + next_pos]
                    v_start = indptr[v]
                    v_end = indptr[v + 1]
                    degree_v = v_end - v_start
                    if degree_center >= degree_u and degree_center >= degree_v:
                        closed = _numba_row_contains(indices, u_start, u_end, v)
                        if closed:
                            c_3[motif_cursor] = center
                            u_3[motif_cursor] = u
                            v_3[motif_cursor] = v
                            t_tau[motif_cursor] = 1
                            motif_cursor += 1

            for pos in range(degree_center):
                u = indices[start + pos]
                u_start = indptr[u]
                u_end = indptr[u + 1]
                degree_u = u_end - u_start
                for next_pos in range(pos + 1, degree_center):
                    v = indices[start + next_pos]
                    v_start = indptr[v]
                    v_end = indptr[v + 1]
                    degree_v = v_end - v_start
                    if degree_center >= degree_u and degree_center >= degree_v and not _numba_row_contains(indices, u_start, u_end, v):
                        c_3[motif_cursor] = center
                        u_3[motif_cursor] = u
                        v_3[motif_cursor] = v
                        t_tau[motif_cursor] = 0
                        motif_cursor += 1


def get_incidence_matrices(num_nodes, indptr, indices, max_motifs_per_node=None):
    indptr_np = np.ascontiguousarray(np.asarray(indptr, dtype=np.int32))
    indices_np = np.ascontiguousarray(np.asarray(indices, dtype=np.int32))
    budget = -1 if max_motifs_per_node is None else int(max_motifs_per_node)

    edge_count, motif_count, closed_counts = _numba_count_incidence_sizes(indptr_np, indices_np, budget)
    c2_arr = np.empty(edge_count, dtype=np.int32)
    u2_arr = np.empty(edge_count, dtype=np.int32)
    c3_arr = np.empty(motif_count, dtype=np.int32)
    u3_arr = np.empty(motif_count, dtype=np.int32)
    v3_arr = np.empty(motif_count, dtype=np.int32)
    t_tau_arr = np.empty(motif_count, dtype=np.int32)

    _numba_fill_incidence_arrays(indptr_np, indices_np, budget, closed_counts, c2_arr, u2_arr, c3_arr, u3_arr, v3_arr, t_tau_arr)

    c_2 = _as_long_tensor(c2_arr)
    u_2 = _as_long_tensor(u2_arr)
    c_3 = _as_long_tensor(c3_arr)
    u_3 = _as_long_tensor(u3_arr)
    v_3 = _as_long_tensor(v3_arr)
    t_tau = _as_long_tensor(t_tau_arr)
    return c_2, u_2, c_3, u_3, v_3, t_tau


def align_pairwise_edge_attr(edges, edge_attr, c_2, u_2):
    if edge_attr is None:
        return None

    edge_pairs = _edge_pairs_array(edges)
    if edge_pairs.shape[0] == 0:
        if c_2.numel() == 0:
            attr = edge_attr.detach() if torch.is_tensor(edge_attr) else torch.as_tensor(edge_attr)
            if attr.dim() == 1:
                attr = attr.view(-1, 1)
            return attr.new_empty((0, attr.size(-1)))
        raise KeyError("Missing edge_attr for incidence edges")

    attr = edge_attr.detach() if torch.is_tensor(edge_attr) else torch.as_tensor(edge_attr)
    if attr.dim() == 1:
        attr = attr.view(-1, 1)
    attr_cpu = attr.cpu()

    edge_keys = _pack_undirected_edge_keys(edge_pairs[:, 0], edge_pairs[:, 1])
    unique_keys, first_idx = np.unique(edge_keys, return_index=True)
    unique_attr = attr_cpu[first_idx]

    c_2_np = c_2.detach().cpu().numpy()
    u_2_np = u_2.detach().cpu().numpy()
    incidence_keys = _pack_undirected_edge_keys(c_2_np, u_2_np)
    positions = np.searchsorted(unique_keys, incidence_keys)
    if positions.size == 0:
        return attr.new_empty((0, attr.size(-1)))
    if np.any(positions >= unique_keys.size) or not np.array_equal(unique_keys[positions], incidence_keys):
        raise KeyError("Missing edge_attr for one or more incidence edges")

    aligned = unique_attr[positions]
    if torch.is_tensor(edge_attr) and edge_attr.device.type != "cpu":
        aligned = aligned.to(device=edge_attr.device)
    return aligned.to(dtype=attr.dtype)


class GETBatch:
    def __init__(
        self,
        x: torch.Tensor,
        c_2: torch.Tensor,
        u_2: torch.Tensor,
        c_3: torch.Tensor,
        u_3: torch.Tensor,
        v_3: torch.Tensor,
        t_tau: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        rwse: torch.Tensor | None = None,
        **extras,
    ):
        self.x = x
        self.c_2 = c_2
        self.u_2 = u_2
        self.c_3 = c_3
        self.u_3 = u_3
        self.v_3 = v_3
        self.t_tau = t_tau
        self.batch = batch
        self.ptr = ptr
        self.edge_attr = edge_attr
        self.y = y
        self.pe = pe
        self.rwse = rwse
        for key, value in extras.items():
            setattr(self, key, value)

    @property
    def num_nodes(self):
        return int(self.x.size(0))

    def to(self, device=None, *args, **kwargs):
        data = {}
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                data[key] = value.to(device=device, *args, **kwargs) if device is not None else value.to(*args, **kwargs)
            else:
                data[key] = value
        return GETBatch(**data)


def _process_one_graph(graph, max_motifs=None, pe_k=0, rwse_k=0):
    item = dict(graph)
    x = item.get("x")
    if x is None:
        raise ValueError("graph must contain 'x'")
    x = torch.as_tensor(x)
    item["x"] = x

    edges = item.get("edges")
    edge_index = item.get("edge_index")
    if edges is None and edge_index is None:
        edges = []
    edge_source = edge_index if edge_index is not None else edges
    num_nodes = int(x.size(0))
    indptr, indices = _numba_edges_to_csr(num_nodes, edge_source)
    c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(num_nodes, indptr, indices, max_motifs_per_node=max_motifs)

    aligned_edge_attr = None
    if item.get("edge_attr") is not None:
        aligned_edge_attr = align_pairwise_edge_attr(edge_source, item["edge_attr"], c_2, u_2)

    if pe_k > 0 and item.get("pe") is not None:
        pe = torch.as_tensor(item["pe"], dtype=x.dtype)
        if pe.size(1) >= int(pe_k):
            item["pe"] = pe[:, : int(pe_k)]
        else:
            pad = torch.zeros(pe.size(0), int(pe_k) - pe.size(1), dtype=pe.dtype, device=pe.device)
            item["pe"] = torch.cat([pe, pad], dim=-1)

    if rwse_k > 0:
        if item.get("rwse") is None:
            from .positional import get_rwse

            item["rwse"] = get_rwse(num_nodes, indptr, indices, int(rwse_k)).to(dtype=x.dtype)
        elif item["rwse"].size(1) != int(rwse_k):
            rwse = torch.as_tensor(item["rwse"], dtype=x.dtype)
            if rwse.size(1) >= int(rwse_k):
                item["rwse"] = rwse[:, : int(rwse_k)]
            else:
                pad = torch.zeros(rwse.size(0), int(rwse_k) - rwse.size(1), dtype=rwse.dtype, device=rwse.device)
                item["rwse"] = torch.cat([rwse, pad], dim=-1)

    item.update(
        {
            "c_2": c_2,
            "u_2": u_2,
            "c_3": c_3,
            "u_3": u_3,
            "v_3": v_3,
            "t_tau": t_tau,
            "edge_attr": aligned_edge_attr,
            "aligned_edge_attr": aligned_edge_attr,
            "num_nodes": num_nodes,
        }
    )
    if "pe" in item and item["pe"] is not None:
        item["pe"] = torch.as_tensor(item["pe"], dtype=x.dtype)
    if "rwse" in item and item["rwse"] is not None:
        item["rwse"] = torch.as_tensor(item["rwse"], dtype=x.dtype)
    return item


def _graph_dataset_cache_fingerprint(dataset, name, max_motifs, pe_k, rwse_k):
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(str(name).encode("utf-8"))
    hasher.update(str(int(max_motifs)).encode("utf-8"))
    hasher.update(str(int(pe_k)).encode("utf-8"))
    hasher.update(str(int(rwse_k)).encode("utf-8"))
    for item in dataset:
        hasher.update(str(sorted(item.keys())).encode("utf-8"))
        for key in ("x", "edges", "edge_index", "edge_attr", "y", "pe", "rwse"):
            value = item.get(key)
            if value is None:
                hasher.update(b"<none>")
                continue
            if torch.is_tensor(value):
                tensor = value.detach().cpu().contiguous()
                hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
                hasher.update(str(tensor.dtype).encode("utf-8"))
                hasher.update(tensor.numpy().tobytes())
            else:
                hasher.update(repr(value).encode("utf-8"))
    return hasher.hexdigest()


def validate_get_batch(batch):
    if batch.x.size(0) != batch.batch.size(0):
        raise ValueError("batch vector must match x rows")
    if batch.ptr[0].item() != 0 or batch.ptr[-1].item() != batch.x.size(0):
        raise ValueError("ptr must span all nodes")
    if batch.c_2.numel() != batch.u_2.numel():
        raise ValueError("c_2/u_2 length mismatch")
    if batch.c_3.numel() != batch.u_3.numel() or batch.c_3.numel() != batch.v_3.numel() or batch.c_3.numel() != batch.t_tau.numel():
        raise ValueError("motif incidence length mismatch")
    if batch.edge_attr is not None:
        if batch.edge_attr.size(0) != batch.c_2.numel():
            raise ValueError("edge_attr must align with c_2/u_2 incidence count")
        if batch.edge_attr.dtype != batch.x.dtype:
            raise ValueError("edge_attr dtype must match x dtype")
    if batch.pe is not None and batch.pe.size(0) != batch.x.size(0):
        raise ValueError("pe must match x rows")
    if batch.rwse is not None and batch.rwse.size(0) != batch.x.size(0):
        raise ValueError("rwse must match x rows")
    return batch


def add_structural_node_features(graph, include_degree=True, include_motif_counts=True):
    item = dict(graph)
    x = item.get("x")
    if x is None:
        raise ValueError("graph must contain 'x'")
    x = torch.as_tensor(x)

    edge_source = item.get("edges", item.get("edge_index", []))
    num_nodes = int(x.size(0))
    indptr, indices = _numba_edges_to_csr(num_nodes, edge_source)
    c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(num_nodes, indptr, indices, max_motifs_per_node=None)

    feats = [x]
    if include_degree:
        degree = torch.as_tensor(np.diff(indptr), dtype=x.dtype, device=x.device).view(-1, 1)
        feats.append(degree)
    if include_motif_counts:
        open_counts = torch.zeros(num_nodes, 1, dtype=x.dtype, device=x.device)
        closed_counts = torch.zeros(num_nodes, 1, dtype=x.dtype, device=x.device)
        if c_3.numel() > 0:
            open_counts.index_add_(0, c_3, (t_tau == 0).to(dtype=x.dtype).view(-1, 1))
            closed_counts.index_add_(0, c_3, (t_tau == 1).to(dtype=x.dtype).view(-1, 1))
        feats.extend([open_counts, closed_counts])

    item["x"] = torch.cat(feats, dim=-1)
    return item


class CachedGraphDataset:
    def __init__(self, dataset, name="dataset", max_motifs=16, pe_k=0, rwse_k=0):
        self.name = str(name)
        self.max_motifs = int(max_motifs)
        self.pe_k = int(pe_k)
        self.rwse_k = int(rwse_k)
        self.cached_data = [
            _process_one_graph(item, max_motifs=self.max_motifs, pe_k=self.pe_k, rwse_k=self.rwse_k)
            for item in list(dataset)
        ]

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, index):
        return self.cached_data[index]


def collate_get_batch(graph_list, max_motifs=None, pe_k=0, rwse_k=0):
    if len(graph_list) == 0:
        raise ValueError("graph_list must not be empty")

    processed = []
    for graph in graph_list:
        if all(key in graph for key in ("c_2", "u_2", "c_3", "u_3", "v_3", "t_tau", "batch", "ptr")):
            # Cached graphs are already normalized; avoid copying the mapping again.
            processed.append(graph)
        else:
            processed.append(
                _process_one_graph(
                    graph,
                    max_motifs=graph.get("max_motifs", max_motifs),
                    pe_k=graph.get("pe_k", pe_k),
                    rwse_k=graph.get("rwse_k", rwse_k),
                )
            )

    xs = []
    c2 = []
    u2 = []
    c3 = []
    u3 = []
    v3 = []
    t_tau = []
    batch = []
    ptr = [0]
    edge_attrs = []
    ys = []
    pes = []
    rwses = []

    node_offset = 0
    for graph_idx, item in enumerate(processed):
        x = item["x"]
        num_nodes = int(x.size(0))
        xs.append(x)
        batch.append(torch.full((num_nodes,), graph_idx, dtype=torch.long))
        ptr.append(ptr[-1] + num_nodes)

        c2.append(item["c_2"] + node_offset)
        u2.append(item["u_2"] + node_offset)
        if item["c_3"].numel() > 0:
            c3.append(item["c_3"] + node_offset)
            u3.append(item["u_3"] + node_offset)
            v3.append(item["v_3"] + node_offset)
            t_tau.append(item["t_tau"])

        if item.get("edge_attr") is not None:
            edge_attrs.append(item["edge_attr"])
        if item.get("y") is not None:
            _y = torch.as_tensor(item["y"])
            ys.append(_y.unsqueeze(0) if _y.dim() == 0 else _y)
        if item.get("pe") is not None:
            pes.append(torch.as_tensor(item["pe"]))
        if item.get("rwse") is not None:
            rwses.append(torch.as_tensor(item["rwse"]))

        node_offset += num_nodes

    edge_attr = torch.cat(edge_attrs, dim=0) if edge_attrs else None
    pe = torch.cat(pes, dim=0) if pes else None
    rwse = torch.cat(rwses, dim=0) if rwses else None

    batch_obj = GETBatch(
        x=torch.cat(xs, dim=0),
        c_2=torch.cat(c2, dim=0),
        u_2=torch.cat(u2, dim=0),
        c_3=torch.cat(c3, dim=0) if c3 else torch.zeros(0, dtype=torch.long),
        u_3=torch.cat(u3, dim=0) if u3 else torch.zeros(0, dtype=torch.long),
        v_3=torch.cat(v3, dim=0) if v3 else torch.zeros(0, dtype=torch.long),
        t_tau=torch.cat(t_tau, dim=0) if t_tau else torch.zeros(0, dtype=torch.long),
        batch=torch.cat(batch, dim=0),
        ptr=torch.as_tensor(ptr, dtype=torch.long),
        edge_attr=edge_attr,
        y=torch.cat(ys, dim=0) if ys else None,
        pe=pe,
        rwse=rwse,
    )
    return batch_obj


__all__ = [
    "GETBatch",
    "CachedGraphDataset",
    "add_structural_node_features",
    "align_pairwise_edge_attr",
    "collate_get_batch",
    "get_incidence_matrices",
    "validate_get_batch",
    "_graph_dataset_cache_fingerprint",
    "_numba_edges_to_csr",
    "_process_one_graph",
]
