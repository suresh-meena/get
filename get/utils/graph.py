"""Graph structure utilities: adjacency, SPD, degree, virtual nodes."""
from __future__ import annotations

import numpy as np
from numba import njit
import torch


@njit
def _edges_to_csr_structural(num_nodes, edges_arr):
    """Directly build CSR from edge array in Numba."""
    indptr = np.zeros(num_nodes + 1, dtype=np.int32)
    for i in range(edges_arr.shape[0]):
        u, v = edges_arr[i, 0], edges_arr[i, 1]
        if u == v or u >= num_nodes or v >= num_nodes:
            continue
        indptr[u + 1] += 1
        indptr[v + 1] += 1

    for i in range(num_nodes):
        indptr[i + 1] += indptr[i]

    indices = np.empty(indptr[num_nodes], dtype=np.int32)
    curr_idx = indptr[:-1].copy()
    for i in range(edges_arr.shape[0]):
        u, v = edges_arr[i, 0], edges_arr[i, 1]
        if u == v or u >= num_nodes or v >= num_nodes:
            continue
        indices[curr_idx[u]] = v
        curr_idx[u] += 1
        indices[curr_idx[v]] = u
        curr_idx[v] += 1

    for i in range(num_nodes):
        indices[indptr[i]:indptr[i + 1]].sort()

    return indptr, indices


@njit
def _numba_msbfs(indptr, indices, max_distance, unreachable):
    num_nodes = len(indptr) - 1
    dist = np.full((num_nodes, num_nodes), unreachable, dtype=np.int32)

    for src in range(num_nodes):
        dist[src, src] = 0
        q = np.empty(num_nodes, dtype=np.int32)
        head = 0
        tail = 0

        q[tail] = src
        tail += 1

        while head < tail:
            u = q[head]
            head += 1

            d_u = dist[src, u]
            if max_distance > 0 and d_u >= max_distance:
                continue

            for i in range(indptr[u], indptr[u + 1]):
                v = indices[i]
                if dist[src, v] == unreachable:
                    dist[src, v] = d_u + 1
                    q[tail] = v
                    tail += 1
    return dist


@njit
def _numba_csr_to_dense(num_nodes, indptr, indices, data):
    """Build a dense matrix from CSR data in Numba."""
    dense = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for idx in range(indptr[i], indptr[i + 1]):
            dense[i, indices[idx]] = data[idx]
    return dense


def build_undirected_adjacency(num_nodes: int, edges_list: list[tuple[int, int]]) -> list[set[int]]:
    """Build adjacency list from edge list (prefer CSR for performance)."""
    adj = [set() for _ in range(int(num_nodes))]
    for u, v in edges_list:
        ui = int(u)
        vi = int(v)
        if ui == vi:
            continue
        adj[ui].add(vi)
        adj[vi].add(ui)
    return adj


def shortest_path_distances(num_nodes: int, edges_list: list[tuple[int, int]], max_distance: int | None = None) -> torch.Tensor:
    edges_arr = np.ascontiguousarray(np.array(edges_list, dtype=np.int32).reshape(-1, 2))
    n = int(num_nodes)
    m_dist = int(max_distance) if max_distance is not None else -1
    unreachable = n if max_distance is None else int(max_distance) + 1

    indptr, indices = _edges_to_csr_structural(n, edges_arr)
    dist_np = _numba_msbfs(indptr, indices, m_dist, unreachable)

    return torch.from_numpy(dist_np)


def degree_centrality(num_nodes: int, edges_list: list[tuple[int, int]]) -> torch.Tensor:
    edges_arr = np.ascontiguousarray(np.array(edges_list, dtype=np.int32).reshape(-1, 2))
    n = int(num_nodes)
    indptr, _ = _edges_to_csr_structural(n, edges_arr)
    return torch.from_numpy(np.diff(indptr)).to(dtype=torch.int32)


def augment_with_virtual_node(x: torch.Tensor, edges_list: list[tuple[int, int]], virtual_token: torch.Tensor):
    num_nodes = int(x.size(0))
    x_aug = torch.cat([x, virtual_token.to(dtype=x.dtype, device=x.device).expand(1, -1)], dim=0)
    v = num_nodes
    aug_edges = list(edges_list)
    aug_edges.extend((i, v) for i in range(num_nodes))
    aug_edges.extend((v, i) for i in range(num_nodes))
    aug_edges.append((v, v))
    return x_aug, aug_edges
