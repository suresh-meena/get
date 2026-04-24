from __future__ import annotations

import numpy as np
from numba import njit

import torch


def _adj_to_csr_structural(adj):
    num_nodes = len(adj)
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for i, neighbors in enumerate(adj):
        indptr[i+1] = indptr[i] + len(neighbors)
    indices = np.empty(indptr[-1], dtype=np.int64)
    for i, neighbors in enumerate(adj):
        indices[indptr[i]:indptr[i+1]] = sorted(neighbors)
    return indptr, indices


@njit
def _numba_msbfs(indptr, indices, max_distance, unreachable):
    num_nodes = len(indptr) - 1
    dist = np.full((num_nodes, num_nodes), unreachable, dtype=np.int64)
    
    for src in range(num_nodes):
        dist[src, src] = 0
        q = np.empty(num_nodes, dtype=np.int64)
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
                
            for i in range(indptr[u], indptr[u+1]):
                v = indices[i]
                if dist[src, v] == unreachable:
                    dist[src, v] = d_u + 1
                    q[tail] = v
                    tail += 1
    return dist


def build_undirected_adjacency(num_nodes: int, edges_list: list[tuple[int, int]]) -> list[set[int]]:
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
    adj = build_undirected_adjacency(num_nodes, edges_list)
    n = int(num_nodes)
    m_dist = int(max_distance) if max_distance is not None else -1
    unreachable = n if max_distance is None else int(max_distance) + 1
    
    indptr, indices = _adj_to_csr_structural(adj)
    dist_np = _numba_msbfs(indptr, indices, m_dist, unreachable)
    
    return torch.from_numpy(dist_np)


def degree_centrality(num_nodes: int, edges_list: list[tuple[int, int]]) -> torch.Tensor:
    adj = build_undirected_adjacency(num_nodes, edges_list)
    return torch.tensor([len(adj[i]) for i in range(int(num_nodes))], dtype=torch.long)


def augment_with_virtual_node(x: torch.Tensor, edges_list: list[tuple[int, int]], virtual_token: torch.Tensor):
    num_nodes = int(x.size(0))
    x_aug = torch.cat([x, virtual_token.to(dtype=x.dtype, device=x.device).expand(1, -1)], dim=0)
    v = num_nodes
    aug_edges = list(edges_list)
    aug_edges.extend((i, v) for i in range(num_nodes))
    aug_edges.extend((v, i) for i in range(num_nodes))
    aug_edges.append((v, v))
    return x_aug, aug_edges
