from __future__ import annotations

from collections import deque

import torch


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
    unreachable = n if max_distance is None else int(max_distance) + 1
    dist = torch.full((n, n), unreachable, dtype=torch.long)
    for src in range(n):
        dist[src, src] = 0
        q = deque([src])
        while q:
            node = q.popleft()
            if max_distance is not None and int(dist[src, node].item()) >= int(max_distance):
                continue
            for nbr in adj[node]:
                if int(dist[src, nbr].item()) > int(dist[src, node].item()) + 1:
                    dist[src, nbr] = dist[src, node] + 1
                    q.append(nbr)
    if max_distance is not None:
        dist = dist.clamp(max=int(max_distance) + 1)
    return dist


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
