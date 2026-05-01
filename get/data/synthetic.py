from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class GraphSample:
    x: torch.Tensor
    y: torch.Tensor
    c_2: torch.Tensor
    u_2: torch.Tensor
    c_3: torch.Tensor
    u_3: torch.Tensor
    v_3: torch.Tensor
    t_tau: torch.Tensor


import numba

@numba.njit
def _extract_motifs_jit(adj, n, max_motifs_per_anchor):
    total_motifs = 0
    for i in range(n):
        n_neigh = 0
        for j in range(n):
            if adj[i, j]:
                n_neigh += 1
        num_pairs = (n_neigh * (n_neigh - 1)) // 2
        total_motifs += min(num_pairs, max_motifs_per_anchor)
        
    c3 = np.empty(total_motifs, dtype=np.int64)
    u3 = np.empty(total_motifs, dtype=np.int64)
    v3 = np.empty(total_motifs, dtype=np.int64)
    tau = np.empty(total_motifs, dtype=np.int64)
    
    ptr = 0
    for i in range(n):
        neigh = []
        for j in range(n):
            if adj[i, j]:
                neigh.append(j)
                
        n_neigh = len(neigh)
        if n_neigh < 2:
            continue
            
        budget = 0
        for j_idx in range(n_neigh):
            for k_idx in range(j_idx + 1, n_neigh):
                if budget >= max_motifs_per_anchor:
                    break
                
                u = neigh[j_idx]
                v = neigh[k_idx]
                
                c3[ptr] = i
                u3[ptr] = u
                v3[ptr] = v
                tau[ptr] = 1 if adj[u, v] else 0
                ptr += 1
                budget += 1
            if budget >= max_motifs_per_anchor:
                break
                
    return c3, u3, v3, tau

def _build_random_graph(
    rng: np.random.Generator,
    min_nodes: int,
    max_nodes: int,
    edge_prob: float,
    in_dim: int,
    max_motifs_per_anchor: int,
) -> GraphSample:
    n = int(rng.integers(min_nodes, max_nodes + 1))
    adj = rng.random((n, n)) < edge_prob
    adj = np.triu(adj, k=1)
    adj = adj | adj.T
    np.fill_diagonal(adj, False)

    # Vectorized triangle counting
    tri_count = int(np.trace(np.linalg.matrix_power(adj.astype(float), 3)) // 6)
    y_val = float(tri_count > 0)
    
    x = rng.standard_normal((n, in_dim)).astype(np.float32)

    # Vectorized edge extraction
    c2, u2 = np.nonzero(adj)

    # JIT compiled motif extraction
    c3, u3, v3, tau = _extract_motifs_jit(adj, n, max_motifs_per_anchor)

    return GraphSample(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y_val, dtype=torch.float32),
        c_2=torch.tensor(c2, dtype=torch.long),
        u_2=torch.tensor(u2, dtype=torch.long),
        c_3=torch.tensor(c3, dtype=torch.long),
        u_3=torch.tensor(u3, dtype=torch.long),
        v_3=torch.tensor(v3, dtype=torch.long),
        t_tau=torch.tensor(tau, dtype=torch.long),
    )


class SyntheticGraphDataset(Dataset):
    """Tiny synthetic graph classification dataset for RAM-safe refactor smoke runs."""

    def __init__(
        self,
        num_graphs: int,
        min_nodes: int,
        max_nodes: int,
        edge_prob: float,
        in_dim: int,
        max_motifs_per_anchor: int,
        seed: int,
    ) -> None:
        self._items: List[GraphSample] = []
        rng = np.random.default_rng(seed)
        for _ in range(num_graphs):
            self._items.append(
                _build_random_graph(
                    rng=rng,
                    min_nodes=min_nodes,
                    max_nodes=max_nodes,
                    edge_prob=edge_prob,
                    in_dim=in_dim,
                    max_motifs_per_anchor=max_motifs_per_anchor,
                )
            )

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        g = self._items[idx]
        return {
            "x": g.x,
            "y": g.y,
            "c_2": g.c_2,
            "u_2": g.u_2,
            "c_3": g.c_3,
            "u_3": g.u_3,
            "v_3": g.v_3,
            "t_tau": g.t_tau,
        }


def collate_graph_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x_all: List[torch.Tensor] = []
    y_all: List[torch.Tensor] = []
    c2_all: List[torch.Tensor] = []
    u2_all: List[torch.Tensor] = []
    c3_all: List[torch.Tensor] = []
    u3_all: List[torch.Tensor] = []
    v3_all: List[torch.Tensor] = []
    tt_all: List[torch.Tensor] = []
    batch_idx: List[torch.Tensor] = []

    node_offset = 0
    for gidx, sample in enumerate(samples):
        x = sample["x"]
        n = x.size(0)
        x_all.append(x)
        y_all.append(sample["y"].view(1))
        batch_idx.append(torch.full((n,), gidx, dtype=torch.long))

        c2 = sample["c_2"]
        u2 = sample["u_2"]
        c3 = sample["c_3"]
        u3 = sample["u_3"]
        v3 = sample["v_3"]
        tt = sample["t_tau"]

        if c2.numel() > 0:
            c2_all.append(c2 + node_offset)
            u2_all.append(u2 + node_offset)
        if c3.numel() > 0:
            c3_all.append(c3 + node_offset)
            u3_all.append(u3 + node_offset)
            v3_all.append(v3 + node_offset)
            tt_all.append(tt)

        node_offset += n

    def _cat_or_empty(parts: List[torch.Tensor], dtype: torch.dtype) -> torch.Tensor:
        if not parts:
            return torch.empty(0, dtype=dtype)
        return torch.cat(parts, dim=0)

    return {
        "x": torch.cat(x_all, dim=0),
        "y": torch.cat(y_all, dim=0),
        "batch": torch.cat(batch_idx, dim=0),
        "num_graphs": torch.tensor(len(samples), dtype=torch.long),
        "c_2": _cat_or_empty(c2_all, torch.long),
        "u_2": _cat_or_empty(u2_all, torch.long),
        "c_3": _cat_or_empty(c3_all, torch.long),
        "u_3": _cat_or_empty(u3_all, torch.long),
        "v_3": _cat_or_empty(v3_all, torch.long),
        "t_tau": _cat_or_empty(tt_all, torch.long),
    }
