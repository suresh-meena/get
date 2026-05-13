from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import numba
from torch_geometric.data import Batch, Data

from get.energy.ops import positional_embeddings_from_edge_index


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
    pos: Optional[torch.Tensor] = None


class GraphSampleData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key in {"c_2", "u_2", "c_3", "u_3", "v_3"}:
            return int(self.num_nodes)
        return super().__inc__(key, value, *args, **kwargs)


@numba.njit(parallel=True)
def _extract_motifs_csr_jit(indptr, indices, max_motifs_per_anchor):
    n = len(indptr) - 1
    # First pass: count motifs per anchor to pre-allocate
    counts = np.zeros(n, dtype=np.int64)
    for i in numba.prange(n):
        start = indptr[i]
        end = indptr[i+1]
        n_neigh = end - start
        num_pairs = (n_neigh * (n_neigh - 1)) // 2
        counts[i] = min(num_pairs, max_motifs_per_anchor)
        
    offsets = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        offsets[i+1] = offsets[i] + counts[i]
        
    total_motifs = offsets[n]
    c3 = np.empty(total_motifs, dtype=np.int64)
    u3 = np.empty(total_motifs, dtype=np.int64)
    v3 = np.empty(total_motifs, dtype=np.int64)
    tau = np.empty(total_motifs, dtype=np.int64)
    
    for i in numba.prange(n):
        start = indptr[i]
        end = indptr[i+1]
        n_neigh = end - start
        if n_neigh < 2:
            continue
            
        ptr = offsets[i]
        budget = 0
        for j_idx in range(start, end):
            if budget >= max_motifs_per_anchor:
                break
            for k_idx in range(j_idx + 1, end):
                if budget >= max_motifs_per_anchor:
                    break
                
                u = indices[j_idx]
                v = indices[k_idx]
                
                # Binary search for edge (u, v)
                u_start = indptr[u]
                u_end = indptr[u+1]
                connected = False
                low = u_start
                high = u_end - 1
                while low <= high:
                    mid = (low + high) // 2
                    if indices[mid] == v:
                        connected = True
                        break
                    elif indices[mid] < v:
                        low = mid + 1
                    else:
                        high = mid - 1

                c3[ptr] = i
                u3[ptr] = u
                v3[ptr] = v
                tau[ptr] = 1 if connected else 0
                ptr += 1
                budget += 1
                
    return c3, u3, v3, tau


def sample_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    x: torch.Tensor, 
    y: torch.Tensor, 
    max_motifs_per_anchor: int,
    pos_k: int = 0,
    edge_attr: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    edge_index_cpu = edge_index.detach().to("cpu", non_blocking=False)
    src_np = edge_index_cpu[0].numpy().astype(np.int64, copy=False)
    dst_np = edge_index_cpu[1].numpy().astype(np.int64, copy=False)

    # Build sorted CSR directly from edge_index to avoid SciPy object overhead.
    if src_np.size > 0:
        order = np.lexsort((dst_np, src_np))
        src_sorted = src_np[order]
        dst_sorted = dst_np[order]
    else:
        src_sorted = src_np
        dst_sorted = dst_np

    counts = np.bincount(src_sorted, minlength=num_nodes)
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    indices = dst_sorted

    c3, u3, v3, tau = _extract_motifs_csr_jit(indptr, indices, max_motifs_per_anchor)
    y = y.float().reshape(-1)
    if y.numel() == 0:
        y = torch.zeros(1, dtype=torch.float32)
        
    sample = GraphSampleData(
        x=x.float(),
        y=y,
        c_2=edge_index[0].clone(),
        u_2=edge_index[1].clone(),
        c_3=torch.tensor(c3, dtype=torch.long),
        u_3=torch.tensor(u3, dtype=torch.long),
        v_3=torch.tensor(v3, dtype=torch.long),
        t_tau=torch.tensor(tau, dtype=torch.long),
    )
    
    if edge_attr is not None:
        edge_attr = edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        elif edge_attr.dim() > 2:
            edge_attr = edge_attr.reshape(edge_attr.size(0), -1)
        sample.edge_attr = edge_attr
        
    if pos_k > 0:
        sample.pos = positional_embeddings_from_edge_index(edge_index, num_nodes, k=pos_k)
        
    return sample


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


def sample_from_adj(
    adj: torch.Tensor, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    max_motifs_per_anchor: int,
    pos_k: int = 0,
) -> Dict[str, torch.Tensor]:
    n = adj.size(0)
    adj_np = adj.cpu().numpy()
    c2, u2 = np.nonzero(adj_np)
    c3, u3, v3, tau = _extract_motifs_jit(adj_np, n, max_motifs_per_anchor)
    y = y.float().reshape(-1)
    if y.numel() == 0:
        y = torch.zeros(1, dtype=torch.float32)
        
    sample = GraphSampleData(
        x=x.float(),
        y=y,
        c_2=torch.tensor(c2, dtype=torch.long),
        u_2=torch.tensor(u2, dtype=torch.long),
        c_3=torch.tensor(c3, dtype=torch.long),
        u_3=torch.tensor(u3, dtype=torch.long),
        v_3=torch.tensor(v3, dtype=torch.long),
        t_tau=torch.tensor(tau, dtype=torch.long),
    )
    
    if pos_k > 0:
        edge_index = torch.stack([sample["c_2"], sample["u_2"]], dim=0)
        sample.pos = positional_embeddings_from_edge_index(edge_index, n, k=pos_k)
        
    return sample


def collate_graph_samples(samples: List[Dict[str, torch.Tensor]]) -> Batch:
    """
    PyG Batch-based graph collator.

    Converts the incoming samples to `Data` objects when needed and lets PyG
    handle batching / index shifting for the custom motif tensors.
    """
    if not samples:
        raise ValueError("collate_graph_samples requires at least one sample")
    data_list: List[GraphSampleData] = []
    for sample in samples:
        if isinstance(sample, GraphSampleData):
            data_list.append(sample)
        elif isinstance(sample, Data):
            data_list.append(GraphSampleData(**sample.to_dict()))
        else:
            data_list.append(GraphSampleData(**sample))

    batch = Batch.from_data_list(data_list)
    batch.num_graphs = torch.tensor(len(data_list), dtype=torch.long, device=batch.x.device)
    return batch


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
        pos_k: int = 0,
        cache_root: str = "data/synthetic_cache",
    ) -> None:
        self._len = num_graphs
        self.cache_dir = Path(cache_root).expanduser() / f"n_graphs_{num_graphs}_n_{min_nodes}_{max_nodes}_p_{edge_prob}_d_{in_dim}_m_{max_motifs_per_anchor}_s_{seed}_k_{pos_k}"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already generated
        if (self.cache_dir / "done.txt").exists():
            return

        from tqdm import tqdm
        rng = np.random.default_rng(seed)
        pbar = tqdm(total=num_graphs, desc="Generating Synthetic Graphs", unit="graph")
        for i in range(num_graphs):
            n = int(rng.integers(min_nodes, max_nodes + 1))
            adj = rng.random((n, n)) < edge_prob
            adj = np.triu(adj, k=1)
            adj = adj | adj.T
            np.fill_diagonal(adj, False)
            adj_t = torch.from_numpy(adj)
            
            # Triangle count
            edge_index = torch.stack(torch.where(adj_t), dim=0)
            import torch_geometric.utils as pyg_utils
            
            if hasattr(pyg_utils, 'triangle_count'):
                tri_count = pyg_utils.triangle_count(edge_index).sum().item() // 3
            else:
                adj_f = adj.astype(np.float32)
                tri_count = int((((adj_f @ adj_f) * adj_f).sum()) / 6.0)
                
            y_val = torch.tensor([1.0 if tri_count > 0 else 0.0])
            x = torch.from_numpy(rng.standard_normal((n, in_dim)).astype(np.float32))
            
            sample = sample_from_adj(adj_t, x, y_val, max_motifs_per_anchor, pos_k=pos_k)
            torch.save(sample, self.cache_dir / f"sample_{i}.pt")
            pbar.update(1)
        pbar.close()
            
        (self.cache_dir / "done.txt").write_text("done")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return torch.load(self.cache_dir / f"sample_{idx}.pt", map_location="cpu", weights_only=False)

    @property
    def items(self) -> List[Dict[str, torch.Tensor]]:
        return [self[i] for i in range(self._len)]
