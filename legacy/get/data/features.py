"""Structural node features and edge attribute alignment."""
import torch
import numpy as np
from .motif_extraction import _numba_edges_to_csr, _numba_count_motifs


def _edge_source_to_tensor(edge_source, device=None):
    if torch.is_tensor(edge_source):
        edges_tensor = edge_source.to(device=device, dtype=torch.long)
        if edges_tensor.dim() != 2:
            raise ValueError("edge_index must be 2D with shape [2, E] or [E, 2].")
        if edges_tensor.size(0) == 2 and edges_tensor.size(1) != 2:
            edges_tensor = edges_tensor.t()
        elif edges_tensor.size(-1) != 2:
            raise ValueError("edge_index must have shape [2, E] or [E, 2].")
        return edges_tensor.contiguous()
    edges_tensor = torch.as_tensor(edge_source, dtype=torch.long, device=device)
    if edges_tensor.numel() == 0:
        return edges_tensor.reshape(0, 2)
    if edges_tensor.dim() == 2 and edges_tensor.size(-1) == 2:
        return edges_tensor.contiguous()
    if edges_tensor.dim() == 2 and edges_tensor.size(0) == 2:
        return edges_tensor.t().contiguous()
    return edges_tensor.reshape(-1, 2).contiguous()


def _graph_edge_source(graph):
    return graph['edge_index'] if 'edge_index' in graph else graph['edges']


def align_pairwise_edge_attr(edge_source, edge_attr, c_2, u_2):
    if edge_attr is None:
        return None
    if edge_attr.size(0) == c_2.numel():
        return edge_attr
    edges_tensor = _edge_source_to_tensor(edge_source, device=c_2.device)
    if edge_attr.size(0) != edges_tensor.size(0):
        raise ValueError(f"edge_attr rows mismatch: {edge_attr.size(0)} vs {edges_tensor.size(0)} edges")
    if c_2.numel() == 0:
        return edge_attr.new_empty((0, *edge_attr.shape[1:]))
    max_node_id = max(
        int(edges_tensor.max().item()) if edges_tensor.numel() > 0 else -1,
        int(c_2.max().item()) if c_2.numel() > 0 else -1,
        int(u_2.max().item()) if u_2.numel() > 0 else -1
    )
    stride = max_node_id + 1
    undirected_key = edges_tensor[:, 0] * stride + edges_tensor[:, 1]
    reverse_key = (undirected_key % stride) * stride + (undirected_key // stride)
    directed_key = torch.cat([undirected_key, reverse_key], dim=0)
    directed_attr = torch.cat([edge_attr, edge_attr], dim=0)
    sort_idx = torch.argsort(directed_key)
    sorted_key = directed_key[sort_idx]
    sorted_attr = directed_attr[sort_idx]
    query_key = c_2.to(dtype=torch.long) * stride + u_2.to(dtype=torch.long)
    pos = torch.searchsorted(sorted_key, query_key)
    safe_pos = pos.clamp(max=max(sorted_key.numel() - 1, 0))
    invalid = (pos >= sorted_key.numel()) | (sorted_key[safe_pos] != query_key)
    if bool(invalid.any()):
        missing_idx = int(torch.nonzero(invalid, as_tuple=False)[0].item())
        raise ValueError(f"Missing edge_attr for edge ({int(c_2[missing_idx])}, {int(u_2[missing_idx])})")
    return sorted_attr[pos]


def add_structural_node_features(graph, include_degree=True, include_motif_counts=False, normalize=True):
    if not include_degree and not include_motif_counts:
        return dict(graph)
    num_nodes = graph["x"].size(0)
    edges_arr = np.ascontiguousarray(np.array(graph["edges"], dtype=np.int64).reshape(-1, 2))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    features = []
    if include_degree:
        degree = torch.from_numpy(np.diff(indptr)).to(dtype=graph["x"].dtype, device=graph["x"].device)
        features.append(degree.view(-1, 1))
    if include_motif_counts:
        counts = _numba_count_motifs(indptr, indices)
        motif_counts = torch.from_numpy(counts).to(dtype=graph["x"].dtype, device=graph["x"].device)
        features.append(motif_counts)
    structural = torch.cat(features, dim=-1).to(device=graph["x"].device)
    if normalize and structural.numel() > 0:
        scale = structural.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
        structural = structural / scale
    enriched = dict(graph)
    enriched["x"] = torch.cat([graph["x"], structural], dim=-1)
    return enriched
