from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from get.energy.ops import positional_embeddings_from_edge_index


@dataclass
class ETBatchContext:
    batch: torch.Tensor
    c_2: torch.Tensor
    u_2: torch.Tensor
    pos: torch.Tensor
    num_graphs: int
    original_node_count: int


def build_et_batch_context(
    batch_data: dict,
    *,
    use_cls_token: bool = True,
    pos_k: int = 15,
    embed_type: str = "eigen",
    flip_sign: bool = False,
) -> ETBatchContext:
    x = batch_data["x"]
    batch = batch_data["batch"]
    c_2 = batch_data["c_2"]
    u_2 = batch_data["u_2"]
    
    device = x.device
    N = x.size(0)

    num_graphs_attr = getattr(batch_data, "num_graphs", None)
    if num_graphs_attr is None and isinstance(batch_data, dict):
        num_graphs_attr = batch_data.get("num_graphs")
    
    if torch.is_tensor(num_graphs_attr):
        num_graphs = int(num_graphs_attr.item())
    elif num_graphs_attr is not None:
        num_graphs = int(num_graphs_attr)
    else:
        num_graphs = int(batch_data["y"].shape[0])

    # 1. Positional Embeddings
    if "pos" in batch_data and batch_data["pos"].numel() > 0:
        pos_cat = batch_data["pos"].to(device=device, dtype=x.dtype)
    elif pos_k > 0:
        embed_type = str(embed_type).lower()
        if embed_type == "svd":
            edge_index = torch.stack([c_2, u_2], dim=0)
            padded_adj = to_dense_adj(edge_index, batch=batch)
            B, max_n, _ = padded_adj.shape
            deg = padded_adj.sum(dim=-1)
            inv_sqrt = deg.clamp_min(1e-12).pow(-0.5)
            ndeg = torch.diag_embed(inv_sqrt)
            lap = torch.eye(max_n, device=device, dtype=x.dtype).unsqueeze(0) - ndeg @ padded_adj @ ndeg
            evals, evecs = torch.linalg.eigh(lap)
            if evecs.size(-1) < pos_k + 1:
                evecs = F.pad(evecs, (0, (pos_k + 1) - evecs.size(-1)))
            pe_padded = evecs[:, :, 1 : pos_k + 1]
            node_counts = torch.bincount(batch, minlength=num_graphs)
            offsets = F.pad(torch.cumsum(node_counts, dim=0)[:-1], (1, 0))
            local_node_ids = torch.arange(N, device=device) - offsets[batch]
            pos_cat = pe_padded[batch, local_node_ids]
            if flip_sign:
                pos_cat = pos_cat * torch.randn_like(pos_cat).sign()
        else:
            pos_cat = torch.zeros((N, pos_k), device=device, dtype=x.dtype)
            node_counts = torch.bincount(batch, minlength=num_graphs)
            node_offsets = torch.cumsum(node_counts, dim=0)
            node_offsets = F.pad(node_offsets[:-1], (1, 0))
            for graph_idx in range(num_graphs):
                node_count = int(node_counts[graph_idx].item())
                if node_count == 0:
                    continue

                start = int(node_offsets[graph_idx].item())
                node_indices = torch.arange(start, start + node_count, device=device)

                edge_mask = batch[c_2] == graph_idx
                if not edge_mask.any():
                    continue

                local_edge_index = torch.stack([
                    c_2[edge_mask] - start,
                    u_2[edge_mask] - start,
                ], dim=0)
                local_pos = positional_embeddings_from_edge_index(
                    local_edge_index,
                    node_count,
                    k=pos_k,
                    flip_sign=flip_sign,
                )
                if local_pos.numel() == 0:
                    continue
                pos_cat[node_indices] = local_pos.to(device=device, dtype=x.dtype)
    else:
        pos_cat = torch.empty((N, 0), device=device, dtype=x.dtype)

    if not use_cls_token:
        return ETBatchContext(batch=batch, c_2=c_2, u_2=u_2, pos=pos_cat, num_graphs=num_graphs, original_node_count=N)

    cls_batch = torch.arange(num_graphs, device=device)
    batch_new = torch.cat([batch, cls_batch], dim=0)
    pos_new = F.pad(pos_cat, (0, 0, 0, num_graphs))
    
    node_ids = torch.arange(N, device=device)
    cls_ids = N + batch
    new_c2 = torch.cat([c_2, node_ids, cls_ids, torch.arange(N, N + num_graphs, device=device)], dim=0)
    new_u2 = torch.cat([u_2, cls_ids, node_ids, torch.arange(N, N + num_graphs, device=device)], dim=0)
    
    return ETBatchContext(
        batch=batch_new,
        c_2=new_c2,
        u_2=new_u2,
        pos=pos_new,
        num_graphs=num_graphs,
        original_node_count=N
    )
