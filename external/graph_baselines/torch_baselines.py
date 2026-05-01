from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, GINConv, global_mean_pool


class ExternalGraphBaseline(nn.Module):
    """Lightweight external baseline path for graph tasks.

    This lives under code/external by design and can be swapped via registry.
    It uses vectorized graph-level pooling + degree statistics.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"]
        b = batch["batch"]
        num_graphs = int(batch["num_graphs"].item())

        pooled = x.new_zeros((num_graphs, x.size(-1)))
        pooled.index_add_(0, b, x)
        counts = torch.bincount(b, minlength=num_graphs).to(x.dtype).unsqueeze(-1).clamp_min(1.0)
        mean_x = pooled / counts

        deg = x.new_zeros((x.size(0),))
        if batch["c_2"].numel() > 0:
            ones = torch.ones_like(batch["c_2"], dtype=x.dtype)
            deg = deg.index_add(0, batch["c_2"], ones)
        deg_sum = x.new_zeros((num_graphs,))
        deg_sum.index_add_(0, b, deg)
        mean_deg = (deg_sum / counts.squeeze(-1)).unsqueeze(-1)

        motif_count = x.new_zeros((num_graphs,))
        if batch["c_3"].numel() > 0:
            motif_ones = torch.ones_like(batch["c_3"], dtype=x.dtype)
            motif_count = motif_count.index_add(0, b[batch["c_3"]], motif_ones)
        motif_per_node = (motif_count / counts.squeeze(-1)).unsqueeze(-1)

        feat = torch.cat([mean_x, mean_deg, motif_per_node], dim=-1)
        out = self.net(feat)
        return out.squeeze(-1) if out.size(-1) == 1 else out


class _PyGGraphModelBase(nn.Module):
    def _graph_from_batch(self, batch: Dict[str, torch.Tensor]):
        x = batch["x"]
        edge_index = torch.stack([batch["c_2"], batch["u_2"]], dim=0) if batch["c_2"].numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=x.device)
        b = batch["batch"]
        return x, edge_index, b


class GCNGraphBaseline(_PyGGraphModelBase):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1) -> None:
        super().__init__()
        self.c1 = GCNConv(in_dim, hidden_dim)
        self.c2 = GCNConv(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, edge_index, b = self._graph_from_batch(batch)
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        g = global_mean_pool(x, b)
        out = self.head(g)
        return out.squeeze(-1) if out.size(-1) == 1 else out


class GATGraphBaseline(_PyGGraphModelBase):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1, heads: int = 4) -> None:
        super().__init__()
        self.c1 = GATConv(in_dim, hidden_dim // heads, heads=heads)
        self.c2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, edge_index, b = self._graph_from_batch(batch)
        x = F.elu(self.c1(x, edge_index))
        x = F.elu(self.c2(x, edge_index))
        g = global_mean_pool(x, b)
        out = self.head(g)
        return out.squeeze(-1) if out.size(-1) == 1 else out


class GINGraphBaseline(_PyGGraphModelBase):
    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 1) -> None:
        super().__init__()
        mlp1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        mlp2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.c1 = GINConv(mlp1)
        self.c2 = GINConv(mlp2)
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x, edge_index, b = self._graph_from_batch(batch)
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        g = global_mean_pool(x, b)
        out = self.head(g)
        return out.squeeze(-1) if out.size(-1) == 1 else out
