"""Traditional message-passing GNN baselines."""

from __future__ import annotations

import torch
import torch.nn as nn


class GINBaseline(nn.Module):
    def __init__(self, in_dim, d, num_classes, num_layers=3, dropout=0.1):
        super().__init__()
        try:
            from torch_geometric.nn import GINConv, global_add_pool
        except ImportError as exc:
            raise ImportError("torch_geometric is required for GINBaseline") from exc
        self.encoder = nn.Linear(in_dim, d)
        self.global_add_pool = global_add_pool
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(d, 2 * d),
                nn.BatchNorm1d(2 * d),
                nn.ReLU(),
                nn.Linear(2 * d, d),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(d))
        self.readout = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(self, batch_data, task_level="graph"):
        x = self.encoder(batch_data.x)
        edge_index = torch.stack([batch_data.c_2, batch_data.u_2], dim=0)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        if task_level == "graph":
            x = self.global_add_pool(x, batch_data.batch)
        return self.readout(x), None


class GCNBaseline(nn.Module):
    def __init__(self, in_dim, d, num_classes, num_layers=3, dropout=0.1):
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv, global_add_pool
        except ImportError as exc:
            raise ImportError("torch_geometric is required for GCNBaseline") from exc
        self.encoder = nn.Linear(in_dim, d)
        self.global_add_pool = global_add_pool
        self.convs = nn.ModuleList()
        self.dropout = dropout
        for _ in range(num_layers):
            self.convs.append(GCNConv(d, d))
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, num_classes))

    def forward(self, batch_data, task_level="graph"):
        x = self.encoder(batch_data.x)
        edge_index = torch.stack([batch_data.c_2, batch_data.u_2], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        if task_level == "graph":
            x = self.global_add_pool(x, batch_data.batch)
        return self.readout(x), None


class GATBaseline(nn.Module):
    def __init__(self, in_dim, d, num_classes, num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        try:
            from torch_geometric.nn import GATConv, global_add_pool
        except ImportError as exc:
            raise ImportError("torch_geometric is required for GATBaseline") from exc
        self.encoder = nn.Linear(in_dim, d)
        self.global_add_pool = global_add_pool
        self.convs = nn.ModuleList()
        self.dropout = dropout
        for i in range(num_layers):
            out_channels = d // heads
            self.convs.append(GATConv(d if i == 0 else d, out_channels, heads=heads, dropout=dropout))
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, num_classes))

    def forward(self, batch_data, task_level="graph"):
        x = self.encoder(batch_data.x)
        edge_index = torch.stack([batch_data.c_2, batch_data.u_2], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        if task_level == "graph":
            x = self.global_add_pool(x, batch_data.batch)
        return self.readout(x), None
