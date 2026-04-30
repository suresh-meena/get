"""Baseline models and factory functions."""
from get.utils.registry import register_model
from get.models.get_model import GETModel
from get.models.et_core import ETFaithfulGraphModel
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
                nn.Linear(2 * d, d)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(d))
        self.readout = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes)
        )

    def forward(self, batch_data, task_level='graph'):
        x = self.encoder(batch_data.x)
        edge_index = torch.stack([batch_data.c_2, batch_data.u_2], dim=0)
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        if task_level == 'graph':
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

    def forward(self, batch_data, task_level='graph'):
        x = self.encoder(batch_data.x)
        edge_index = torch.stack([batch_data.c_2, batch_data.u_2], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        if task_level == 'graph':
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
            # For GAT, we divide d by heads to keep total hidden dimension constant if possible, 
            # or just use d/heads if we want to match GET's head structure.
            # Here we'll use d as the total output dimension of the multi-head attention.
            out_channels = d // heads
            self.convs.append(GATConv(d if i == 0 else d, out_channels, heads=heads, dropout=dropout))
        self.readout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, num_classes))

    def forward(self, batch_data, task_level='graph'):
        x = self.encoder(batch_data.x)
        edge_index = torch.stack([batch_data.c_2, batch_data.u_2], dim=0)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        if task_level == 'graph':
            x = self.global_add_pool(x, batch_data.batch)
        return self.readout(x), None


@register_model("pairwise")
@register_model("pairwiseget")
def PairwiseGET(in_dim, d=256, num_classes=1, **kwargs):
    """Pairwise-only baseline (lambda_3=0, lambda_m=0)"""
    return GETModel(in_dim, d, num_classes, lambda_3=0.0, lambda_m=0.0,
                    use_motif=False, use_memory=False, **kwargs)


@register_model("full")
@register_model("fullget")
def FullGET(in_dim, d=256, num_classes=1,
            lambda_2=1.0, lambda_3=0.5, lambda_m=1.0,
            beta_2=1.0, beta_3=1.0, beta_m=1.0, num_motif_types=2, **kwargs):
    """Full GET model with motif and memory branches active"""
    return GETModel(in_dim, d, num_classes,
                    lambda_2=lambda_2, lambda_3=lambda_3, lambda_m=lambda_m,
                    beta_2=beta_2, beta_3=beta_3, beta_m=beta_m,
                    num_motif_types=num_motif_types, **kwargs)


@register_model("etfaithful")
@register_model("etfaithfulgraphmodel")
def ETFaithful(in_dim, d, num_classes, **kwargs):
    """Paper-inspired ET with CLS token, Laplacian PE, masked energy attention, and HN memory."""
    return ETFaithfulGraphModel(in_dim, d, num_classes, **kwargs)


@register_model("gin")
@register_model("ginbaseline")
def _build_gin(in_dim, d, num_classes, **kwargs):
    return GINBaseline(in_dim, d, num_classes, **kwargs)


@register_model("gcn")
@register_model("gcnbaseline")
def _build_gcn(in_dim, d, num_classes, **kwargs):
    return GCNBaseline(in_dim, d, num_classes, **kwargs)


@register_model("gat")
@register_model("gatbaseline")
def _build_gat(in_dim, d, num_classes, **kwargs):
    return GATBaseline(in_dim, d, num_classes, **kwargs)
