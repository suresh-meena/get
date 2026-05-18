from __future__ import annotations

import scipy.special
import sympy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


def calculate_theta2(d: int):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) / (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d + 1):
            inv_coeff.append(float(coeff[d - i]))
        thetas.append(inv_coeff)
    return thetas


class PolyConv(MessagePassing):
    def __init__(self, in_feats, out_feats, theta, activation=F.leaky_relu, lin=False, bias=False):
        super().__init__(aggr='add')
        self._theta = theta
        self._k = len(self._theta)
        self.activation = activation
        self.lin = lin
        if self.lin:
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)

    def forward(self, x, edge_index):
        # Compute D^-1/2 A D^-1/2 x
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        h = self._theta[0] * x
        curr_x = x
        for k in range(1, self._k):
            # L_norm x = x - (D^-1/2 A D^-1/2) x
            ax = self.propagate(edge_index, x=curr_x, norm=norm)
            curr_x = curr_x - ax
            h += self._theta[k] * curr_x
        
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class BWGNNBaseline(nn.Module):
    """
    BWGNN baseline ported from the external anomaly detection repo.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, d: int = 2):
        super().__init__()
        self.thetas = calculate_theta2(d=d)
        self.convs = nn.ModuleList([
            PolyConv(hidden_dim, hidden_dim, theta, lin=False) 
            for theta in self.thetas
        ])
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim * len(self.convs), hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, num_classes)
        self.act = nn.ReLU()

    def forward(self, batch_data: dict[str, torch.Tensor], **kwargs):
        x = batch_data["x"]
        edge_index = torch.stack([batch_data["c_2"], batch_data["u_2"]], dim=0) if batch_data["c_2"].numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=x.device)
        batch = batch_data["batch"]
        num_graphs = int(batch_data["y"].shape[0])

        h = self.act(self.linear1(x))
        h = self.act(self.linear2(h))
        
        h_all = []
        for conv in self.convs:
            h_all.append(conv(h, edge_index))
        
        h_final = torch.cat(h_all, dim=-1)
        h = self.act(self.linear3(h_final))
        
        # Pooling for graph classification
        pooled = h.new_zeros((num_graphs, h.size(-1)))
        pooled.index_add_(0, batch, h)
        counts = torch.bincount(batch, minlength=num_graphs).to(h.dtype).unsqueeze(-1).clamp_min(1.0)
        h_graph = pooled / counts
        
        logits = self.linear4(h_graph)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits
