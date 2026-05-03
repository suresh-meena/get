from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool


class GraphTransformerLayer(nn.Module):
    def __init__(self, d: int, num_heads: int, dropout: float = 0.0, ffn_ratio: int = 4, layer_norm: bool = True, residual: bool = True):
        super().__init__()
        self.attn = TransformerConv(d, d // num_heads, heads=num_heads, dropout=dropout)
        self.residual = residual
        self.layer_norm = layer_norm
        
        if layer_norm:
            self.norm1 = nn.LayerNorm(d)
            self.norm2 = nn.LayerNorm(d)
        
        # FFN: Linear -> ReLU -> Linear with shared/tied weights (mirroring external repo's matmul(h, W) and matmul(h, W.t()))
        # Actually the external repo uses h2 = matmul(h2, self.W), h2 = relu(h2), h2 = matmul(h2, self.W.t())
        # We'll implement it faithfully.
        self.W = nn.Parameter(torch.Tensor(d, d * ffn_ratio))
        nn.init.xavier_uniform_(self.W)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = x
        if self.layer_norm:
            h = self.norm1(h)
        
        # Attention
        attn_out = self.attn(h, edge_index)
        
        # FFN
        h2 = attn_out
        h2 = torch.matmul(h2, self.W)
        h2 = F.relu(h2)
        h2 = self.dropout(h2)
        h2 = torch.matmul(h2, self.W.t())
        
        out = attn_out + h2
        
        if self.residual:
            out = x + out
            
        if self.layer_norm:
            out = self.norm2(out)
            
        return out


class GraphTransformerBaseline(nn.Module):
    """
    Graph Transformer baseline mirroring the external repo's GraphTransformerNet.
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, num_heads: int = 4, n_layers: int = 2, dropout: float = 0.0, ffn_ratio: int = 2, layer_norm: bool = True, residual: bool = True):
        super().__init__()
        self.embedding = nn.Linear(in_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(hidden_dim, num_heads, dropout, ffn_ratio, layer_norm, residual)
            for _ in range(n_layers)
        ])
        
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, batch_data: dict[str, torch.Tensor]):
        x = batch_data["x"]
        edge_index = torch.stack([batch_data["c_2"], batch_data["u_2"]], dim=0) if batch_data["c_2"].numel() > 0 else torch.empty((2, 0), dtype=torch.long, device=x.device)
        batch = batch_data["batch"]
        
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h, edge_index)
            
        g = global_mean_pool(h, batch)
        logits = self.head(g)
        return logits.squeeze(-1) if logits.size(-1) == 1 else logits
