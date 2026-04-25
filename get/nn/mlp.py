"""Reusable MLP building blocks."""
import torch.nn as nn


class StableMLP(nn.Module):
    """Two-layer MLP with optional LayerNorm, GELU activation, and dropout.

    Used as the input encoder (Eq 20 in the writeup) and task readout heads.
    """

    def __init__(self, in_dim, out_dim, hidden_dim=None, activation=nn.GELU, dropout=0.1, final_norm=True):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim) if final_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.net(x))
