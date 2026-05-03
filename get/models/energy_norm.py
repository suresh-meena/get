from __future__ import annotations

import torch
import torch.nn as nn


class EnergyLayerNorm(nn.Module):
    """
    ET-style normalization used as a forward activation before the energy block.

    This matches the external JAX implementation more closely than nn.LayerNorm:
    - scalar gamma
    - optional bias
    - normalization over the last dimension only
    """

    def __init__(self, in_dim: int, *, use_bias: bool = True, eps: float = 1e-5) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.use_bias = bool(use_bias)
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.ones(1))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(self.in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = x - x.mean(dim=-1, keepdim=True)
        norm = torch.sqrt(centered.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = self.gamma * centered / norm
        if self.use_bias:
            out = out + self.bias
        return out
