"""ET-style LayerNorm with scalar gamma, optional vector bias, and analytical pullback."""
import torch
import torch.nn as nn


class EnergyLayerNorm(nn.Module):
    """ET-style LayerNorm with scalar gamma and optional vector bias.

    Provides an analytical backward method for efficient gradient pullback
    without requiring autograd, maintaining theoretical guarantees of the
    energy-based framework.
    """

    def __init__(self, dim, use_bias=True, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self.use_bias = bool(use_bias)
        self.gamma = nn.Parameter(torch.ones(()))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        xmeaned = x - mean
        rstd = torch.rsqrt((xmeaned.pow(2)).mean(dim=-1, keepdim=True) + self.eps)
        v = self.gamma * xmeaned * rstd
        if self.bias is not None:
            return v + self.bias
        return v

    def backward(self, x, grad_v):
        """Analytical pullback: dE/dx = (dE/dv) * (dv/dx)"""
        mean = x.mean(dim=-1, keepdim=True)
        xmeaned = x - mean
        var = (xmeaned.pow(2)).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(var + self.eps)
        rstd_sq = rstd ** 2

        term1 = grad_v
        term2 = grad_v.mean(dim=-1, keepdim=True)
        term3 = xmeaned * (grad_v * xmeaned).mean(dim=-1, keepdim=True) * rstd_sq

        grad_x = self.gamma * rstd * (term1 - term2 - term3)
        return grad_x
