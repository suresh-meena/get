"""
ETGraphClassifier — native PyTorch ET graph model.

Reference: external/energy-transformer-graph/src/model/{core,et}.py
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from get.energy.ops import segment_logsumexp
from .energy_norm import EnergyLayerNorm
from .et_utils import build_et_batch_context


class _AttentionEnergy(nn.Module):
    """Graph-masked attention energy block."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, use_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        self.Wq = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.Wk = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.Hw = nn.Parameter(torch.empty(num_heads, num_heads))
        
        if use_bias:
            self.Bq = nn.Parameter(torch.empty(num_heads, head_dim))
            self.Bk = nn.Parameter(torch.empty(num_heads, head_dim))
        else:
            self.register_parameter("Bq", None)
            self.register_parameter("Bk", None)

        init_beta = 1.0 / math.sqrt(head_dim)
        self.betas = nn.Parameter(torch.full((num_heads,), init_beta))

        nn.init.normal_(self.Wq, std=0.002)
        nn.init.normal_(self.Wk, std=0.002)
        nn.init.normal_(self.Hw, std=0.002)
        if use_bias:
            nn.init.zeros_(self.Bq)
            nn.init.zeros_(self.Bk)

    def energy(
        self,
        g: torch.Tensor,        # [N, D]
        c_2: torch.Tensor,        # [E] source nodes
        u_2: torch.Tensor,        # [E] dest nodes
        adj: Optional[torch.Tensor] = None,
        compute_corr: bool = False,
    ) -> torch.Tensor:
        N, D = g.shape
        E = c_2.numel()
        
        Q = torch.einsum("nd, hzd -> nhz", g, self.Wq)
        K = torch.einsum("nd, hzd -> nhz", g, self.Wk)
        
        if self.use_bias:
            Q = Q + self.Bq.unsqueeze(0)
            K = K + self.Bk.unsqueeze(0)

        H = self.num_heads
        betas = self.betas.abs().clamp_min(1e-6)

        if adj is not None and adj.dim() == 3:
            a1 = torch.einsum("h, qhz, khz -> hqk", betas, Q, K)
            a11 = torch.einsum("qkh,hm->qkm", a1.permute(1, 2, 0), self.Hw) * adj
            a11 = torch.where(a11 == 0, torch.full_like(a11, float("-inf")), a11)
            a21 = torch.logsumexp(a11, dim=1)
            a21 = torch.where(a21 == float("-inf"), torch.zeros_like(a21), a21)
            return ((-1.0 / betas) * a21.sum(dim=0)).sum()

        if E == 0:
            return g.new_zeros(1).squeeze()

        chunk_size = 500_000
        scores_max = torch.full((N, H), float('-inf'), device=g.device, dtype=g.dtype)
        for i in range(0, E, chunk_size):
            end = min(i + chunk_size, E)
            c_chunk = c_2[i:end]
            u_chunk = u_2[i:end]
            s_chunk = (Q[c_chunk] * K[u_chunk]).sum(dim=-1)
            if compute_corr:
                corr = (g[c_chunk] * g[u_chunk]).sum(dim=-1, keepdim=True)
                s_chunk = s_chunk * corr
            s_final = s_chunk * betas.unsqueeze(0)
            scores_max.scatter_reduce_(0, c_chunk.unsqueeze(-1).expand(-1, H), s_final, reduce="amax", include_self=True)

        exp_sum = g.new_zeros(N, H)
        for i in range(0, E, chunk_size):
            end = min(i + chunk_size, E)
            c_chunk = c_2[i:end]
            u_chunk = u_2[i:end]
            s_chunk = (Q[c_chunk] * K[u_chunk]).sum(dim=-1)
            if compute_corr:
                corr = (g[c_chunk] * g[u_chunk]).sum(dim=-1, keepdim=True)
                s_chunk = s_chunk * corr
            s_final = s_chunk * betas.unsqueeze(0)
            s_shifted = s_final - scores_max[c_chunk]
            exp_sum.scatter_add_(0, c_chunk.unsqueeze(-1).expand(-1, H), torch.exp(s_shifted))

        lse = torch.log(exp_sum.clamp_min(1e-12)) + scores_max
        return -(1.0 / betas.unsqueeze(0) * lse).sum()


class _HNNEnergy(nn.Module):
    """Hopfield network channel energy."""

    def __init__(self, hidden_dim: int, multiplier: float = 4.0, chn_type: str = "relu", use_bias: bool = True) -> None:
        super().__init__()
        hid_dim = int(multiplier * hidden_dim)
        self.W = nn.Linear(hidden_dim, hid_dim, bias=use_bias)
        self.chn_type = chn_type.lower()
        self.hid_dim = hid_dim

    def energy(self, g: torch.Tensor) -> torch.Tensor:
        h = self.W(g)
        if self.chn_type == "relu":
            A = F.relu(h)
            return -0.5 * (A * A).sum()
        elif self.chn_type == "gelu":
            A = F.gelu(h)
            return -0.5 * (A * A).sum()
        elif self.chn_type == "lse":
            beta = self.hid_dim ** 0.5
            return (-1.0 / beta) * torch.logsumexp(beta * h, dim=-1).sum()
        else:
            raise ValueError(f"Unknown chn_type: {self.chn_type}")


class _ETBlock(nn.Module):
    """A single ET energy block."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        multiplier: float,
        chn_type: str,
        use_bias_attn: bool,
        use_bias_norm: bool,
        use_bias_chn: bool,
    ) -> None:
        super().__init__()
        self.norm = EnergyLayerNorm(hidden_dim, use_bias=use_bias_norm)
        self.attn = _AttentionEnergy(hidden_dim, num_heads, head_dim, use_bias=use_bias_attn)
        self.hnn = _HNNEnergy(hidden_dim, multiplier=multiplier, chn_type=chn_type, use_bias=use_bias_chn)

    def step(
        self,
        x: torch.Tensor,
        c_2: torch.Tensor,
        u_2: torch.Tensor,
        adj: Optional[torch.Tensor],
        alpha: float,
        compute_corr: bool = True,
        noise_std: float = 0.02,
        vary_noise: bool = False,
        step_idx: int = 0,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.norm(x)
        g = g.requires_grad_(True)
        e = self.energy_from_g(g, c_2, u_2, adj=adj)
        grad, = torch.autograd.grad(e, g, create_graph=create_graph)
        
        if vary_noise and noise_std > 0.0:
            current_noise = noise_std / pow(1.0 + float(step_idx), 0.55)
        else:
            current_noise = noise_std
            
        noise = torch.zeros_like(grad)
        if current_noise > 0.0:
            noise = torch.randn_like(grad) * (alpha ** 0.5) * current_noise
            
        x_new = x - alpha * grad + noise
        return x_new, e.detach()

    def energy_from_g(
        self,
        g: torch.Tensor,
        c_2: torch.Tensor,
        u_2: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.attn.energy(g, c_2, u_2, adj=adj) + self.hnn.energy(g)


class ETGraphClassifier(nn.Module):
    """Energy Transformer for graph classification."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        head_dim: int,
        num_steps: int = 5,
        num_blocks: int = 1,
        alpha: float = 0.1,
        multiplier: float = 4.0,
        chn_type: str = "relu",
        use_bias_attn: bool = False,
        use_bias_chn: bool = False,
        use_bias_norm: bool = True,
        use_cls_token: bool = True,
        pos_k: int = 15,
        embed_type: str = "eigen",
        flip_sign: bool = False,
        compute_corr: bool = True,
        noise_std: float = 0.02,
        vary_noise: bool = False,
        readout_mode: str = "cls",
        update_damping: float = 0.0,
        inference_mode_train: str = "fixed",
        inference_mode_eval: str = "fixed",
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.num_blocks = num_blocks
        self.alpha = float(alpha) * (1.0 - update_damping)
        self.use_cls_token = bool(use_cls_token)
        self.pos_k = int(pos_k)
        self.embed_type = str(embed_type)
        self.flip_sign = bool(flip_sign)
        self.compute_corr = bool(compute_corr)
        self.noise_std = float(noise_std)
        self.vary_noise = bool(vary_noise)
        self.readout_mode = str(readout_mode).lower()
        self.requires_double_backward = True

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.blocks = nn.ModuleList([
            _ETBlock(hidden_dim, num_heads, head_dim, multiplier, chn_type, use_bias_attn, use_bias_norm, use_bias_chn)
            for _ in range(num_blocks)
        ])

        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        pos_dim = self.pos_k if self.embed_type.lower() != "svd" else 2 * self.pos_k
        self.pos_proj = nn.Linear(pos_dim, hidden_dim) if pos_dim > 0 else None
        self.readout = nn.Linear(hidden_dim, num_classes)

    def _pool_graph_mean(self, x_state: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        out = x_state.new_zeros((num_graphs, x_state.size(-1)))
        out.index_add_(0, batch_idx, x_state)
        counts = torch.bincount(batch_idx, minlength=num_graphs).to(dtype=x_state.dtype).unsqueeze(-1).clamp_min(1.0)
        return out / counts

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        inference_mode: Optional[str] = None,
        return_solver_stats: bool = False,
    ):
        ctx = build_et_batch_context(
            batch_data,
            use_cls_token=self.use_cls_token,
            pos_k=self.pos_k,
            embed_type=self.embed_type,
            flip_sign=self.flip_sign,
        )
        create_graph = self.training
        energy_trace: List[float] = []

        x = self.encoder(batch_data["x"])
        if self.use_cls_token:
            x = torch.cat([x, self.cls_token.expand(ctx.num_graphs, -1)], dim=0)
        if self.pos_proj is not None and ctx.pos.numel() > 0:
            x = x + self.pos_proj(ctx.pos.to(dtype=x.dtype, device=x.device))

        for b_idx, block in enumerate(self.blocks):
            for s_idx in range(self.num_steps):
                x, e_val = block.step(
                    x, ctx.c_2, ctx.u_2, None, self.alpha,
                    compute_corr=self.compute_corr, noise_std=self.noise_std,
                    vary_noise=self.vary_noise, step_idx=s_idx + b_idx * self.num_steps,
                    create_graph=create_graph
                )
                energy_trace.append(float(e_val))

        if self.readout_mode == "cls" and self.use_cls_token:
            graph_repr = x[ctx.original_node_count:]
        else:
            graph_repr = self._pool_graph_mean(x[:ctx.original_node_count], batch_data["batch"], ctx.num_graphs)

        logits = self.readout(graph_repr)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_solver_stats:
            return logits, energy_trace, {"mode": "et_fixed"}
        return logits
