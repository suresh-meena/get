"""
ETGraphClassifier — faithful PyTorch port of the JAX Energy Transformer graph model.

Reference: external/energy-transformer-graph/src/model/{core,et}.py

Architecture
------------
The full ET energy at each step is:

    E(g, adj) = E_att(g, adj) + E_hnn(g)

where:
    g            = EnergyLayerNorm(x)              [N, D]
    E_att(g,adj) = (-1/β) * Σ_i Σ_h lse_h(β * A_adj[i])
    E_hnn(g)     = -0.5 * ||ReLU(W g)||²   (summed over nodes and hidden units)

The update rule at each step t is:
    x ← x - α * ∇_g E(g, adj)   where g = ELN(x)

Because g = ELN(x) is differentiable w.r.t. x, we use torch.autograd.grad to get
∇_x E (chain rule through ELN) and apply the update to x directly.

For graph classification the final x is mean-pooled per graph, then projected to logits.

Batch format
------------
Same batch dict as EnergyGraphClassifier:
    x          [total_nodes, in_dim]
    batch      [total_nodes]           node → graph index
    num_graphs scalar
    c_2, u_2   edge incidence (used to build adjacency mask)

Interface
---------
Same __init__ signature conventions as EnergyGraphClassifier where parameters are shared.
New parameters: alpha (step size), multiplier (HNN hidden expansion), chn_type ('relu'|'gelu'|'lse').
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .energy_norm import EnergyLayerNorm


# ---------------------------------------------------------------------------
# Energy sub-modules
# ---------------------------------------------------------------------------

class _AttentionEnergy(nn.Module):
    """
    Graph-masked attention energy block.

    E_att = (-1/β) * Σ_i Σ_h logsumexp_{j ∈ N(i)} ( β * (Q_i · K_j) / sqrt(d) )

    Q, K projections share a learnable per-head inverse-temperature β (scalar per head).
    The adjacency mask is derived from the batch's edge index so that logsumexp only
    runs over actual neighbours, matching the JAX `A1 * adj` masking logic.
    """

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, use_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        # [H, d, D] — weight matrices, matching JAX Wq/Wk shape
        self.Wq = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.Wk = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        
        # Head-mixing weight Hw: [H, H]
        self.Hw = nn.Parameter(torch.empty(num_heads, num_heads))
        
        if use_bias:
            self.Bq = nn.Parameter(torch.empty(num_heads, head_dim))
            self.Bk = nn.Parameter(torch.empty(num_heads, head_dim))
        else:
            self.register_parameter("Bq", None)
            self.register_parameter("Bk", None)

        # Per-head inverse temperature, initialised to 1/sqrt(d) like the JAX default
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
        g: torch.Tensor,          # [N, D]
        c_2: torch.Tensor,        # [E] source nodes (edge list)
        u_2: torch.Tensor,        # [E] dest nodes
    ) -> torch.Tensor:
        """Scalar attention energy (summed over all nodes and heads)."""
        N, D = g.shape
        H, d = self.num_heads, self.head_dim

        # Q, K: [N, H, d]
        Q = torch.einsum("nd, hzd -> nhz", g, self.Wq)
        K = torch.einsum("nd, hzd -> nhz", g, self.Wk)
        
        if self.use_bias:
            Q = Q + self.Bq.unsqueeze(0)
            K = K + self.Bk.unsqueeze(0)

        # Raw scores per edge: dot(Q_src, K_dst) per head  →  [E, H]
        betas = self.betas.abs().clamp_min(1e-6)  # keep positive

        if c_2.numel() == 0:
            return g.new_zeros(1).squeeze()

        # [E, H]
        scores = (Q[c_2] * K[u_2]).sum(dim=-1)
        scores = scores * betas.unsqueeze(0)
        
        # Apply head mixing Hw: [E, H] @ [H, H] -> [E, H]
        # This matches the JAX logic: (A1.transpose(1, 2, 0) @ self.Hw)
        scores = scores @ self.Hw

        # Segment logsumexp per (dst_node, head).
        lse = _segment_logsumexp_2d(scores, c_2, N)     # [N, H]

        # Mask out isolated nodes
        in_degree = torch.bincount(c_2, minlength=N)
        lse = lse.masked_fill((in_degree == 0).unsqueeze(1), 0.0)

        # E_att = (-1/β) * Σ_i Σ_h lse_h(i)
        inv_beta = (1.0 / betas).unsqueeze(0)     # [1, H]
        E_att = -(inv_beta * lse).sum()
        return E_att


class _HNNEnergy(nn.Module):
    """
    Hopfield network channel energy.

    E_hnn = -0.5 * ||f(W g)||²

    where f is relu (default), gelu, or logsumexp-based (lse).
    W: [D, D_hid]
    """

    def __init__(self, hidden_dim: int, multiplier: float = 4.0, chn_type: str = "relu") -> None:
        super().__init__()
        hid_dim = int(multiplier * hidden_dim)
        self.W = nn.Linear(hidden_dim, hid_dim, bias=True)
        self.chn_type = chn_type.lower()
        self.hid_dim = hid_dim

    def energy(self, g: torch.Tensor) -> torch.Tensor:
        """Scalar Hopfield channel energy."""
        h = self.W(g)   # [N, hid_dim]
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _segment_logsumexp_2d(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Compute logsumexp over segments for a 2D source tensor [E, H].
    Returns [N, H] output.

    Numerically stable: subtracts per-segment max before exponentiating.
    """
    N, H = num_segments, src.size(1)
    E = src.size(0)

    idx = segment_ids.unsqueeze(1).expand(E, H)  # [E, H]

    # Max per segment
    seg_max = src.new_full((N, H), float('-inf'))
    seg_max.scatter_reduce_(0, idx, src, reduce="amax", include_self=True)

    # Expand max back to edges, shift
    max_expanded = seg_max[segment_ids]   # [E, H]
    # Replace -inf max (empty segment) with 0 to avoid NaN
    max_expanded = torch.where(max_expanded == float('-inf'), src.new_zeros(()), max_expanded)
    x_shifted = src - max_expanded

    # Sum of exps
    sum_exp = src.new_zeros(N, H)
    sum_exp.scatter_add_(0, idx, x_shifted.exp())

    lse = sum_exp.log() + seg_max
    # Where seg_max is still -inf (empty segments), force lse to -inf
    lse = torch.where(seg_max == float('-inf'), seg_max, lse)
    return lse


# ---------------------------------------------------------------------------
# ET Block (one block = one ELN + attention energy + HNN energy)
# ---------------------------------------------------------------------------

class _ETBlock(nn.Module):
    """A single ET energy block: attention energy + Hopfield channel energy."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        multiplier: float,
        chn_type: str,
        use_bias_attn: bool,
    ) -> None:
        super().__init__()
        self.norm = EnergyLayerNorm(hidden_dim, use_bias=True)
        self.attn = _AttentionEnergy(hidden_dim, num_heads, head_dim, use_bias=use_bias_attn)
        self.hnn = _HNNEnergy(hidden_dim, multiplier=multiplier, chn_type=chn_type)

    def step(
        self,
        x: torch.Tensor,
        c_2: torch.Tensor,
        u_2: torch.Tensor,
        alpha: float,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """
        One gradient descent step.
        Computes grad w.r.t. normalized state g, applies update to original state x.
        """
        g = self.norm(x)
        g = g.requires_grad_(True)
        e = self.energy_from_g(g, c_2, u_2)
        grad, = torch.autograd.grad(e, g, create_graph=create_graph)
        x_new = x - alpha * grad
        return x_new, e.detach().item()

    def energy_from_g(self, g: torch.Tensor, c_2: torch.Tensor, u_2: torch.Tensor) -> torch.Tensor:
        """Total energy from normalized state g."""
        return self.attn.energy(g, c_2, u_2) + self.hnn.energy(g)

    def energy(self, x: torch.Tensor, c_2: torch.Tensor, u_2: torch.Tensor) -> torch.Tensor:
        """Total energy from original state x (includes ELN)."""
        g = self.norm(x)
        return self.energy_from_g(g, c_2, u_2)


# ---------------------------------------------------------------------------
# Full ET Graph Classifier
# ---------------------------------------------------------------------------

class ETGraphClassifier(nn.Module):
    """
    Energy Transformer for graph classification.

    Follows the same forward interface as EnergyGraphClassifier so the existing
    trainers, DataLoaders, and evaluation scripts work without modification.

    Key differences from GET:
    - Energy is the sum of a graph-masked attention term and a Hopfield channel term
    - No motif (3-body) or memory branch
    - Update rule: x ← x - α * ∇_x E  (single gradient step per iteration)
    - Multiple ET blocks (depth) can be stacked; each block has its own ELN + params
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_heads: int,
        head_dim: int,
        num_steps: int = 5,            # gradient descent iterations per block
        num_blocks: int = 1,           # number of stacked ET blocks
        alpha: float = 0.1,            # step size
        multiplier: float = 4.0,       # HNN hidden expansion ratio
        chn_type: str = "relu",        # "relu" | "gelu" | "lse"
        use_bias_attn: bool = False,   # bias in Q/K projections (matches JAX default)
        update_damping: float = 0.0,   # optional damping: effective_alpha = alpha * (1 - damping)
        inference_mode_train: str = "fixed",   # always "fixed" for ET (no Armijo)
        inference_mode_eval: str = "fixed",
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_steps = num_steps
        self.num_blocks = num_blocks
        self.alpha = float(alpha)
        damping = float(max(0.0, min(update_damping, 1.0)))
        self.effective_alpha = self.alpha * (1.0 - damping)
        self.inference_mode_train = inference_mode_train
        self.inference_mode_eval = inference_mode_eval

        # Training uses create_graph=True for higher-order autodiff
        self.requires_double_backward = True

        # Input encoder: linear → ReLU → linear, same as GET
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stack of ET blocks
        self.blocks = nn.ModuleList([
            _ETBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                multiplier=multiplier,
                chn_type=chn_type,
                use_bias_attn=use_bias_attn,
            )
            for _ in range(num_blocks)
        ])

        self.readout = nn.Linear(hidden_dim, num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        inference_mode: Optional[str] = None,
        return_solver_stats: bool = False,
    ):
        x = self.encoder(batch_data["x"])
        c_2 = batch_data["c_2"]
        u_2 = batch_data["u_2"]
        num_graphs = int(batch_data["num_graphs"].item())
        batch_idx = batch_data["batch"]

        create_graph = self.training
        energy_trace: List[float] = []
        alpha = self.effective_alpha

        for block in self.blocks:
            for _ in range(self.num_steps):
                x, e_val = block.step(x, c_2, u_2, alpha=alpha, create_graph=create_graph)
                energy_trace.append(e_val)

        # Mean pool per graph
        pooled = self._pool(x, batch_idx, num_graphs)
        logits = self.readout(pooled)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_solver_stats:
            stats = {
                "mode": "et_fixed",
                "alpha": self.effective_alpha,
                "num_steps": self.num_steps * self.num_blocks,
                "grad_norms": [],  # not tracked to avoid CUDA syncs
            }
            return logits, energy_trace, stats
        return logits

    def _pool(self, x: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        out = x.new_zeros(num_graphs, x.size(-1))
        out.index_add(0, batch_idx, x)
        counts = torch.bincount(batch_idx, minlength=num_graphs).to(dtype=x.dtype).unsqueeze(-1).clamp_min(1.0)
        return out / counts
