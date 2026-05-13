from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from get.energy.ops import (
    positive_param, inverse_temperature,
    segment_logsumexp, scatter_add_nd, fused_motif_dot,
)


class EnergyBranch(nn.Module):
    name: str = "base"

    def forward(self, state: Dict[str, torch.Tensor], batch: Any, context: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError


class QuadraticBranch(EnergyBranch):
    name = "quadratic"

    def forward(self, state, batch, context):
        H = state["H"]
        batch_idx = batch.batch if hasattr(batch, "batch") else context.get("batch")
        num_graphs = context.get("num_graphs", int(batch_idx.max().item() + 1) if batch_idx is not None else 1)
        per_graph = H.new_zeros(num_graphs)
        per_graph.scatter_add_(0, batch_idx, (H * H).sum(dim=-1))
        return 0.5 * per_graph


class PairwiseBranch(EnergyBranch):
    name = "pairwise"

    def forward(self, state, batch, context):
        H = state["H"]
        params = context.get("params", {})
        if not params.get("use_pairwise", True):
            return H.new_zeros(context.get("num_graphs", 1))
        lambda_2 = positive_param(params, "lambda_2")
        if lambda_2 <= 1e-6:
            return H.new_zeros(context.get("num_graphs", 1))

        proj = context.get("projections", {})
        Q2 = proj.get("Q2")
        K2 = proj.get("K2")
        c_2 = batch.c_2 if hasattr(batch, "c_2") else context.get("c_2")
        u_2 = batch.u_2 if hasattr(batch, "u_2") else context.get("u_2")
        batch_idx = batch.batch if hasattr(batch, "batch") else context.get("batch")
        num_graphs = context.get("num_graphs", int(batch_idx.max().item() + 1) if batch_idx is not None else 1)
        num_nodes = H.size(0)

        if Q2 is None or c_2 is None or c_2.numel() == 0:
            return H.new_zeros(num_graphs)

        scale = Q2.size(-1) ** 0.5
        ell_2 = (Q2[c_2] * K2[u_2]).sum(dim=-1) / scale

        a_2 = proj.get("a_2")
        if a_2 is not None:
            ell_2 = ell_2 + a_2

        beta_2 = inverse_temperature(params, "beta_2")
        lse_2 = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes, dim=0)
        if c_2.numel() > 0:
            counts = torch.bincount(c_2, minlength=num_nodes)
            empty = counts.eq(0)
            if empty.any():
                lse_2 = lse_2.masked_fill(empty.view(-1, *([1] * (lse_2.dim() - 1))), 0.0)

        scaler = context.get("degree_scaler")
        if scaler is not None:
            lse_2 = lse_2 * scaler.unsqueeze(-1)

        graph_lse = scatter_add_nd(H.new_zeros((num_graphs, lse_2.shape[-1])), batch_idx, lse_2, dim=0)
        return (lambda_2 / beta_2) * graph_lse


class MotifBranch(EnergyBranch):
    name = "motif"

    def forward(self, state, batch, context):
        H = state["H"]
        params = context.get("params", {})
        if not params.get("use_motif", True):
            return H.new_zeros(context.get("num_graphs", 1))
        lambda_3 = positive_param(params, "lambda_3")
        if lambda_3 <= 1e-6:
            return H.new_zeros(context.get("num_graphs", 1))

        proj = context.get("projections", {})
        Q3 = proj.get("Q3")
        K3 = proj.get("K3")
        c_3 = batch.c_3 if hasattr(batch, "c_3") else context.get("c_3")
        u_3 = batch.u_3 if hasattr(batch, "u_3") else context.get("u_3")
        v_3 = batch.v_3 if hasattr(batch, "v_3") else context.get("v_3")
        t_tau = batch.t_tau if hasattr(batch, "t_tau") else context.get("t_tau")
        batch_idx = batch.batch if hasattr(batch, "batch") else context.get("batch")
        num_graphs = context.get("num_graphs", int(batch_idx.max().item() + 1) if batch_idx is not None else 1)
        num_nodes = H.size(0)
        R = int(params.get("R", 1))

        if Q3 is None or c_3 is None or c_3.numel() == 0:
            return H.new_zeros(num_graphs)

        scale = (R * Q3.size(-1)) ** 0.5
        T_params = params.get("T_tau", context.get("T_tau"))
        if T_params is None or t_tau.numel() == 0:
            T_selected = None
        else:
            if t_tau.max() >= T_params.size(0):
                t_tau = torch.clamp(t_tau, max=T_params.size(0) - 1)
            T_selected = T_params[t_tau]
        ell_3 = fused_motif_dot(Q3[c_3], K3[u_3], K3[v_3], T_selected) / scale

        beta_3 = inverse_temperature(params, "beta_3")
        lse_3 = segment_logsumexp(beta_3 * ell_3, c_3, num_nodes, dim=0)
        if c_3.numel() > 0:
            counts = torch.bincount(c_3, minlength=num_nodes)
            empty = counts.eq(0)
            if empty.any():
                lse_3 = lse_3.masked_fill(empty.view(-1, *([1] * (lse_3.dim() - 1))), 0.0)

        scaler = context.get("degree_scaler")
        if scaler is not None:
            lse_3 = lse_3 * scaler.unsqueeze(-1)

        graph_lse = scatter_add_nd(H.new_zeros((num_graphs, lse_3.shape[-1])), batch_idx, lse_3, dim=0)
        return (lambda_3 / beta_3) * graph_lse


class MemoryBranch(EnergyBranch):
    name = "memory"

    def forward(self, state, batch, context):
        H = state["H"]
        params = context.get("params", {})
        if not params.get("use_memory", True):
            return H.new_zeros(context.get("num_graphs", 1))
        lambda_m = positive_param(params, "lambda_m")
        if lambda_m <= 1e-6 or params.get("K", 0) <= 0:
            return H.new_zeros(context.get("num_graphs", 1))

        proj = context.get("projections", {})
        Qm = proj.get("Qm")
        Km = proj.get("Km")
        batch_idx = batch.batch if hasattr(batch, "batch") else context.get("batch")
        num_graphs = context.get("num_graphs", int(batch_idx.max().item() + 1) if batch_idx is not None else 1)

        if Qm is None or Km is None:
            return H.new_zeros(num_graphs)

        scale = Qm.size(-1) ** 0.5
        ell_m = torch.einsum("nhd,hkd->nhk", Qm, Km) / scale

        beta_m = inverse_temperature(params, "beta_m")
        lse_m = torch.logsumexp(beta_m * ell_m, dim=-1)
        per_node = (lambda_m / beta_m) * lse_m
        per_graph = scatter_add_nd(H.new_zeros((num_graphs, per_node.shape[-1])), batch_idx, per_node, dim=0)
        return per_graph


class GlobalAttentionBranch(EnergyBranch):
    name = "global_attention"

    def __init__(self, max_global_nodes: int = 512):
        super().__init__()
        self.max_global_nodes = max_global_nodes

    def forward(self, state, batch, context):
        H = state["H"]
        params = context.get("params", {})
        if not params.get("use_global_attention", False):
            return H.new_zeros(context.get("num_graphs", 1))
        lambda_g = positive_param(params, "lambda_g")
        if lambda_g <= 1e-6:
            return H.new_zeros(context.get("num_graphs", 1))

        proj = context.get("projections", {})
        Qg = proj.get("Qg")
        Kg = proj.get("Kg")
        batch_idx = batch.batch if hasattr(batch, "batch") else context.get("batch")
        num_graphs = context.get("num_graphs", int(batch_idx.max().item() + 1) if batch_idx is not None else 1)

        if Qg is None or Kg is None:
            return H.new_zeros(num_graphs)

        scale = Qg.size(-1) ** 0.5
        beta_g = inverse_temperature(params, "beta_g")

        # Avoid dense N x N masking across the whole batch. Compute per-graph
        # attention blocks using contiguous node ranges from batch assignments.
        counts = torch.bincount(batch_idx, minlength=num_graphs)
        if counts.numel() == 0:
            return H.new_zeros(num_graphs)
        max_nodes_per_graph = int(counts.max().item())
        if max_nodes_per_graph > self.max_global_nodes:
            raise RuntimeError(
                f"Global attention graph size {max_nodes_per_graph} exceeds "
                f"max_global_nodes={self.max_global_nodes}. Use sparse global "
                "attention or reduce per-graph size."
            )

        if batch_idx.numel() > 1 and not torch.all(batch_idx[:-1] <= batch_idx[1:]):
            same_graph = batch_idx[:, None].eq(batch_idx[None, :])
            if Qg.dim() == 2:
                attn = torch.mm(Qg, Kg.t()) / scale
                attn = attn.masked_fill(~same_graph, float("-inf"))
                lse_g = torch.logsumexp(beta_g * attn, dim=-1)
                per_graph = H.new_zeros(num_graphs)
                per_graph.scatter_add_(0, batch_idx, lse_g)
            else:
                attn = torch.einsum("nhd,mhd->nhm", Qg, Kg) / scale
                attn = attn.masked_fill(~same_graph[:, None, :], float("-inf"))
                lse_g = torch.logsumexp(beta_g * attn, dim=-1)
                per_graph = H.new_zeros((num_graphs, lse_g.size(-1)))
                per_graph.scatter_add_(0, batch_idx[:, None].expand_as(lse_g), lse_g)
            return (lambda_g / beta_g) * per_graph

        chunk_size = int(params.get("global_chunk_size", 256))
        chunk_size = max(16, chunk_size)

        starts = torch.cumsum(counts, dim=0) - counts
        if Qg.dim() == 2:
            per_graph = H.new_zeros(num_graphs)
            for gidx in range(num_graphs):
                ng = int(counts[gidx].item())
                if ng == 0:
                    continue
                s = int(starts[gidx].item())
                e = s + ng
                qg = Qg[s:e]
                kg = Kg[s:e]
                lse_g = qg.new_full((ng,), float("-inf"))
                for ks in range(0, ng, chunk_size):
                    ke = min(ks + chunk_size, ng)
                    scores = torch.mm(qg, kg[ks:ke].t()) / scale
                    chunk_lse = torch.logsumexp(beta_g * scores, dim=-1)
                    lse_g = torch.logaddexp(lse_g, chunk_lse)
                per_graph[gidx] = lse_g.sum()
        elif Qg.dim() == 3:
            per_graph = H.new_zeros((num_graphs, Qg.size(1)))
            for gidx in range(num_graphs):
                ng = int(counts[gidx].item())
                if ng == 0:
                    continue
                s = int(starts[gidx].item())
                e = s + ng
                qg = Qg[s:e]  # [ng, H, d]
                kg = Kg[s:e]  # [ng, H, d]
                lse_g = qg.new_full((ng, qg.size(1)), float("-inf"))
                for ks in range(0, ng, chunk_size):
                    ke = min(ks + chunk_size, ng)
                    scores = torch.einsum("nhd,mhd->nhm", qg, kg[ks:ke]) / scale
                    chunk_lse = torch.logsumexp(beta_g * scores, dim=-1)
                    lse_g = torch.logaddexp(lse_g, chunk_lse)
                per_graph[gidx] = lse_g.sum(dim=0)
        else:
            raise ValueError(f"Expected Qg/Kg to be 2D or 3D, got Qg.dim()={Qg.dim()}")
        return (lambda_g / beta_g) * per_graph


# Registry
ENERGY_BRANCHES: Dict[str, EnergyBranch] = {}


def register_branch(branch: EnergyBranch) -> EnergyBranch:
    ENERGY_BRANCHES[branch.name] = branch
    return branch


def get_branch(name: str) -> EnergyBranch:
    branch = ENERGY_BRANCHES.get(name)
    if branch is None:
        raise KeyError(f"Unknown energy branch '{name}'. Available: {list(ENERGY_BRANCHES.keys())}")
    return branch


def enabled_branches_from_config(model_cfg: Any) -> Dict[str, bool]:
    """Extract enabled branch flags from a config object."""
    return {
        "pairwise": bool(getattr(model_cfg, "pairwise", True)),
        "motif": bool(getattr(model_cfg, "motif", True)),
        "memory": bool(getattr(model_cfg, "memory", True)),
        "global_attention": bool(getattr(model_cfg, "global_attention", False)),
        "structural_memory": bool(getattr(model_cfg, "structural_memory", False)),
        "dynamic_edges": bool(getattr(model_cfg, "dynamic_edges", False)),
    }


class _QuadraticBranchCached(QuadraticBranch):
    """Single cached instance to avoid re-creating on every forward call."""

_quad_branch = _QuadraticBranchCached()


class ComposedEnergy(nn.Module):
    def __init__(self, enabled_names: List[str], **branch_kwargs):
        super().__init__()
        self.enabled_names = list(enabled_names)
        self.branches = nn.ModuleDict()
        for name in enabled_names:
            if name == "global_attention":
                max_nodes = branch_kwargs.get("max_global_nodes", 512)
                self.branches[name] = GlobalAttentionBranch(max_global_nodes=max_nodes)
            elif name in ENERGY_BRANCHES:
                self.branches[name] = ENERGY_BRANCHES[name]
            else:
                raise ValueError(f"Energy branch '{name}' not registered. Available: {list(ENERGY_BRANCHES.keys())}")

    def forward(self, state: Dict[str, torch.Tensor], batch: Any, context: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = None
        branch_energies = {}
        quad_term = _quad_branch(state, batch, context)
        for name, branch in self.branches.items():
            e = branch(state, batch, context)
            if e.dim() > quad_term.dim():
                e = e.mean(dim=tuple(range(quad_term.dim(), e.dim())))
            branch_energies[name] = e.detach()
            if total is None:
                total = e
            else:
                total = total + e

        branch_energies["quadratic"] = quad_term.detach()
        if total is None:
            total = quad_term
        else:
            total = quad_term - total
        return total, branch_energies


# Register built-in branches
register_branch(PairwiseBranch())
register_branch(MotifBranch())
register_branch(MemoryBranch())
