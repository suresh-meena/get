from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from get.energy import build_energy, ComposedEnergy, enabled_branches_from_config
from get.energy.ops import get_degree_from_incidence, compute_degree_scaler
from get.solvers import ArmijoSolver, FixedStepSolver
from .energy_norm import EnergyLayerNorm


def _maybe_add_cls_token(x: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    extra = x.new_zeros((num_graphs, x.size(-1)))
    x_with_cls = torch.cat([x, extra], dim=0)
    cls_batch = torch.arange(num_graphs, device=x.device, dtype=batch_idx.dtype)
    batch_with_cls = torch.cat([batch_idx, cls_batch], dim=0)
    cls_mask = torch.cat([
        torch.zeros(x.size(0), dtype=torch.bool, device=x.device),
        torch.ones(num_graphs, dtype=torch.bool, device=x.device),
    ], dim=0)
    return x_with_cls, batch_with_cls, cls_mask


def _extend_pairwise_for_cls(c_2: torch.Tensor, u_2: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int, n_orig: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add edges from each CLS token to every node in the same graph. Fully vectorized."""
    if num_graphs == 0 or n_orig == 0:
        return c_2, u_2
    cls_pos = n_orig + batch_idx[:n_orig]
    node_idx = torch.arange(n_orig, device=c_2.device, dtype=c_2.dtype)
    if c_2.numel() == 0:
        return torch.cat([cls_pos, node_idx]), torch.cat([node_idx, cls_pos])
    return torch.cat([c_2, cls_pos, node_idx]), torch.cat([u_2, node_idx, cls_pos])


class _GETBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        R: int,
        K: int,
        num_motif_types: int,
        energy_name: str,
        use_bias_norm: bool = True,
        use_cls_token: bool = False,
        edge_attr_dim: int = 0,
        lambda_g: float = 0.0,
        beta_g: float = 1.0,
        max_global_nodes: int = 512,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.R = R
        self.K = K
        self.num_motif_types = num_motif_types
        self.energy_name = energy_name
        self.use_cls_token = use_cls_token
        self.lambda_g = lambda_g
        self.beta_g = beta_g
        self.max_global_nodes = max_global_nodes

        self.norm = EnergyLayerNorm(hidden_dim, use_bias=use_bias_norm)

        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.k2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.k3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.qm = nn.Linear(hidden_dim, hidden_dim)
        self.qg = nn.Linear(hidden_dim, hidden_dim)
        self.kg = nn.Linear(hidden_dim, hidden_dim)

        self.km = nn.Parameter(torch.randn(num_heads, K, head_dim) * 0.05)
        self.t_tau = nn.Parameter(torch.randn(num_motif_types, num_heads, R, head_dim) * 0.05)

        if edge_attr_dim > 0:
            self.edge_proj = nn.Linear(edge_attr_dim, num_heads)
        else:
            self.edge_proj = None

        self.energy_fn = build_energy(self.energy_name)

    def _enabled_branches(self) -> tuple[bool, bool, bool, bool]:
        if self.energy_name == "quadratic_only":
            return False, False, False, False
        if self.energy_name == "pairwise_only":
            return True, False, False, False
        return True, True, True, self.lambda_g > 0.0

    def _build_projections(self, g: torch.Tensor, batch_data: Dict) -> Dict:
        n = g.size(0)
        pairwise_on, motif_on, memory_on, global_on = self._enabled_branches()
        proj = {}
        if pairwise_on:
            q2 = self.q2(g).view(n, self.num_heads, self.head_dim)
            k2 = self.k2(g).view(n, self.num_heads, self.head_dim)
            proj.update({"Q2": q2, "K2": k2})
        if motif_on:
            q3 = self.q3(g).view(n, self.num_heads, self.R, self.head_dim)
            k3 = self.k3(g).view(n, self.num_heads, self.R, self.head_dim)
            proj.update({"Q3": q3, "K3": k3})
        if memory_on:
            qm = self.qm(g).view(n, self.num_heads, self.head_dim)
            proj.update({"Qm": qm, "Km": self.km})
        if global_on:
            qg = self.qg(g).view(n, self.num_heads, self.head_dim)
            kg = self.kg(g).view(n, self.num_heads, self.head_dim)
            proj.update({"Qg": qg, "Kg": kg})
        if self.edge_proj is not None and "edge_attr" in batch_data:
            edge_attr = batch_data["edge_attr"]
            if edge_attr is not None and edge_attr.numel() > 0:
                proj["a_2"] = self.edge_proj(edge_attr)
        return proj

    def energy_from_g(
        self,
        x: torch.Tensor,
        batch_data: Dict,
        cfg_params: Dict,
        scaler: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        g = self.norm(x)
        projections = self._build_projections(g, batch_data)

        pairwise_on, motif_on, memory_on, global_on = self._enabled_branches()
        params = {
            "d": self.hidden_dim,
            "R": self.R,
            "K": self.K,
            "lambda_2": cfg_params["lambda_2"],
            "lambda_3": cfg_params["lambda_3"],
            "lambda_m": cfg_params["lambda_m"],
            "lambda_g": self.lambda_g,
            "beta_2": cfg_params["beta_2"],
            "beta_3": cfg_params["beta_3"],
            "beta_m": cfg_params["beta_m"],
            "beta_g": self.beta_g,
            "global_chunk_size": cfg_params.get("global_chunk_size", 256),
            "use_pairwise": pairwise_on and cfg_params["lambda_2"] > 0.0,
            "use_motif": motif_on and cfg_params["lambda_3"] > 0.0,
            "use_memory": memory_on and cfg_params["lambda_m"] > 0.0,
            "use_global_attention": global_on,
            "pairwise_symmetric": False,
            "lambda_sum": 0.0,
            "T_tau": self.t_tau,
            "agg_mode": cfg_params["agg_mode"],
        }

        return self.energy_fn(
            x, g,
            batch_data["c_2"], batch_data["u_2"],
            batch_data["c_3"], batch_data["u_3"], batch_data["v_3"], batch_data["t_tau"],
            batch_data["batch"], num_graphs,
            params, projections,
            degree_scaler=scaler,
        ).sum()


class _GETBlockBranch(nn.Module):
    """Block using the branch-based ComposedEnergy."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        R: int,
        K: int,
        num_motif_types: int,
        use_bias_norm: bool = True,
        use_cls_token: bool = False,
        edge_attr_dim: int = 0,
        lambda_g: float = 0.0,
        lambda_s: float = 0.0,
        beta_g: float = 1.0,
        max_global_nodes: int = 512,
        branch_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.R = R
        self.K = K
        self.num_motif_types = num_motif_types
        self.use_cls_token = use_cls_token
        self.lambda_g = lambda_g
        self.lambda_s = lambda_s
        self.beta_g = beta_g
        self.max_global_nodes = max_global_nodes

        self.norm = EnergyLayerNorm(hidden_dim, use_bias=use_bias_norm)

        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.k2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.k3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.qm = nn.Linear(hidden_dim, hidden_dim)
        self.qg = nn.Linear(hidden_dim, hidden_dim)
        self.kg = nn.Linear(hidden_dim, hidden_dim)

        self.km = nn.Parameter(torch.randn(num_heads, K, head_dim) * 0.05)
        self.t_tau = nn.Parameter(torch.randn(num_motif_types, num_heads, R, head_dim) * 0.05)

        if edge_attr_dim > 0:
            self.edge_proj = nn.Linear(edge_attr_dim, num_heads)
        else:
            self.edge_proj = None

        if branch_names is None:
            branch_names = ["pairwise", "motif", "memory"]
        if lambda_g > 0.0 and "global_attention" not in branch_names:
            branch_names = list(branch_names) + ["global_attention"]
        self.composed = ComposedEnergy(branch_names, max_global_nodes=max_global_nodes)

    def _build_projections(self, g: torch.Tensor, batch_data: Dict) -> Dict:
        n = g.size(0)
        enabled = self.composed.enabled_names
        proj = {}
        if "pairwise" in enabled:
            q2 = self.q2(g).view(n, self.num_heads, self.head_dim)
            k2 = self.k2(g).view(n, self.num_heads, self.head_dim)
            proj.update({"Q2": q2, "K2": k2})
        if "motif" in enabled:
            q3 = self.q3(g).view(n, self.num_heads, self.R, self.head_dim)
            k3 = self.k3(g).view(n, self.num_heads, self.R, self.head_dim)
            proj.update({"Q3": q3, "K3": k3})
        if "memory" in enabled:
            qm = self.qm(g).view(n, self.num_heads, self.head_dim)
            proj.update({"Qm": qm, "Km": self.km})
        if "global_attention" in enabled:
            qg = self.qg(g).view(n, self.num_heads, self.head_dim)
            kg = self.kg(g).view(n, self.num_heads, self.head_dim)
            proj.update({"Qg": qg, "Kg": kg})
        if self.edge_proj is not None and "edge_attr" in batch_data:
            edge_attr = batch_data["edge_attr"]
            if edge_attr is not None and edge_attr.numel() > 0:
                proj["a_2"] = self.edge_proj(edge_attr)
        return proj

    def energy_from_g(self, x, batch_data, cfg_params, scaler, num_graphs):
        g = self.norm(x)
        projections = self._build_projections(g, batch_data)

        params = {
            "d": self.hidden_dim, "R": self.R, "K": self.K,
            "lambda_2": cfg_params["lambda_2"], "lambda_3": cfg_params["lambda_3"],
            "lambda_m": cfg_params["lambda_m"], "lambda_g": self.lambda_g,
            "beta_2": cfg_params["beta_2"], "beta_3": cfg_params["beta_3"],
            "beta_m": cfg_params["beta_m"], "beta_g": self.beta_g,
            "global_chunk_size": cfg_params.get("global_chunk_size", 256),
            "use_pairwise": cfg_params["lambda_2"] > 0.0,
            "use_motif": cfg_params["lambda_3"] > 0.0,
            "use_memory": cfg_params["lambda_m"] > 0.0,
            "use_global_attention": self.lambda_g > 0.0,
            "pairwise_symmetric": False, "lambda_sum": 0.0,
            "T_tau": self.t_tau, "agg_mode": cfg_params["agg_mode"],
        }

        context = {"params": params, "projections": projections,
                   "num_graphs": num_graphs, "degree_scaler": scaler,
                   "batch": batch_data["batch"]}

        state = {"H": g}
        total, _branch_energies = self.composed(state, batch_data, context)
        return total.sum()


class EnergyGraphClassifier(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_steps: int,
        num_heads: int,
        head_dim: int,
        R: int,
        K: int,
        num_motif_types: int,
        lambda_2: float,
        lambda_3: float,
        lambda_m: float,
        beta_2: float,
        beta_3: float,
        beta_m: float,
        update_damping: float,
        fixed_step_size: float = 0.1,
        armijo_eta0: float = 0.2,
        armijo_gamma: float = 0.5,
        armijo_c: float = 1e-4,
        armijo_max_backtracks: int = 20,
        armijo_eval_max_backtracks: int = 5,
        inference_mode_train: str = "fixed",
        inference_mode_eval: str = "armijo",
        energy_name: str = "get_full",
        use_energy_norm: bool = True,
        agg_mode: str = "softmax",
        num_blocks: int = 1,
        readout_mode: str = "graph",
        use_cls_token: bool = False,
        edge_attr_dim: int = 0,
        lambda_g: float = 0.0,
        beta_g: float = 1.0,
        max_global_nodes: int = 512,
        global_chunk_size: int = 256,
        lambda_s: float = 0.0,
        use_branch_registry: bool = False,
        branch_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        if hidden_dim != num_heads * head_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal num_heads*head_dim ({num_heads * head_dim})"
            )

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.R = R
        self.K = K
        self.num_motif_types = num_motif_types
        self.readout_mode = str(readout_mode).lower()
        self.use_cls_token = use_cls_token

        self.lambda_2 = float(lambda_2)
        self.lambda_3 = float(lambda_3)
        self.lambda_m = float(lambda_m)
        self.beta_2 = float(beta_2)
        self.beta_3 = float(beta_3)
        self.beta_m = float(beta_m)
        self.lambda_g = float(lambda_g)
        self.beta_g = float(beta_g)
        self.global_chunk_size = int(global_chunk_size)
        self.update_damping = float(update_damping)
        self.inference_mode_train = inference_mode_train
        self.inference_mode_eval = inference_mode_eval
        self.energy_name = str(energy_name).strip().lower()
        self.use_energy_norm = use_energy_norm
        self.agg_mode = agg_mode
        self.num_blocks = max(1, int(num_blocks))
        self.armijo_eval_max_backtracks = max(1, int(armijo_eval_max_backtracks))
        self._global_avg_degree: float | None = None
        self.requires_double_backward = True

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if use_branch_registry:
            self.blocks = nn.ModuleList([
                _GETBlockBranch(
                    hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                    R=R, K=K, num_motif_types=num_motif_types,
                    use_bias_norm=use_energy_norm, use_cls_token=use_cls_token,
                    edge_attr_dim=edge_attr_dim, lambda_g=self.lambda_g, beta_g=self.beta_g,
                    max_global_nodes=max_global_nodes,
                    branch_names=branch_names,
                )
                for _ in range(self.num_blocks)
            ])
        else:
            self.blocks = nn.ModuleList([
                _GETBlock(
                    hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                    R=R, K=K, num_motif_types=num_motif_types,
                    energy_name=self.energy_name, use_bias_norm=use_energy_norm,
                    use_cls_token=use_cls_token, edge_attr_dim=edge_attr_dim,
                    lambda_g=self.lambda_g, beta_g=self.beta_g,
                    max_global_nodes=max_global_nodes,
                )
                for _ in range(self.num_blocks)
            ])

        self.graph_readout = nn.Linear(hidden_dim * 3, num_classes)
        self.cls_readout = nn.Linear(hidden_dim, num_classes)
        self.node_readout = nn.Linear(hidden_dim, num_classes)

        self.fixed_solver = FixedStepSolver(
            num_steps=num_steps, step_size=fixed_step_size,
            update_damping=self.update_damping,
        )
        self.armijo_solver = ArmijoSolver(
            num_steps=num_steps, eta0=armijo_eta0,
            gamma=armijo_gamma, c=armijo_c,
            max_backtracks=armijo_max_backtracks,
            update_damping=self.update_damping,
        )

    def set_global_avg_degree(self, avg_degree: float | torch.Tensor | None) -> None:
        if avg_degree is None:
            self._global_avg_degree = None
            return
        if torch.is_tensor(avg_degree):
            avg_degree = float(avg_degree.detach().item())
        self._global_avg_degree = float(avg_degree)

    def _build_params(self) -> Dict[str, float | str]:
        return {
            "lambda_2": self.lambda_2, "lambda_3": self.lambda_3,
            "lambda_m": self.lambda_m,
            "beta_2": self.beta_2, "beta_3": self.beta_3, "beta_m": self.beta_m,
            "global_chunk_size": self.global_chunk_size,
            "agg_mode": self.agg_mode,
        }

    def _pool_graph_concat(self, x_state: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int, cls_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if cls_mask is not None:
            node_mask = ~cls_mask
            x_state = x_state[node_mask]
            batch_idx = batch_idx[node_mask]
        d = x_state.size(-1)
        mean_out = x_state.new_zeros((num_graphs, d))
        mean_out.index_add_(0, batch_idx, x_state)
        counts = torch.bincount(batch_idx, minlength=num_graphs).to(dtype=x_state.dtype).unsqueeze(-1).clamp_min(1.0)
        mean_pool = mean_out / counts
        sum_pool = mean_out
        max_pool = x_state.new_full((num_graphs, d), float("-inf"))
        max_pool.scatter_reduce_(0, batch_idx.unsqueeze(-1).expand_as(x_state), x_state, reduce="amax")
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))
        return torch.cat([mean_pool, sum_pool, max_pool], dim=-1)

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        inference_mode: Optional[str] = None,
        return_solver_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list[float], Dict[str, list[float]]]:
        x0 = self.encoder(batch_data["x"])
        mode = inference_mode or (self.inference_mode_train if self.training else self.inference_mode_eval)
        num_graphs = int(batch_data["y"].shape[0])

        if self.use_cls_token:
            n_orig = x0.size(0)
            old_num_edges = batch_data["c_2"].numel()
            x0, batch_data["batch"], cls_mask = _maybe_add_cls_token(x0, batch_data["batch"], num_graphs)
            batch_data["c_2"], batch_data["u_2"] = _extend_pairwise_for_cls(
                batch_data["c_2"], batch_data["u_2"], batch_data["batch"], num_graphs, n_orig,
            )
            edge_attr = batch_data.get("edge_attr")
            if edge_attr is not None and edge_attr.numel() > 0 and edge_attr.size(0) == old_num_edges:
                extra_edges = batch_data["c_2"].numel() - old_num_edges
                if extra_edges > 0:
                    pad = edge_attr.new_zeros((extra_edges, *edge_attr.shape[1:]))
                    batch_data["edge_attr"] = torch.cat([edge_attr, pad], dim=0)
        else:
            cls_mask = None

        params_cache = self._build_params()
        scaler = None
        if self.agg_mode == "softmax" and batch_data["c_2"].numel() > 0:
            num_nodes = x0.size(0)
            degs = get_degree_from_incidence(batch_data["c_2"], num_nodes)
            avg_deg = self._global_avg_degree if self._global_avg_degree is not None else degs.mean()
            scaler = compute_degree_scaler(degs, avg_deg, mode="pna")

        x = x0
        collect_solver_stats = bool(return_solver_stats)
        for block in self.blocks:
            def energy_fn(curr_x: torch.Tensor) -> torch.Tensor:
                return block.energy_from_g(curr_x, batch_data, params_cache, scaler, num_graphs)

            def energy_and_grad_fn(curr_x: torch.Tensor, create_graph: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
                curr_x = curr_x.requires_grad_(True)
                e = energy_fn(curr_x)
                grad, = torch.autograd.grad(e, curr_x, create_graph=create_graph)
                return e.detach(), grad

            if mode == "fixed":
                x, energy_trace, solver_stats = self.fixed_solver.run(
                    x, energy_fn, energy_and_grad_fn,
                    create_graph=self.training,
                    collect_stats=collect_solver_stats,
                )
            elif mode == "armijo":
                max_backtracks = self.armijo_solver.max_backtracks if self.training else min(self.armijo_solver.max_backtracks, self.armijo_eval_max_backtracks)
                x, energy_trace, solver_stats = self.armijo_solver.run(
                    x, energy_fn, energy_and_grad_fn,
                    max_backtracks=max_backtracks,
                    collect_stats=collect_solver_stats,
                )
            else:
                raise ValueError(f"Unsupported inference mode: {mode}")

        x_final = x

        if self.readout_mode == "node":
            logits = self.node_readout(x_final)
        elif self.readout_mode == "cls" and cls_mask is not None:
            cls_state = x_final[cls_mask]
            logits = self.cls_readout(cls_state)
        else:
            pooled = self._pool_graph_concat(x_final, batch_data["batch"], num_graphs, cls_mask=cls_mask)
            logits = self.graph_readout(pooled)

        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_solver_stats:
            return logits, energy_trace, solver_stats
        return logits
