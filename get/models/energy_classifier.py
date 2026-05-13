from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from get.energy import ComposedEnergy
from get.energy.ops import get_degree_from_incidence, compute_degree_scaler
from get.solvers import ArmijoSolver, FixedStepSolver
from .energy_norm import EnergyLayerNorm


class GETBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        R: int,
        K: int,
        num_motif_types: int,
        use_bias_norm: bool = True,
        edge_attr_dim: int = 0,
        lambda_g: float = 0.0,
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
            "pairwise_symmetric": False,
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
        edge_attr_dim: int = 0,
        lambda_g: float = 0.0,
        beta_g: float = 1.0,
        max_global_nodes: int = 512,
        global_chunk_size: int = 256,
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
        self.requires_double_backward = False

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if branch_names is None:
            branch_names = ["pairwise", "motif", "memory"]
        if lambda_g > 0.0 and "global_attention" not in branch_names:
            branch_names = list(branch_names) + ["global_attention"]
        self.blocks = nn.ModuleList([
            GETBlock(
                hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                R=R, K=K, num_motif_types=num_motif_types,
                use_bias_norm=use_energy_norm,
                edge_attr_dim=edge_attr_dim, lambda_g=self.lambda_g, beta_g=self.beta_g,
                max_global_nodes=max_global_nodes,
                branch_names=branch_names,
            )
            for _ in range(self.num_blocks)
        ])

        self.graph_readout = nn.Linear(hidden_dim * 3, num_classes)
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

    def _pool_graph_concat(self, x_state: torch.Tensor, batch_idx: torch.Tensor, num_graphs: int) -> torch.Tensor:
        return torch.cat([
            global_mean_pool(x_state, batch_idx),
            global_add_pool(x_state, batch_idx),
            global_max_pool(x_state, batch_idx),
        ], dim=-1)

    def forward(
        self,
        batch_data: Dict[str, torch.Tensor],
        inference_mode: Optional[str] = None,
        return_solver_stats: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list[float], Dict[str, list[float]]]:
        x0 = self.encoder(batch_data["x"])
        mode = inference_mode or (self.inference_mode_train if self.training else self.inference_mode_eval)
        num_graphs = int(batch_data["y"].shape[0])

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
            def _energy_fn(curr_x: torch.Tensor) -> torch.Tensor:
                return block.energy_from_g(curr_x, batch_data, params_cache, scaler, num_graphs)

            energy_fn = torch.compile(_energy_fn, dynamic=True, fullgraph=False)
            grad_energy_fn = torch.func.grad(energy_fn)

            def energy_and_grad_fn(curr_x: torch.Tensor, create_graph: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
                e = energy_fn(curr_x)
                del create_graph
                grad = grad_energy_fn(curr_x)
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
        else:
            pooled = self._pool_graph_concat(x_final, batch_data["batch"], num_graphs)
            logits = self.graph_readout(pooled)

        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_solver_stats:
            return logits, energy_trace, solver_stats
        return logits
