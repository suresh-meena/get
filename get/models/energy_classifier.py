from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from get.energy import build_energy
from get.energy.ops import get_degree_from_incidence, compute_degree_scaler
from get.solvers import ArmijoSolver, FixedStepSolver
from .energy_norm import EnergyLayerNorm


class _GETBlock(nn.Module):
    """A single GET energy block: pairwise + motif + memory energies."""

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
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.R = R
        self.K = K
        self.num_motif_types = num_motif_types
        self.energy_name = energy_name

        self.norm = EnergyLayerNorm(hidden_dim, use_bias=use_bias_norm)

        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.k2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.k3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.qm = nn.Linear(hidden_dim, hidden_dim)

        self.km = nn.Parameter(torch.randn(num_heads, K, head_dim) * 0.05)
        self.t_tau = nn.Parameter(torch.randn(num_motif_types, num_heads, R, head_dim) * 0.05)
        self.energy_fn = build_energy(self.energy_name)

    def _enabled_branches(self) -> tuple[bool, bool, bool]:
        if self.energy_name == "quadratic_only":
            return False, False, False
        if self.energy_name == "pairwise_only":
            return True, False, False
        return True, True, True

    def energy_from_g(
        self, 
        x: torch.Tensor, 
        batch_data: Dict, 
        cfg_params: Dict, 
        scaler: torch.Tensor | None, 
        num_graphs: int
    ) -> torch.Tensor:
        g = self.norm(x)
        n = g.size(0)
        
        projections = {
            "Q2": self.q2(g).view(n, self.num_heads, self.head_dim),
            "K2": self.k2(g).view(n, self.num_heads, self.head_dim),
            "Q3": self.q3(g).view(n, self.num_heads, self.R, self.head_dim),
            "K3": self.k3(g).view(n, self.num_heads, self.R, self.head_dim),
            "Qm": self.qm(g).view(n, self.num_heads, self.head_dim),
            "Km": self.km,
        }
        
        pairwise_on, motif_on, memory_on = self._enabled_branches()
        params = {
            "d": self.hidden_dim,
            "R": self.R,
            "K": self.K,
            "lambda_2": cfg_params["lambda_2"],
            "lambda_3": cfg_params["lambda_3"],
            "lambda_m": cfg_params["lambda_m"],
            "beta_2": cfg_params["beta_2"],
            "beta_3": cfg_params["beta_3"],
            "beta_m": cfg_params["beta_m"],
            "use_pairwise": pairwise_on and cfg_params["lambda_2"] > 0.0,
            "use_motif": motif_on and cfg_params["lambda_3"] > 0.0,
            "use_memory": memory_on and cfg_params["lambda_m"] > 0.0,
            "pairwise_symmetric": False,
            "lambda_sum": 0.0,
            "T_tau": self.t_tau,
            "agg_mode": cfg_params["agg_mode"],
        }
        
        return self.energy_fn(
            x,                   # X
            g,                   # G
            batch_data["c_2"],   # c_2
            batch_data["u_2"],   # u_2
            batch_data["c_3"],   # c_3
            batch_data["u_3"],   # u_3
            batch_data["v_3"],   # v_3
            batch_data["t_tau"], # t_tau
            batch_data["batch"], # batch
            num_graphs,          # num_graphs
            params,              # params
            projections,         # projections
            degree_scaler=scaler # degree_scaler
        ).sum()


class EnergyGraphClassifier(nn.Module):
    """
    Refactored GET wrapper:
    - pure scalar energy via `GETEnergy`
    - decoupled state-update solvers in `get/solvers`
    """

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
        self.update_damping = float(update_damping)
        self.inference_mode_train = inference_mode_train
        self.inference_mode_eval = inference_mode_eval
        self.energy_name = str(energy_name).strip().lower()
        self.use_energy_norm = use_energy_norm
        self.agg_mode = agg_mode
        self.num_blocks = max(1, int(num_blocks))
        self.armijo_eval_max_backtracks = max(1, int(armijo_eval_max_backtracks))
        self._global_avg_degree: float | None = None
        # Training unroll uses create_graph=True, i.e. higher-order autodiff.
        self.requires_double_backward = True

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Stack of blocks
        self.blocks = nn.ModuleList([
            _GETBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                R=R,
                K=K,
                num_motif_types=num_motif_types,
                energy_name=self.energy_name,
                use_bias_norm=use_energy_norm,
            )
            for _ in range(self.num_blocks)
        ])

        self.readout = nn.Linear(hidden_dim, num_classes)

        self.fixed_solver = FixedStepSolver(
            num_steps=num_steps,
            step_size=fixed_step_size,
            update_damping=self.update_damping,
        )
        self.armijo_solver = ArmijoSolver(
            num_steps=num_steps,
            eta0=armijo_eta0,
            gamma=armijo_gamma,
            c=armijo_c,
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
            "lambda_2": self.lambda_2,
            "lambda_3": self.lambda_3,
            "lambda_m": self.lambda_m,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "beta_m": self.beta_m,
            "agg_mode": self.agg_mode,
        }

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
                    x, energy_fn, energy_and_grad_fn, create_graph=self.training
                )
            elif mode == "armijo":
                max_backtracks = self.armijo_solver.max_backtracks if self.training else min(self.armijo_solver.max_backtracks, self.armijo_eval_max_backtracks)
                x, energy_trace, solver_stats = self.armijo_solver.run(
                    x,
                    energy_fn,
                    energy_and_grad_fn,
                    max_backtracks=max_backtracks,
                )
            else:
                raise ValueError(f"Unsupported inference mode: {mode}")

        x_final = x

        if self.readout_mode == "node":
            logits = self.readout(x_final)
        else:
            pooled = self._pool_graph_mean(x_final, batch_data["batch"], num_graphs)
            logits = self.readout(pooled)
            
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_solver_stats:
            return logits, energy_trace, solver_stats
        return logits
