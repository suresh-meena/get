from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from get.energy import build_energy
from get.energy.ops import get_degree_from_incidence, compute_degree_scaler
from get.solvers import ArmijoSolver, FixedStepSolver


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
        inference_mode_train: str = "fixed",
        inference_mode_eval: str = "armijo",
        energy_name: str = "get_full",
        use_energy_norm: bool = True,
        agg_mode: str = "softmax",
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
        # Training unroll uses create_graph=True, i.e. higher-order autodiff.
        self.requires_double_backward = True

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.state_norm = nn.LayerNorm(hidden_dim)

        self.q2 = nn.Linear(hidden_dim, hidden_dim)
        self.k2 = nn.Linear(hidden_dim, hidden_dim)
        self.q3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.k3 = nn.Linear(hidden_dim, num_heads * R * head_dim)
        self.qm = nn.Linear(hidden_dim, hidden_dim)

        self.km = nn.Parameter(torch.randn(num_heads, K, head_dim) * 0.05)
        self.t_tau = nn.Parameter(torch.randn(num_motif_types, num_heads, R, head_dim) * 0.05)
        self.readout = nn.Linear(hidden_dim, num_classes)

        self.energy = build_energy(self.energy_name)
        self.fixed_solver = FixedStepSolver(num_steps=num_steps, step_size=fixed_step_size)
        self.armijo_solver = ArmijoSolver(
            num_steps=num_steps,
            eta0=armijo_eta0,
            gamma=armijo_gamma,
            c=armijo_c,
            max_backtracks=armijo_max_backtracks,
        )

    def _enabled_branches(self) -> tuple[bool, bool, bool]:
        if self.energy_name == "quadratic_only":
            return False, False, False
        if self.energy_name == "pairwise_only":
            return True, False, False
        return True, True, True

    def _build_params(self, dtype: torch.dtype, device: torch.device) -> Dict[str, torch.Tensor | float | int | bool]:
        pairwise_on, motif_on, memory_on = self._enabled_branches()
        return {
            "d": self.hidden_dim,
            "R": self.R,
            "K": self.K,
            "lambda_2": self.lambda_2,
            "lambda_3": self.lambda_3,
            "lambda_m": self.lambda_m,
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "beta_m": self.beta_m,
            "use_pairwise": pairwise_on and self.lambda_2 > 0.0,
            "use_motif": motif_on and self.lambda_3 > 0.0,
            "use_memory": memory_on and self.lambda_m > 0.0,
            "pairwise_symmetric": False,
            "lambda_sum": 0.0,
            "T_tau": self.t_tau.to(device=device, dtype=dtype),
            "agg_mode": self.agg_mode,
        }

    def _build_projections(self, g: torch.Tensor) -> Dict[str, torch.Tensor]:
        n = g.size(0)
        q2 = self.q2(g).view(n, self.num_heads, self.head_dim)
        k2 = self.k2(g).view(n, self.num_heads, self.head_dim)
        q3 = self.q3(g).view(n, self.num_heads, self.R, self.head_dim)
        k3 = self.k3(g).view(n, self.num_heads, self.R, self.head_dim)
        qm = self.qm(g).view(n, self.num_heads, self.head_dim)
        return {
            "Q2": q2,
            "K2": k2,
            "a_2": None,
            "Q3": q3,
            "K3": k3,
            "Qm": qm,
            "Km": self.km,
        }

    def _energy_sum(self, x_state: torch.Tensor, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        g = self.state_norm(x_state) if self.use_energy_norm else x_state
        params = self._build_params(dtype=g.dtype, device=g.device)
        projections = self._build_projections(g)
        num_graphs = int(batch_data["num_graphs"].item())
        
        # Compute degree scaler if using LogSumExp
        scaler = None
        if self.agg_mode == "softmax" and batch_data["c_2"].numel() > 0:
            num_nodes = x_state.size(0)
            degs = get_degree_from_incidence(batch_data["c_2"], num_nodes)
            avg_deg = degs.mean().item()
            scaler = compute_degree_scaler(degs, avg_deg, mode="pna")

        e_vec = self.energy(
            x_state,
            g,
            batch_data["c_2"],
            batch_data["u_2"],
            batch_data["c_3"],
            batch_data["u_3"],
            batch_data["v_3"],
            batch_data["t_tau"],
            batch_data["batch"],
            num_graphs,
            params,
            projections,
            degree_scaler=scaler,
        )
        return e_vec.sum()

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

        def closure(x: torch.Tensor) -> torch.Tensor:
            return self._energy_sum(x, batch_data)

        if mode == "fixed":
            x_final, energy_trace, solver_stats = self.fixed_solver.run(
                x0, closure, create_graph=self.training
            )
        elif mode == "armijo":
            x_final, energy_trace, solver_stats = self.armijo_solver.run(x0, closure)
        else:
            raise ValueError(f"Unsupported inference mode: {mode}")

        z = self.state_norm(x_final)
        num_graphs = int(batch_data["num_graphs"].item())
        pooled = self._pool_graph_mean(z, batch_data["batch"], num_graphs)
        logits = self.readout(pooled)
        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_solver_stats:
            return logits, energy_trace, solver_stats
        return logits
