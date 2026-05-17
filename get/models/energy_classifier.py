from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from get.energy import ComposedEnergy
from get.energy.ops import get_degree_from_incidence, compute_degree_scaler
from get.solvers import ArmijoSolver, FixedStepSolver
from .energy_norm import EnergyLayerNorm




def _make_block_energy(block: GETBlock):
    """Pre-compiled per-block energy function for torch.compile caching."""
    def _energy_eager(x, batch_data, cfg_params, scaler, num_graphs):
        return block.energy_from_g(x, batch_data, cfg_params, scaler, num_graphs)

    try:
        compiled = torch.compile(_energy_eager, dynamic=True, fullgraph=False)
        compiled(torch.randn(1, block.hidden_dim), {}, {}, None, 1)
        return compiled
    except Exception:
        return _energy_eager


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

        beta_init = 1.0 / math.sqrt(head_dim)
        self.hw = nn.Parameter(torch.randn(num_heads, num_heads) * 0.002)
        self.beta_2 = nn.Parameter(torch.full((num_heads,), beta_init))
        self.beta_3 = nn.Parameter(torch.full((num_heads,), beta_init))
        self.beta_m = nn.Parameter(torch.full((num_heads,), beta_init))
        self.beta_g = nn.Parameter(torch.full((num_heads,), beta_init))

        if edge_attr_dim > 0:
            self.edge_proj = nn.Linear(edge_attr_dim, num_heads)
        else:
            self.edge_proj = None

        if branch_names is None:
            branch_names = ["pairwise", "motif", "memory"]
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
            "beta_2": self.beta_2,
            "beta_3": self.beta_3,
            "beta_m": self.beta_m,
            "beta_g": self.beta_g,
            "beta_max": cfg_params.get("beta_max", None),
            "use_pairwise": cfg_params["lambda_2"] > 0.0,
            "use_motif": cfg_params["lambda_3"] > 0.0,
            "use_memory": cfg_params["lambda_m"] > 0.0,
            "use_global_attention": self.lambda_g > 0.0,
            "pairwise_symmetric": False,
            "T_tau": self.t_tau, "agg_mode": cfg_params["agg_mode"],
        }

        num_nodes = g.size(0)
        c_2 = batch_data.get("c_2")
        c_3 = batch_data.get("c_3")
        empty_2 = None
        empty_3 = None
        if c_2 is not None and c_2.numel() > 0:
            empty_2 = (torch.bincount(c_2.view(-1), minlength=num_nodes) == 0)
        if c_3 is not None and c_3.numel() > 0:
            empty_3 = (torch.bincount(c_3.view(-1), minlength=num_nodes) == 0)

        context = {"params": params, "projections": projections,
                   "num_graphs": num_graphs, "degree_scaler": scaler,
                   "batch": batch_data["batch"], "hw": self.hw,
                   "c_2": c_2, "u_2": batch_data["u_2"],
                   "c_3": c_3, "u_3": batch_data["u_3"],
                   "v_3": batch_data["v_3"], "t_tau": batch_data["t_tau"],
                   "empty_2": empty_2, "empty_3": empty_3}

        state = {"H": g}
        total, branch_energies = self.composed(state, batch_data, context)
        self._last_branch_energies = {k: v for k, v in branch_energies.items()}
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
        update_damping: float,
        fixed_step_size: float = 0.1,
        pos_k: int = 0,
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
        max_global_nodes: int = 512,
        branch_names: Optional[List[str]] = None,
        canonical_model_name: str = "",
        model_alias: str = "",
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
        self.lambda_g = float(lambda_g)
        self.update_damping = float(update_damping)
        self.inference_mode_train = inference_mode_train
        self.inference_mode_eval = inference_mode_eval
        self.energy_name = str(energy_name).strip().lower()
        self.canonical_model_name = str(canonical_model_name or self.energy_name).strip().lower()
        self.model_alias = str(model_alias or self.canonical_model_name).strip().lower()
        self.use_energy_norm = use_energy_norm
        self.agg_mode = agg_mode
        self.num_blocks = max(1, int(num_blocks))
        self.armijo_eval_max_backtracks = max(1, int(armijo_eval_max_backtracks))
        self._global_avg_degree: float | None = None
        self.requires_double_backward = False
        self.pos_k = int(pos_k)

        encoder_in_dim = in_dim + self.pos_k
        self.encoder = nn.Sequential(
            nn.Linear(encoder_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if branch_names is None:
            branch_names = ["pairwise", "motif", "memory"]
        self.branch_names = list(branch_names)
        self.blocks = nn.ModuleList([
            GETBlock(
                hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                R=R, K=K, num_motif_types=num_motif_types,
                use_bias_norm=use_energy_norm,
                edge_attr_dim=edge_attr_dim, lambda_g=self.lambda_g,
                max_global_nodes=max_global_nodes,
                branch_names=branch_names,
            )
            for _ in range(self.num_blocks)
        ])

        self.graph_readout = nn.Linear(hidden_dim * 3, num_classes)
        self.node_readout = nn.Linear(hidden_dim, num_classes)

        self._compiled_energy_fns = [_make_block_energy(blk) for blk in self.blocks]

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
            "agg_mode": self.agg_mode,
        }

    def _active_branch_names(self) -> List[str]:
        if getattr(self, "branch_names", None):
            return list(self.branch_names)
        if len(self.blocks) > 0:
            return list(self.blocks[0].composed.enabled_names)
        return []

    def energy_metadata(self) -> Dict[str, object]:
        branch_names = self._active_branch_names()
        return {
            "canonical_model_name": self.canonical_model_name,
            "model_alias": self.model_alias,
            "branch_names": branch_names,
            "enabled_branches": {
                "quadratic": True,
                "pairwise": ("pairwise" in branch_names) and self.lambda_2 > 0.0,
                "motif": ("motif" in branch_names) and self.lambda_3 > 0.0,
                "memory": ("memory" in branch_names) and self.lambda_m > 0.0,
                "global_attention": ("global_attention" in branch_names) and self.lambda_g > 0.0,
            },
            "energy_lambdas": {
                "lambda_2": float(self.lambda_2),
                "lambda_3": float(self.lambda_3),
                "lambda_m": float(self.lambda_m),
                "lambda_g": float(self.lambda_g),
            },
            "readout_mode": self.readout_mode,
            "num_blocks": int(self.num_blocks),
            "num_inference_steps": int(self.fixed_solver.num_steps),
            "inference_mode_train": self.inference_mode_train,
            "inference_mode_eval": self.inference_mode_eval,
            "pos_k": int(self.pos_k),
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
        x_input = batch_data["x"]
        
        # Concatenate positional encodings if present and requested
        if self.pos_k > 0 and "pos" in batch_data:
            pos = batch_data["pos"]
            if pos is not None and torch.is_tensor(pos) and pos.numel() > 0:
                if pos.size(0) == x_input.size(0):
                    if pos.dim() == 1:
                        pos = pos.unsqueeze(-1)
                    # Trim or pad PE to match pos_k dimensions
                    pos_trimmed = pos[:, :self.pos_k] if pos.size(-1) >= self.pos_k else torch.nn.functional.pad(pos, (0, self.pos_k - pos.size(-1)))
                    x_input = torch.cat([x_input, pos_trimmed], dim=-1)
        
        x0 = self.encoder(x_input)
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
        branch_energies_initial: List[Dict[str, float]] | None = [] if collect_solver_stats else None
        branch_energies_final: List[Dict[str, float]] | None = [] if collect_solver_stats else None
        for idx, block in enumerate(self.blocks):
            compiled_energy = self._compiled_energy_fns[idx]

            def _energy_fn(curr_x: torch.Tensor) -> torch.Tensor:
                return compiled_energy(curr_x, batch_data, params_cache, scaler, num_graphs)

            if collect_solver_stats and branch_energies_initial is not None:
                with torch.no_grad():
                    compiled_energy(x, batch_data, params_cache, scaler, num_graphs)
                    branch_energies_initial.append({
                        k: v.sum().item() for k, v in block._last_branch_energies.items()
                    })

            def energy_and_grad_fn(curr_x: torch.Tensor, create_graph: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
                e, vjp_fn = torch.func.vjp(_energy_fn, curr_x)
                grad, = vjp_fn(torch.ones_like(e))
                return (e.detach() if not create_graph else e), grad

            if mode == "fixed":
                x, energy_trace, solver_stats = self.fixed_solver.run(
                    x, _energy_fn, energy_and_grad_fn,
                    create_graph=self.training,
                    collect_stats=collect_solver_stats,
                )
            elif mode == "armijo":
                max_backtracks = self.armijo_solver.max_backtracks if self.training else min(self.armijo_solver.max_backtracks, self.armijo_eval_max_backtracks)
                x, energy_trace, solver_stats = self.armijo_solver.run(
                    x, _energy_fn, energy_and_grad_fn,
                    max_backtracks=max_backtracks,
                    collect_stats=collect_solver_stats,
                    create_graph=self.training,
                )
            else:
                raise ValueError(f"Unsupported inference mode: {mode}")

        x_final = x

        z_final = self.blocks[-1].norm(x_final)
        if self.readout_mode == "node":
            logits = self.node_readout(z_final)
        else:
            pooled = self._pool_graph_concat(z_final, batch_data["batch"], num_graphs)
            logits = self.graph_readout(pooled)

        if self.num_classes == 1:
            logits = logits.squeeze(-1)

        if return_solver_stats:
            solver_stats = dict(solver_stats)
            solver_stats["latent_displacement"] = torch.linalg.vector_norm((x_final - x0).detach()).item()
            if branch_energies_initial is not None:
                for block in self.blocks:
                    branch_energies_final.append({
                        k: v.sum().item() for k, v in block._last_branch_energies.items()
                    })
                solver_stats["branch_energies_initial"] = branch_energies_initial
                solver_stats["branch_energies_final"] = branch_energies_final
            return logits, energy_trace, solver_stats
        return logits
