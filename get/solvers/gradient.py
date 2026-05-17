from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch


EnergyClosure = Callable[[torch.Tensor], torch.Tensor]
EnergyAndGradClosure = Callable[[torch.Tensor, bool], Tuple[torch.Tensor, torch.Tensor]]


@dataclass
class FixedStepSolver:
    num_steps: int
    step_size: float
    update_damping: float = 0.0

    def run(
        self,
        x0: torch.Tensor,
        energy_fn: EnergyClosure,
        energy_and_grad_fn: EnergyAndGradClosure | None = None,
        create_graph: bool = False,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, List[float], Dict[str, List[float]]]:
        x = x0
        energy_trace_tensors: List[torch.Tensor] | None = [] if collect_stats else None
        grad_norm_tensors: List[torch.Tensor] | None = [] if collect_stats else None
        initial_energy_tensor: torch.Tensor | None = None
        damping = min(max(float(self.update_damping), 0.0), 1.0)
        step_scale = 1.0 - damping

        for _ in range(self.num_steps):
            x = x.requires_grad_(True)
            if energy_and_grad_fn is None:
                e = energy_fn(x)
                grad, = torch.autograd.grad(e, x, create_graph=create_graph)
            else:
                e, grad = energy_and_grad_fn(x, create_graph)
            if collect_stats and initial_energy_tensor is None:
                initial_energy_tensor = e.detach()

            if collect_stats and grad_norm_tensors is not None:
                grad_norm_tensors.append(torch.linalg.vector_norm(grad).detach())
            x = x - (self.step_size * step_scale) * grad
            if collect_stats and energy_trace_tensors is not None:
                energy_trace_tensors.append(e.detach())

        final_energy_tensor = None
        if collect_stats:
            with torch.no_grad():
                final_energy_tensor = energy_fn(x.detach()).detach()
            if initial_energy_tensor is None:
                initial_energy_tensor = final_energy_tensor

        stats = {
            "mode": "fixed",
            "update_damping": self.update_damping,
            "step_sizes": [self.step_size * step_scale for _ in range(self.num_steps)] if collect_stats else [],
            "grad_norms": torch.stack(grad_norm_tensors).tolist() if (collect_stats and grad_norm_tensors) else [],
            "energy_initial": float(initial_energy_tensor.item()) if initial_energy_tensor is not None else None,
            "energy_final": float(final_energy_tensor.item()) if final_energy_tensor is not None else None,
            "energy_drop": float((initial_energy_tensor - final_energy_tensor).item()) if (initial_energy_tensor is not None and final_energy_tensor is not None) else None,
        }
        return x, torch.stack(energy_trace_tensors).tolist() if (collect_stats and energy_trace_tensors) else [], stats


@dataclass
class ArmijoSolver:
    num_steps: int
    eta0: float
    gamma: float = 0.5
    c: float = 1e-4
    max_backtracks: int = 25
    update_damping: float = 0.0

    def run(
        self,
        x0: torch.Tensor,
        energy_fn: EnergyClosure,
        energy_and_grad_fn: EnergyAndGradClosure | None = None,
        max_backtracks: int | None = None,
        collect_stats: bool = True,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, List[float], Dict[str, List[float]]]:
        x = x0.detach()
        energy_trace_tensors: List[torch.Tensor] | None = [] if collect_stats else None
        step_sizes: List[float] | None = [] if collect_stats else None
        backtracks: List[int] | None = [] if collect_stats else None
        accepted: List[bool] | None = [] if collect_stats else None
        grad_norm_tensors: List[torch.Tensor] | None = [] if collect_stats else None
        initial_energy_tensor: torch.Tensor | None = None
        damping = min(max(float(self.update_damping), 0.0), 1.0)
        step_scale = 1.0 - damping
        backtrack_limit = self.max_backtracks if max_backtracks is None else max(0, int(max_backtracks))

        for _ in range(self.num_steps):
            x = x.requires_grad_(True)
            if energy_and_grad_fn is None:
                e = energy_fn(x)
                grad, = torch.autograd.grad(e, x, create_graph=create_graph)
            else:
                e, grad = energy_and_grad_fn(x, create_graph)
            if collect_stats and initial_energy_tensor is None:
                initial_energy_tensor = e.detach()
            
            grad_norm_sq = (grad * grad).sum().detach()
            if collect_stats and grad_norm_tensors is not None:
                grad_norm_tensors.append(torch.sqrt(grad_norm_sq).detach())

            eta = self.eta0
            found = False
            chosen = x.detach()
            e_current = e.detach()

            for bt in range(backtrack_limit):
                effective_eta = eta * step_scale
                cand = (x - effective_eta * grad).detach()
                with torch.no_grad():
                    e_cand = energy_fn(cand)
                rhs = e_current - self.c * effective_eta * grad_norm_sq
                if e_cand <= rhs:
                    found = True
                    chosen = cand
                    e_current = e_cand.detach()
                    if collect_stats and backtracks is not None:
                        backtracks.append(bt)
                    break
                eta *= self.gamma

            if not found:
                if collect_stats and backtracks is not None:
                    backtracks.append(backtrack_limit)

            x = chosen
            if collect_stats and energy_trace_tensors is not None:
                energy_trace_tensors.append(e_current.detach())
            if collect_stats and step_sizes is not None:
                step_sizes.append(eta * step_scale if found else 0.0)
            if collect_stats and accepted is not None:
                accepted.append(found)

        final_energy_tensor = None
        if collect_stats:
            with torch.no_grad():
                final_energy_tensor = energy_fn(x.detach()).detach()
            if initial_energy_tensor is None:
                initial_energy_tensor = final_energy_tensor

        stats = {
            "mode": "armijo",
            "update_damping": self.update_damping,
            "step_scale": step_scale,
            "max_backtracks": backtrack_limit,
            "step_sizes": step_sizes if (collect_stats and step_sizes is not None) else [],
            "backtracks": [float(v) for v in backtracks] if (collect_stats and backtracks is not None) else [],
            "accepted": [1.0 if v else 0.0 for v in accepted] if (collect_stats and accepted is not None) else [],
            "grad_norms": torch.stack(grad_norm_tensors).tolist() if (collect_stats and grad_norm_tensors) else [],
            "energy_initial": float(initial_energy_tensor.item()) if initial_energy_tensor is not None else None,
            "energy_final": float(final_energy_tensor.item()) if final_energy_tensor is not None else None,
            "energy_drop": float((initial_energy_tensor - final_energy_tensor).item()) if (initial_energy_tensor is not None and final_energy_tensor is not None) else None,
        }
        return x, torch.stack(energy_trace_tensors).tolist() if (collect_stats and energy_trace_tensors) else [], stats
