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
    ) -> Tuple[torch.Tensor, List[float], Dict[str, List[float]]]:
        x = x0
        energy_trace: List[float] = []
        grad_norms: List[float] = []
        damping = min(max(float(self.update_damping), 0.0), 1.0)
        step_scale = 1.0 - damping

        for _ in range(self.num_steps):
            x = x.requires_grad_(True)
            if energy_and_grad_fn is None:
                e = energy_fn(x)
                grad, = torch.autograd.grad(e, x, create_graph=create_graph)
            else:
                e, grad = energy_and_grad_fn(x, create_graph)
            grad_norm = torch.linalg.vector_norm(grad).detach().item()
            x = x - (self.step_size * step_scale) * grad
            energy_trace.append(e.detach().item())
            grad_norms.append(grad_norm)

        stats = {
            "mode": "fixed",
            "update_damping": self.update_damping,
            "step_sizes": [self.step_size * step_scale for _ in range(self.num_steps)],
            "grad_norms": grad_norms,
        }
        return x, energy_trace, stats


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
    ) -> Tuple[torch.Tensor, List[float], Dict[str, List[float]]]:
        x = x0.detach()
        energy_trace: List[float] = []
        step_sizes: List[float] = []
        backtracks: List[float] = []
        accepted: List[float] = []
        grad_norms: List[float] = []
        damping = min(max(float(self.update_damping), 0.0), 1.0)
        step_scale = 1.0 - damping
        backtrack_limit = self.max_backtracks if max_backtracks is None else max(0, int(max_backtracks))

        for _ in range(self.num_steps):
            x = x.requires_grad_(True)
            if energy_and_grad_fn is None:
                e = energy_fn(x)
                grad, = torch.autograd.grad(e, x, create_graph=False)
            else:
                e, grad = energy_and_grad_fn(x, False)
            grad_norm_sq = (grad * grad).sum().detach()
            grad_norms.append(torch.sqrt(grad_norm_sq).item())

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
                    backtracks.append(float(bt))
                    break
                eta *= self.gamma

            if not found:
                backtracks.append(float(backtrack_limit))

            x = chosen
            energy_trace.append(e_current.item())
            step_sizes.append((eta * step_scale) if found else 0.0)
            accepted.append(1.0 if found else 0.0)

        stats = {
            "mode": "armijo",
            "update_damping": self.update_damping,
            "step_scale": step_scale,
            "max_backtracks": backtrack_limit,
            "step_sizes": step_sizes,
            "backtracks": backtracks,
            "accepted": accepted,
            "grad_norms": grad_norms,
        }
        return x, energy_trace, stats
