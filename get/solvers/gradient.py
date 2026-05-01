from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch


EnergyClosure = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class FixedStepSolver:
    num_steps: int
    step_size: float

    def run(
        self,
        x0: torch.Tensor,
        energy_fn: EnergyClosure,
        create_graph: bool = False,
    ) -> Tuple[torch.Tensor, List[float], Dict[str, List[float]]]:
        x = x0
        energy_trace: List[float] = []
        grad_norms: List[float] = []

        for _ in range(self.num_steps):
            x = x.requires_grad_(True)
            e = energy_fn(x)
            grad, = torch.autograd.grad(e, x, create_graph=create_graph)
            grad_norm = torch.linalg.vector_norm(grad).detach().item()
            x = x - self.step_size * grad
            energy_trace.append(e.detach().item())
            grad_norms.append(grad_norm)

        stats = {
            "mode": "fixed",
            "step_sizes": [self.step_size for _ in range(self.num_steps)],
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

    def run(
        self,
        x0: torch.Tensor,
        energy_fn: EnergyClosure,
    ) -> Tuple[torch.Tensor, List[float], Dict[str, List[float]]]:
        x = x0.detach()
        energy_trace: List[float] = []
        step_sizes: List[float] = []
        backtracks: List[float] = []
        accepted: List[float] = []
        grad_norms: List[float] = []

        for _ in range(self.num_steps):
            x = x.requires_grad_(True)
            e = energy_fn(x)
            grad, = torch.autograd.grad(e, x, create_graph=False)
            grad_norm_sq = (grad * grad).sum().detach()
            grad_norms.append(torch.sqrt(grad_norm_sq).item())

            eta = self.eta0
            found = False
            chosen = x.detach()
            e_current = e.detach()

            for bt in range(self.max_backtracks):
                cand = (x - eta * grad).detach()
                with torch.no_grad():
                    e_cand = energy_fn(cand)
                rhs = e_current - self.c * eta * grad_norm_sq
                if e_cand <= rhs:
                    found = True
                    chosen = cand
                    e_current = e_cand.detach()
                    backtracks.append(float(bt))
                    break
                eta *= self.gamma

            if not found:
                backtracks.append(float(self.max_backtracks))

            x = chosen
            energy_trace.append(e_current.item())
            step_sizes.append(eta if found else 0.0)
            accepted.append(1.0 if found else 0.0)

        stats = {
            "mode": "armijo",
            "step_sizes": step_sizes,
            "backtracks": backtracks,
            "accepted": accepted,
            "grad_norms": grad_norms,
        }
        return x, energy_trace, stats
