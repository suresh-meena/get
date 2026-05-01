from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .core import GETEnergy
from .pairwise import PairwiseEnergy
from .quadratic import QuadraticEnergy


@dataclass(frozen=True)
class EnergySpec:
    name: str
    description: str


class QuadraticOnlyEnergy(nn.Module):
    """E = E_quad."""

    def __init__(self) -> None:
        super().__init__()
        self.quadratic = QuadraticEnergy()

    def forward(self, X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, degree_scaler=None):
        del G, c_2, u_2, c_3, u_3, v_3, t_tau, params, projections, degree_scaler
        return self.quadratic(X, batch, num_graphs)


class PairwiseOnlyEnergy(nn.Module):
    """E = E_quad - E_pairwise."""

    def __init__(self) -> None:
        super().__init__()
        self.quadratic = QuadraticEnergy()
        self.pairwise = PairwiseEnergy()

    def forward(self, X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, degree_scaler=None):
        del c_3, u_3, v_3, t_tau
        num_nodes = X.size(-2)
        e_quad = self.quadratic(X, batch, num_graphs)
        e_pair = self.pairwise(
            G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler
        )
        if e_pair.dim() > e_quad.dim():
            e_pair = e_pair.mean(dim=-1)
        return e_quad - e_pair


_ENERGY_REGISTRY = {
    "get_full": GETEnergy,
    "quadratic_only": QuadraticOnlyEnergy,
    "pairwise_only": PairwiseOnlyEnergy,
}

ENERGY_SPECS = [
    EnergySpec("get_full", "Quadratic - pairwise - motif - memory"),
    EnergySpec("quadratic_only", "Quadratic energy only"),
    EnergySpec("pairwise_only", "Quadratic - pairwise"),
]


def build_energy(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key not in _ENERGY_REGISTRY:
        supported = ", ".join(sorted(_ENERGY_REGISTRY.keys()))
        raise ValueError(f"Unknown energy function '{name}'. Supported: {supported}")
    return _ENERGY_REGISTRY[key]()
