from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from .core import GETEnergy


@dataclass(frozen=True)
class EnergySpec:
    name: str
    description: str


_ENERGY_REGISTRY = {
    "get_full": GETEnergy,
    "pairwise_only": GETEnergy,
    "quadratic_only": GETEnergy,
}

ENERGY_SPECS = [
    EnergySpec("get_full", "Quadratic - pairwise - motif - memory"),
    EnergySpec("pairwise_only", "Quadratic - pairwise (motif/memory disabled via lambda=0)"),
    EnergySpec("quadratic_only", "Quadratic only (pairwise/motif/memory disabled via lambda=0)"),
]


def build_energy(name: str) -> nn.Module:
    key = str(name).strip().lower()
    if key not in _ENERGY_REGISTRY:
        supported = ", ".join(sorted(_ENERGY_REGISTRY.keys()))
        raise ValueError(f"Unknown energy function '{name}'. Supported: {supported}")
    return _ENERGY_REGISTRY[key]()
