"""GET energy functions: scalar energy branches and their gradients."""

from .ops import (
    segment_logsumexp,
    segment_softmax,
    segment_reduce_1d,
    scatter_add_nd,
    fused_motif_dot,
    positive_param,
    inverse_temperature,
)
from .quadratic import QuadraticEnergy, compute_quadratic_energy
from .pairwise import PairwiseEnergy, compute_pairwise_energy
from .motif import MotifEnergy, compute_motif_energy
from .memory import MemoryEnergy, compute_memory_energy, compute_memory_entropy
from .core import GETEnergy, GETEnergyWithGrad, compute_energy_GET, compute_energy_and_grad_GET

# Backward-compatible alias
_scatter_add_nd = scatter_add_nd

__all__ = [
    "segment_logsumexp",
    "segment_softmax",
    "segment_reduce_1d",
    "scatter_add_nd",
    "_scatter_add_nd",
    "fused_motif_dot",
    "positive_param",
    "inverse_temperature",
    "QuadraticEnergy",
    "PairwiseEnergy",
    "MotifEnergy",
    "MemoryEnergy",
    "GETEnergy",
    "GETEnergyWithGrad",
    "compute_quadratic_energy",
    "compute_pairwise_energy",
    "compute_motif_energy",
    "compute_memory_energy",
    "compute_memory_entropy",
    "compute_energy_GET",
    "compute_energy_and_grad_GET",
]
