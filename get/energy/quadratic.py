from functools import lru_cache

import torch
import torch.nn as nn
from .ops import scatter_add_nd

class QuadraticEnergy(nn.Module):
    """Per-graph quadratic energy branch normalized by graph size."""

    def forward(self, X: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        """Returns [num_graphs] quadratic energy."""
        node_energies = 0.5 * (X ** 2).sum(dim=-1)
        graph_energy = scatter_add_nd(X.new_zeros(num_graphs), batch, node_energies, dim=0)
        counts = torch.bincount(batch, minlength=num_graphs).to(dtype=X.dtype, device=X.device)
        return graph_energy / counts.clamp_min(1.0)

@lru_cache(maxsize=1)
def _cached_quadratic_energy() -> QuadraticEnergy:
    return QuadraticEnergy()


def compute_quadratic_energy(X, batch, num_graphs):
    return _cached_quadratic_energy()(X, batch, num_graphs)
