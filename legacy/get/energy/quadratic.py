"""Quadratic energy term: (1/2)||X||^2 per graph."""
import torch.nn as nn
from .ops import scatter_add_nd


class QuadraticEnergy(nn.Module):
    """Per-graph quadratic energy branch."""

    def forward(self, X, batch, num_graphs):
        """Returns [..., num_graphs] quadratic energy."""
        node_energies = 0.5 * (X ** 2).sum(dim=-1)
        return scatter_add_nd(X.new_zeros((*X.shape[:-2], num_graphs)), batch, node_energies, dim=-1)


_QUADRATIC_ENERGY = QuadraticEnergy()


def compute_quadratic_energy(X, batch, num_graphs):
    return _QUADRATIC_ENERGY(X, batch, num_graphs)
