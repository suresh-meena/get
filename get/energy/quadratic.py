"""Quadratic energy term: (1/2)||X||^2 per graph."""
import torch
from .ops import scatter_add_nd


def compute_quadratic_energy(X, batch, num_graphs):
    """Returns [..., num_graphs] quadratic energy."""
    node_energies = 0.5 * (X ** 2).sum(dim=-1)
    return scatter_add_nd(X.new_zeros((*X.shape[:-2], num_graphs)), batch, node_energies, dim=-1)
