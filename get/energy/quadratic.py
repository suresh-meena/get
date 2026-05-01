import torch
import torch.nn as nn
from .ops import scatter_add_nd

class QuadraticEnergy(nn.Module):
    """Per-graph quadratic energy branch."""

    def forward(self, X: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        """Returns [num_graphs] quadratic energy."""
        node_energies = 0.5 * (X ** 2).sum(dim=-1)
        return scatter_add_nd(X.new_zeros(num_graphs), batch, node_energies, dim=0)

def compute_quadratic_energy(X, batch, num_graphs):
    return QuadraticEnergy()(X, batch, num_graphs)
