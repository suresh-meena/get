"""Linear Aggregation energy: - \sum_{i,j} X_i^T X_j."""
import torch
import torch.nn as nn
from .ops import scatter_add_nd

class LinearAggregationEnergy(nn.Module):
    """Linear aggregation energy branch: - \sum_{i,j} X_i^T X_j."""
    
    def forward(self, X, c_2, u_2, batch, num_graphs, return_grad=False):
        # Gather neighbor states
        X_u = X[u_2]
        # Dot product with self
        scores = (X[c_2] * X_u).sum(dim=-1)
        # Energy per graph (sum over edges)
        # Note: Factor 0.5 because each edge is counted twice in undirected c_2, u_2
        energy = -0.5 * scatter_add_nd(X.new_zeros((*X.shape[:-2], num_graphs)), batch[c_2], scores, dim=-1)
        
        if not return_grad:
            return energy
            
        # Gradient is -\sum X_j
        grad = -scatter_add_nd(torch.zeros_like(X), c_2, X_u, dim=0)
        return energy, grad

def compute_linear_aggregation_energy(X, c_2, u_2, batch, num_graphs, return_grad=False):
    return LinearAggregationEnergy()(X, c_2, u_2, batch, num_graphs, return_grad=return_grad)
