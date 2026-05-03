from functools import lru_cache

import torch.nn as nn
from .quadratic import QuadraticEnergy
from .pairwise import PairwiseEnergy
from .motif import MotifEnergy
from .memory import MemoryEnergy
from .linear_agg import LinearAggregationEnergy

class GETEnergy(nn.Module):
    """
    Pure Functional GET energy as composition of branch modules.
    Computes E_quad - E_att2 - E_att3 - E_mem
    No manual gradients; relies on torch.autograd + torch.compile.
    """
    def __init__(self):
        super().__init__()
        self.quadratic = QuadraticEnergy()
        self.pairwise = PairwiseEnergy()
        self.motif = MotifEnergy()
        self.memory = MemoryEnergy()
        self.linear_agg = LinearAggregationEnergy()

    def forward(self, X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, degree_scaler=None):
        num_nodes = X.size(-2)
        
        E_quad = self.quadratic(X, batch, num_graphs)
        E_att2 = self.pairwise(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
        E_att3 = self.motif(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
        E_mem = self.memory(G, batch, num_graphs, params, projections)
        
        lambda_sum = params.get('lambda_sum', 0.0)
        if hasattr(lambda_sum, 'item'): lambda_sum = lambda_sum.item()
        E_sum = 0.0
        if lambda_sum > 0:
            E_sum = lambda_sum * self.linear_agg(X, c_2, u_2, batch, num_graphs)

        if E_att2.dim() > E_quad.dim():
            E_att2 = E_att2.mean(dim=-1)
        if E_att3.dim() > E_quad.dim():
            E_att3 = E_att3.mean(dim=-1)
        if E_mem.dim() > E_quad.dim():
            E_mem = E_mem.mean(dim=-1)
            
        return E_quad - E_att2 - E_att3 - E_mem - E_sum

@lru_cache(maxsize=1)
def _cached_get_energy() -> GETEnergy:
    return GETEnergy()


def compute_energy_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, degree_scaler=None):
    return _cached_get_energy()(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, degree_scaler=degree_scaler)
