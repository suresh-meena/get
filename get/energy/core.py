"""Full GET energy composition: E_quad - E_att2 - E_att3 - E_mem."""
import torch.nn as nn
from .quadratic import QuadraticEnergy
from .pairwise import PairwiseEnergy
from .motif import MotifEnergy
from .memory import MemoryEnergy
from .linear_agg import LinearAggregationEnergy


class GETEnergy(nn.Module):
    """Full GET energy as composition of branch modules."""
    def __init__(self):
        super().__init__()
        self.quadratic = QuadraticEnergy()
        self.pairwise = PairwiseEnergy()
        self.motif = MotifEnergy()
        self.memory = MemoryEnergy()
        self.linear_agg = LinearAggregationEnergy()

    def forward(self, X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=None, degree_scaler=None):
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
            E_att2 = E_att2.mean(dim=-2)
        if E_att3.dim() > E_quad.dim():
            E_att3 = E_att3.mean(dim=-2)
        if E_mem.dim() > E_quad.dim():
            E_mem = E_mem.mean(dim=-2)
        return E_quad - E_att2 - E_att3 - E_mem - E_sum


class GETEnergyWithGrad(nn.Module):
    def __init__(self):
        super().__init__()
        self.quadratic = QuadraticEnergy()
        self.pairwise = PairwiseEnergy()
        self.motif = MotifEnergy()
        self.memory = MemoryEnergy()
        self.linear_agg = LinearAggregationEnergy()

    def forward(self, X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=None, degree_scaler=None):
        num_nodes = X.size(-2)
        E_quad = self.quadratic(X, batch, num_graphs)
        E_att2, (gQ2, gK2) = self.pairwise(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, return_grad=True, degree_scaler=degree_scaler)
        E_att3, (gQ3, gK3) = self.motif(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, return_grad=True, degree_scaler=degree_scaler)
        E_mem, (gQm, _) = self.memory(G, batch, num_graphs, params, projections, return_grad=True)
        H = params.get('num_heads', 1)
        grads = {
            'grad_Q2': (gQ2 / H) if gQ2 is not None else None,
            'grad_K2': (gK2 / H) if gK2 is not None else None,
            'grad_Q3': (gQ3 / H) if gQ3 is not None else None,
            'grad_K3': (gK3 / H) if gK3 is not None else None,
            'grad_Qm': (gQm / H) if gQm is not None else None
        }
        E2 = E_att2.mean(dim=-2) if E_att2.dim() > E_quad.dim() else E_att2
        E3 = E_att3.mean(dim=-2) if E_att3.dim() > E_quad.dim() else E_att3
        Em = E_mem.mean(dim=-2) if E_mem.dim() > E_quad.dim() else E_mem
        
        lambda_sum = params.get('lambda_sum', 0.0)
        if hasattr(lambda_sum, 'item'): lambda_sum = lambda_sum.item()
        E_sum = 0.0
        grad_X_sum = 0.0
        if lambda_sum > 0:
            E_sum_val, gX_sum = self.linear_agg(X, c_2, u_2, batch, num_graphs, return_grad=True)
            E_sum = lambda_sum * E_sum_val
            grad_X_sum = lambda_sum * gX_sum
            
        return E_quad - E2 - E3 - Em - E_sum, X + grad_X_sum, grads

def compute_energy_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=None, degree_scaler=None):
    return _GET_ENERGY(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=projections, degree_scaler=degree_scaler)


def compute_energy_and_grad_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=None, degree_scaler=None):
    return _GET_ENERGY_WITH_GRAD(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=projections, degree_scaler=degree_scaler)


_GET_ENERGY = GETEnergy()
_GET_ENERGY_WITH_GRAD = GETEnergyWithGrad()
