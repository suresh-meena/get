import torch.nn as nn
from .quadratic import QuadraticEnergy
from .pairwise import PairwiseEnergy
from .motif import MotifEnergy
from .memory import MemoryEnergy

class GETEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.quadratic = QuadraticEnergy()
        self.pairwise = PairwiseEnergy()
        self.motif = MotifEnergy()
        self.memory = MemoryEnergy()

    def forward(self, X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, degree_scaler=None):
        num_nodes = X.size(-2)

        E_quad = self.quadratic(X, batch, num_graphs)
        E_att2 = self.pairwise(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
        E_att3 = self.motif(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
        E_mem = self.memory(G, batch, num_graphs, params, projections)

        if E_att2.dim() > E_quad.dim():
            E_att2 = E_att2.mean(dim=-1)
        if E_att3.dim() > E_quad.dim():
            E_att3 = E_att3.mean(dim=-1)
        if E_mem.dim() > E_quad.dim():
            E_mem = E_mem.mean(dim=-1)

        return E_quad - E_att2 - E_att3 - E_mem
