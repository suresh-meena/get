import torch.nn as nn
from torch_geometric.utils import scatter

class LinearAggregationEnergy(nn.Module):
    def forward(self, X, c_2, u_2, batch, num_graphs):
        scores = (X[c_2] * X[u_2]).sum(dim=-1)
        energy = -0.5 * scatter(scores, batch[c_2], dim=0, dim_size=num_graphs, reduce="sum")
        return energy
