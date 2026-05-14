import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from .ops import positive_param, inverse_temperature

class MemoryEnergy(nn.Module):
    def forward(self, G, batch, num_graphs, params, projections):
        if not params.get('use_memory', True):
            return G.new_zeros((*G.shape[:-2], num_graphs))

        lambda_m = positive_param(params, 'lambda_m')
        if (torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or (not torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or params.get('K', 0) <= 0:
            return G.new_zeros((*G.shape[:-2], num_graphs))

        Qm, Km = projections['Qm'], projections['Km']
        beta_m = inverse_temperature(params, 'beta_m', beta_max=params.get('beta_max', None))

        Lm = torch.einsum("nhd, hkd -> nhk", Qm, Km) / (params['d'] ** 0.5)
        lse_m = torch.logsumexp(beta_m * Lm, dim=-1)

        return (lambda_m / beta_m) * scatter(lse_m, batch, dim=0, dim_size=num_graphs, reduce="sum")
