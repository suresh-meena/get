from functools import lru_cache

import torch
import torch.nn as nn
from .ops import positive_param, inverse_temperature, scatter_add_nd

class MemoryEnergy(nn.Module):
    def forward(self, G, batch, num_graphs, params, projections, return_grad=False):
        # We enforce pure functional energy; return_grad is removed to rely on autograd.
        d = params['d']
        beta_max = params.get('beta_max', None)
        
        if not params.get('use_memory', True):
            return G.new_zeros((*G.shape[:-2], num_graphs))

        lambda_m = positive_param(params, 'lambda_m')
        if (torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or (not torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or params.get('K', 0) <= 0:
            return G.new_zeros((*G.shape[:-2], num_graphs))

        Qm, Km = projections['Qm'], projections['Km']
        scale = d ** 0.5
        beta_m = inverse_temperature(params, 'beta_m', beta_max=beta_max)
        
        # Qm is [num_nodes, num_heads, head_dim]
        # Km is [num_heads, K, head_dim]
        Lm = torch.einsum("nhd, hkd -> nhk", Qm, Km) / scale
        lse_m = torch.logsumexp(beta_m * Lm, dim=-1)
        
        return (lambda_m / beta_m) * scatter_add_nd(lse_m.new_zeros((num_graphs, lse_m.shape[-1])), batch, lse_m, dim=0)

@lru_cache(maxsize=1)
def _cached_memory_energy() -> MemoryEnergy:
    return MemoryEnergy()


def compute_memory_energy(G, batch, num_graphs, params, projections):
    return _cached_memory_energy()(G, batch, num_graphs, params, projections)
