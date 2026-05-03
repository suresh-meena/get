import torch
import torch.nn as nn
from .ops import positive_param, inverse_temperature, segment_logsumexp, scatter_add_nd

class PairwiseEnergy(nn.Module):
    def forward(self, G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=None):
        d = params['d']
        beta_max = params.get('beta_max', None)
        
        if not params.get('use_pairwise', True) or c_2.numel() == 0:
            return G.new_zeros((*G.shape[:-2], num_graphs))

        lambda_2 = positive_param(params, 'lambda_2')
        if (torch.is_tensor(lambda_2) and lambda_2 <= 1e-6) or (not torch.is_tensor(lambda_2) and lambda_2 <= 1e-6):
            return G.new_zeros((*G.shape[:-2], num_graphs))

        Q2, K2 = projections['Q2'], projections['K2']
        scale = d ** 0.5
        
        ell_2 = (Q2[c_2] * K2[u_2]).sum(dim=-1) / scale
        if params.get("pairwise_symmetric", False):
            ell_2 = ell_2 + (Q2[u_2] * K2[c_2]).sum(dim=-1) / scale

        a_2 = projections.get('a_2')
        if a_2 is not None:
            ell_2 = ell_2 + a_2

        beta_2 = inverse_temperature(params, 'beta_2', beta_max=beta_max)
        
        mode = params.get('agg_mode', 'softmax')
        if mode == 'sum':
            scores = torch.exp(beta_2 * ell_2) if params.get('sum_exp', False) else ell_2
            agg_2 = scatter_add_nd(ell_2.new_zeros((num_nodes, ell_2.shape[-1])), c_2, scores, dim=0)
            if degree_scaler is not None:
                agg_2 = agg_2 * degree_scaler.unsqueeze(-1)
            graph_agg = scatter_add_nd(ell_2.new_zeros((num_graphs, ell_2.shape[-1])), batch, agg_2, dim=0)
            return lambda_2 * graph_agg
        else:
            lse_2 = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes, dim=0)
            if degree_scaler is not None:
                lse_2 = lse_2 * degree_scaler.unsqueeze(-1)
            graph_lse = scatter_add_nd(ell_2.new_zeros((num_graphs, ell_2.shape[-1])), batch, lse_2, dim=0)
            return (lambda_2 / beta_2) * graph_lse

def compute_pairwise_energy(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=None):
    return PairwiseEnergy()(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
