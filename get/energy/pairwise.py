import torch
import torch.nn as nn
from torch_geometric.utils import scatter
from .ops import positive_param, inverse_temperature, segment_logsumexp

class PairwiseEnergy(nn.Module):
    def forward(self, G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=None):
        beta_max = params.get('beta_max', None)

        if not params.get('use_pairwise', True) or c_2.numel() == 0:
            return G.new_zeros((*G.shape[:-2], num_graphs))

        lambda_2 = positive_param(params, 'lambda_2')
        if (torch.is_tensor(lambda_2) and lambda_2 <= 1e-6) or (not torch.is_tensor(lambda_2) and lambda_2 <= 1e-6):
            return G.new_zeros((*G.shape[:-2], num_graphs))

        Q2, K2 = projections['Q2'], projections['K2']
        d_actual = Q2.size(2)
        scale = d_actual ** 0.5

        ell_2 = (Q2[c_2] * K2[u_2]).sum(dim=-1) / scale

        if params.get("pairwise_symmetric", False):
            src = torch.cat([c_2, u_2], dim=0)
            dst = torch.cat([u_2, c_2], dim=0)
            ell_sym = (Q2[src] * K2[dst]).sum(dim=-1) / scale
            ell_2 = ell_2 + ell_sym[c_2.numel():]

        a_2 = projections.get('a_2')
        if a_2 is not None:
            ell_2 = ell_2 + a_2

        beta_2 = inverse_temperature(params, 'beta_2', beta_max=beta_max)

        mode = params.get('agg_mode', 'softmax')
        if mode == 'sum':
            scores = torch.exp(beta_2 * ell_2) if params.get('sum_exp', False) else ell_2
            agg_2 = scatter(scores, c_2, dim=0, dim_size=num_nodes, reduce="sum")
            if degree_scaler is not None:
                agg_2 = agg_2 * degree_scaler.unsqueeze(-1)
            graph_agg = scatter(agg_2, batch, dim=0, dim_size=num_graphs, reduce="sum")
            return lambda_2 * graph_agg
        else:
            lse_2 = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes, dim=0)
            lse_2 = torch.where(
                torch.bincount(c_2, minlength=num_nodes).eq(0).view(-1, *([1] * (lse_2.dim() - 1))),
                torch.zeros_like(lse_2),
                lse_2,
            )
            if degree_scaler is not None:
                lse_2 = lse_2 * degree_scaler.unsqueeze(-1)
            graph_lse = scatter(lse_2, batch, dim=0, dim_size=num_graphs, reduce="sum")
            return (lambda_2 / beta_2) * graph_lse
