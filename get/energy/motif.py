import torch
import torch.nn as nn
from .ops import (
    positive_param, inverse_temperature,
    segment_logsumexp, scatter_add_nd, fused_motif_dot,
)

class MotifEnergy(nn.Module):
    def forward(self, G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, degree_scaler=None):
        d = params['d']
        R = params.get('R', 1)
        beta_max = params.get('beta_max', None)
        
        if not params.get('use_motif', True) or c_3.numel() == 0:
            return G.new_zeros((*G.shape[:-2], num_graphs))

        lambda_3 = positive_param(params, 'lambda_3')
        if (torch.is_tensor(lambda_3) and lambda_3 <= 1e-6) or (not torch.is_tensor(lambda_3) and lambda_3 <= 1e-6):
            return G.new_zeros((*G.shape[:-2], num_graphs))

        Q3, K3 = projections['Q3'], projections['K3']

        T_tau_selected = projections.get('T_tau_selected')
        if T_tau_selected is None:
            T_params = params['T_tau']
            if t_tau.numel() > 0 and t_tau.max() >= T_params.size(0):
                t_tau = torch.clamp(t_tau, max=T_params.size(0) - 1)
            T_tau_selected = T_params[t_tau]

        scale = (R * d) ** 0.5
        beta_3 = inverse_temperature(params, 'beta_3', beta_max=beta_max)
        
        ell_3 = fused_motif_dot(Q3[c_3], K3[u_3], K3[v_3], T_tau_selected) / scale
        
        mode = params.get('agg_mode', 'softmax')
        if mode == 'sum':
            scores = torch.exp(beta_3 * ell_3) if params.get('sum_exp', False) else ell_3
            agg_3 = scatter_add_nd(ell_3.new_zeros((num_nodes, ell_3.shape[-1])), c_3, scores, dim=0)
            if degree_scaler is not None:
                agg_3 = agg_3 * degree_scaler.unsqueeze(-1)
            graph_agg = scatter_add_nd(ell_3.new_zeros((num_graphs, ell_3.shape[-1])), batch, agg_3, dim=0)
            return lambda_3 * graph_agg
        else:
            lse_3 = segment_logsumexp(beta_3 * ell_3, c_3, num_nodes, dim=0)
            if degree_scaler is not None:
                lse_3 = lse_3 * degree_scaler.unsqueeze(-1)
            graph_lse = scatter_add_nd(ell_3.new_zeros((num_graphs, ell_3.shape[-1])), batch, lse_3, dim=0)
            return (lambda_3 / beta_3) * graph_lse

def compute_motif_energy(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, degree_scaler=None):
    return MotifEnergy()(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
