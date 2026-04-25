"""Motif (order-3) energy branch: sparse anchored wedge/triangle scores."""
import torch
from .ops import (
    positive_param, inverse_temperature,
    segment_logsumexp, scatter_add_nd, fused_motif_dot,
)


def compute_motif_energy(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, return_grad=False, degree_scaler=None):
    d = params['d']
    R = params.get('R', 1)
    beta_max = params.get('beta_max', None)
    if not params.get('use_motif', True) or c_3.numel() == 0:
        zero_E = G.new_zeros((*G.shape[:-2], num_graphs))
        return (zero_E, (None, None)) if return_grad else zero_E

    lambda_3 = positive_param(params, 'lambda_3')
    if (torch.is_tensor(lambda_3) and lambda_3 <= 1e-6) or (not torch.is_tensor(lambda_3) and lambda_3 <= 1e-6):
        zero_E = G.new_zeros((*G.shape[:-2], num_graphs))
        return (zero_E, (None, None)) if return_grad else zero_E

    Q3, K3 = projections['Q3'], projections['K3']

    # Static gather optimization: use pre-gathered bias if available
    T_tau_selected = projections.get('T_tau_selected')
    if T_tau_selected is None:
        T_params = params['T_tau']
        if t_tau.numel() > 0 and t_tau.max() >= T_params.size(0):
            t_tau = torch.clamp(t_tau, max=T_params.size(0) - 1)
        T_tau_selected = T_params[t_tau].transpose(0, 1)  # [H, L, R, d_h]

    scale = (R * d) ** 0.5
    ell_3 = fused_motif_dot(Q3[..., c_3, :, :], K3[..., u_3, :, :], K3[..., v_3, :, :], T_tau_selected) / scale
    beta_3 = inverse_temperature(params, 'beta_3', beta_max=beta_max)
    lse_3 = segment_logsumexp(beta_3 * ell_3, c_3, num_nodes)
    
    if degree_scaler is not None:
        lse_3 = lse_3 * degree_scaler
        
    graph_lse = scatter_add_nd(ell_3.new_zeros((*ell_3.shape[:-1], num_graphs)), batch, lse_3, dim=-1)
    E = (lambda_3 / beta_3) * graph_lse
    if not return_grad:
        return E

    probs = torch.exp(beta_3 * ell_3 - lse_3[..., c_3] / (degree_scaler[..., c_3] if degree_scaler is not None else 1.0))
    coeff_val = lambda_3 * probs / scale
    if degree_scaler is not None:
        coeff_val = coeff_val * degree_scaler[..., c_3]
        
    coeff = coeff_val.unsqueeze(-1).unsqueeze(-1)
    src_Q = coeff * (K3[..., u_3, :, :] * K3[..., v_3, :, :] + T_tau_selected)
    src_Ku = coeff * (Q3[..., c_3, :, :] * K3[..., v_3, :, :])
    src_Kv = coeff * (Q3[..., c_3, :, :] * K3[..., u_3, :, :])

    grad_Q3 = scatter_add_nd(torch.zeros_like(Q3), c_3, src_Q, dim=-3)
    grad_K3 = scatter_add_nd(torch.zeros_like(K3), u_3, src_Ku, dim=-3)
    grad_K3 = grad_K3 + scatter_add_nd(torch.zeros_like(K3), v_3, src_Kv, dim=-3)

    return E, (grad_Q3, grad_K3)
