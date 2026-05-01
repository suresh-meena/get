"""Pairwise energy branch: graph-local log-sum-exp over edge scores."""
import torch
import torch.nn as nn
from .ops import (
    positive_param, inverse_temperature,
    segment_logsumexp, scatter_add_nd,
)


def _pairwise_dot_fused(Q, K, c, u, num_nodes, scale):
    return (Q[..., c, :] * K[..., u, :]).sum(dim=-1) / scale


def _pairwise_pullback_fused(coeff, Q, K, c, u, num_nodes):
    src = coeff.unsqueeze(-1) * K[..., u, :]
    return scatter_add_nd(torch.zeros_like(Q), c, src, dim=-2)


class PairwiseEnergy(nn.Module):
    def forward(self, G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, return_grad=False, degree_scaler=None):
        d = params['d']
        beta_max = params.get('beta_max', None)
        if not params.get('use_pairwise', True) or c_2.numel() == 0:
            zero_E = G.new_zeros((*G.shape[:-2], num_graphs))
            return (zero_E, (None, None)) if return_grad else zero_E

        lambda_2 = positive_param(params, 'lambda_2')
        if (torch.is_tensor(lambda_2) and lambda_2 <= 1e-6) or (not torch.is_tensor(lambda_2) and lambda_2 <= 1e-6):
            zero_E = G.new_zeros((*G.shape[:-2], num_graphs))
            return (zero_E, (None, None)) if return_grad else zero_E

        Q2, K2 = projections['Q2'], projections['K2']
        scale = d ** 0.5
        ell_2 = _pairwise_dot_fused(Q2, K2, c_2, u_2, num_nodes, scale)
        if params.get("pairwise_symmetric", False):
            ell_2 = ell_2 + _pairwise_dot_fused(Q2, K2, u_2, c_2, num_nodes, scale)

        a_2 = projections.get('a_2')
        if a_2 is not None:
            ell_2 = ell_2 + a_2

        beta_2 = inverse_temperature(params, 'beta_2', beta_max=beta_max)
        lse_2 = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes)
        if degree_scaler is not None:
            lse_2 = lse_2 * degree_scaler

        graph_lse = scatter_add_nd(ell_2.new_zeros((*ell_2.shape[:-1], num_graphs)), batch, lse_2, dim=-1)
        E = (lambda_2 / beta_2) * graph_lse
        if not return_grad:
            return E

        probs = torch.exp(beta_2 * ell_2 - lse_2[..., c_2] / (degree_scaler[..., c_2] if degree_scaler is not None else 1.0))
        coeff = lambda_2 * probs / scale
        if degree_scaler is not None:
            coeff = coeff * degree_scaler[..., c_2]

        grad_Q2 = _pairwise_pullback_fused(coeff, Q2, K2, c_2, u_2, num_nodes)
        grad_K2 = _pairwise_pullback_fused(coeff, K2, Q2, u_2, c_2, num_nodes)
        if params.get("pairwise_symmetric", False):
            grad_Q2 = grad_Q2 + _pairwise_pullback_fused(coeff, Q2, K2, u_2, c_2, num_nodes)
            grad_K2 = grad_K2 + _pairwise_pullback_fused(coeff, K2, Q2, c_2, u_2, num_nodes)

        return E, (grad_Q2, grad_K2)


_PAIRWISE_ENERGY = PairwiseEnergy()


def compute_pairwise_energy(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, return_grad=False, degree_scaler=None):
    return _PAIRWISE_ENERGY(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, return_grad=return_grad, degree_scaler=degree_scaler)
