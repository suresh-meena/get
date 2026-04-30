"""Motif (order-3) energy branch: sparse anchored wedge/triangle scores."""
import torch
import torch.nn as nn
from .ops import (
    positive_param, inverse_temperature,
    segment_logsumexp, segment_reduce_1d, scatter_add_nd, fused_motif_dot,
)


def _chunked_segment_logsumexp(beta_ell, segment_ids, num_segments):
    """Segment logsumexp that preserves empty segments as -inf."""
    max_val, counts = segment_reduce_1d(beta_ell, segment_ids, num_segments, reduce="max")
    empty = counts == 0
    max_safe = torch.where(empty, torch.zeros_like(max_val), max_val)
    centered = beta_ell - max_safe[..., segment_ids]
    exp_centered = torch.exp(centered)
    sum_exp, _ = segment_reduce_1d(exp_centered, segment_ids, num_segments, reduce="sum")
    lse = torch.log(sum_exp) + max_safe
    return torch.where(empty, torch.full_like(lse, float("-inf")), lse)


class MotifEnergy(nn.Module):
    def forward(self, G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, return_grad=False, degree_scaler=None):
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
        beta_3 = inverse_temperature(params, 'beta_3', beta_max=beta_max)
        motif_count = int(c_3.numel())
        chunk_size = int(params.get("motif_chunk_size", 8192))
        use_chunked = motif_count > max(0, chunk_size)

        if not use_chunked:
            ell_3 = fused_motif_dot(Q3[..., c_3, :, :], K3[..., u_3, :, :], K3[..., v_3, :, :], T_tau_selected) / scale
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

        total_lse = None
        chunk_cache = [] if return_grad else None
        for start in range(0, motif_count, chunk_size):
            end = min(start + chunk_size, motif_count)
            c_chunk = c_3[start:end]
            u_chunk = u_3[start:end]
            v_chunk = v_3[start:end]
            q_chunk = Q3.index_select(-3, c_chunk)
            ku_chunk = K3.index_select(-3, u_chunk)
            kv_chunk = K3.index_select(-3, v_chunk)
            t_chunk = T_tau_selected.index_select(1, c_chunk)
            ell_chunk = fused_motif_dot(q_chunk, ku_chunk, kv_chunk, t_chunk) / scale
            lse_chunk = _chunked_segment_logsumexp(beta_3 * ell_chunk, c_chunk, num_nodes)
            total_lse = lse_chunk if total_lse is None else torch.logaddexp(total_lse, lse_chunk)
            if chunk_cache is not None:
                chunk_cache.append((c_chunk, u_chunk, v_chunk, q_chunk, ku_chunk, kv_chunk, t_chunk, ell_chunk))

        total_lse = torch.where(torch.isfinite(total_lse), total_lse, torch.zeros_like(total_lse))
        if degree_scaler is not None:
            total_lse = total_lse * degree_scaler

        graph_lse = scatter_add_nd(ell_chunk.new_zeros((*ell_chunk.shape[:-1], num_graphs)), batch, total_lse, dim=-1)
        E = (lambda_3 / beta_3) * graph_lse
        if not return_grad:
            return E

        grad_Q3 = torch.zeros_like(Q3)
        grad_K3 = torch.zeros_like(K3)
        for c_chunk, u_chunk, v_chunk, q_chunk, ku_chunk, kv_chunk, t_chunk, ell_chunk in chunk_cache:
            denom = total_lse[..., c_chunk] / (degree_scaler[..., c_chunk] if degree_scaler is not None else 1.0)
            probs = torch.exp(beta_3 * ell_chunk - denom)
            coeff_val = lambda_3 * probs / scale
            if degree_scaler is not None:
                coeff_val = coeff_val * degree_scaler[..., c_chunk]
            coeff = coeff_val.unsqueeze(-1).unsqueeze(-1)

            src_Q = ku_chunk * kv_chunk
            src_Q.add_(t_chunk)
            src_Q.mul_(coeff)
            grad_Q3.add_(scatter_add_nd(torch.zeros_like(Q3), c_chunk, src_Q, dim=-3))

            src_Ku = q_chunk * kv_chunk
            src_Ku.mul_(coeff)
            grad_K3.add_(scatter_add_nd(torch.zeros_like(K3), u_chunk, src_Ku, dim=-3))

            src_Kv = q_chunk * ku_chunk
            src_Kv.mul_(coeff)
            grad_K3.add_(scatter_add_nd(torch.zeros_like(K3), v_chunk, src_Kv, dim=-3))

        return E, (grad_Q3, grad_K3)


_MOTIF_ENERGY = MotifEnergy()


def compute_motif_energy(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, return_grad=False, degree_scaler=None):
    return _MOTIF_ENERGY(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, return_grad=return_grad, degree_scaler=degree_scaler)
