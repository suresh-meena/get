import torch
from .fused_ops import segment_reduce_1d, fused_motif_dot

try:
    from torch_scatter import scatter as pyg_scatter
    from torch_scatter import scatter_logsumexp as pyg_scatter_lse
except (ImportError, OSError):
    pyg_scatter = None
    pyg_scatter_lse = None


def segment_logsumexp(x, segment_ids, num_segments, return_intermediates=False):
    if pyg_scatter_lse is not None and not return_intermediates:
        try:
            lse = pyg_scatter_lse(x, segment_ids, dim=-1, dim_size=num_segments)
            return torch.where(torch.isinf(lse) & (lse < 0), torch.zeros_like(lse), lse)
        except Exception:
            pass
    max_val, counts = segment_reduce_1d(x, segment_ids, num_segments, reduce="max")
    is_empty_bool = counts == 0
    max_val = torch.where(is_empty_bool, torch.zeros_like(max_val), max_val)
    x_centered = x - max_val[..., segment_ids]
    exp_x = torch.exp(x_centered)
    sum_exp, _ = segment_reduce_1d(exp_x, segment_ids, num_segments, reduce="sum")
    is_empty = is_empty_bool.to(dtype=x.dtype)
    denom = sum_exp + is_empty
    lse = torch.log(denom) + max_val
    if return_intermediates:
        return lse, exp_x, denom
    return lse


def segment_softmax(x, segment_ids, num_segments):
    lse, exp_x, denom = segment_logsumexp(x, segment_ids, num_segments, return_intermediates=True)
    return exp_x / denom[..., segment_ids]


def _scatter_add_nd(grad_buffer, indices, src, dim):
    """Robustly scatter src into grad_buffer using 1D indices along a specific dimension."""
    view_shape = [1] * src.dim()
    view_shape[dim] = -1
    idx = indices.view(*view_shape).expand_as(src)
    return grad_buffer.scatter_add_(dim, idx, src)


def _pairwise_dot_fused(Q, K, c, u, num_nodes, scale):
    return (Q[..., c, :] * K[..., u, :]).sum(dim=-1) / scale


def _pairwise_pullback_fused(coeff, Q, K, c, u, num_nodes):
    src = coeff.unsqueeze(-1) * K[..., u, :]
    return _scatter_add_nd(torch.zeros_like(Q), c, src, dim=-2)


def positive_param(params, name):
    val = params[name]
    if isinstance(val, (float, int)):
        return val
    return torch.nn.functional.softplus(val) + 1e-8


def inverse_temperature(params, name, beta_max=None):
    beta = positive_param(params, name)
    if beta_max is not None:
        if torch.is_tensor(beta):
            beta = beta.clamp(max=beta_max)
        else:
            beta = min(beta, beta_max)
    return beta


def compute_quadratic_energy(X):
    return 0.5 * (X ** 2).sum(dim=(-2, -1))


def compute_pairwise_energy(G, c_2, u_2, params, projections, num_nodes, return_grad=False):
    d = params['d']
    beta_max = params.get('beta_max', None)
    if not params.get('use_pairwise', True):
        return (G.new_zeros(G.shape[:-2]), (G.new_zeros(G.shape), G.new_zeros(G.shape))) if return_grad else G.new_zeros(G.shape[:-2])
    lambda_2 = positive_param(params, 'lambda_2')
    if (torch.is_tensor(lambda_2) and lambda_2 <= 1e-6) or (not torch.is_tensor(lambda_2) and lambda_2 <= 1e-6) or c_2.numel() == 0:
        return (G.new_zeros(G.shape[:-2]), (G.new_zeros(G.shape), G.new_zeros(G.shape))) if return_grad else G.new_zeros(G.shape[:-2])
    Q2, K2 = projections['Q2'], projections['K2']
    scale = d ** 0.5
    ell_2 = _pairwise_dot_fused(Q2, K2, c_2, u_2, num_nodes, scale)
    if params.get("pairwise_symmetric", False):
        ell_2 = ell_2 + _pairwise_dot_fused(Q2, K2, u_2, c_2, num_nodes, scale)
    a_2 = projections.get('a_2')
    if a_2 is not None:
        ell_2 = ell_2 + a_2
    beta_2 = inverse_temperature(params, 'beta_2', beta_max=beta_max)
    lse_2, exp_x, denom = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes, return_intermediates=True)
    E = (lambda_2 / beta_2) * lse_2.sum(dim=-1)
    if not return_grad:
        return E
    probs = exp_x / denom[..., c_2]
    coeff = lambda_2 * probs / scale
    grad_Q2 = _pairwise_pullback_fused(coeff, Q2, K2, c_2, u_2, num_nodes)
    grad_K2 = _pairwise_pullback_fused(coeff, K2, Q2, u_2, c_2, num_nodes)
    if params.get("pairwise_symmetric", False):
        grad_Q2 = grad_Q2 + _pairwise_pullback_fused(coeff, Q2, K2, u_2, c_2, num_nodes)
        grad_K2 = grad_K2 + _pairwise_pullback_fused(coeff, K2, Q2, c_2, u_2, num_nodes)
    return E, (grad_Q2, grad_K2)


def compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, num_nodes, return_grad=False):
    d = params['d']
    R = params.get('R', 1)
    beta_max = params.get('beta_max', None)
    if not params.get('use_motif', True):
        dummy = G.new_zeros((*G.shape[:-1], R, d))
        return (G.new_zeros(G.shape[:-2]), (dummy, dummy)) if return_grad else G.new_zeros(G.shape[:-2])
    lambda_3 = positive_param(params, 'lambda_3')
    if (torch.is_tensor(lambda_3) and lambda_3 <= 1e-6) or (not torch.is_tensor(lambda_3) and lambda_3 <= 1e-6) or c_3.numel() == 0:
        dummy = G.new_zeros((*G.shape[:-1], R, d))
        return (G.new_zeros(G.shape[:-2]), (dummy, dummy)) if return_grad else G.new_zeros(G.shape[:-2])
    Q3, K3 = projections['Q3'], projections['K3']
    T_params = params['T_tau']
    if t_tau.numel() > 0 and t_tau.max() >= T_params.size(0):
        t_tau = torch.clamp(t_tau, max=T_params.size(0) - 1)
    T_tau_selected = T_params[t_tau].transpose(0, 1)  # [H, L, R, d_h]
    scale = (R * d) ** 0.5
    ell_3 = fused_motif_dot(Q3[..., c_3, :, :], K3[..., u_3, :, :], K3[..., v_3, :, :], T_tau_selected) / scale
    beta_3 = inverse_temperature(params, 'beta_3', beta_max=beta_max)
    lse_3, exp_x, denom = segment_logsumexp(beta_3 * ell_3, c_3, num_nodes, return_intermediates=True)
    E = (lambda_3 / beta_3) * lse_3.sum(dim=-1)
    if not return_grad:
        return E
    probs = exp_x / denom[..., c_3]
    coeff = (lambda_3 * probs / scale).unsqueeze(-1).unsqueeze(-1)
    src_Q = coeff * (K3[..., u_3, :, :] * K3[..., v_3, :, :] + T_tau_selected)
    src_Ku = coeff * (Q3[..., c_3, :, :] * K3[..., v_3, :, :])
    src_Kv = coeff * (Q3[..., c_3, :, :] * K3[..., u_3, :, :])
    grad_Q3 = _scatter_add_nd(torch.zeros_like(Q3), c_3, src_Q, dim=-3)
    grad_K3 = _scatter_add_nd(torch.zeros_like(K3), u_3, src_Ku, dim=-3)
    grad_K3 = grad_K3 + _scatter_add_nd(torch.zeros_like(K3), v_3, src_Kv, dim=-3)
    return E, (grad_Q3, grad_K3)


def compute_memory_energy(G, params, projections, return_grad=False):
    d = params['d']
    beta_max = params.get('beta_max', None)
    if not params.get('use_memory', True):
        return (G.new_zeros(G.shape[:-2]), (G.new_zeros(G.shape), None)) if return_grad else G.new_zeros(G.shape[:-2])
    lambda_m = positive_param(params, 'lambda_m')
    if (torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or (not torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or params.get('K', 0) <= 0:
        return (G.new_zeros(G.shape[:-2]), (G.new_zeros(G.shape), None)) if return_grad else G.new_zeros(G.shape[:-2])
    Qm, Km = projections['Qm'], projections['Km']
    scale = d ** 0.5
    beta_m = inverse_temperature(params, 'beta_m', beta_max=beta_max)
    Lm = (Qm @ Km.transpose(-2, -1)) / scale
    lse_m = torch.logsumexp(beta_m * Lm, dim=-1)
    E = (lambda_m / beta_m) * lse_m.sum(dim=-1)
    if not return_grad:
        return E
    probs = torch.softmax(beta_m * Lm, dim=-1)
    grad_Qm = (lambda_m / scale) * (probs @ Km)
    return E, (grad_Qm, None)


def compute_energy_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, params, projections=None):
    N = X.size(-2)
    E_quad = compute_quadratic_energy(X)
    E_att2 = compute_pairwise_energy(G, c_2, u_2, params, projections, N)
    E_att3 = compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, N)
    E_mem = compute_memory_energy(G, params, projections)
    if E_att2.dim() > E_quad.dim():
        E_att2 = E_att2.mean(dim=-1)
    if E_att3.dim() > E_quad.dim():
        E_att3 = E_att3.mean(dim=-1)
    if E_mem.dim() > E_quad.dim():
        E_mem = E_mem.mean(dim=-1)
    return E_quad - E_att2 - E_att3 - E_mem


def compute_energy_and_grad_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, params, projections=None):
    N = X.size(-2)
    E_quad = compute_quadratic_energy(X)
    E_att2, (gQ2, gK2) = compute_pairwise_energy(G, c_2, u_2, params, projections, N, return_grad=True)
    E_att3, (gQ3, gK3) = compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, N, return_grad=True)
    E_mem, (gQm, _) = compute_memory_energy(G, params, projections, return_grad=True)
    H = params.get('num_heads', 1)
    grads = {
        'grad_Q2': gQ2/H, 'grad_K2': gK2/H,
        'grad_Q3': gQ3/H, 'grad_K3': gK3/H,
        'grad_Qm': (gQm/H) if gQm is not None else None
    }
    E2 = E_att2.mean(dim=-1) if E_att2.dim() > E_quad.dim() else E_att2
    E3 = E_att3.mean(dim=-1) if E_att3.dim() > E_quad.dim() else E_att3
    Em = E_mem.mean(dim=-1) if E_mem.dim() > E_quad.dim() else E_mem
    return E_quad - E2 - E3 - Em, X, grads


def compute_memory_entropy(G, params, projections, eps=1e-12):
    if not params.get('use_memory', True) or params.get('K', 0) <= 0:
        return torch.zeros(G.shape[:-2], dtype=G.dtype, device=G.device)
    Qm, Km = projections['Qm'], projections['Km']
    d = params['d']
    Lm = (Qm @ Km.transpose(-2, -1)) / (d ** 0.5)
    beta_m = inverse_temperature(params, 'beta_m', beta_max=params.get('beta_max', None))
    probs = torch.softmax(beta_m * Lm, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
    return entropy.mean(dim=-1)
