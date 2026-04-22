import torch
from .fused_ops import segment_reduce_1d, fused_motif_dot

def segment_logsumexp(x, segment_ids, num_segments):
    """
    Computes log( 1[|N(i)|=0] + sum_j exp(x_j) ) efficiently.
    Supports optional batch dimension in x: [L] or [B, L].
    """
    # 1. Find max per segment for numerical stability
    max_val, counts = segment_reduce_1d(x, segment_ids, num_segments, reduce="max")
    is_empty_bool = counts == 0
    # Where max_val is -inf (empty segments), set it to 0 so exp doesn't blow up
    max_val = torch.where(is_empty_bool, torch.zeros_like(max_val), max_val)
    
    # 2. Subtract max_val and exponentiate
    if x.dim() == 2:
        x_centered = x - max_val[:, segment_ids]
    else:
        x_centered = x - max_val[segment_ids]
    exp_x = torch.exp(x_centered)
    
    # 3. Sum exponentials per segment
    sum_exp, _ = segment_reduce_1d(exp_x, segment_ids, num_segments, reduce="sum")
    
    # 4. Handle empty segments (masked indicator convention from paper)
    is_empty = is_empty_bool.to(dtype=x.dtype)
    
    # log( is_empty + sum_exp ) + max_val
    lse = torch.log(sum_exp + is_empty) + max_val
    return lse


def positive_param(params, name):
    return torch.nn.functional.softplus(params[name]) + 1e-8


def _branch_enabled(params, name):
    return bool(params.get(name, True))


def inverse_temperature(params, name, beta_max=None):
    beta = positive_param(params, name)
    if beta_max is not None:
        beta = beta.clamp(max=beta_max)
    return beta


def compute_quadratic_energy(X):
    # Returns [B] if X is [B, N, D], else scalar.
    if X.dim() == 3:
        return 0.5 * (X ** 2).sum(dim=(1, 2))
    return 0.5 * (X ** 2).sum()


def compute_pairwise_energy(G, c_2, u_2, params, projections, num_nodes):
    d = params['d']
    beta_max = params.get('beta_max', None)
    if not _branch_enabled(params, 'use_pairwise'):
        return G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
    lambda_2 = positive_param(params, 'lambda_2')
    if lambda_2 <= 1e-6 or c_2.numel() == 0:
        return G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)

    if projections is not None and 'Q2' in projections:
        Q2, K2 = projections['Q2'], projections['K2']
    else:
        # G is [N, D] or [B, N, D]
        Q2 = G @ params['W_Q2']
        K2 = G @ params['W_K2']

    if Q2.dim() == 3:
        # Batched: [B, N, D]
        ell_2 = (Q2[:, c_2] * K2[:, u_2]).sum(dim=-1) / (d ** 0.5)
        if params.get("pairwise_symmetric", False):
            ell_2 = ell_2 + (Q2[:, u_2] * K2[:, c_2]).sum(dim=-1) / (d ** 0.5)
    else:
        ell_2 = (Q2[c_2] * K2[u_2]).sum(dim=-1) / (d ** 0.5)
        if params.get("pairwise_symmetric", False):
            ell_2 = ell_2 + (Q2[u_2] * K2[c_2]).sum(dim=-1) / (d ** 0.5)

    a_2 = projections.get('a_2') if projections else params.get('a_2')
    if a_2 is not None:
        ell_2 = ell_2 + a_2

    beta_2 = inverse_temperature(params, 'beta_2', beta_max=beta_max)
    lse_2 = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes)
    return (lambda_2 / beta_2) * lse_2.sum(dim=-1)


def compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, num_nodes):
    d = params['d']
    R = params.get('R', 1)
    beta_max = params.get('beta_max', None)
    if not _branch_enabled(params, 'use_motif'):
        return G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
    lambda_3 = positive_param(params, 'lambda_3')
    if lambda_3 <= 1e-6 or c_3.numel() == 0:
        return G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)

    if projections is not None and 'Q3' in projections:
        Q3, K3 = projections['Q3'], projections['K3']
    else:
        if G.dim() == 3:
            Q3 = (G @ params['W_Q3']).view(G.size(0), num_nodes, R, d)
            K3 = (G @ params['W_K3']).view(G.size(0), num_nodes, R, d)
        else:
            Q3 = (G @ params['W_Q3']).view(num_nodes, R, d)
            K3 = (G @ params['W_K3']).view(num_nodes, R, d)

    T_tau = params['T_tau'][t_tau]
    ell_3 = fused_motif_dot(Q3[:, c_3], K3[:, u_3], K3[:, v_3], T_tau) / ((R * d) ** 0.5) if G.dim() == 3 else \
            fused_motif_dot(Q3[c_3], K3[u_3], K3[v_3], T_tau) / ((R * d) ** 0.5)

    beta_3 = inverse_temperature(params, 'beta_3', beta_max=beta_max)
    lse_3 = segment_logsumexp(beta_3 * ell_3, c_3, num_nodes)
    return (lambda_3 / beta_3) * lse_3.sum(dim=-1)


def compute_memory_energy(G, params, projections):
    d = params['d']
    beta_max = params.get('beta_max', None)
    if not _branch_enabled(params, 'use_memory'):
        return G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
    lambda_m = positive_param(params, 'lambda_m')
    if lambda_m <= 1e-6 or params.get('K', 0) <= 0:
        return G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)

    if projections is not None and 'Qm' in projections:
        Qm, Km = projections['Qm'], projections['Km']
    else:
        Qm = G @ params['W_Qm']
        Km = params['B_mem'] @ params['W_Km']

    if Qm.dim() == 3:
        # [B, N, K]
        Lm = (Qm @ Km.t()) / (d ** 0.5)
        beta_m = inverse_temperature(params, 'beta_m', beta_max=beta_max)
        lse_m = torch.logsumexp(beta_m * Lm, dim=2)
        return (lambda_m / beta_m) * lse_m.sum(dim=1)
    else:
        Lm = (Qm @ Km.t()) / (d ** 0.5)
        beta_m = inverse_temperature(params, 'beta_m', beta_max=beta_max)
        lse_m = torch.logsumexp(beta_m * Lm, dim=1)
        return (lambda_m / beta_m) * lse_m.sum()


def compute_memory_entropy(G, params, projections, eps=1e-12):
    d = params['d']
    if not _branch_enabled(params, 'use_memory') or params.get('K', 0) <= 0:
        return torch.zeros((), dtype=G.dtype, device=G.device)

    Qm = projections['Qm'] if (projections is not None and 'Qm' in projections) else (G @ params['W_Qm'])
    Km = projections['Km'] if (projections is not None and 'Km' in projections) else (params['B_mem'] @ params['W_Km'])
    
    if Qm.dim() == 3:
        Lm = (Qm @ Km.t()) / (d ** 0.5)
        beta_m = inverse_temperature(params, 'beta_m', beta_max=params.get('beta_max', None))
        probs = torch.softmax(beta_m * Lm, dim=2)
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=2)
        return entropy.mean(dim=1)
    else:
        Lm = (Qm @ Km.t()) / (d ** 0.5)
        beta_m = inverse_temperature(params, 'beta_m', beta_max=params.get('beta_max', None))
        probs = torch.softmax(beta_m * Lm, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
        return entropy.mean()

def compute_energy_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, params, projections=None):
    """
    Computes the total scalar energy of the GET model.
    Supports batched state iterates [B, N, D] returning [B] energies.
    """
    N = X.size(-2)
    E_quad = compute_quadratic_energy(X)
    E_att2 = compute_pairwise_energy(G, c_2, u_2, params, projections, N)
    E_att3 = compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, N)
    E_mem = compute_memory_energy(G, params, projections)

    return E_quad - E_att2 - E_att3 - E_mem
