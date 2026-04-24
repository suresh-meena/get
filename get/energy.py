import torch
from .fused_ops import segment_reduce_1d, fused_motif_dot

try:
    import dgl.ops as dgl_ops
    import dgl
except (ImportError, OSError):
    dgl_ops = None
    dgl = None

try:
    from torch_scatter import scatter as pyg_scatter
except (ImportError, OSError):
    pyg_scatter = None

def segment_logsumexp(x, segment_ids, num_segments, return_intermediates=False):
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
    denom = sum_exp + is_empty
    lse = torch.log(denom) + max_val
    if return_intermediates:
        return lse, exp_x, denom
    return lse

def segment_softmax(x, segment_ids, num_segments):
    lse, exp_x, denom = segment_logsumexp(x, segment_ids, num_segments, return_intermediates=True)
    if x.dim() == 2:
        return exp_x / denom[:, segment_ids]
    return exp_x / denom[segment_ids]

def _get_dgl_graph(c, u, num_nodes):
    """Internal helper to create a temporary DGL graph for fused ops."""
    return dgl.graph((u, c), num_nodes=num_nodes).to(c.device)

def _pairwise_dot_fused(Q, K, c, u, num_nodes, scale):
    """Fused gather-dot-product using DGL."""
    if dgl_ops is not None and Q.dim() == 2:
        g = _get_dgl_graph(c, u, num_nodes)
        return dgl_ops.u_dot_v(g, Q, K).squeeze(-1) / scale
    
    if Q.dim() == 3:
        return (Q[:, c] * K[:, u]).sum(dim=-1) / scale
    return (Q[c] * K[u]).sum(dim=-1) / scale

def _pairwise_pullback_fused(coeff, Q, K, c, u, num_nodes):
    """Fused gather-scatter pullback using DGL or PyG."""
    if dgl_ops is not None and Q.dim() == 2:
        g = _get_dgl_graph(c, u, num_nodes)
        return dgl_ops.u_mul_e_sum(g, K, coeff.unsqueeze(-1))
    
    if pyg_scatter is not None:
        if Q.dim() == 3:
            idx = c.view(1, -1).expand(Q.size(0), -1)
            return pyg_scatter(coeff.unsqueeze(-1) * K[:, u], idx, dim=1, dim_size=num_nodes, reduce="sum")
        return pyg_scatter(coeff.unsqueeze(-1) * K[u], c, dim=0, dim_size=num_nodes, reduce="sum")
        
    if Q.dim() == 3:
        grad = torch.zeros_like(Q)
        idx = c.view(1, -1, 1).expand(Q.size(0), -1, Q.size(2))
        grad.scatter_add_(1, idx, coeff.unsqueeze(-1) * K[:, u])
        return grad
    else:
        grad = torch.zeros_like(Q)
        grad.scatter_add_(0, c.view(-1, 1).expand(-1, Q.size(1)), coeff.unsqueeze(-1) * K[u])
        return grad

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


def compute_pairwise_energy(G, c_2, u_2, params, projections, num_nodes, return_grad=False):
    d = params['d']
    beta_max = params.get('beta_max', None)
    if not _branch_enabled(params, 'use_pairwise'):
        zero_E = G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
        return (zero_E, G.new_zeros(G.shape)) if return_grad else zero_E
    
    lambda_2 = positive_param(params, 'lambda_2')
    if lambda_2 <= 1e-6 or c_2.numel() == 0:
        zero_E = G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
        return (zero_E, G.new_zeros(G.shape)) if return_grad else zero_E

    if projections is not None and 'Q2' in projections:
        Q2, K2 = projections['Q2'], projections['K2']
    else:
        if G.dim() == 3:
            Q2 = G @ params['W_Q2'].transpose(-2, -1)
            K2 = G @ params['W_K2'].transpose(-2, -1)
        else:
            Q2 = G @ params['W_Q2'].transpose(-2, -1)
            K2 = G @ params['W_K2'].transpose(-2, -1)

    scale = d ** 0.5
    ell_2 = _pairwise_dot_fused(Q2, K2, c_2, u_2, num_nodes, scale)
    if params.get("pairwise_symmetric", False):
        ell_2 = ell_2 + _pairwise_dot_fused(Q2, K2, u_2, c_2, num_nodes, scale)

    a_2 = projections.get('a_2') if projections else params.get('a_2')
    if a_2 is not None:
        ell_2 = ell_2 + a_2

    beta_2 = inverse_temperature(params, 'beta_2', beta_max=beta_max)
    lse_2, exp_x, denom = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes, return_intermediates=True)
    E = (lambda_2 / beta_2) * lse_2.sum(dim=-1)
    
    if not return_grad:
        return E

    # Grad computation
    if ell_2.dim() == 2:
        probs = exp_x / denom[:, c_2]
    else:
        probs = exp_x / denom[c_2]
    
    coeff = lambda_2 * probs / scale
    grad_Q2 = _pairwise_pullback_fused(coeff, Q2, K2, c_2, u_2, num_nodes)
    grad_K2 = _pairwise_pullback_fused(coeff, K2, Q2, u_2, c_2, num_nodes)
    if params.get("pairwise_symmetric", False):
        grad_Q2 = grad_Q2 + _pairwise_pullback_fused(coeff, Q2, K2, u_2, c_2, num_nodes)
        grad_K2 = grad_K2 + _pairwise_pullback_fused(coeff, K2, Q2, c_2, u_2, num_nodes)
            
    grad_G = grad_Q2 @ params['W_Q2'] + grad_K2 @ params['W_K2']
    return E, grad_G


def compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, num_nodes, return_grad=False):
    d = params['d']
    R = params.get('R', 1)
    beta_max = params.get('beta_max', None)
    if not _branch_enabled(params, 'use_motif'):
        zero_E = G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
        return (zero_E, G.new_zeros(G.shape)) if return_grad else zero_E
    lambda_3 = positive_param(params, 'lambda_3')
    if lambda_3 <= 1e-6 or c_3.numel() == 0:
        zero_E = G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
        return (zero_E, G.new_zeros(G.shape)) if return_grad else zero_E

    if projections is not None and 'Q3' in projections:
        Q3, K3 = projections['Q3'], projections['K3']
    else:
        if G.dim() == 3:
            Q3 = (G @ params['W_Q3'].transpose(-2, -1)).view(G.size(0), num_nodes, R, d)
            K3 = (G @ params['W_K3'].transpose(-2, -1)).view(G.size(0), num_nodes, R, d)
        else:
            Q3 = (G @ params['W_Q3'].transpose(-2, -1)).view(num_nodes, R, d)
            K3 = (G @ params['W_K3'].transpose(-2, -1)).view(num_nodes, R, d)

    T_params = params['T_tau']
    if G.dim() == 3:
        if T_params.dim() == 4:
            T_tau = T_params[:, t_tau] # [Batch, M, R, d]
        else:
            T_tau = T_params[t_tau]
    else:
        T_tau = T_params[t_tau]
        
    scale = (R * d) ** 0.5
    if G.dim() == 3:
        Q3_c = torch.index_select(Q3, 1, c_3)
        K3_u = torch.index_select(K3, 1, u_3)
        K3_v = torch.index_select(K3, 1, v_3)
        ell_3 = fused_motif_dot(Q3_c, K3_u, K3_v, T_tau) / scale
    else:
        Q3_c = torch.index_select(Q3, 0, c_3)
        K3_u = torch.index_select(K3, 0, u_3)
        K3_v = torch.index_select(K3, 0, v_3)
        ell_3 = fused_motif_dot(Q3_c, K3_u, K3_v, T_tau) / scale

    beta_3 = inverse_temperature(params, 'beta_3', beta_max=beta_max)
    lse_3, exp_x, denom = segment_logsumexp(beta_3 * ell_3, c_3, num_nodes, return_intermediates=True)
    E = (lambda_3 / beta_3) * lse_3.sum(dim=-1)
    
    if not return_grad:
        return E
        
    if ell_3.dim() == 2:
        probs = exp_x / denom[:, c_3]
    else:
        probs = exp_x / denom[c_3]
        
    coeff = (lambda_3 * probs / scale).unsqueeze(-1).unsqueeze(-1)
    
    if Q3.dim() == 4:
        grad_Q3 = torch.zeros_like(Q3)
        val_Q = coeff * (K3[:, u_3] * K3[:, v_3] + T_tau)
        grad_Q3.scatter_add_(1, c_3.view(1, -1, 1, 1).expand(Q3.size(0), -1, R, d), val_Q)
        grad_K3 = torch.zeros_like(K3)
        grad_K3.scatter_add_(1, u_3.view(1, -1, 1, 1).expand(K3.size(0), -1, R, d), coeff * (Q3[:, c_3] * K3[:, v_3]))
        grad_K3.scatter_add_(1, v_3.view(1, -1, 1, 1).expand(K3.size(0), -1, R, d), coeff * (Q3[:, c_3] * K3[:, u_3]))
    else:
        if pyg_scatter is not None:
            grad_Q3 = pyg_scatter(coeff * (K3[u_3] * K3[v_3] + T_tau), c_3, dim=0, dim_size=Q3.size(0), reduce="sum")
            grad_K3 = pyg_scatter(coeff * (Q3[c_3] * K3[v_3]), u_3, dim=0, dim_size=K3.size(0), reduce="sum")
            grad_K3 = grad_K3 + pyg_scatter(coeff * (Q3[c_3] * K3[u_3]), v_3, dim=0, dim_size=K3.size(0), reduce="sum")
        else:
            grad_Q3 = torch.zeros_like(Q3)
            grad_Q3.scatter_add_(0, c_3.view(-1, 1, 1).expand(-1, R, d), coeff * (K3[u_3] * K3[v_3] + T_tau))
            grad_K3 = torch.zeros_like(K3)
            grad_K3.scatter_add_(0, u_3.view(-1, 1, 1).expand(-1, R, d), coeff * (Q3[c_3] * K3[v_3]))
            grad_K3.scatter_add_(0, v_3.view(-1, 1, 1).expand(-1, R, d), coeff * (Q3[c_3] * K3[u_3]))

    if Q3.dim() == 4:
        grad_G = grad_Q3.reshape(grad_Q3.size(0), grad_Q3.size(1), -1) @ params['W_Q3'] + \
                 grad_K3.reshape(grad_K3.size(0), grad_K3.size(1), -1) @ params['W_K3']
    else:
        grad_G = grad_Q3.reshape(grad_Q3.size(0), -1) @ params['W_Q3'] + \
                 grad_K3.reshape(grad_K3.size(0), -1) @ params['W_K3']
    return E, grad_G


def compute_memory_energy(G, params, projections, return_grad=False):
    d = params['d']
    beta_max = params.get('beta_max', None)
    if not _branch_enabled(params, 'use_memory'):
        zero_E = G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
        return (zero_E, G.new_zeros(G.shape)) if return_grad else zero_E
    lambda_m = positive_param(params, 'lambda_m')
    if lambda_m <= 1e-6 or params.get('K', 0) <= 0:
        zero_E = G.new_zeros(G.shape[0]) if G.dim() == 3 else torch.zeros((), dtype=G.dtype, device=G.device)
        return (zero_E, G.new_zeros(G.shape)) if return_grad else zero_E

    if projections is not None and 'Qm' in projections:
        Qm, Km = projections['Qm'], projections['Km']
    else:
        if G.dim() == 3:
            Qm = G @ params['W_Qm'].transpose(-2, -1)
            Km = params['B_mem'] @ params['W_Km'].transpose(-2, -1)
        else:
            Qm = G @ params['W_Qm'].transpose(-2, -1)
            Km = params['B_mem'] @ params['W_Km'].transpose(-2, -1)

    scale = d ** 0.5
    beta_m = inverse_temperature(params, 'beta_m', beta_max=beta_max)
    if Qm.dim() == 3:
        Lm = (Qm @ Km.transpose(-2, -1)) / scale
        lse_m = torch.logsumexp(beta_m * Lm, dim=2)
        E = (lambda_m / beta_m) * lse_m.sum(dim=1)
    else:
        Lm = (Qm @ Km.transpose(-2, -1)) / scale
        lse_m = torch.logsumexp(beta_m * Lm, dim=1)
        E = (lambda_m / beta_m) * lse_m.sum()
        
    if not return_grad:
        return E
        
    probs = torch.softmax(beta_m * Lm, dim=-1)
    grad_Qm = (lambda_m / scale) * (probs @ Km)
    grad_G = grad_Qm @ params['W_Qm']
    return E, grad_G


def compute_memory_entropy(G, params, projections, eps=1e-12):
    d = params['d']
    if not _branch_enabled(params, 'use_memory') or params.get('K', 0) <= 0:
        return torch.zeros((), dtype=G.dtype, device=G.device)
    if projections is not None and 'Qm' in projections:
        Qm, Km = projections['Qm'], projections['Km']
    else:
        if G.dim() == 3:
            Qm = G @ params['W_Qm'].transpose(-2, -1)
            Km = params['B_mem'] @ params['W_Km'].transpose(-2, -1)
        else:
            Qm = G @ params['W_Qm'].transpose(-2, -1)
            Km = params['B_mem'] @ params['W_Km'].transpose(-2, -1)
    
    if Qm.dim() == 3:
        Lm = (Qm @ Km.transpose(-2, -1)) / (d ** 0.5)
        beta_m = inverse_temperature(params, 'beta_m', beta_max=params.get('beta_max', None))
        probs = torch.softmax(beta_m * Lm, dim=2)
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=2)
        return entropy.mean(dim=1)
    else:
        Lm = (Qm @ Km.transpose(-2, -1)) / (d ** 0.5)
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

def compute_energy_and_grad_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, params, projections=None):
    N = X.size(-2)
    E_quad = compute_quadratic_energy(X)
    grad_X_quad = X # d(0.5||X||^2)/dX = X
    
    E_att2, grad_G_att2 = compute_pairwise_energy(G, c_2, u_2, params, projections, N, return_grad=True)
    E_att3, grad_G_att3 = compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, N, return_grad=True)
    E_mem, grad_G_mem = compute_memory_energy(G, params, projections, return_grad=True)
    
    E = E_quad - E_att2 - E_att3 - E_mem
    return E, grad_X_quad, grad_G_att2 + grad_G_att3 + grad_G_mem
