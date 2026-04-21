import torch

def segment_logsumexp(x, segment_ids, num_segments):
    """
    Computes log( 1[|N(i)|=0] + sum_j exp(x_j) ) efficiently using scatter_reduce.
    
    Args:
        x: (L,) values to reduce
        segment_ids: (L,) segment indices
        num_segments: int, total number of segments (nodes)
    """
    # 1. Find max per segment for numerical stability
    max_val = torch.full((num_segments,), float('-inf'), dtype=x.dtype, device=x.device)
    max_val.scatter_reduce_(0, segment_ids, x, reduce="amax", include_self=False)
    
    # Where max_val is -inf (empty segments), set it to 0 so exp doesn't blow up
    max_val = torch.where(max_val == float('-inf'), torch.zeros_like(max_val), max_val)
    
    # 2. Subtract max_val and exponentiate
    x_centered = x - max_val[segment_ids]
    exp_x = torch.exp(x_centered)
    
    # 3. Sum exponentials per segment
    sum_exp = torch.zeros(num_segments, dtype=x.dtype, device=x.device)
    sum_exp.scatter_add_(0, segment_ids, exp_x)
    
    # 4. Handle empty segments (masked indicator convention from paper)
    # If empty, sum_exp == 0. We want the output to be 0.
    is_empty = (sum_exp == 0).float()
    
    # log( is_empty + sum_exp ) + max_val
    # If empty: log( 1 + 0 ) + 0 = 0
    # If not empty: log( 0 + sum_exp ) + max_val
    lse = torch.log(sum_exp + is_empty) + max_val
    return lse


def positive_param(params, name):
    return torch.nn.functional.softplus(params[name]) + 1e-8


def inverse_temperature(params, name, beta_max=None):
    beta = positive_param(params, name)
    if beta_max is not None:
        beta = beta.clamp(max=beta_max)
    return beta


def compute_quadratic_energy(X):
    return 0.5 * (X ** 2).sum()


def compute_pairwise_energy(G, c_2, u_2, params, projections, num_nodes):
    d = params['d']
    beta_max = params.get('beta_max', None)
    lambda_2 = positive_param(params, 'lambda_2')
    if lambda_2 <= 1e-6 or c_2.numel() == 0:
        return torch.zeros((), dtype=G.dtype, device=G.device)

    if projections is not None and 'Q2' in projections:
        Q2, K2 = projections['Q2'], projections['K2']
    else:
        Q2 = G @ params['W_Q2']
        K2 = G @ params['W_K2']

    ell_2 = (Q2[c_2] * K2[u_2]).sum(dim=-1) / (d ** 0.5)
    if params.get("pairwise_symmetric", False):
        ell_2 = ell_2 + (Q2[u_2] * K2[c_2]).sum(dim=-1) / (d ** 0.5)

    a_2 = projections.get('a_2') if projections else params.get('a_2')
    if a_2 is not None:
        ell_2 = ell_2 + a_2

    beta_2 = inverse_temperature(params, 'beta_2', beta_max=beta_max)
    lse_2 = segment_logsumexp(beta_2 * ell_2, c_2, num_nodes)
    return (lambda_2 / beta_2) * lse_2.sum()


def compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, num_nodes):
    d = params['d']
    R = params.get('R', 1)
    beta_max = params.get('beta_max', None)
    lambda_3 = positive_param(params, 'lambda_3')
    if lambda_3 <= 1e-6 or c_3.numel() == 0:
        return torch.zeros((), dtype=G.dtype, device=G.device)

    if projections is not None and 'Q3' in projections:
        Q3, K3 = projections['Q3'], projections['K3']
    else:
        Q3 = (G @ params['W_Q3']).view(num_nodes, R, d)
        K3 = (G @ params['W_K3']).view(num_nodes, R, d)

    T_tau = params['T_tau'][t_tau]
    Phi_3 = K3[u_3] * K3[v_3] + T_tau
    ell_3 = (Q3[c_3] * Phi_3).sum(dim=(1, 2)) / ((R * d) ** 0.5)

    beta_3 = inverse_temperature(params, 'beta_3', beta_max=beta_max)
    lse_3 = segment_logsumexp(beta_3 * ell_3, c_3, num_nodes)
    return (lambda_3 / beta_3) * lse_3.sum()


def compute_memory_energy(G, params, projections):
    d = params['d']
    beta_max = params.get('beta_max', None)
    lambda_m = positive_param(params, 'lambda_m')
    if lambda_m <= 1e-6 or params.get('K', 0) <= 0:
        return torch.zeros((), dtype=G.dtype, device=G.device)

    Qm = projections['Qm'] if (projections is not None and 'Qm' in projections) else (G @ params['W_Qm'])
    Km = projections['Km'] if (projections is not None and 'Km' in projections) else (params['B_mem'] @ params['W_Km'])

    Lm = (Qm @ Km.t()) / (d ** 0.5)
    beta_m = inverse_temperature(params, 'beta_m', beta_max=beta_max)
    lse_m = torch.logsumexp(beta_m * Lm, dim=1)
    return (lambda_m / beta_m) * lse_m.sum()

def compute_energy_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, params, projections=None):
    """
    Computes the total scalar energy of the GET model.
    Optimized to accept pre-computed projections and biases.
    """
    N = X.size(0)
    E_quad = compute_quadratic_energy(X)
    E_att2 = compute_pairwise_energy(G, c_2, u_2, params, projections, N)
    E_att3 = compute_motif_energy(G, c_3, u_3, v_3, t_tau, params, projections, N)
    E_mem = compute_memory_energy(G, params, projections)

    return E_quad - E_att2 - E_att3 - E_mem
