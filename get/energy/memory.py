"""Memory (Hopfield) energy branch: global prototype retrieval."""
import torch
from .ops import positive_param, inverse_temperature, scatter_add_nd


def compute_memory_energy(G, batch, num_graphs, params, projections, return_grad=False):
    d = params['d']
    beta_max = params.get('beta_max', None)
    if not params.get('use_memory', True):
        zero_E = G.new_zeros((*G.shape[:-2], num_graphs))
        return (zero_E, (G.new_zeros(G.shape), None)) if return_grad else zero_E

    lambda_m = positive_param(params, 'lambda_m')
    if (torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or (not torch.is_tensor(lambda_m) and lambda_m <= 1e-6) or params.get('K', 0) <= 0:
        zero_E = G.new_zeros((*G.shape[:-2], num_graphs))
        return (zero_E, (G.new_zeros(G.shape), None)) if return_grad else zero_E

    Qm, Km = projections['Qm'], projections['Km']
    scale = d ** 0.5
    beta_m = inverse_temperature(params, 'beta_m', beta_max=beta_max)
    Lm = (Qm @ Km.transpose(-2, -1)) / scale
    lse_m = torch.logsumexp(beta_m * Lm, dim=-1)
    E = (lambda_m / beta_m) * scatter_add_nd(lse_m.new_zeros((*lse_m.shape[:-1], num_graphs)), batch, lse_m, dim=-1)
    if not return_grad:
        return E

    probs = torch.softmax(beta_m * Lm, dim=-1)
    grad_Qm = (lambda_m / scale) * (probs @ Km)
    return E, (grad_Qm, None)


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
