import torch
import torch.nn.functional as F

def segment_reduce_1d(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, reduce="sum", dim: int = -1):
    """
    Segmented reduction using native PyTorch operations.
    src: [..., L, ...]
    segment_ids: [L]
    """
    if dim < 0:
        dim = src.dim() + dim

    counts = torch.bincount(segment_ids, minlength=num_segments)

    idx_shape = [1] * src.dim()
    idx_shape[dim] = len(segment_ids)
    idx = segment_ids.view(*idx_shape).expand_as(src)

    out_shape = list(src.shape)
    out_shape[dim] = num_segments

    if reduce == "sum":
        out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, idx, src)
    elif reduce == "max":
        out = torch.full(out_shape, float("-inf"), dtype=src.dtype, device=src.device)
        if src.numel() > 0:
            out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=False)
    else:
        raise ValueError(f"Unsupported reduction: {reduce}")

    return out, counts

def segment_logsumexp(x: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int = -1):
    """
    Computes logsumexp over segments using native PyTorch scatter_reduce.
    """
    if dim < 0:
        dim = x.dim() + dim

    max_val, counts = segment_reduce_1d(x, segment_ids, num_segments, reduce="max", dim=dim)
    is_empty_bool = (counts == 0)
    
    idx_shape = [1] * max_val.dim()
    idx_shape[dim] = num_segments
    is_empty_bool = is_empty_bool.view(*idx_shape)
    
    max_safe = torch.where(is_empty_bool, torch.zeros_like(max_val), max_val)
    
    # We need to expand max_safe back to the shape of x to subtract
    idx_shape2 = [1] * x.dim()
    idx_shape2[dim] = len(segment_ids)
    idx = segment_ids.view(*idx_shape2).expand_as(x)
    
    x_centered = x - torch.gather(max_safe, dim, idx)
    exp_x = torch.exp(x_centered)
    
    sum_exp, _ = segment_reduce_1d(exp_x, segment_ids, num_segments, reduce="sum", dim=dim)
    is_empty = is_empty_bool.to(dtype=x.dtype)
    denom = sum_exp + is_empty
    
    lse = torch.log(denom) + max_safe
    return torch.where(is_empty_bool, torch.full_like(lse, float("-inf")), lse)

def scatter_add_nd(grad_buffer: torch.Tensor, indices: torch.Tensor, src: torch.Tensor, dim: int):
    """
    Memory-efficient N-dimensional scatter-add using index_add.
    """
    perm = list(range(src.dim()))
    perm[0], perm[dim] = perm[dim], perm[0]

    src_p = src.permute(perm)
    if src_p.dtype != grad_buffer.dtype:
        src_p = src_p.to(dtype=grad_buffer.dtype)
        
    return grad_buffer.permute(perm).index_add_(0, indices, src_p).permute(perm)

def fused_motif_dot_baseline(
    Q3_c: torch.Tensor,
    K3_u: torch.Tensor,
    K3_v: torch.Tensor,
    T_tau: torch.Tensor,
) -> torch.Tensor:
    """Baseline trilinear motif score contraction with explicit elementwise chain."""
    return (Q3_c * (K3_u * K3_v + T_tau)).sum(dim=(-1, -2))


def fused_motif_dot(
    Q3_c: torch.Tensor,
    K3_u: torch.Tensor,
    K3_v: torch.Tensor,
    T_tau: torch.Tensor,
) -> torch.Tensor:
    """
    Optimized trilinear motif contraction.

    Uses addcmul to materialize one intermediate and einsum for contraction,
    reducing explicit temporary tensors compared to baseline chaining.
    """
    # CPU kernels are typically faster with the plain elementwise chain.
    if Q3_c.device.type != "cuda":
        return fused_motif_dot_baseline(Q3_c, K3_u, K3_v, T_tau)

    mixed = torch.addcmul(T_tau, K3_u, K3_v)
    return torch.einsum("...rd,...rd->...", Q3_c, mixed)

def positive_param(params: dict, name: str):
    val = params[name]
    if isinstance(val, (float, int)):
        return val
    return F.softplus(val) + 1e-8

def inverse_temperature(params: dict, name: str, beta_max=None):
    beta = positive_param(params, name)
    if beta_max is not None:
        if torch.is_tensor(beta):
            beta = beta.clamp(max=beta_max)
        else:
            beta = min(beta, beta_max)
    return beta

def get_degree_from_incidence(c_2: torch.Tensor, num_nodes: int):
    counts = torch.bincount(c_2, minlength=num_nodes)
    return counts.to(dtype=torch.float32)

def compute_degree_scaler(degrees: torch.Tensor, avg_degree: float, mode="pna"):
    if mode == "pna":
        return torch.log(degrees + 1.0) / torch.log(torch.tensor(avg_degree + 1.0, device=degrees.device))
    return torch.ones_like(degrees)
