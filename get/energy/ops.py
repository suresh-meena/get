import torch
import torch.nn.functional as F

def segment_reduce_1d(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, reduce="sum", dim: int = -1):
    """
    Segmented reduction using native PyTorch.
    """
    if dim < 0:
        dim = src.dim() + dim
        
    idx_shape = [1] * src.dim()
    idx_shape[dim] = len(segment_ids)
    idx = segment_ids.view(*idx_shape).expand_as(src)

    out = torch.zeros(
        *src.shape[:dim], num_segments, *src.shape[dim+1:], 
        dtype=src.dtype, device=src.device
    )
    
    if reduce == "sum":
        out.scatter_add_(dim, idx, src)
    elif reduce == "max":
        out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=False)
    elif reduce == "min":
        out.scatter_reduce_(dim, idx, src, reduce="amin", include_self=False)
    elif reduce == "mean":
        out.scatter_reduce_(dim, idx, src, reduce="mean", include_self=False)
    else:
        raise ValueError(f"Unsupported reduction type: {reduce}")
        
    counts = torch.bincount(segment_ids, minlength=num_segments)
    return out, counts

def segment_logsumexp(x: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int = -1):
    """
    Computes logsumexp over segments using native PyTorch to support torch.compile.
    """
    if dim < 0:
        dim = x.dim() + dim

    idx_shape = [1] * x.dim()
    idx_shape[dim] = len(segment_ids)
    idx = segment_ids.view(*idx_shape).expand_as(x)

    # Use native scatter_reduce to find max per segment
    out_max = torch.zeros(
        *x.shape[:dim], num_segments, *x.shape[dim+1:], 
        dtype=x.dtype, device=x.device
    ).fill_(float('-inf'))
    out_max.scatter_reduce_(dim, idx, x, reduce="amax", include_self=False)

    # Calculate x - max(x)
    max_expanded = torch.gather(out_max, dim, idx)
    # Clamp to avoid nan when x is -inf and max is -inf
    max_expanded = torch.where(max_expanded == float('-inf'), torch.zeros_like(max_expanded), max_expanded)
    x_shifted = x - max_expanded

    # Exponential and scatter_add
    exp_x_shifted = torch.exp(x_shifted)
    out_sum_exp = torch.zeros_like(out_max).fill_(0.0)
    out_sum_exp.scatter_add_(dim, idx, exp_x_shifted)

    # Log and add back max
    out = torch.log(out_sum_exp) + out_max
    # Fill empty segments with -inf
    out = torch.where(out_sum_exp == 0, torch.tensor(float('-inf'), dtype=out.dtype, device=out.device), out)
    return out

def scatter_add_nd(grad_buffer: torch.Tensor, indices: torch.Tensor, src: torch.Tensor, dim: int):
    """
    Memory-efficient N-dimensional scatter-add using index_add directly.
    """
    if src.dtype != grad_buffer.dtype:
        src = src.to(dtype=grad_buffer.dtype)
        
    return grad_buffer.index_add_(dim, indices, src)

def fused_motif_dot(
    Q3_c: torch.Tensor,
    K3_u: torch.Tensor,
    K3_v: torch.Tensor,
    T_tau: torch.Tensor,
) -> torch.Tensor:
    """
    Trilinear motif score contraction.
    
    Written as an explicit element-wise chain. This is mathematically optimal for 
    memory and speed because it allows PyTorch Inductor (torch.compile) to intercept 
    and fuse the entire operation into a single Triton kernel without intermediate allocations.
    Opaque ops like einsum block this fusion.
    """
    return (Q3_c * (K3_u * K3_v + T_tau)).sum(dim=(-1, -2))


def fused_motif_dot_baseline(
    Q3_c: torch.Tensor,
    K3_u: torch.Tensor,
    K3_v: torch.Tensor,
    T_tau: torch.Tensor,
) -> torch.Tensor:
    """Baseline alias kept for API compatibility with older imports/tests."""
    return fused_motif_dot(Q3_c=Q3_c, K3_u=K3_u, K3_v=K3_v, T_tau=T_tau)

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

def compute_degree_scaler(degrees: torch.Tensor, avg_degree: torch.Tensor | float, mode="pna"):
    if mode == "pna":
        if not torch.is_tensor(avg_degree):
            avg_degree = torch.tensor(float(avg_degree), device=degrees.device, dtype=degrees.dtype)
        else:
            avg_degree = avg_degree.to(device=degrees.device, dtype=degrees.dtype)
        avg_degree = avg_degree.clamp_min(1e-6)
        return torch.log(degrees + 1.0) / torch.log(avg_degree + 1.0)
    return torch.ones_like(degrees)
