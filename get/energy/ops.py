import torch
import torch.nn.functional as F

from torch_scatter import scatter, scatter_logsumexp

def segment_reduce_1d(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, reduce="sum", dim: int = -1):
    """
    Segmented reduction using torch_scatter.
    """
    if dim < 0:
        dim = src.dim() + dim
        
    idx_shape = [1] * src.dim()
    idx_shape[dim] = len(segment_ids)
    idx = segment_ids.view(*idx_shape).expand_as(src)

    out = scatter(src, idx, dim=dim, dim_size=num_segments, reduce=reduce)
    counts = torch.bincount(segment_ids, minlength=num_segments)
    return out, counts

def segment_logsumexp(x: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int = -1):
    """
    Computes logsumexp over segments using torch_scatter.
    """
    if dim < 0:
        dim = x.dim() + dim

    idx_shape = [1] * x.dim()
    idx_shape[dim] = len(segment_ids)
    idx = segment_ids.view(*idx_shape).expand_as(x)

    return scatter_logsumexp(x, idx, dim=dim, dim_size=num_segments)

def scatter_add_nd(grad_buffer: torch.Tensor, indices: torch.Tensor, src: torch.Tensor, dim: int):
    """
    Memory-efficient N-dimensional scatter-add using index_add directly.
    """
    if src.dtype != grad_buffer.dtype:
        src = src.to(dtype=grad_buffer.dtype)
        
    return grad_buffer.index_add_(dim, indices, src)

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
