"""Low-level sparse operations for energy computation.

Provides segment_reduce, segment_logsumexp, segment_softmax,
scatter_add, and fused_motif_dot.
"""
import torch

try:
    from torch_scatter import scatter as _torch_scatter
except (ImportError, OSError):
    _torch_scatter = None

try:
    from torch_scatter import scatter as pyg_scatter
    from torch_scatter import scatter_logsumexp as pyg_scatter_lse
except (ImportError, OSError):
    pyg_scatter = None
    pyg_scatter_lse = None


def _is_compiling():
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is None:
        return False
    return bool(getattr(dynamo, "is_compiling", lambda: False)())


# ---------------------------------------------------------------------------
# Segment reduce
# ---------------------------------------------------------------------------

def _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce):
    """Robust segmented reduction using torch.scatter_reduce."""
    counts = torch.bincount(segment_ids, minlength=num_segments)

    dim = -1
    if src.dim() > 2 and src.size(-1) != len(segment_ids):
        dim = -3

    if dim < 0:
        dim = src.dim() + dim

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

    return out, counts


def segment_reduce_1d(src, segment_ids, num_segments, reduce="sum"):
    """Segmented reduction for GNNs.

    src: [..., L, ...] tensor of values.
    segment_ids: [L] tensor of target segment indices.
    """
    if _torch_scatter is not None and not _is_compiling():
        try:
            dim = -1
            if src.dim() > 2 and src.size(-1) != len(segment_ids):
                dim = -3
            out = _torch_scatter(src, segment_ids, dim=dim, dim_size=num_segments, reduce=reduce)
            counts = torch.bincount(segment_ids, minlength=num_segments)
            return out, counts
        except Exception:
            pass

    return _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce)


# ---------------------------------------------------------------------------
# Segment logsumexp / softmax
# ---------------------------------------------------------------------------

def segment_logsumexp(x, segment_ids, num_segments):
    if pyg_scatter_lse is not None and not _is_compiling():
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
    return lse


def segment_softmax(x, segment_ids, num_segments):
    lse = segment_logsumexp(x, segment_ids, num_segments)
    return torch.exp(x - lse[..., segment_ids])


# ---------------------------------------------------------------------------
# Scatter add
# ---------------------------------------------------------------------------

def scatter_add_nd(grad_buffer, indices, src, dim):
    """Memory-efficient N-dimensional scatter-add."""
    if pyg_scatter is not None and not _is_compiling():
        try:
            return pyg_scatter(src, indices, dim=dim, dim_size=grad_buffer.size(dim), reduce="sum")
        except Exception:
            pass

    perm = list(range(src.dim()))
    perm[0], perm[dim] = perm[dim], perm[0]

    src_p = src.permute(perm)
    if src_p.dtype != grad_buffer.dtype:
        src_p = src_p.to(dtype=grad_buffer.dtype)
    return grad_buffer.permute(perm).index_add_(0, indices, src_p).permute(perm)


# ---------------------------------------------------------------------------
# Fused motif dot
# ---------------------------------------------------------------------------

def fused_motif_dot(Q3_c, K3_u, K3_v, T_tau):
    """Trilinear motif score contraction: (Q * (K_u * K_v + T)).sum(dim=(-1, -2))"""
    return (Q3_c * (K3_u * K3_v + T_tau)).sum(dim=(-1, -2))


# ---------------------------------------------------------------------------
# Parameter helpers
# ---------------------------------------------------------------------------

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

def compute_degree_scaler(degrees, avg_degree, mode="pna"):
    """
    Returns PNA-style scalers: [num_nodes].
    mode='pna' uses log(d+1)/log(avg+1).
    """
    if mode == "pna":
        return torch.log(degrees + 1.0) / torch.log(torch.tensor(avg_degree + 1.0, device=degrees.device))
    return torch.ones_like(degrees)


def get_degree_from_incidence(c_2, num_nodes):
    """Computes degrees from directed incidence c_2."""
    counts = torch.bincount(c_2, minlength=num_nodes)
    return counts.to(dtype=torch.float32)
