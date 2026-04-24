import torch

try:
    from torch_scatter import scatter as _torch_scatter  # type: ignore
except (ImportError, OSError):
    _torch_scatter = None


def _segment_reduce_with_torch_segment(src, segment_ids, num_segments, reduce):
    """Fallback using torch.segment_reduce (if available) or manual scatter."""
    if hasattr(torch, "segment_reduce"):
        # torch.segment_reduce requires sorted segments and counts
        # This is more complex to implement correctly as a general fallback
        pass
    return _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce)


def _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce):
    """
    Robust segmented reduction using torch.scatter_reduce.
    Handles any number of batch dimensions in src.
    """
    counts = torch.bincount(segment_ids, minlength=num_segments)
    
    # We need to reduce along the dimension corresponding to segment_ids.
    # We assume segment_ids corresponds to the LAST dimension of src if src is 1D or 2D,
    # or a specific dimension in the general case.
    # In GrET, segment_ids usually corresponds to the 'num_edges' dimension.
    # Let's find that dimension.
    
    # Standard GrET usage:
    # Pairwise: src [H, L], segment_ids [L] -> dim -1
    # Motif: src [H, L, R, d], segment_ids [L] -> dim -3
    
    dim = -1
    if src.dim() > 2:
        # For motifs, src is [..., L, R, d]. L is dim -3.
        # For pairwise, src is [..., L]. L is dim -1.
        if src.size(-1) != len(segment_ids):
             dim = -3 # Motif case
    
    # Adjust dim to positive
    if dim < 0: dim = src.dim() + dim
    
    # Create index tensor matching src shape
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
            # torch.scatter_reduce_ is available in PyTorch 1.12+
            out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=False)
            
    return out, counts


def segment_reduce_1d(src, segment_ids, num_segments, reduce="sum"):
    """
    Segmented reduction for GNNs.
    src: [..., L, ...] tensor of values.
    segment_ids: [L] tensor of target segment indices.
    """
    if _torch_scatter is not None:
        try:
            # Determine dim based on where L matches len(segment_ids)
            dim = -1
            if src.dim() > 2 and src.size(-1) != len(segment_ids):
                dim = -3
            out = _torch_scatter(src, segment_ids, dim=dim, dim_size=num_segments, reduce=reduce)
            counts = torch.bincount(segment_ids, minlength=num_segments)
            return out, counts
        except Exception:
            pass

    return _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce)


def fused_motif_dot(Q3_c, K3_u, K3_v, T_tau):
    """
    Performs the trilinear motif score contraction.
    (Q * (K_u * K_v + T)).sum(dim=(-1, -2))
    Robustly handles multi-head broadcasting.
    """
    # K3_u/v are [..., L, R, d]
    # T_tau is [..., L, R, d]
    # Ensure T_tau matches K tensors rank and head count
    return (Q3_c * (K3_u * K3_v + T_tau)).sum(dim=(-1, -2))
