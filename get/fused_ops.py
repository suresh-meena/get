import torch

try:
    from torch_scatter import scatter as _torch_scatter  # type: ignore
except (ImportError, OSError):
    _torch_scatter = None


def _segment_reduce_with_torch_segment(src, segment_ids, num_segments, reduce):
    if src.dim() == 2:
        B, L = src.shape
        counts = torch.bincount(segment_ids, minlength=num_segments)
        if L == 0:
            if reduce == "max":
                return torch.full((B, num_segments), float("-inf"), dtype=src.dtype, device=src.device), counts
            return torch.zeros((B, num_segments), dtype=src.dtype, device=src.device), counts
        return _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce)

    counts = torch.bincount(segment_ids, minlength=num_segments)
    if src.numel() == 0:
        if reduce == "max":
            return torch.full((num_segments,), float("-inf"), dtype=src.dtype, device=src.device), counts
        return torch.zeros((num_segments,), dtype=src.dtype, device=src.device), counts
    perm = torch.argsort(segment_ids)
    src_sorted = src[perm]
    out = torch.segment_reduce(src_sorted, reduce=reduce, lengths=counts)
    return out, counts


def _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce):
    counts = torch.bincount(segment_ids, minlength=num_segments)
    if src.dim() == 2:
        B, L = src.shape
        expanded_ids = segment_ids.unsqueeze(0).expand(B, L)
        if reduce == "sum":
            out = torch.zeros((B, num_segments), dtype=src.dtype, device=src.device)
            out.scatter_add_(1, expanded_ids, src)
        elif reduce == "max":
            out = torch.full((B, num_segments), float("-inf"), dtype=src.dtype, device=src.device)
            if src.numel() > 0:
                out.scatter_reduce_(1, expanded_ids, src, reduce="amax", include_self=False)
        return out, counts

    if reduce == "sum":
        out = torch.zeros((num_segments,), dtype=src.dtype, device=src.device)
        out.scatter_add_(0, segment_ids, src)
    elif reduce == "max":
        out = torch.full((num_segments,), float("-inf"), dtype=src.dtype, device=src.device)
        if src.numel() > 0:
            out.scatter_reduce_(0, segment_ids, src, reduce="amax", include_self=False)
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")
    return out, counts


def segment_reduce_1d(src, segment_ids, num_segments, reduce="sum"):
    """
    Segment reduction with best available backend.
    """
    if _torch_scatter is not None:
        dim = 1 if src.dim() == 2 else 0
        idx = segment_ids
        if src.dim() == 2:
            idx = segment_ids.unsqueeze(0).expand(src.size(0), -1)
        
        if reduce == "sum":
            out = _torch_scatter(src, idx, dim=dim, dim_size=num_segments, reduce="sum")
            counts = torch.bincount(segment_ids, minlength=num_segments)
            return out, counts
        if reduce == "max":
            out = _torch_scatter(src, idx, dim=dim, dim_size=num_segments, reduce="max")
            counts = torch.bincount(segment_ids, minlength=num_segments)
            out = torch.where(counts > 0, out, torch.full_like(out, float("-inf")))
            return out, counts
        raise ValueError(f"Unsupported reduce: {reduce}")

    if hasattr(torch, "segment_reduce") and not src.requires_grad and src.dim() == 1:
        return _segment_reduce_with_torch_segment(src, segment_ids, num_segments, reduce)
    return _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce)


def fused_motif_dot(Q3_c, K3_u, K3_v, T_tau):
    """
    Performs the trilinear motif score contraction.
    Factorizing as Q * (K_u * K_v + T) allows torch.compile to fuse this into a 
    single CUDA kernel, avoiding the allocation of intermediate triplet tensors.
    """
    # Use in-place capable operations to hint fusion to the compiler
    res = K3_u * K3_v
    res.add_(T_tau)
    res.mul_(Q3_c)
    return res.sum(dim=(-1, -2))
