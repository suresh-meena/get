import torch
import triton
import triton.language as tl
from triton import Config

try:
    from torch_scatter import scatter as _torch_scatter  # type: ignore
except Exception:
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
        if reduce == "sum":
            out = _torch_scatter(src, segment_ids, dim=dim, dim_size=num_segments, reduce="sum")
            counts = torch.bincount(segment_ids, minlength=num_segments)
            return out, counts
        if reduce == "max":
            out = _torch_scatter(src, segment_ids, dim=dim, dim_size=num_segments, reduce="max")
            counts = torch.bincount(segment_ids, minlength=num_segments)
            out = torch.where(counts > 0, out, torch.full_like(out, float("-inf")))
            return out, counts
        raise ValueError(f"Unsupported reduce: {reduce}")

    if hasattr(torch, "segment_reduce") and not src.requires_grad and src.dim() == 1:
        return _segment_reduce_with_torch_segment(src, segment_ids, num_segments, reduce)
    return _segment_reduce_with_scatter_reduce(src, segment_ids, num_segments, reduce)


@triton.autotune(
    configs=[
        Config({"BLOCK_SIZE_M": 16, "BLOCK_SIZE_R": 1, "BLOCK_SIZE_D": 32}, num_warps=2, num_stages=2),
        Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_R": 1, "BLOCK_SIZE_D": 32}, num_warps=4, num_stages=2),
        Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_R": 2, "BLOCK_SIZE_D": 32}, num_warps=4, num_stages=2),
        Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_R": 2, "BLOCK_SIZE_D": 64}, num_warps=4, num_stages=3),
        Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_R": 2, "BLOCK_SIZE_D": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "R", "D"],
)
@triton.jit
def _fused_motif_dot_kernel(
    Q_ptr, K_u_ptr, K_v_ptr, T_ptr, Out_ptr,
    stride_qb, stride_qm, stride_qr, stride_qd,
    stride_kub, stride_kum, stride_kur, stride_kud,
    stride_kvb, stride_kvm, stride_kvr, stride_kvd,
    stride_tm, stride_tr, stride_td,
    stride_ob, stride_om,
    B, M, R, D,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_R: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel for fused motif dot product.

    The CUDA path receives already-indexed tensors with shapes:
    Q_indexed: [B, M, R, D]
    K_u_indexed: [B, M, R, D]
    K_v_indexed: [B, M, R, D]
    T_indexed: [M, R, D] broadcast across batch.

    The contraction is:
    Out[b, m] = sum_{r, d} Q[b, m, r, d] * (K_u[b, m, r, d] * K_v[b, m, r, d] + T[m, r, d])
    """
    idx_m = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    idx_b = tl.program_id(1)

    mask_m = idx_m < M
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    for r in range(0, R, BLOCK_SIZE_R):
        idx_r = r + tl.arange(0, BLOCK_SIZE_R)
        mask_r = idx_r < R
        for d in range(0, D, BLOCK_SIZE_D):
            idx_d = d + tl.arange(0, BLOCK_SIZE_D)
            mask_d = idx_d < D

            mask = mask_m[:, None, None] & mask_r[None, :, None] & mask_d[None, None, :]

            off_q = (
                idx_b * stride_qb
                + idx_m[:, None, None] * stride_qm
                + idx_r[None, :, None] * stride_qr
                + idx_d[None, None, :] * stride_qd
            )
            off_k_u = (
                idx_b * stride_kub
                + idx_m[:, None, None] * stride_kum
                + idx_r[None, :, None] * stride_kur
                + idx_d[None, None, :] * stride_kud
            )
            off_k_v = (
                idx_b * stride_kvb
                + idx_m[:, None, None] * stride_kvm
                + idx_r[None, :, None] * stride_kvr
                + idx_d[None, None, :] * stride_kvd
            )
            off_t = (
                idx_m[:, None, None] * stride_tm
                + idx_r[None, :, None] * stride_tr
                + idx_d[None, None, :] * stride_td
            )

            q = tl.load(Q_ptr + off_q, mask=mask, other=0.0)
            ku = tl.load(K_u_ptr + off_k_u, mask=mask, other=0.0)
            kv = tl.load(K_v_ptr + off_k_v, mask=mask, other=0.0)
            t = tl.load(T_ptr + off_t, mask=mask, other=0.0)

            val = q * (ku * kv + t)
            acc += tl.sum(tl.sum(val, axis=2), axis=1)

    off_out = idx_b * stride_ob + idx_m * stride_om
    tl.store(Out_ptr + off_out, acc, mask=mask_m)


def _fused_motif_dot_reference(Q3_c, K3_u, K3_v, T_tau):
    if Q3_c.dim() == 4:
        return torch.einsum("bmrd,bmrd->bm", Q3_c, K3_u * K3_v + T_tau)
    return torch.einsum("mrd,mrd->m", Q3_c, K3_u * K3_v + T_tau)


def _can_use_fused_motif_triton(Q3_c, K3_u, K3_v, T_tau):
    if not (Q3_c.is_cuda and K3_u.is_cuda and K3_v.is_cuda and T_tau.is_cuda):
        return False
    if Q3_c.dim() != 4 or K3_u.dim() != 4 or K3_v.dim() != 4 or T_tau.dim() != 3:
        return False
    if Q3_c.shape != K3_u.shape or Q3_c.shape != K3_v.shape:
        return False
    if T_tau.shape != Q3_c.shape[1:]:
        return False
    if not (Q3_c.dtype == K3_u.dtype == K3_v.dtype == T_tau.dtype):
        return False
    return Q3_c.dtype in (torch.float16, torch.bfloat16, torch.float32)


def fused_motif_dot(Q3_c, K3_u, K3_v, T_tau):
    """
    Performs the trilinear motif score contraction.
    Uses Triton if on CUDA and dimensions are suitable, otherwise fallbacks to optimized einsum.
    """
    if _can_use_fused_motif_triton(Q3_c, K3_u, K3_v, T_tau):
        Q3_c = Q3_c.contiguous()
        K3_u = K3_u.contiguous()
        K3_v = K3_v.contiguous()
        T_tau = T_tau.contiguous()

        B, M, R, D = Q3_c.shape
        out = torch.empty((B, M), device=Q3_c.device, dtype=Q3_c.dtype)

        def grid(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B)

        # Autotune chooses BLOCK_SIZE_* and launch geometry at compile time.
        
        _fused_motif_dot_kernel[grid](
            Q3_c, K3_u, K3_v, T_tau, out,
            Q3_c.stride(0), Q3_c.stride(1), Q3_c.stride(2), Q3_c.stride(3),
            K3_u.stride(0), K3_u.stride(1), K3_u.stride(2), K3_u.stride(3),
            K3_v.stride(0), K3_v.stride(1), K3_v.stride(2), K3_v.stride(3),
            T_tau.stride(0), T_tau.stride(1), T_tau.stride(2),
            out.stride(0), out.stride(1),
            B, M, R, D,
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_R=BLOCK_SIZE_R, BLOCK_SIZE_D=BLOCK_SIZE_D,
        )
        return out

    return _fused_motif_dot_reference(Q3_c, K3_u, K3_v, T_tau)
