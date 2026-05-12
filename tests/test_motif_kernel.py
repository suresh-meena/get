from __future__ import annotations

import torch

from get.energy.ops import fused_motif_dot


def test_fused_motif_dot_matches_einsum_reference():
    """
    Verify that the fused motif contraction matches a manual einsum-based reference.
    """
    torch.manual_seed(11)
    # [E, H, R, D]
    q = torch.randn(64, 4, 2, 16, dtype=torch.float32)
    ku = torch.randn(64, 4, 2, 16, dtype=torch.float32)
    kv = torch.randn(64, 4, 2, 16, dtype=torch.float32)
    t = torch.randn(64, 4, 2, 16, dtype=torch.float32)

    out_fast = fused_motif_dot(q, ku, kv, t)
    
    # Reference: sum_{h,r,d} Q * (K1 * K2 + T)
    out_ref = (q * (ku * kv + t)).sum(dim=(-1, -2))
    
    assert torch.allclose(out_fast, out_ref, atol=1e-5, rtol=1e-4)


def test_fused_motif_dot_gradients():
    """
    Verify gradients of the fused motif contraction.
    """
    torch.manual_seed(13)
    q = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)
    ku = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)
    kv = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)
    t = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)

    out_fast = fused_motif_dot(q, ku, kv, t).sum()
    grads_fast = torch.autograd.grad(out_fast, (q, ku, kv, t))

    out_ref = (q * (ku * kv + t)).sum()
    grads_ref = torch.autograd.grad(out_ref, (q, ku, kv, t))

    for g_fast, g_ref in zip(grads_fast, grads_ref):
        assert torch.allclose(g_fast, g_ref, atol=1e-9, rtol=1e-7)
