from __future__ import annotations

import torch

from get.energy.ops import fused_motif_dot, fused_motif_dot_baseline


def test_fused_motif_dot_matches_baseline_values():
    torch.manual_seed(11)
    q = torch.randn(64, 4, 2, 16, dtype=torch.float32)
    ku = torch.randn(64, 4, 2, 16, dtype=torch.float32)
    kv = torch.randn(64, 4, 2, 16, dtype=torch.float32)
    t = torch.randn(64, 4, 2, 16, dtype=torch.float32)

    out_fast = fused_motif_dot(q, ku, kv, t)
    out_ref = fused_motif_dot_baseline(q, ku, kv, t)
    assert torch.allclose(out_fast, out_ref, atol=1e-5, rtol=1e-4)


def test_fused_motif_dot_matches_baseline_gradients():
    torch.manual_seed(13)
    q = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)
    ku = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)
    kv = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)
    t = torch.randn(24, 3, 2, 8, dtype=torch.float64, requires_grad=True)

    out_fast = fused_motif_dot(q, ku, kv, t).sum()
    grads_fast = torch.autograd.grad(out_fast, (q, ku, kv, t), retain_graph=False, create_graph=False)

    out_ref = fused_motif_dot_baseline(q, ku, kv, t).sum()
    grads_ref = torch.autograd.grad(out_ref, (q, ku, kv, t), retain_graph=False, create_graph=False)

    for g_fast, g_ref in zip(grads_fast, grads_ref):
        assert torch.allclose(g_fast, g_ref, atol=1e-9, rtol=1e-7)
