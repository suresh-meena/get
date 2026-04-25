import torch
import pytest
from get.energy.ops import (
    fused_motif_dot, 
    segment_logsumexp, 
    segment_reduce_1d,
    get_degree_from_incidence,
    compute_degree_scaler
)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fused_motif_dot_correctness():
    B, M, R, D = 2, 10, 2, 32
    Q3 = torch.randn(B, M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)
    K3u = torch.randn(B, M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)
    K3v = torch.randn(B, M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)
    T = torch.randn(M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)

    # Forward
    out_ref = (Q3 * (K3u * K3v + T)).sum(dim=(-1, -2))
    out_fused = fused_motif_dot(Q3, K3u, K3v, T)
    assert (out_ref - out_fused).abs().max() < 1e-6

    # Backward
    loss_ref = out_ref.sum()
    grad_Q3_ref = torch.autograd.grad(loss_ref, Q3, retain_graph=True)[0]
    loss_fused = out_fused.sum()
    grad_Q3_fused = torch.autograd.grad(loss_fused, Q3, retain_graph=True)[0]
    assert (grad_Q3_ref - grad_Q3_fused).abs().max() < 1e-6

def test_segment_ops():
    src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ids = torch.tensor([0, 0, 1, 1, 2])
    
    # segment_reduce
    out_sum, counts = segment_reduce_1d(src, ids, 3, reduce="sum")
    assert torch.allclose(out_sum, torch.tensor([3.0, 7.0, 5.0]))
    assert torch.allclose(counts, torch.tensor([2, 2, 1]))
    
    # segment_logsumexp
    lse = segment_logsumexp(src, ids, 3)
    ref = torch.tensor([
        torch.log(torch.exp(src[0]) + torch.exp(src[1])),
        torch.log(torch.exp(src[2]) + torch.exp(src[3])),
        torch.log(torch.exp(src[4]))
    ])
    assert torch.allclose(lse, ref)

def test_degree_utilities():
    c2 = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.long)
    num_nodes = 4
    degrees = get_degree_from_incidence(c2, num_nodes)
    assert torch.allclose(degrees, torch.tensor([2.0, 1.0, 3.0, 0.0]))
    
    scaler = compute_degree_scaler(degrees, avg_degree=2.0)
    # log(d+1)/log(3)
    ref = torch.log(degrees + 1.0) / torch.log(torch.tensor(3.0))
    assert torch.allclose(scaler, ref)
