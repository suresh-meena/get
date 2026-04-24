import torch
from get.fused_ops import fused_motif_dot

def test_fused_motif_dot_correctness():
    print("Verifying fused_motif_dot correctness...")
    B, M, R, D = 2, 10, 2, 32
    Q3 = torch.randn(B, M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)
    K3u = torch.randn(B, M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)
    K3v = torch.randn(B, M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)
    T = torch.randn(M, R, D, device="cuda", dtype=torch.float32, requires_grad=True)

    # Test forward matches reference
    out_ref = (Q3 * (K3u * K3v + T)).sum(dim=(-1, -2))
    out_fused = fused_motif_dot(Q3, K3u, K3v, T)
    
    diff_fw = (out_ref - out_fused).abs().max().item()
    print(f"  Forward diff: {diff_fw:.2e}")
    assert diff_fw < 1e-6

    # Test backward
    loss_ref = out_ref.sum()
    grad_Q3_ref = torch.autograd.grad(loss_ref, Q3, create_graph=True)[0]

    loss_fused = out_fused.sum()
    grad_Q3_fused = torch.autograd.grad(loss_fused, Q3, create_graph=True)[0]

    diff_bw = (grad_Q3_ref - grad_Q3_fused).abs().max().item()
    print(f"  Backward diff: {diff_bw:.2e}")
    assert diff_bw < 1e-6

    # Test double backward
    loss2_ref = grad_Q3_ref.sum()
    grad2_K3u_ref = torch.autograd.grad(loss2_ref, K3u)[0]

    loss2_fused = grad_Q3_fused.sum()
    grad2_K3u_fused = torch.autograd.grad(loss2_fused, K3u)[0]

    diff_dbw = (grad2_K3u_ref - grad2_K3u_fused).abs().max().item()
    print(f"  Double Backward diff: {diff_dbw:.2e}")
    assert diff_dbw < 1e-6
    print("fused_motif_dot verification passed!")

if __name__ == "__main__":
    if torch.cuda.is_available():
        test_fused_motif_dot_correctness()
    else:
        print("CUDA not available, skipping fused_motif_dot verification.")
