import torch
import pytest
from get.fused_ops import fused_motif_dot

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for this smoke test", allow_module_level=True)

B, M, R, D = 2, 10, 2, 32
Q3 = torch.randn(B, M, R, D, device='cuda', dtype=torch.float32, requires_grad=True)
K3u = torch.randn(B, M, R, D, device='cuda', dtype=torch.float32, requires_grad=True)
K3v = torch.randn(B, M, R, D, device='cuda', dtype=torch.float32, requires_grad=True)
T = torch.randn(M, R, D, device='cuda', dtype=torch.float32, requires_grad=True)

out = fused_motif_dot(Q3, K3u, K3v, T)
print(f"Output requires grad: {out.requires_grad}")
if out.requires_grad:
    loss = out.sum()
    loss.backward()
    print("Backward pass succeeded.")
else:
    print("BACKWARD PASS FAILED: No gradients.")
