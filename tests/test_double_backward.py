import torch
import pytest
from get import FullGET
from get.data import collate_get_batch

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for this smoke test", allow_module_level=True)

B, N, D = 1, 10, 32
X = torch.randn(N, D, requires_grad=True, device='cuda')
edges = [(0, 1), (1, 2), (0, 2)]
batch = collate_get_batch([{'x': X, 'edges': edges}])
batch = batch.to('cuda')

model = FullGET(in_dim=D, d=D, num_classes=1, num_steps=1).cuda()
out, energy_trace = model(batch, is_training=True)

loss = out.sum()
print("Computing loss backward...")
try:
    loss.backward()
    print("SUCCESS: Loss backward worked.")
except Exception as e:
    print(f"FAILED: {e}")
