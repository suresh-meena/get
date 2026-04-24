import torch

# Case 1: Broken double backward
x1 = torch.tensor([1.0, 2.0], requires_grad=True)
x_batch1 = torch.zeros(2)
x_batch1[:] = x1
x_batch1 = x_batch1.requires_grad_(True)
e1 = (x_batch1 * x_batch1).sum()
grad1 = torch.autograd.grad(e1, x_batch1, create_graph=True)[0]
loss1 = grad1.sum()
loss1.backward()

# Case 2: Correct double backward
x2 = torch.tensor([1.0, 2.0], requires_grad=True)
x_batch2 = torch.zeros(2)
x_batch2 = x_batch2 + x2
e2 = (x_batch2 * x_batch2).sum()
grad2 = torch.autograd.grad(e2, x_batch2, create_graph=True)[0]
loss2 = grad2.sum()
loss2.backward()

print(f"Broken grad: {x1.grad}")
print(f"Correct grad: {x2.grad}")
