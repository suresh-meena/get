import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
x_batch = torch.zeros(2)
x_batch[:] = x
x_batch = x_batch.requires_grad_(True)

e = x_batch.sum()
grad_x_batch = torch.autograd.grad(e, x_batch, create_graph=True)[0]
x_next = x - 0.1 * grad_x_batch

loss = x_next.sum()
try:
    loss.backward()
    print(f"Gradient of x: {x.grad}")
except Exception as err:
    print(f"ERROR: {err}")
