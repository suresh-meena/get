import torch
from get import FullGET
from get.data import collate_get_batch
import networkx as nx

# Generate a tiny graph
G = nx.erdos_renyi_graph(5, 0.5)
batch = collate_get_batch([{'x': torch.randn(5, 4), 'edges': list(G.edges()), 'y': torch.tensor([1.0])}])

model = FullGET(in_dim=4, d=4, num_classes=1, num_steps=2)
out, _ = model(batch, task_level='graph')

loss = out.sum()
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
    else:
        print(f"{name}: NO GRADIENT")
