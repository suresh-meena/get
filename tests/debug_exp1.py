import torch
import torch.nn as nn
import torch.optim as optim
from get import FullGET, PairwiseGET
from get.data import collate_get_batch
import networkx as nx
import random
from tqdm.auto import tqdm

# Generate two graphs with same degrees but different triangles
def generate_matched_pair(n=20):
    while True:
        G1 = nx.random_regular_graph(3, n)
        G2 = G1.copy()
        # Double edge swap to change structure but keep degrees
        # (u,v), (x,y) -> (u,x), (v,y)
        edges = list(G2.edges())
        e1, e2 = random.sample(edges, 2)
        u, v = e1
        x, y = e2
        if len({u,v,x,y}) == 4 and not G2.has_edge(u,x) and not G2.has_edge(v,y):
            G2.remove_edge(u, v)
            G2.remove_edge(x, y)
            G2.add_edge(u, x)
            G2.add_edge(v, y)
            
            t1 = sum(nx.triangles(G1).values()) // 3
            t2 = sum(nx.triangles(G2).values()) // 3
            if t1 != t2:
                return G1, G2, t1, t2

G1, G2, t1, t2 = generate_matched_pair()
print(f"Graph 1 Triangles: {t1}, Graph 2 Triangles: {t2}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FullGET(in_dim=1, d=32, num_classes=1, num_steps=8).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-2)

batch1 = collate_get_batch([{'x': torch.ones(20, 1), 'edges': list(G1.edges()), 'y': torch.tensor([1.0])}]).to(device)
batch2 = collate_get_batch([{'x': torch.ones(20, 1), 'edges': list(G2.edges()), 'y': torch.tensor([0.0])}]).to(device)

for i in range(100):
    model.train()
    optimizer.zero_grad()
    
    out1, e_trace1 = model(batch1, task_level='graph')
    out2, e_trace2 = model(batch2, task_level='graph')
    
    loss = nn.BCEWithLogitsLoss()(out1, batch1.y.view(-1, 1)) + \
           nn.BCEWithLogitsLoss()(out2, batch2.y.view(-1, 1))
           
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i}: Loss {loss.item():.4f}, Out1 {out1.item():.4f}, Out2 {out2.item():.4f}")
        print(f"  Energy Trace 1: {[e.item() for e in e_trace1[:3]]}")
        # Check grad norm of W_Q3
        print(f"  W_Q3 Grad Norm: {model.get_layer.W_Q3.grad.norm().item():.6f}")
