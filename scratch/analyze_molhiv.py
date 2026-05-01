import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import MoleculeNet
import numpy as np

def count_wedges(edge_index, num_nodes):
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True
    
    # Degrees
    deg = torch.bincount(edge_index[0], minlength=num_nodes)
    
    # Wedges at node i: deg[i] * (deg[i] - 1) / 2
    wedges = (deg * (deg - 1)) // 2
    return wedges.sum().item()

def main():
    dataset = MoleculeNet(root='data', name='HIV')
    
    wedge_counts = []
    for data in dataset[:1000]:
        wedge_counts.append(count_wedges(data.edge_index, data.num_nodes))
    
    print(f"MolHIV (sample 1000) - Avg Wedges: {np.mean(wedge_counts):.1f}, Max Wedges: {np.max(wedge_counts)}")

if __name__ == "__main__":
    main()
