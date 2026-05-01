import torch
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np

def count_wedges(edge_index, num_nodes):
    # Degrees
    deg = torch.bincount(edge_index[0], minlength=num_nodes)
    # Wedges at node i: deg[i] * (deg[i] - 1) / 2
    wedges = (deg * (deg - 1)) // 2
    return wedges.sum().item()

def main():
    dataset = PygGraphPropPredDataset(name='ogbg-molhiv', root='data/OGB')
    
    wedge_counts = []
    for i in range(min(1000, len(dataset))):
        data = dataset[i]
        wedge_counts.append(count_wedges(data.edge_index, data.num_nodes))
    
    print(f"MolHIV (sample 1000) - Avg Wedges: {np.mean(wedge_counts):.1f}, Max Wedges: {np.max(wedge_counts)}")

if __name__ == "__main__":
    main()
