import torch
from experiments.stage1.wedge_discrimination import generate_matched_dataset

dataset = generate_matched_dataset(num_pairs=5, rwse_k=8)
for i, data in enumerate(dataset):
    print(f"\nGraph {i} (Label {data['y']}):")
    print(f"Nodes: {data['x'].shape[0]}")
    if 'rwse' in data:
        print(f"RWSE Mean: {data['rwse'].mean(dim=0)}")
        print(f"RWSE Std: {data['rwse'].std(dim=0)}")
        print(f"RWSE Sample Node 0: {data['rwse'][0]}")
    else:
        print("NO RWSE FOUND!")
