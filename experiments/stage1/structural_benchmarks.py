import torch
import networkx as nx
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Batch
from get.models.get_model import GETModel
from experiments.shared.common import GETTrainer, set_seed, save_results
from experiments.stage1.wedge_discrimination import PairwiseGET, FullGET, GINBaseline

def generate_srg_dataset(num_samples=500):
    """
    Generates a dataset of Strongly Regular Graphs (SRG) that are 1-WL indistinguishable.
    Graph 0: Shrikhande Graph (16 nodes, degree 6)
    Graph 1: 4x4 Rook's Graph (16 nodes, degree 6)
    """
    # Nodes are (x, y) for x, y in {0, 1, 2, 3}
    nodes = [(x, y) for x in range(4) for y in range(4)]
    node_to_idx = {p: i for i, p in enumerate(nodes)}

    # Shrikhande Generators
    gens0 = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,-1)]
    edges0_base = []
    for x, y in nodes:
        for dx, dy in gens0:
            nx, ny = (x + dx) % 4, (y + dy) % 4
            u, v = node_to_idx[(x, y)], node_to_idx[(nx, ny)]
            if u < v: edges0_base.append((u, v))

    # Rook's Generators (All nodes in same row or column)
    edges1_base = []
    for x, y in nodes:
        for i in range(1, 4):
            # Row
            nx, ny = (x + i) % 4, y
            u, v = node_to_idx[(x, y)], node_to_idx[(nx, ny)]
            if u < v: edges1_base.append((u, v))
            # Column
            nx, ny = x, (y + i) % 4
            u, v = node_to_idx[(x, y)], node_to_idx[(nx, ny)]
            if u < v: edges1_base.append((u, v))
    
    # Remove duplicates from Rook's (since i=1 and i=3 might hit same edge)
    edges1_base = list(set(tuple(sorted(e)) for e in edges1_base))

    dataset = []
    for _ in range(num_samples):
        # Shrikhande (Label 0)
        perm = np.random.permutation(16)
        mapping = {i: int(perm[i]) for i in range(16)}
        edges0 = [(mapping[u], mapping[v]) for u, v in edges0_base]
        dataset.append({"x": torch.ones(16, 1), "edges": edges0, "y": 0, "graph_id": 0})
        
        # Rook (Label 1)
        perm = np.random.permutation(16)
        mapping = {i: int(perm[i]) for i in range(16)}
        edges1 = [(mapping[u], mapping[v]) for u, v in edges1_base]
        dataset.append({"x": torch.ones(16, 1), "edges": edges1, "y": 1, "graph_id": 1})
        
    return dataset

def generate_cycle_parity_dataset(num_samples=500, n=20):
    """
    Generates graphs that are either purely Bipartite (Even cycles only) 
    or Non-Bipartite (Contains odd cycles), while matching degrees.
    """
    dataset = []
    for _ in range(num_samples):
        # Label 0: Bipartite 3-regular graph
        g0 = nx.random_regular_graph(3, n)
        while not nx.is_bipartite(g0):
            g0 = nx.random_regular_graph(3, n)
        dataset.append({"x": torch.ones(n, 1), "edges": list(g0.edges()), "y": 0, "graph_id": 0})
        
        # Label 1: Non-Bipartite 3-regular graph
        g1 = nx.random_regular_graph(3, n)
        while nx.is_bipartite(g1):
            g1 = nx.random_regular_graph(3, n)
        dataset.append({"x": torch.ones(n, 1), "edges": list(g1.edges()), "y": 1, "graph_id": 1})
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="srg", choices=["srg", "cycle"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--pe_k", type=int, default=8)
    parser.add_argument("--no_pe", action="store_true", help="Test without positional encodings (Strict 1-WL check)")
    args = parser.parse_args()

    if args.no_pe:
        args.pe_k = 0
        print("!!! WARNING: Running in NO-PE mode. This is a strict 1-WL expressivity test. !!!")

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Generating {args.task.upper()} Dataset ---")
    if args.task == "srg":
        full_ds = generate_srg_dataset(100)
    else:
        full_ds = generate_cycle_parity_dataset(100, n=20)

    # Split
    np.random.shuffle(full_ds)
    train_ds = full_ds[:400]
    val_ds = full_ds[400:450]
    test_ds = full_ds[450:]
    
    # Symmetry-breaking noise for strict 1-WL tests
    for d in train_ds + val_ds + test_ds:
        d["x"] = torch.ones_like(d["x"]) + torch.randn_like(d["x"]) * 0.01

    # Models (Using the same stable hyperparams from Wedge Discrimination)
    # We use in_dim=1 (constant ones)
    in_dim = 1
    pe_k = 0 if args.no_pe else args.pe_k
    models = [
        ("PairwiseGET", PairwiseGET(in_dim, 128, 1, num_steps=16, pe_k=pe_k, lambda_2=10.0, update_damping=0.05, grad_clip_norm=0.1), 5e-4),
        ("FullGET", FullGET(in_dim, 128, 1, num_steps=16, pe_k=pe_k, lambda_3=10.0, beta_3=5.0, update_damping=0.05, grad_clip_norm=0.1), 5e-4),
        ("GIN", GINBaseline(in_dim, 128, 1, num_layers=4), 5e-4),
    ]

    results = {}
    for name, model, lr in models:
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(
            model,
            task_type='binary',
            device=device,
            lr=lr,
            max_grad_val=0.05,
            weight_decay=1e-4
        )
        res = trainer.run(train_ds, val_ds, test_ds, args.epochs, args.batch_size)
        results[name] = res

    save_results(f"exp1_{args.task}_bench", results)

if __name__ == "__main__":
    main()
