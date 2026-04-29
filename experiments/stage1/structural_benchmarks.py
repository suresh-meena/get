import torch
import networkx as nx
import numpy as np
import argparse
from experiments.shared.common import GETTrainer, set_seed, save_results
from get import PairwiseGET, FullGET, GINBaseline
from sklearn.model_selection import train_test_split

def generate_srg_dataset(num_pairs=500, seed=0):
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
            if u < v:
                edges0_base.append((u, v))

    # Rook's Generators (All nodes in same row or column)
    edges1_base = []
    for x, y in nodes:
        for i in range(1, 4):
            # Row
            nx, ny = (x + i) % 4, y
            u, v = node_to_idx[(x, y)], node_to_idx[(nx, ny)]
            if u < v:
                edges1_base.append((u, v))
            # Column
            nx, ny = x, (y + i) % 4
            u, v = node_to_idx[(x, y)], node_to_idx[(nx, ny)]
            if u < v:
                edges1_base.append((u, v))
    
    # Remove duplicates from Rook's (since i=1 and i=3 might hit same edge)
    edges1_base = list(set(tuple(sorted(e)) for e in edges1_base))

    rng = np.random.default_rng(seed)
    dataset = []
    for pair_id in range(num_pairs):
        # Shrikhande (Label 0)
        perm = rng.permutation(16)
        mapping = {i: int(perm[i]) for i in range(16)}
        edges0 = [(mapping[u], mapping[v]) for u, v in edges0_base]
        dataset.append({"x": torch.ones(16, 1), "edges": edges0, "y": torch.tensor([0.0]), "graph_id": f"shrikhande_{pair_id}"})
        
        # Rook (Label 1)
        perm = rng.permutation(16)
        mapping = {i: int(perm[i]) for i in range(16)}
        edges1 = [(mapping[u], mapping[v]) for u, v in edges1_base]
        dataset.append({"x": torch.ones(16, 1), "edges": edges1, "y": torch.tensor([1.0]), "graph_id": f"rook_{pair_id}"})
        
    return dataset

def generate_cycle_parity_dataset(num_pairs=500, n=20, seed=0):
    """
    Generates graphs that are either purely Bipartite (Even cycles only) 
    or Non-Bipartite (Contains odd cycles), while matching degrees.
    """
    rng = np.random.default_rng(seed)
    dataset = []
    for pair_id in range(num_pairs):
        # Label 0: Bipartite 3-regular graph
        g0 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        while not nx.is_bipartite(g0):
            g0 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        dataset.append({"x": torch.ones(n, 1), "edges": list(g0.edges()), "y": torch.tensor([0.0]), "graph_id": f"bipartite_{pair_id}"})
        
        # Label 1: Non-Bipartite 3-regular graph
        g1 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        while nx.is_bipartite(g1):
            g1 = nx.random_regular_graph(3, n, seed=int(rng.integers(0, 2**32 - 1)))
        dataset.append({"x": torch.ones(n, 1), "edges": list(g1.edges()), "y": torch.tensor([1.0]), "graph_id": f"nonbipartite_{pair_id}"})
    return dataset


def stratified_split(dataset, seed):
    labels = np.array([int(float(item["y"].view(-1)[0].item()) >= 0.5) for item in dataset])
    idx = np.arange(len(dataset))
    train_idx, tmp_idx, y_train, y_tmp = train_test_split(
        idx,
        labels,
        train_size=0.70,
        random_state=seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        tmp_idx,
        train_size=0.50,
        random_state=seed + 1,
        stratify=y_tmp,
    )
    return [dataset[i] for i in train_idx], [dataset[i] for i in val_idx], [dataset[i] for i in test_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="srg", choices=["srg", "cycle"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pe_k", type=int, default=0)
    parser.add_argument("--inject_feature_noise", type=float, default=0.0, help="Std of Gaussian noise added to node features.")
    args = parser.parse_args()

    if args.pe_k > 0:
        print(f"Using positional encodings with pe_k={args.pe_k}.")
    else:
        print("Running without positional encodings (default Stage-1 setting).")

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Generating {args.task.upper()} Dataset ---")
    if args.task == "srg":
        full_ds = generate_srg_dataset(args.num_pairs, seed=args.seed)
    else:
        full_ds = generate_cycle_parity_dataset(args.num_pairs, n=20, seed=args.seed)

    train_ds, val_ds, test_ds = stratified_split(full_ds, seed=args.seed)
    print(f"Split sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    if args.inject_feature_noise > 0.0:
        print(f"Injecting node feature noise std={args.inject_feature_noise}.")
        for d in train_ds + val_ds + test_ds:
            d["x"] = torch.ones_like(d["x"]) + torch.randn_like(d["x"]) * float(args.inject_feature_noise)

    # Models (Using the same stable hyperparams from Wedge Discrimination)
    # We use in_dim=1 (constant ones)
    in_dim = 1
    pe_k = args.pe_k
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

    metadata = {
        "interpretation": "Supervised structural diagnostic. SRG and cycle parity are evidence of realized discrimination only; they are not formal proofs of WL expressivity.",
        "task": args.task,
        "num_pairs": args.num_pairs,
        "seed": args.seed,
        "pe_k": args.pe_k,
        "inject_feature_noise": args.inject_feature_noise,
    }
    save_results(f"exp1_{args.task}_bench", results, metadata=metadata)

if __name__ == "__main__":
    main()
