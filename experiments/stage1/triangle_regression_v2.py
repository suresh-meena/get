import argparse
import random
import torch
import networkx as nx
from tqdm.auto import tqdm
import numpy as np

from get import FullGET, PairwiseGET, GINBaseline
from experiments.common import build_dataloader_kwargs, set_seed, GETTrainer, save_results, split_grouped_dataset

def generate_true_triangle_regression_dataset(num_graphs=2000, n_nodes=24, degree_range=(2, 6), seed=0):
    dataset = []
    pbar = tqdm(total=num_graphs, desc="Generating Graphs")
    rng = random.Random(seed)
    
    for i in range(num_graphs):
        degree = rng.randint(degree_range[0], degree_range[1])
        base_seed = rng.randint(0, 10**9)
        
        try:
            G = nx.random_regular_graph(degree, n_nodes, seed=base_seed)
        except nx.NetworkXError:
            G = nx.fast_gnp_random_graph(n_nodes, degree/n_nodes, seed=base_seed)
            
        nswap = rng.randint(4, degree * n_nodes)
        try:
            nx.double_edge_swap(G, nswap=nswap, max_tries=nswap * 10, seed=rng.randint(0, 10**9))
        except:
            pass
            
        tri_count = sum(nx.triangles(G).values()) // 3
        
        dataset.append({
            "x": torch.ones(n_nodes, 1), 
            "edges": list(G.edges()), 
            "y": torch.tensor([float(tri_count)], dtype=torch.float32),
            "degree": float(degree),
            "graph_id": i,
            "tri_count": float(tri_count),
        })
        pbar.update(1)
        
    counts = np.array([g["tri_count"] for g in dataset])
    print(f"Triangle range: [{counts.min():.0f}, {counts.max():.0f}], mean: {counts.mean():.2f}")
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_graphs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_nodes", type=int, default=24)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = generate_true_triangle_regression_dataset(
        num_graphs=args.num_graphs,
        n_nodes=args.n_nodes,
        seed=args.seed,
    )
    
    train_ds, val_ds, test_ds = split_grouped_dataset(dataset, "graph_id", seed=args.seed)

    results = {}
    models = [
        ("FullGET", FullGET(1, 64, 1, num_steps=16, R=2, lambda_3=1.0, update_damping=0.1), 5e-4),
        ("PairwiseGET", PairwiseGET(1, 110, 1, num_steps=8), 5e-4),
        ("GIN", GINBaseline(1, 64, 1), 5e-4),
    ]

    for name, model, lr in models:
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(model, task_type='regression', device=device, lr=lr)
        res = trainer.run(train_ds, val_ds, test_ds, args.epochs, args.batch_size)
        results[name] = res
        print(f"{name} Test MAE: {res['metric']:.4f}")

    save_results("exp1_triangle_regression_v2", results)

if __name__ == "__main__":
    main()
