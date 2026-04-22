import argparse
import torch
import networkx as nx
from tqdm.auto import tqdm
import numpy as np

from get import FullGET, PairwiseGET, GINBaseline, ETFaithful
from experiments.common import set_seed, GETTrainer, save_results, mean_std

def generate_matched_dataset(n_nodes=20, num_pairs=500, degree=3, nswap=40):
    dataset = []
    pair_id = 0
    pbar = tqdm(total=num_pairs, desc="Generating Matched Pairs")
    while pair_id < num_pairs:
        G = nx.random_regular_graph(degree, n_nodes)
        G2 = G.copy()
        try:
            nx.double_edge_swap(G2, nswap=nswap, max_tries=200)
        except nx.NetworkXException: continue
        t1, t2 = sum(nx.triangles(G).values()), sum(nx.triangles(G2).values())
        if t1 == t2: continue
        def to_dict(g, y): return {"x": torch.ones(n_nodes, 1), "edges": list(g.edges()), "y": torch.tensor([y]), "pair_id": pair_id}
        if t1 > t2:
            dataset.extend([to_dict(G, 1.0), to_dict(G2, 0.0)])
        else:
            dataset.extend([to_dict(G2, 1.0), to_dict(G, 0.0)])
        pair_id += 1; pbar.update(1)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = generate_matched_dataset(num_pairs=args.num_pairs)
    
    # Split: 70/15/15 by pair_id
    ids = sorted({g["pair_id"] for g in dataset})
    train_ids, val_ids = ids[:int(0.7*len(ids))], ids[int(0.7*len(ids)):int(0.85*len(ids))]
    test_ids = ids[int(0.85*len(ids)):]
    
    train_ds = [g for g in dataset if g["pair_id"] in train_ids]
    val_ds = [g for g in dataset if g["pair_id"] in val_ids]
    test_ds = [g for g in dataset if g["pair_id"] in test_ids]

    results = {}
    models = [
        ("PairwiseGET", PairwiseGET(1, 64, 1)),
        ("FullGET", FullGET(1, 64, 1, lambda_3=1.0)),
        ("ETFaithful_sparse", ETFaithful(1, 64, 1, num_steps=6, pe_k=8, mask_mode="sparse")),
        ("ETFaithful_dense", ETFaithful(1, 64, 1, num_steps=6, pe_k=8, mask_mode="official_dense", et_official_mode=True)),
        ("GIN", GINBaseline(1, 64, 1))
    ]

    for name, model in models:
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(model, task_type='binary', device=device, lr=1e-4)
        results[name] = trainer.run(train_ds, val_ds, test_ds, args.epochs, 32)
        print(f"{name} Test AUC: {results[name]['metric']:.4f}")

    save_results("exp1_wedge_results", results)

if __name__ == "__main__":
    main()
