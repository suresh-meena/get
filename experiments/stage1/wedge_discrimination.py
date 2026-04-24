import argparse
import torch
import networkx as nx
from tqdm.auto import tqdm

from get import FullGET, PairwiseGET, GINBaseline
from experiments.common import set_seed, GETTrainer, save_results, split_grouped_dataset

def generate_matched_dataset(n_nodes=20, num_pairs=500, degree=3, nswap=40):
    dataset = []
    pair_id = 0
    pbar = tqdm(total=num_pairs, desc="Generating Matched Pairs")
    while pair_id < num_pairs:
        G = nx.random_regular_graph(degree, n_nodes)
        G2 = G.copy()
        try:
            nx.double_edge_swap(G2, nswap=nswap, max_tries=200)
        except nx.NetworkXException:
            continue

        t1 = sum(nx.triangles(G).values())
        t2 = sum(nx.triangles(G2).values())
        if t1 == t2:
            continue

        def to_dict(g, y):
            return {
                "x": torch.ones(n_nodes, 1),
                "edges": list(g.edges()),
                "y": torch.tensor([y]),
                "pair_id": pair_id,
            }

        if t1 > t2:
            dataset.extend([to_dict(G, 1.0), to_dict(G2, 0.0)])
        else:
            dataset.extend([to_dict(G2, 1.0), to_dict(G, 0.0)])
        pair_id += 1
        pbar.update(1)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = generate_matched_dataset(num_pairs=args.num_pairs)
    train_ds, val_ds, test_ds = split_grouped_dataset(dataset, "pair_id", seed=args.seed)

    results = {}
    models = [
        ("PairwiseGET", PairwiseGET(1, int(96 * 1.73), 1, num_steps=8, eta=0.01, eta_max=0.05, beta_2=1.0, grad_clip_norm=0.5, state_clip_norm=5.0, beta_max=3.0), 1e-4, 0.5),
        ("FullGET", FullGET(1, 96, 1, num_steps=8, R=2, lambda_3=0.8, lambda_m=1.0, beta_2=1.0, beta_3=1.2, eta=0.008, eta_max=0.04, grad_clip_norm=0.3, state_clip_norm=5.0, beta_max=3.0, update_damping=0.5, dropout=0.0, compile=False), 3e-5, 0.3),
        ("GIN", GINBaseline(1, 96, 1, num_layers=4), 2e-4, 1.0),
    ]

    for name, model, lr, max_grad_norm in models:
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(
            model,
            task_type='binary',
            device=device,
            lr=lr,
            max_grad_norm=max_grad_norm,
            margin_loss_weight=0.05,
            logit_margin=1.0,
        )
        results[name] = trainer.run(train_ds, val_ds, test_ds, args.epochs, args.batch_size)
        print(f"{name} Test AUC: {results[name]['metric']:.4f}, Accuracy: {results[name].get('accuracy', 0.0):.4f}")

    save_results("exp1_wedge_results", results)

if __name__ == "__main__":
    main()
