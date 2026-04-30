import argparse
import torch
import networkx as nx
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from get import ETFaithful, FullGET, PairwiseGET, GINBaseline, GCNBaseline, GATBaseline, CachedGraphDataset
from experiments.shared.common import set_seed, GETTrainer, save_results, get_num_params

def generate_csl_dataset(graphs_per_class=15, seed=42):
    rng = np.random.default_rng(seed)
    dataset, labels = [], []
    ks = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
    for label, k in enumerate(ks):
        G = nx.Graph()
        G.add_nodes_from(range(41))
        for i in range(41):
            G.add_edge(i, (i + 1) % 41)
            G.add_edge(i, (i + k) % 41)
        for _ in range(graphs_per_class):
            perm = rng.permutation(41)
            G_perm = nx.relabel_nodes(G, {i: perm[i] for i in range(41)})
            dataset.append({"x": torch.ones(41, 1), "edges": list(G_perm.edges()), "y": torch.tensor([label])})
            labels.append(label)
    return dataset, np.array(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--rwse_k", type=int, default=0, help="Keep 0 for the pure expressivity diagnostic; use >0 only for GET+RWSE.")
    parser.add_argument("--graphs_per_class", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_dataset, labels = generate_csl_dataset(graphs_per_class=args.graphs_per_class, seed=0)
    
    if args.rwse_k > 0:
        print(f"Computing RWSE (k={args.rwse_k})...")
        dataset = CachedGraphDataset(raw_dataset, name=f"CSL_rwse{args.rwse_k}", rwse_k=args.rwse_k)
    else:
        dataset = raw_dataset

    model_factories = [
        ("PairwiseGET", lambda: PairwiseGET(1, 96, 10, rwse_k=args.rwse_k)),
        ("FullGET", lambda: FullGET(1, 64, 10, R=2, lambda_3=0.5, rwse_k=args.rwse_k)),
        ("ETFaithful", lambda: ETFaithful(1, 64, 10, num_steps=8, rwse_k=args.rwse_k, pe_k=0, mask_mode="sparse", et_official_mode=False)),
        ("GIN", lambda: GINBaseline(1, 64, 10)),
        ("GCN", lambda: GCNBaseline(1, 64, 10)),
        ("GAT", lambda: GATBaseline(1, 64, 10))
    ]

    results = {}
    for name, factory in model_factories:
        print(f"\n--- CV for {name} ---")
        seed_accs = []
        param_count = None
        for seed in args.seeds:
            set_seed(seed)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            fold_accs = []
            for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                y_train_full = labels[train_idx]
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + fold)
                train_sub_idx, val_sub_idx = next(splitter.split(np.zeros(len(train_idx)), y_train_full))
                train_ids = np.asarray(train_idx)[train_sub_idx]
                val_ids = np.asarray(train_idx)[val_sub_idx]

                train_ds = [dataset[i] for i in train_ids]
                val_ds = [dataset[i] for i in val_ids]
                test_ds = [dataset[i] for i in test_idx]
                model = factory()
                param_count = param_count or get_num_params(model)
                trainer = GETTrainer(model, task_type='multiclass', device=device, lr=1e-3)
                res = trainer.run(train_ds, val_ds, test_ds, args.epochs, 16)
                fold_accs.append(res['metric'])
                print(f"Seed {seed} Fold {fold+1} Acc: {res['metric']:.4f}")
            seed_accs.append(float(np.mean(fold_accs)))

        results[name] = {
            "mean": float(np.mean(seed_accs)),
            "std": float(np.std(seed_accs)),
            "per_seed_fold_mean": seed_accs,
            "params": param_count,
        }
        print(f"{name} Mean Acc: {results[name]['mean']:.4f} ± {results[name]['std']:.4f}")

    metadata = {
        "interpretation": "CSL supervised realized-expressivity diagnostic. rwse_k=0 is the pure architectural test; rwse_k>0 is a GET+structural-encoding practical variant.",
        "rwse_k": args.rwse_k,
        "seeds": args.seeds,
        "graphs_per_class": args.graphs_per_class,
    }
    save_results("exp3_csl_results", results, metadata=metadata)

if __name__ == "__main__":
    main()
