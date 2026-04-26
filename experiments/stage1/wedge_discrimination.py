import argparse
import torch
import networkx as nx
import numpy as np
import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import os

from get import FullGET, PairwiseGET, GINBaseline
from experiments.common import set_seed, GETTrainer, save_results, split_grouped_dataset

def to_dict(G, x, y, pe_k=16):
    from torch_geometric.utils import from_networkx
    data = from_networkx(G)
    
    # Compute RWSE (Random Walk Structural Encodings)
    # A_rw = A D^-1
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(np.sum(A, axis=1))
    D_inv = np.linalg.inv(D)
    A_rw = A @ D_inv
    
    rw_list = []
    if pe_k > 0:
        curr_A = np.eye(G.number_of_nodes())
        for k in range(1, pe_k + 1):
            curr_A = curr_A @ A_rw
            rw_list.append(np.diag(curr_A))
        pe = torch.from_numpy(np.stack(rw_list, axis=1)).float()
    else:
        pe = torch.zeros((G.number_of_nodes(), 0))
    
    # Combine RWSE with the provided features x
    data.x = torch.cat([x, pe], dim=-1)
    data.pe = pe # Also for projection branch
    data.y = torch.tensor([y])
    return data


def generate_matched_dataset(num_pairs=500, pe_k=8):
    dataset = []
    pair_id = 0
    while len(dataset) < num_pairs * 2:
        # Generate two graphs with different triangle counts but same edges/nodes
        n, m = 20, 50
        G = nx.gnm_random_graph(n, m)
        if not nx.is_connected(G): continue
        
        # Swap edges to change triangle count while keeping degrees (mostly)
        G2 = G.copy()
        edges = list(G2.edges())
        for _ in range(20):
            e1, e2 = random.sample(edges, 2)
            u, v = e1
            x, y = e2
            if len({u, v, x, y}) == 4:
                if not G2.has_edge(u, x) and not G2.has_edge(v, y):
                    G2.remove_edge(u, v)
                    G2.remove_edge(x, y)
                    G2.add_edge(u, x)
                    G2.add_edge(v, y)
                    edges = list(G2.edges())

        t1 = sum(nx.triangles(G).values()) // 3
        t2 = sum(nx.triangles(G2).values()) // 3
        
        if t1 == t2: continue
        
        # Features: Just constant ones + tiny noise to break symmetry
        X = torch.ones(n, 1) + torch.randn(n, 1) * 0.01
        if t1 > t2:
            dataset.extend([to_dict(G, X, 1.0, pe_k), to_dict(G2, X, 0.0, pe_k)])
        else:
            dataset.extend([to_dict(G, X, 0.0, pe_k), to_dict(G2, X, 1.0, pe_k)])
        
        for d in dataset[-2:]: d.pair_id = pair_id
        pair_id += 1
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pe_k", type=int, default=16)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = generate_matched_dataset(num_pairs=args.num_pairs, pe_k=args.pe_k)
    train_ds, val_ds, test_ds = split_grouped_dataset(dataset, "pair_id", seed=args.seed)

    in_dim = 1 + args.pe_k
    results = {}
    models = [
        ("PairwiseGET", PairwiseGET(in_dim, 128, 1, num_steps=16, pe_k=args.pe_k, lambda_2=10.0, update_damping=0.05, grad_clip_norm=0.1), 1e-4, 0.5),
        ("FullGET", FullGET(in_dim, 128, 1, num_steps=16, pe_k=args.pe_k, lambda_3=50.0, beta_3=5.0, update_damping=0.05, grad_clip_norm=0.1), 1e-4, 0.3),
        ("GIN", GINBaseline(in_dim, 128, 1, num_layers=4), 1e-4, 1.0),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # Pre-build a sample batch for energy descent plot
    from get import collate_get_batch
    sample_loader = torch.utils.data.DataLoader(test_ds[:16], batch_size=16, collate_fn=collate_get_batch)
    sample_batch = next(iter(sample_loader)).to(device)

    for name, model, lr, max_grad_norm in models:
        if "cuda" in device:
            torch.cuda.empty_cache()
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(
            model,
            task_type='binary',
            device=device,
            lr=lr,
            max_grad_norm=max_grad_norm,
            max_grad_val=0.05,
            weight_decay=1e-4,
        )
        
        # We wrap trainer.run logic to print gradients during training
        # For simplicity, we just look at the final history but add a print hook
        res = trainer.run(train_ds, val_ds, test_ds, args.epochs, args.batch_size)
        
        results[name] = res
        print(f"{name} Final Test AUC: {res['metric']:.4f}")
        
        # 1. Plot Loss and AUC
        history = res['history']
        epochs_range = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs_range, history['train_loss'], label=name)
        axes[1].plot(epochs_range, history['val_metric'], label=name)

        # 2. Capture Energy Descent (only for GET models)
        if "GET" in name:
            model.eval()
            with torch.no_grad():
                _, energy_trace = model(sample_batch)
                energies = torch.stack(energy_trace).mean(dim=1).cpu().numpy()
                axes[2].plot(range(len(energies)), energies, marker='o', label=name)
                print(f"{name} Energy Trace: {energies.tolist()}")

    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].set_title('Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()

    axes[2].set_title('Inference Energy Descent')
    axes[2].set_xlabel('Unrolled Step')
    axes[2].set_ylabel('Energy E(X)')
    axes[2].legend()
    
    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/exp1_wedge_plots.png")
    print("\nPlots saved to outputs/exp1_wedge_plots.png")

    save_results("exp1_wedge_results", results)

if __name__ == "__main__":
    main()
