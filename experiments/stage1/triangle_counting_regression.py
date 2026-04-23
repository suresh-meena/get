import argparse
import random
import torch
import networkx as nx
from tqdm.auto import tqdm
import numpy as np

from get import FullGET, PairwiseGET, GINBaseline
from experiments.common import build_dataloader_kwargs, set_seed, GETTrainer, save_results, split_grouped_dataset

def generate_degree_controlled_triangle_dataset(num_graphs=2000, n_nodes=24, degree=4, seed=0):
    rows = []
    pbar = tqdm(total=num_graphs, desc="Generating Graphs")
    rng = random.Random(seed)
    while len(rows) < num_graphs:
        base_seed = rng.randint(0, 10**9)
        G = nx.random_regular_graph(degree, n_nodes, seed=base_seed)
        nswap = rng.randint(max(4, degree * 2), max(8, degree * n_nodes))
        try:
            nx.double_edge_swap(G, nswap=nswap, max_tries=max(100, nswap * 20), seed=rng.randint(0, 10**9))
        except nx.NetworkXException: continue
        tri_count = sum(nx.triangles(G).values()) // 3
        rows.append({"graph": G, "tri_count": tri_count})
        pbar.update(1)

    counts = np.array([r["tri_count"] for r in rows], dtype=np.float64)
    median_count = float(np.median(counts))
    target_pos = num_graphs // 2
    ranked = list(range(num_graphs))
    ranked.sort(key=lambda i: (counts[i], rng.random()))
    pos_set = set(ranked[-target_pos:])

    dataset = []
    for i, r in enumerate(rows):
        y = 1.0 if i in pos_set else 0.0
        dataset.append({
            "x": torch.ones(n_nodes, 1), 
            "edges": list(r["graph"].edges()), 
            "y": torch.tensor([y], dtype=torch.float32),
            "degree": float(degree),
            "graph_id": i,
            "tri_count": r["tri_count"],
        })

    pos_rate = float(np.mean([g["y"].item() for g in dataset]))
    print(
        f"Triangle threshold (median): {median_count:.1f}, "
        f"count range: [{counts.min():.0f}, {counts.max():.0f}], "
        f"positive rate: {pos_rate:.3f}"
    )
    return dataset


def _predict(model, dataset, batch_size=256, device="cpu"):
    model.eval()
    preds = []
    ys = []
    from torch.utils.data import DataLoader
    from get import collate_get_batch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **build_dataloader_kwargs(device))
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out, _ = model(batch, task_level='graph')
            preds.extend(out.view(-1).cpu().numpy().tolist())
            ys.extend(batch.y.view(-1).cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def _rankdata(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        if j > i:
            avg = 0.5 * (i + j)
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    r1 = _rankdata(y_true)
    r2 = _rankdata(y_pred)
    c1 = r1 - r1.mean()
    c2 = r2 - r2.mean()
    denom = np.sqrt((c1 * c1).sum() * (c2 * c2).sum())
    if denom <= 1e-12:
        return 0.0
    return float((c1 * c2).sum() / denom)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _degree_regressed_scores(y_true, y_pred, degrees):
    # Regress out degree from target and predictions before correlation metrics.
    X = np.stack([np.ones_like(degrees), degrees], axis=1)
    beta_y = np.linalg.lstsq(X, y_true, rcond=None)[0]
    beta_p = np.linalg.lstsq(X, y_pred, rcond=None)[0]
    y_res = y_true - X @ beta_y
    p_res = y_pred - X @ beta_p
    return {
        "rmse_deg_residual": _rmse(y_res, p_res),
        "spearman_deg_residual": _spearman(y_res, p_res),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_graphs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_nodes", type=int, default=24)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--margin_loss_weight", type=float, default=0.05)
    parser.add_argument("--logit_margin", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = generate_degree_controlled_triangle_dataset(
        num_graphs=args.num_graphs,
        n_nodes=args.n_nodes,
        degree=args.degree,
        seed=args.seed,
    )
    train_ds, val_ds, test_ds = split_grouped_dataset(dataset, "graph_id", seed=args.seed)

    results = {}
    models = [
        ("PairwiseGET", PairwiseGET(1, 96, 1, num_steps=8, eta=0.01, eta_max=0.05, beta_2=1.0, grad_clip_norm=0.5, state_clip_norm=5.0, beta_max=3.0), 1e-4, 0.5),
        ("FullGET", FullGET(1, 96, 1, num_steps=8, R=2, lambda_3=0.8, lambda_m=0.0, beta_2=1.0, beta_3=1.2, eta=0.008, eta_max=0.04, grad_clip_norm=0.3, state_clip_norm=5.0, beta_max=3.0, update_damping=0.5, dropout=0.0, compile=False), 3e-5, 0.3),
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
            margin_loss_weight=args.margin_loss_weight,
            logit_margin=args.logit_margin,
        )
        results[name] = trainer.run(train_ds, val_ds, test_ds, args.epochs, args.batch_size)
        print(f"{name} Test AUC: {results[name]['metric']:.4f}")

    save_results("exp2_triangle_results", results)

if __name__ == "__main__":
    main()
