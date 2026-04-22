import argparse
import torch
import networkx as nx
from tqdm.auto import tqdm
import numpy as np

from get import FullGET, PairwiseGET, GINBaseline
from experiments.common import build_dataloader_kwargs, set_seed, GETTrainer, save_results

def generate_controlled_triangles(num_graphs=500, n_nodes=24, degree=4):
    rows = []
    pbar = tqdm(total=num_graphs, desc="Generating Graphs")
    while len(rows) < num_graphs:
        G = nx.random_regular_graph(degree, n_nodes)
        try: nx.double_edge_swap(G, nswap=20, max_tries=100)
        except nx.NetworkXException: continue
        tri_count = sum(nx.triangles(G).values()) // 3
        rows.append({"graph": G, "tri_count": tri_count})
        pbar.update(1)
    
    # Regression target: triangle count per node.
    dataset = []
    for i, r in enumerate(rows):
        y = float(r["tri_count"]) / float(n_nodes)
        dataset.append({
            "x": torch.ones(n_nodes, 1), 
            "edges": list(r["graph"].edges()), 
            "y": torch.tensor([y], dtype=torch.float32),
            "degree": float(degree),
            "id": i
        })
    return dataset


def _predict(model, dataset, batch_size=64, device="cpu"):
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
    parser.add_argument("--num_graphs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = generate_controlled_triangles(num_graphs=args.num_graphs)
    
    # Split: 80/20
    split = int(0.8 * len(dataset))
    train_ds, test_ds = dataset[:split], dataset[split:]
    val_ds = test_ds

    results = {}
    models = [
        ("PairwiseGET", PairwiseGET(1, 64, 1)),
        ("FullGET", FullGET(1, 64, 1, lambda_3=1.0)),
        ("GIN", GINBaseline(1, 64, 1))
    ]

    for name, model in models:
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(model, task_type='regression', device=device, lr=1e-4)
        base_res = trainer.run(train_ds, val_ds, test_ds, args.epochs, 32)
        y_pred, y_true = _predict(trainer.model, test_ds, batch_size=64, device=device)
        degrees = np.asarray([float(g["degree"]) for g in test_ds], dtype=np.float64)
        metrics = {
            "mae": float(np.mean(np.abs(y_true - y_pred))),
            "rmse": _rmse(y_true, y_pred),
            "spearman": _spearman(y_true, y_pred),
        }
        metrics.update(_degree_regressed_scores(y_true, y_pred, degrees))
        results[name] = {"train_eval": base_res, **metrics}
        print(
            f"{name} Test MAE: {metrics['mae']:.4f} | RMSE: {metrics['rmse']:.4f} | "
            f"Spearman: {metrics['spearman']:.4f}"
        )

    save_results("exp2_triangle_results", results)

if __name__ == "__main__":
    main()
