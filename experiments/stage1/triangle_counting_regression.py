import argparse
import numpy as np
import torch

from experiments.shared.common import GETTrainer, build_dataloader_kwargs, save_results, set_seed, split_grouped_dataset
from experiments.stage1.common import (
    generate_degree_controlled_triangle_dataset,
    match_pairwise_width,
    summarize_stage1_support,
)
from get import ETFaithful, FullGET, GINBaseline, PairwiseGET


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
    parser.add_argument("--rwse_k", type=int, default=0)
    parser.add_argument("--include_degree", action="store_true")
    parser.add_argument("--include_motif_counts", action="store_true")
    parser.add_argument(
        "--support_mode",
        type=str,
        default="exact",
        choices=["exact", "topB_closed_first", "topB_open_first", "random", "oracle", "full"],
    )
    parser.add_argument("--max_motifs", type=int, default=-1, help="Per-node support budget; negative means unlimited.")
    parser.add_argument(
        "--feature_mode",
        type=str,
        default="core",
        choices=["core", "rwse", "static_motif"],
        help="Stage-1 feature ablation mode.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_motifs = None if int(args.max_motifs) < 0 else int(args.max_motifs)
    dataset = generate_degree_controlled_triangle_dataset(
        num_graphs=args.num_graphs,
        n_nodes=args.n_nodes,
        degree=args.degree,
        seed=args.seed,
        rwse_k=args.rwse_k,
        include_degree=args.include_degree,
        include_motif_counts=args.include_motif_counts,
        support_mode=args.support_mode,
        max_motifs=max_motifs,
        feature_mode=args.feature_mode,
    )
    train_ds, val_ds, test_ds = split_grouped_dataset(dataset, "graph_id", seed=args.seed)
    support_summary = summarize_stage1_support(dataset)
    print(
        "Support summary: "
        f"candidate={int(support_summary['candidate_motif_count'])}, "
        f"retained={int(support_summary['retained_motif_count'])}, "
        f"retained_fraction={support_summary['retained_fraction']:.3f}"
    )

    in_dim = int(train_ds[0]["x"].size(-1)) if train_ds else int(dataset[0]["x"].size(-1))
    match = match_pairwise_width(
        in_dim,
        1,
        96,
        full_kwargs={
            "num_steps": 8,
            "R": 2,
            "lambda_3": 0.8,
            "lambda_m": 1.0,
            "beta_2": 1.0,
            "beta_3": 1.2,
            "eta": 0.008,
            "eta_max": 0.04,
            "grad_clip_norm": 0.3,
            "state_clip_norm": 5.0,
            "beta_max": 3.0,
            "update_damping": 0.5,
            "dropout": 0.0,
            "compile": False,
        },
        pairwise_kwargs={
            "num_steps": 8,
            "eta": 0.01,
            "eta_max": 0.05,
            "beta_2": 1.0,
            "grad_clip_norm": 0.5,
            "state_clip_norm": 5.0,
            "beta_max": 3.0,
        },
    )
    pairwise_d = match["pairwise_width"]
    print(
        f"Matched PairwiseGET width={pairwise_d} against FullGET d=96 "
        f"(params {match['pairwise_params']} vs {match['full_params']}, rel err {match['relative_error']:.3%})"
    )

    results = {}
    models = [
        ("PairwiseGET", lambda: PairwiseGET(in_dim, pairwise_d, 1, num_steps=8, eta=0.01, eta_max=0.05, beta_2=1.0, grad_clip_norm=0.5, state_clip_norm=5.0, beta_max=3.0), 1e-4, 0.5),
        ("FullGET", lambda: FullGET(in_dim, 96, 1, num_steps=8, R=2, lambda_3=0.8, lambda_m=1.0, beta_2=1.0, beta_3=1.2, eta=0.008, eta_max=0.04, grad_clip_norm=0.3, state_clip_norm=5.0, beta_max=3.0, update_damping=0.5, dropout=0.0, compile=False), 3e-5, 0.3),
        ("ETFaithful", lambda: ETFaithful(in_dim, 96, 1, num_steps=8, rwse_k=args.rwse_k, mask_mode="sparse", et_official_mode=False), 3e-5, 0.3),
        ("GIN", lambda: GINBaseline(in_dim, 96, 1, num_layers=4, dropout=0.0), 5e-5, 0.5),
    ]

    for name, model_fn, lr, max_grad_norm in models:
        model = model_fn()
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(
            model,
            task_type='regression',
            device=device,
            lr=lr,
            max_grad_norm=max_grad_norm,
        )
        train_res = trainer.run(train_ds, val_ds, test_ds, args.epochs, args.batch_size)
        
        y_pred, y_true = _predict(model, test_ds, batch_size=args.batch_size, device=device)
        degs = np.array([g["degree"] for g in test_ds], dtype=np.float64)
        
        rmse_val = _rmse(y_true, y_pred)
        spearman_val = _spearman(y_true, y_pred)
        deg_res = _degree_regressed_scores(y_true, y_pred, degs)
        
        res = {
            'mae': float(np.mean(np.abs(y_true - y_pred))),
            'rmse': rmse_val,
            'spearman': spearman_val,
            'rmse_deg_residual': deg_res['rmse_deg_residual'],
            'spearman_deg_residual': deg_res['spearman_deg_residual'],
            'history': train_res.get('history', {}),
        }
        if train_res.get('energy_trace'):
            res['energy_trace'] = train_res['energy_trace']
        results[name] = res
        print(f"{name} Test MAE: {res['mae']:.4f}, RMSE: {res['rmse']:.4f}, Spearman: {res['spearman']:.4f}, Deg-Resid Spearman: {res['spearman_deg_residual']:.4f}")

    results["_support_summary"] = support_summary
    save_results("exp2_triangle_results", results)

if __name__ == "__main__":
    main()
