import argparse
import json
import os

import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm

from get import FullGET, PairwiseGET
from stage1_common import history_mean_std, mean_std, parse_seeds, set_seed, train_and_eval_binary


def generate_matched_dataset(n_nodes=20, num_pairs=500, degree=3, nswap=40, max_tries=200):
    if (degree * n_nodes) % 2 != 0:
        raise ValueError("degree * n_nodes must be even for regular graphs.")

    dataset = []
    pair_id = 0
    print("Generating matched graph pairs (same degree sequence, different triangle counts)...")
    pbar = tqdm(total=num_pairs, desc="Generating data")
    while pair_id < num_pairs:
        G = nx.random_regular_graph(degree, n_nodes)
        triangles_1 = sum(nx.triangles(G).values()) // 3

        G2 = G.copy()
        try:
            nx.double_edge_swap(G2, nswap=nswap, max_tries=max_tries)
        except nx.NetworkXException:
            continue

        triangles_2 = sum(nx.triangles(G2).values()) // 3
        if triangles_1 == triangles_2:
            continue

        def add_graph(graph, label, pid):
            x = torch.ones(n_nodes, 1, dtype=torch.float32)
            dataset.append(
                {"x": x, "edges": list(graph.edges()), "y": torch.tensor([label], dtype=torch.float32), "pair_id": pid}
            )

        if triangles_1 > triangles_2:
            add_graph(G, 1.0, pair_id)
            add_graph(G2, 0.0, pair_id)
        else:
            add_graph(G2, 1.0, pair_id)
            add_graph(G, 0.0, pair_id)
        pair_id += 1
        pbar.update(1)
    pbar.close()
    return dataset


def run_fullget_sweep(
    dataset,
    epochs,
    batch_size,
    device,
    seed,
    compile_model=False,
    num_workers=8,
    margin_loss_weight=0.05,
    logit_margin=1.0,
):
    sweep = [
        {
            "name": "FullGET-R1",
            "model_kwargs": dict(
                in_dim=1,
                d=96,
                num_classes=1,
                num_steps=8,
                R=1,
                lambda_3=0.6,
                lambda_m=0.0,
                beta_2=1.0,
                beta_3=1.0,
                eta=0.006,
                eta_max=0.03,
                grad_clip_norm=0.3,
                state_clip_norm=5.0,
                beta_max=3.0,
                update_damping=0.5,
                dropout=0.0,
                compile=False,
            ),
            "train_kwargs": dict(lr=3e-5, max_grad_norm=0.3),
        },
        {
            "name": "FullGET-R2",
            "model_kwargs": dict(
                in_dim=1,
                d=96,
                num_classes=1,
                num_steps=8,
                R=2,
                lambda_3=0.8,
                lambda_m=0.0,
                beta_2=1.0,
                beta_3=1.2,
                eta=0.008,
                eta_max=0.04,
                grad_clip_norm=0.3,
                state_clip_norm=5.0,
                beta_max=3.0,
                update_damping=0.5,
                dropout=0.0,
                compile=False,
            ),
            "train_kwargs": dict(lr=3e-5, max_grad_norm=0.3),
        },
    ]

    results = []
    best = None
    for i, cfg in enumerate(sweep):
        set_seed(seed + i)
        model = FullGET(**cfg["model_kwargs"])
        auc, hist, trained_model = train_and_eval_binary(
            cfg["name"],
            model,
            dataset,
            split_key="pair_id",
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed + i,
            lr=cfg["train_kwargs"]["lr"],
            max_grad_norm=cfg["train_kwargs"]["max_grad_norm"],
            compile_model=compile_model,
            apply_sigmoid_eval=False,
            track_grad_norm=True,
            num_workers=num_workers,
            use_amp=False,
            margin_loss_weight=margin_loss_weight,
            logit_margin=logit_margin,
        )
        bad_total = int(sum(hist["bad_batches"]))
        result = {"name": cfg["name"], "auc": float(auc), "bad_total": bad_total, "history": hist, "model": trained_model}
        results.append(result)
        if best is None or result["auc"] > best["auc"]:
            best = result
    return results, best


def run_single_seed(seed, args, device):
    set_seed(seed)
    dataset = generate_matched_dataset(num_pairs=args.num_pairs)

    model_pw = PairwiseGET(
        in_dim=1,
        d=96,
        num_classes=1,
        num_steps=8,
        eta=0.01,
        eta_max=0.05,
        beta_2=1.0,
        grad_clip_norm=0.5,
        state_clip_norm=5.0,
        beta_max=3.0,
    )
    auc_pw, hist_pw, _ = train_and_eval_binary(
        "PairwiseGET",
        model_pw,
        dataset,
        split_key="pair_id",
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        max_grad_norm=0.5,
        lr=1e-4,
        seed=seed,
        compile_model=args.compile,
        num_workers=args.num_workers,
        margin_loss_weight=args.margin_loss_weight,
        logit_margin=args.logit_margin,
    )

    from get import GINBaseline

    model_gin = GINBaseline(in_dim=1, d=96, num_classes=1, num_layers=4)
    auc_gin, hist_gin, _ = train_and_eval_binary(
        "GIN (1-WL)",
        model_gin,
        dataset,
        split_key="pair_id",
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        max_grad_norm=1.0,
        lr=2e-4,
        seed=seed + 1,
        compile_model=args.compile,
        num_workers=args.num_workers,
        margin_loss_weight=args.margin_loss_weight,
        logit_margin=args.logit_margin,
    )

    if args.no_sweep:
        model_full = FullGET(
            in_dim=1,
            d=96,
            num_classes=1,
            num_steps=8,
            R=2,
            lambda_3=0.8,
            lambda_m=0.0,
            beta_2=1.0,
            beta_3=1.2,
            eta=0.008,
            eta_max=0.04,
            grad_clip_norm=0.3,
            state_clip_norm=5.0,
            beta_max=3.0,
            update_damping=0.5,
            dropout=0.0,
            compile=False,
        )
        auc_full, hist_full, _ = train_and_eval_binary(
            "FullGET",
            model_full,
            dataset,
            split_key="pair_id",
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            max_grad_norm=0.3,
            lr=3e-5,
            seed=seed + 2,
            compile_model=args.compile,
            apply_sigmoid_eval=False,
            track_grad_norm=True,
            num_workers=args.num_workers,
            use_amp=False,
            margin_loss_weight=args.margin_loss_weight,
            logit_margin=args.logit_margin,
        )
        sweep_results = [{"name": "single", "auc": auc_full, "bad_total": int(sum(hist_full["bad_batches"]))}]
    else:
        sweep_results, best = run_fullget_sweep(
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=seed + 2,
            compile_model=args.compile,
            num_workers=args.num_workers,
            margin_loss_weight=args.margin_loss_weight,
            logit_margin=args.logit_margin,
        )
        auc_full = best["auc"]
        hist_full = best["history"]

    return {
        "seed": seed,
        "metrics": {"pairwise_auc": float(auc_pw), "gin_auc": float(auc_gin), "fullget_auc": float(auc_full)},
        "histories": {"pairwise": hist_pw, "gin": hist_gin, "fullget": hist_full},
        "sweep_results": [{"name": r["name"], "auc": float(r["auc"]), "bad_total": int(r["bad_total"])} for r in sweep_results],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="123,124,125", help="Comma-separated seeds for multi-run averaging.")
    parser.add_argument("--seed", type=int, default=None, help="Single-seed fallback when --seeds is empty.")
    parser.add_argument("--no_sweep", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for all models during training.")
    parser.add_argument("--margin_loss_weight", type=float, default=0.05)
    parser.add_argument("--logit_margin", type=float, default=1.0)
    args = parser.parse_args()

    if args.seeds.strip():
        seeds = parse_seeds(args.seeds)
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        raise ValueError("Provide --seeds or --seed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Running seeds: {seeds}")

    runs = []
    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        run = run_single_seed(seed, args, device)
        runs.append(run)
        print(
            f"Seed {seed}: Pairwise AUC={run['metrics']['pairwise_auc']:.4f}, "
            f"GIN AUC={run['metrics']['gin_auc']:.4f}, FullGET AUC={run['metrics']['fullget_auc']:.4f}"
        )

    pairwise_vals = [r["metrics"]["pairwise_auc"] for r in runs]
    gin_vals = [r["metrics"]["gin_auc"] for r in runs]
    full_vals = [r["metrics"]["fullget_auc"] for r in runs]
    pairwise_mean, pairwise_std = mean_std(pairwise_vals)
    gin_mean, gin_std = mean_std(gin_vals)
    full_mean, full_std = mean_std(full_vals)

    print("\n========================================")
    print("RESULTS: WEDGE DISCRIMINATION (MULTI-SEED)")
    print("========================================")
    print(f"PairwiseGET -> selected test AUC: {pairwise_mean:.4f} ± {pairwise_std:.4f}")
    print(f"GIN Baseline -> selected test AUC: {gin_mean:.4f} ± {gin_std:.4f}")
    print(f"FullGET     -> selected test AUC: {full_mean:.4f} ± {full_std:.4f}")
    print("========================================")

    if not args.no_sweep:
        cfg_names = [cfg["name"] for cfg in runs[0]["sweep_results"]]
        print("FullGET sweep (across seeds):")
        for cfg_name in cfg_names:
            vals = []
            bad = 0
            for run in runs:
                rec = next(x for x in run["sweep_results"] if x["name"] == cfg_name)
                vals.append(rec["auc"])
                bad += rec["bad_total"]
            m, s = mean_std(vals)
            print(f"  {cfg_name}: AUC={m:.4f} ± {s:.4f}, bad_batches_total={bad}")

    os.makedirs("outputs", exist_ok=True)
    raw_out = {
        "args": vars(args),
        "seeds": seeds,
        "runs": runs,
        "summary": {
            "pairwise_auc_mean": pairwise_mean,
            "pairwise_auc_std": pairwise_std,
            "gin_auc_mean": gin_mean,
            "gin_auc_std": gin_std,
            "fullget_auc_mean": full_mean,
            "fullget_auc_std": full_std,
        },
    }
    with open("outputs/exp1_wedge_discrimination_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_out, f, indent=2)
    print("Raw logs saved to outputs/exp1_wedge_discrimination_raw.json")

    import matplotlib.pyplot as plt

    colors = {"pairwise": "#1f77b4", "gin": "#ff7f0e", "fullget": "#2ca02c"}
    labels = {"pairwise": "PairwiseGET", "gin": "GIN (1-WL)", "fullget": "FullGET"}

    plot_specs = [
        (1, "train_bce_loss", "Training BCE"),
        (2, "val_loss", "Validation BCE"),
        (3, "val_auc", "Validation AUC"),
        (4, "val_logit_margin", "Validation Logit Margin"),
    ]
    plt.figure(figsize=(16, 8))
    for subplot_idx, key, title in plot_specs:
        ax = plt.subplot(2, 2, subplot_idx)
        for model_key in ["pairwise", "gin", "fullget"]:
            model_hists = [r["histories"][model_key] for r in runs]
            for h in model_hists:
                ax.plot(h[key], color=colors[model_key], alpha=0.15, linewidth=1.0)
            mean_curve, std_curve = history_mean_std(model_hists, key)
            x = np.arange(len(mean_curve))
            ax.fill_between(
                x,
                np.array(mean_curve) - np.array(std_curve),
                np.array(mean_curve) + np.array(std_curve),
                color=colors[model_key],
                alpha=0.12,
            )
            ax.plot(mean_curve, color=colors[model_key], linewidth=2.2, label=f"{labels[model_key]} (mean)")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.savefig("outputs/exp1_wedge_discrimination.png")
    print("Plot saved to outputs/exp1_wedge_discrimination.png")
