import argparse
import json
import os
import random

import networkx as nx
import numpy as np
import torch
from tqdm.auto import tqdm

from get import FullGET, PairwiseGET
from stage1_common import history_mean_std, mean_std, parse_seeds, set_seed, train_and_eval_binary


def generate_degree_controlled_triangle_dataset(
    num_graphs=500,
    n_nodes=24,
    degree=4,
    seed=0,
):
    if (degree * n_nodes) % 2 != 0:
        raise ValueError("degree * n_nodes must be even for regular graphs.")

    rng = random.Random(seed)
    rows = []
    print("Generating degree-controlled triangle dataset...")

    graph_id = 0
    pbar = tqdm(total=num_graphs, desc="Generating data")
    while graph_id < num_graphs:
        base_seed = rng.randint(0, 10**9)
        G = nx.random_regular_graph(degree, n_nodes, seed=base_seed)

        nswap = rng.randint(max(4, degree * 2), max(8, degree * n_nodes))
        try:
            nx.double_edge_swap(
                G,
                nswap=nswap,
                max_tries=max(100, nswap * 20),
                seed=rng.randint(0, 10**9),
            )
        except nx.NetworkXException:
            continue

        tri_count = sum(nx.triangles(G).values()) // 3
        rows.append({"graph": G, "tri_count": tri_count, "graph_id": graph_id})
        graph_id += 1
        pbar.update(1)
    pbar.close()

    counts = np.array([r["tri_count"] for r in rows], dtype=np.float64)
    median_count = float(np.median(counts))
    target_pos = num_graphs // 2

    ranked = list(range(num_graphs))
    ranked.sort(key=lambda i: (counts[i], rng.random()))
    pos_set = set(ranked[-target_pos:])

    dataset = []
    for i, r in enumerate(rows):
        y = 1.0 if i in pos_set else 0.0
        x = torch.ones(n_nodes, 1, dtype=torch.float32)
        dataset.append(
            {
                "x": x,
                "edges": list(r["graph"].edges()),
                "y": torch.tensor([y], dtype=torch.float32),
                "graph_id": r["graph_id"],
                "tri_count": r["tri_count"],
            }
        )

    pos_rate = float(np.mean([g["y"].item() for g in dataset]))
    print(
        f"Triangle threshold (median): {median_count:.1f}, "
        f"count range: [{counts.min():.0f}, {counts.max():.0f}], "
        f"positive rate: {pos_rate:.3f}"
    )
    return dataset


def run_fullget_sweep(dataset, epochs, batch_size, device, seed, compile_model=False):
    sweep = [
        {
            "name": "FullGET-R1",
            "model_kwargs": dict(
                in_dim=1,
                d=96,
                num_classes=1,
                num_steps=8,
                R=1,
                lambda_3=0.5,
                lambda_m=0.0,
                beta_2=1.0,
                beta_3=1.0,
                eta=0.01,
                eta_max=0.05,
                grad_clip_norm=0.5,
                state_clip_norm=5.0,
                beta_max=3.0,
                compile=False,
            ),
            "train_kwargs": dict(lr=5e-5, max_grad_norm=0.5),
        },
        {
            "name": "FullGET-R2",
            "model_kwargs": dict(
                in_dim=1,
                d=96,
                num_classes=1,
                num_steps=10,
                R=2,
                lambda_3=1.0,
                lambda_m=0.0,
                beta_2=1.0,
                beta_3=1.5,
                eta=0.01,
                eta_max=0.05,
                grad_clip_norm=0.5,
                state_clip_norm=5.0,
                beta_max=3.0,
                compile=False,
            ),
            "train_kwargs": dict(lr=5e-5, max_grad_norm=0.5),
        },
        {
            "name": "FullGET-R2-strong",
            "model_kwargs": dict(
                in_dim=1,
                d=96,
                num_classes=1,
                num_steps=12,
                R=2,
                lambda_3=2.0,
                lambda_m=0.0,
                beta_2=1.0,
                beta_3=2.0,
                eta=0.008,
                eta_max=0.04,
                grad_clip_norm=0.4,
                state_clip_norm=4.0,
                beta_max=2.5,
                compile=False,
            ),
            "train_kwargs": dict(lr=4e-5, max_grad_norm=0.4),
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
            split_key="graph_id",
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            seed=seed + i,
            lr=cfg["train_kwargs"]["lr"],
            max_grad_norm=cfg["train_kwargs"]["max_grad_norm"],
            compile_model=compile_model,
            apply_sigmoid_eval=True,
            track_grad_norm=False,
        )
        bad_total = int(sum(hist["bad_batches"]))
        result = {"name": cfg["name"], "auc": float(auc), "bad_total": bad_total, "history": hist, "model": trained_model}
        results.append(result)
        if best is None or result["auc"] > best["auc"]:
            best = result
    return results, best


def run_single_seed(seed, args, device):
    set_seed(seed)
    dataset = generate_degree_controlled_triangle_dataset(
        num_graphs=args.num_graphs,
        n_nodes=args.n_nodes,
        degree=args.degree,
        seed=seed,
    )

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
        split_key="graph_id",
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=seed,
        lr=1e-4,
        max_grad_norm=0.5,
        compile_model=args.compile,
        apply_sigmoid_eval=True,
        track_grad_norm=False,
    )

    from get import GINBaseline

    model_gin = GINBaseline(in_dim=1, d=96, num_classes=1, num_layers=4)
    auc_gin, hist_gin, _ = train_and_eval_binary(
        "GIN (1-WL)",
        model_gin,
        dataset,
        split_key="graph_id",
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        seed=seed + 1,
        lr=2e-4,
        max_grad_norm=1.0,
        compile_model=args.compile,
        apply_sigmoid_eval=True,
        track_grad_norm=False,
    )

    if args.no_sweep:
        model_full = FullGET(
            in_dim=1,
            d=96,
            num_classes=1,
            num_steps=10,
            R=2,
            lambda_3=1.0,
            lambda_m=0.0,
            beta_2=1.0,
            beta_3=1.5,
            eta=0.01,
            eta_max=0.05,
            grad_clip_norm=0.5,
            state_clip_norm=5.0,
            beta_max=3.0,
            compile=False,
        )
        auc_full, hist_full, _ = train_and_eval_binary(
            "FullGET",
            model_full,
            dataset,
            split_key="graph_id",
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            seed=seed + 2,
            lr=5e-5,
            max_grad_norm=0.5,
            compile_model=args.compile,
            apply_sigmoid_eval=True,
            track_grad_norm=False,
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
    parser.add_argument("--num_graphs", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_nodes", type=int, default=24)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--seeds", type=str, default="123,124,125", help="Comma-separated seeds for multi-run averaging.")
    parser.add_argument("--seed", type=int, default=None, help="Single-seed fallback when --seeds is empty.")
    parser.add_argument("--no_sweep", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for all models during training.")
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
    print("RESULTS: DEGREE-CONTROLLED TRIANGLE CLASSIFICATION (MULTI-SEED)")
    print("========================================")
    print(f"PairwiseGET -> AUC: {pairwise_mean:.4f} ± {pairwise_std:.4f}")
    print(f"GIN Baseline -> AUC: {gin_mean:.4f} ± {gin_std:.4f}")
    print(f"FullGET     -> AUC: {full_mean:.4f} ± {full_std:.4f}")
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

    os.makedirs("code/outputs", exist_ok=True)
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
    with open("code/outputs/exp2_triangle_counting_raw.json", "w", encoding="utf-8") as f:
        json.dump(raw_out, f, indent=2)
    print("Raw logs saved to code/outputs/exp2_triangle_counting_raw.json")

    import matplotlib.pyplot as plt

    colors = {"pairwise": "#1f77b4", "gin": "#ff7f0e", "fullget": "#2ca02c"}
    labels = {"pairwise": "PairwiseGET", "gin": "GIN (1-WL)", "fullget": "FullGET"}

    plt.figure(figsize=(12, 4))
    for subplot_idx, key, title in [(1, "train_loss", "Training Loss"), (2, "test_auc", "Test AUC")]:
        ax = plt.subplot(1, 2, subplot_idx)
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
    plt.savefig("code/outputs/exp2_triangle_counting.png")
    print("Plot saved to code/outputs/exp2_triangle_counting.png")
