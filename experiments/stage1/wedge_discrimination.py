import argparse
import random
from pathlib import Path

import networkx as nx
import torch

from experiments.common import GETTrainer, save_results, set_seed, split_grouped_dataset
from experiments.shared.plotting import plot_payload
from experiments.stage1.common import (
    build_stage1_graph_item,
    match_pairwise_width,
    prepare_stage1_graph,
    summarize_stage1_support,
)
from get import FullGET, GINBaseline, PairwiseGET


def generate_matched_dataset(
    num_pairs=500,
    rwse_k=0,
    include_degree=False,
    include_motif_counts=False,
    support_mode="exact",
    max_motifs=None,
    feature_mode="core",
    seed=0,
):
    dataset = []
    pair_id = 0
    while len(dataset) < num_pairs * 2:
        n, m = 20, 50
        graph = nx.gnm_random_graph(n, m)
        if not nx.is_connected(graph):
            continue

        candidate = graph.copy()
        edges = list(candidate.edges())
        for _ in range(20):
            edge_1, edge_2 = random.sample(edges, 2)
            u, v = edge_1
            x, y = edge_2
            if len({u, v, x, y}) != 4:
                continue
            if candidate.has_edge(u, x) or candidate.has_edge(v, y):
                continue
            candidate.remove_edge(u, v)
            candidate.remove_edge(x, y)
            candidate.add_edge(u, x)
            candidate.add_edge(v, y)
            edges = list(candidate.edges())

        if sorted(d for _, d in graph.degree()) != sorted(d for _, d in candidate.degree()):
            continue

        tri_base = sum(nx.triangles(graph).values()) // 3
        tri_candidate = sum(nx.triangles(candidate).values()) // 3
        if tri_base == tri_candidate:
            continue

        if tri_base > tri_candidate:
            first, second = graph, candidate
            first_label, second_label = 1.0, 0.0
        else:
            first, second = candidate, graph
            first_label, second_label = 1.0, 0.0

        first_item = build_stage1_graph_item(
            first,
            first_label,
            pair_id,
            rwse_k=0,
            include_degree=include_degree,
            include_motif_counts=include_motif_counts,
        )
        second_item = build_stage1_graph_item(
            second,
            second_label,
            pair_id,
            rwse_k=0,
            include_degree=include_degree,
            include_motif_counts=include_motif_counts,
        )

        dataset.extend(
            [
                prepare_stage1_graph(
                    first_item,
                    feature_mode=feature_mode,
                    support_mode=support_mode,
                    max_motifs=max_motifs,
                    rwse_k=rwse_k,
                    seed=seed + (2 * pair_id),
                ),
                prepare_stage1_graph(
                    second_item,
                    feature_mode=feature_mode,
                    support_mode=support_mode,
                    max_motifs=max_motifs,
                    rwse_k=rwse_k,
                    seed=seed + (2 * pair_id) + 1,
                ),
            ]
        )
        pair_id += 1

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
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
    dataset = generate_matched_dataset(
        num_pairs=args.num_pairs,
        rwse_k=args.rwse_k,
        include_degree=args.include_degree,
        include_motif_counts=args.include_motif_counts,
        support_mode=args.support_mode,
        max_motifs=max_motifs,
        feature_mode=args.feature_mode,
        seed=args.seed,
    )
    train_ds, val_ds, test_ds = split_grouped_dataset(dataset, "pair_id", seed=args.seed)
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
        128,
        full_kwargs={
            "num_steps": 16,
            "lambda_3": 1.0,
            "beta_3": 5.0,
            "update_damping": 0.05,
            "grad_clip_norm": 0.1,
        },
        pairwise_kwargs={
            "num_steps": 16,
            "update_damping": 0.05,
            "grad_clip_norm": 0.1,
        },
    )
    pairwise_d = match["pairwise_width"]
    print(
        f"Matched PairwiseGET width={pairwise_d} against FullGET d=128 "
        f"(params {match['pairwise_params']} vs {match['full_params']}, rel err {match['relative_error']:.3%})"
    )

    results = {}
    models = [
        ("PairwiseGET", lambda: PairwiseGET(in_dim, pairwise_d, 1, num_steps=16, update_damping=0.05, grad_clip_norm=0.1), 1e-4, 0.5),
        ("FullGET", lambda: FullGET(in_dim, 128, 1, num_steps=16, lambda_3=1.0, beta_3=5.0, update_damping=0.05, grad_clip_norm=0.1), 1e-4, 0.3),
        ("GIN", lambda: GINBaseline(in_dim, 128, 1, num_layers=4), 1e-4, 1.0),
    ]

    for name, model_fn, lr, max_grad_norm in models:
        model = model_fn()
        if "cuda" in device:
            torch.cuda.empty_cache()
        print(f"\n--- Training {name} ---")
        trainer = GETTrainer(
            model,
            task_type="binary",
            device=device,
            lr=lr,
            max_grad_norm=max_grad_norm,
            max_grad_val=0.05,
            weight_decay=1e-4,
        )
        res = trainer.run(train_ds, val_ds, test_ds, args.epochs, args.batch_size)
        results[name] = res
        print(f"{name} Final Test AUC: {res['metric']:.4f}")

    plot_path = plot_payload(results, Path("outputs/exp1_wedge_plots.png"), title="Wedge Discrimination")
    print(f"\nPlots saved to {plot_path}")

    results["_support_summary"] = support_summary

    save_results("exp1_wedge_results", results)


if __name__ == "__main__":
    main()
