import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get import ETFaithful, ETInspiredGET, FullGET, PairwiseGET
from stage1_common import mean_std, parse_seeds, set_seed
from stage2_common import (
    build_anomaly_protocol_split,
    generate_synth_graph_anomaly,
    generate_synth_graph_classification,
    load_node_anomaly_dataset,
    load_tu_dataset,
    train_eval_graph_anomaly,
    train_eval_graph_classification,
)
from plot_stage2 import plot_stage2_graph_anomaly, plot_stage2_graph_classification


def _try_build_gin(in_dim, d, num_classes):
    try:
        from get import GINBaseline

        return GINBaseline(in_dim=in_dim, d=d, num_classes=num_classes, num_layers=4)
    except Exception:
        return None


def _save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_stage2_graph_classification(args, device):
    if args.dataset == "synth":
        dataset = generate_synth_graph_classification(
            num_graphs=args.num_graphs,
            n_nodes=args.n_nodes,
            seed=args.seed,
            num_classes=args.num_classes,
        )
        num_classes = args.num_classes
        in_dim = int(dataset[0]["x"].size(1))
    else:
        dataset = load_tu_dataset(args.dataset, limit=args.limit)
        labels = [int(g["y"].item()) for g in dataset]
        num_classes = int(np.max(labels)) + 1
        in_dim = int(dataset[0]["x"].size(1))

    seeds = parse_seeds(args.seeds)
    results = []
    motif_budget_other = args.max_motifs_other
    motif_budget_full = args.max_motifs_full

    for seed in seeds:
        set_seed(seed)
        pw = PairwiseGET(in_dim=in_dim, d=args.hidden_dim, num_classes=num_classes, num_steps=args.num_steps)
        pw_acc, pw_hist = train_eval_graph_classification(
            "PairwiseGET",
            pw,
            dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            lr=args.lr_pairwise,
            max_grad_norm=0.5,
            seed=seed,
            compile_model=args.compile,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            max_motifs=motif_budget_other,
        )

        fg = FullGET(
            in_dim=in_dim,
            d=args.hidden_dim,
            num_classes=num_classes,
            num_steps=args.num_steps,
            R=args.rank,
            lambda_3=args.lambda3,
            lambda_m=0.0,
        )
        fg_acc, fg_hist = train_eval_graph_classification(
            "FullGET",
            fg,
            dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            lr=args.lr_full,
            max_grad_norm=0.5,
            seed=seed + 1,
            compile_model=args.compile,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            max_motifs=motif_budget_full,
        )

        et_get = ETInspiredGET(in_dim=in_dim, d=args.hidden_dim, num_classes=num_classes, num_steps=args.num_steps)
        et_get_acc, et_get_hist = train_eval_graph_classification(
            "ETInspiredGET",
            et_get,
            dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            lr=args.lr_pairwise,
            max_grad_norm=0.5,
            seed=seed + 10,
            compile_model=args.compile,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            max_motifs=motif_budget_other,
        )

        et_faithful = ETFaithful(
            in_dim=in_dim,
            d=args.hidden_dim,
            num_classes=num_classes,
            num_steps=args.num_steps,
            num_heads=args.et_num_heads,
            head_dim=args.et_head_dim,
            pe_k=args.et_pe_k,
            K=args.et_memory_slots,
        )
        et_faithful_acc, et_faithful_hist = train_eval_graph_classification(
            "ETFaithful",
            et_faithful,
            dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            lr=args.lr_et_faithful,
            max_grad_norm=0.5,
            seed=seed + 11,
            compile_model=args.compile,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            max_motifs=motif_budget_other,
        )

        gin = _try_build_gin(in_dim=in_dim, d=args.hidden_dim, num_classes=num_classes)
        if gin is not None:
            gin_acc, gin_hist = train_eval_graph_classification(
                "GIN",
                gin,
                dataset,
                num_classes=num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_gin,
                max_grad_norm=1.0,
                seed=seed + 2,
                compile_model=args.compile,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_other,
            )
        else:
            gin_acc, gin_hist = None, None

        results.append(
            {
                "seed": seed,
                "pairwise_acc": float(pw_acc),
                "fullget_acc": float(fg_acc),
                "gin_acc": None if gin_acc is None else float(gin_acc),
                "et_get_acc": float(et_get_acc),
                "et_faithful_acc": float(et_faithful_acc),
                "histories": {
                    "pairwise": pw_hist,
                    "fullget": fg_hist,
                    "gin": gin_hist,
                    "et_get": et_get_hist,
                    "et_faithful": et_faithful_hist,
                },
            }
        )

    pw_vals = [r["pairwise_acc"] for r in results]
    fg_vals = [r["fullget_acc"] for r in results]
    et_get_vals = [r["et_get_acc"] for r in results]
    et_faithful_vals = [r["et_faithful_acc"] for r in results]
    out = {
        "task": "graph_classification",
        "dataset": args.dataset,
        "summary": {
            "pairwise_mean": mean_std(pw_vals)[0],
            "pairwise_std": mean_std(pw_vals)[1],
            "fullget_mean": mean_std(fg_vals)[0],
            "fullget_std": mean_std(fg_vals)[1],
            "et_get_mean": mean_std(et_get_vals)[0],
            "et_get_std": mean_std(et_get_vals)[1],
            "et_faithful_mean": mean_std(et_faithful_vals)[0],
            "et_faithful_std": mean_std(et_faithful_vals)[1],
        },
        "runs": results,
    }
    out_path = Path("outputs/stage2_graph_classification.json")
    _save_json(out_path, out)
    print(f"Saved {out_path}")
    if args.plot:
        fig_path = Path(args.classification_plot_path)
        plot_stage2_graph_classification(out_path, fig_path)
        print(f"Saved {fig_path}")
    print(
        f"Pairwise acc: {out['summary']['pairwise_mean']:.4f} ± {out['summary']['pairwise_std']:.4f} | "
        f"FullGET acc: {out['summary']['fullget_mean']:.4f} ± {out['summary']['fullget_std']:.4f} | "
        f"ETInspiredGET acc: {out['summary']['et_get_mean']:.4f} ± {out['summary']['et_get_std']:.4f} | "
        f"ETFaithful acc: {out['summary']['et_faithful_mean']:.4f} ± {out['summary']['et_faithful_std']:.4f}"
    )


def run_stage2_graph_anomaly(args, device):
    if args.dataset == "synth":
        dataset = generate_synth_graph_anomaly(num_graphs=args.num_graphs, n_nodes=args.n_nodes, seed=args.seed)
    else:
        effective_limit = args.limit
        if effective_limit is None and args.dataset.lower() in {"yelp", "amazon", "amazonproducts", "amazon_products"}:
            effective_limit = 5000
            print(
                "No --limit provided for a large node dataset. "
                f"Using default --limit={effective_limit} for tractable ego-graph conversion."
            )
        dataset = load_node_anomaly_dataset(
            dataset_name=args.dataset,
            root=args.data_root,
            num_hops=args.ego_hops,
            limit=effective_limit,
        )
    in_dim = int(dataset[0]["x"].size(1))
    seeds = parse_seeds(args.seeds)
    label_rates = [float(x.strip()) for x in args.anomaly_label_rates.split(",") if x.strip()]
    results = {rate: [] for rate in label_rates}
    motif_budget_other = args.max_motifs_other
    motif_budget_full = args.max_motifs_full

    for rate in label_rates:
        for seed in seeds:
            set_seed(seed)
            split_data = build_anomaly_protocol_split(
                dataset,
                seed=seed,
                labeled_rate=rate,
                val_ratio=args.anomaly_val_ratio,
                test_ratio=args.anomaly_test_ratio,
            )
            pw = PairwiseGET(in_dim=in_dim, d=args.hidden_dim, num_classes=1, num_steps=args.num_steps)
            pw_auc, pw_hist = train_eval_graph_anomaly(
                "PairwiseGET",
                pw,
                dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_pairwise,
                max_grad_norm=0.5,
                seed=seed,
                compile_model=args.compile,
                split_data=split_data,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_other,
            )

            fg = FullGET(
                in_dim=in_dim,
                d=args.hidden_dim,
                num_classes=1,
                num_steps=args.num_steps,
                R=args.rank,
                lambda_3=args.lambda3,
                lambda_m=0.0,
            )
            fg_auc, fg_hist = train_eval_graph_anomaly(
                "FullGET",
                fg,
                dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_full,
                max_grad_norm=0.5,
                seed=seed + 1,
                compile_model=args.compile,
                split_data=split_data,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_full,
            )

            et_get = ETInspiredGET(in_dim=in_dim, d=args.hidden_dim, num_classes=1, num_steps=args.num_steps)
            et_get_auc, et_get_hist = train_eval_graph_anomaly(
                "ETInspiredGET",
                et_get,
                dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_pairwise,
                max_grad_norm=0.5,
                seed=seed + 10,
                compile_model=args.compile,
                split_data=split_data,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_other,
            )

            et_faithful = ETFaithful(
                in_dim=in_dim,
                d=args.hidden_dim,
                num_classes=1,
                num_steps=args.num_steps,
                num_heads=args.et_num_heads,
                head_dim=args.et_head_dim,
                pe_k=args.et_pe_k,
                K=args.et_memory_slots,
            )
            et_faithful_auc, et_faithful_hist = train_eval_graph_anomaly(
                "ETFaithful",
                et_faithful,
                dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_et_faithful,
                max_grad_norm=0.5,
                seed=seed + 11,
                compile_model=args.compile,
                split_data=split_data,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_other,
            )

            gin = _try_build_gin(in_dim=in_dim, d=args.hidden_dim, num_classes=1)
            if gin is not None:
                gin_auc, gin_hist = train_eval_graph_anomaly(
                    "GIN",
                    gin,
                    dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    lr=args.lr_gin,
                    max_grad_norm=1.0,
                    seed=seed + 2,
                    compile_model=args.compile,
                    split_data=split_data,
                    use_amp=args.amp,
                    amp_dtype=args.amp_dtype,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    max_motifs=motif_budget_other,
                )
            else:
                gin_auc, gin_hist = None, None

            results[rate].append(
                {
                    "seed": seed,
                    "pairwise_auc": float(pw_auc),
                    "fullget_auc": float(fg_auc),
                    "gin_auc": None if gin_auc is None else float(gin_auc),
                    "et_get_auc": float(et_get_auc),
                    "et_faithful_auc": float(et_faithful_auc),
                    "histories": {
                        "pairwise": pw_hist,
                        "fullget": fg_hist,
                        "gin": gin_hist,
                        "et_get": et_get_hist,
                        "et_faithful": et_faithful_hist,
                    },
                }
            )

    summary = {}
    for rate in label_rates:
        runs = results[rate]
        pw_vals = [r["pairwise_auc"] for r in runs]
        fg_vals = [r["fullget_auc"] for r in runs]
        et_get_vals = [r["et_get_auc"] for r in runs]
        et_faithful_vals = [r["et_faithful_auc"] for r in runs]
        summary[str(rate)] = {
            "pairwise_mean": mean_std(pw_vals)[0],
            "pairwise_std": mean_std(pw_vals)[1],
            "fullget_mean": mean_std(fg_vals)[0],
            "fullget_std": mean_std(fg_vals)[1],
            "et_get_mean": mean_std(et_get_vals)[0],
            "et_get_std": mean_std(et_get_vals)[1],
            "et_faithful_mean": mean_std(et_faithful_vals)[0],
            "et_faithful_std": mean_std(et_faithful_vals)[1],
        }

    out = {
        "task": "graph_anomaly",
        "dataset": args.dataset,
        "summary": summary,
        "runs": results,
    }
    out_path = Path("outputs/stage2_graph_anomaly.json")
    _save_json(out_path, out)
    print(f"Saved {out_path}")
    if args.plot:
        fig_path = Path(args.anomaly_plot_path)
        plot_stage2_graph_anomaly(out_path, fig_path)
        print(f"Saved {fig_path}")
    for rate in label_rates:
        s = out["summary"][str(rate)]
        print(
            f"label_rate={rate:.2f} | Pairwise AUC: {s['pairwise_mean']:.4f} ± {s['pairwise_std']:.4f} | "
            f"FullGET AUC: {s['fullget_mean']:.4f} ± {s['fullget_std']:.4f} | "
            f"ETInspiredGET AUC: {s['et_get_mean']:.4f} ± {s['et_get_std']:.4f} | "
            f"ETFaithful AUC: {s['et_faithful_mean']:.4f} ± {s['et_faithful_std']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Unified Stage-2 runner.")
    parser.add_argument("--task", choices=["graph_classification", "graph_anomaly"], required=True)
    parser.add_argument(
        "--dataset",
        default="synth",
        help=(
            "For graph_classification: synth or TU dataset name. "
            "For graph_anomaly: synth, yelp, or amazon."
        ),
    )
    parser.add_argument("--data_root", default="data", help="Root directory for downloaded datasets.")
    parser.add_argument("--ego_hops", type=int, default=2, help="Ego hop radius for node anomaly adapters.")
    parser.add_argument(
        "--anomaly_label_rates",
        type=str,
        default="0.01,0.4",
        help="Comma-separated labeled training rates for anomaly protocols.",
    )
    parser.add_argument("--anomaly_val_ratio", type=int, default=1)
    parser.add_argument("--anomaly_test_ratio", type=int, default=2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Optional dataset size limit. For yelp/amazon anomaly tasks, "
            "if unset, a default of 5000 centers is applied."
        ),
    )
    parser.add_argument("--num_graphs", type=int, default=200)
    parser.add_argument("--n_nodes", type=int, default=24)
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=96)
    parser.add_argument("--num_steps", type=int, default=8)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--lambda3", type=float, default=1.0)
    parser.add_argument("--lr_pairwise", type=float, default=1e-4)
    parser.add_argument("--lr_full", type=float, default=5e-5)
    parser.add_argument("--lr_et_faithful", type=float, default=1e-4)
    parser.add_argument("--lr_gin", type=float, default=2e-4)
    parser.add_argument("--et_num_heads", type=int, default=2)
    parser.add_argument("--et_head_dim", type=int, default=None)
    parser.add_argument("--et_pe_k", type=int, default=16)
    parser.add_argument("--et_memory_slots", type=int, default=32)
    parser.add_argument(
        "--max_motifs_full",
        type=int,
        default=32,
        help="Maximum anchored motifs per node for FullGET batching; lower is faster/cheaper.",
    )
    parser.add_argument(
        "--max_motifs_other",
        type=int,
        default=0,
        help="Motif budget for non-motif baselines (0 disables motif extraction).",
    )
    parser.add_argument("--seeds", type=str, default="123,124,125")
    parser.add_argument("--seed", type=int, default=123, help="Generator seed for synthetic datasets.")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Enable AMP autocast where supported.")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots after writing JSON outputs.")
    parser.add_argument("--classification_plot_path", default="outputs/stage2_graph_classification.png")
    parser.add_argument("--anomaly_plot_path", default="outputs/stage2_graph_anomaly.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.task == "graph_classification":
        run_stage2_graph_classification(args, device)
    elif args.task == "graph_anomaly":
        run_stage2_graph_anomaly(args, device)


if __name__ == "__main__":
    main()
