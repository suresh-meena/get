import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get import ETComplete, ETLocal, FullGET, PairwiseGET
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
        )

        et_local = ETLocal(in_dim=in_dim, d=args.hidden_dim, num_classes=num_classes, num_steps=args.num_steps)
        et_local_acc, et_local_hist = train_eval_graph_classification(
            "ET-Local",
            et_local,
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
        )

        et_complete = ETComplete(in_dim=in_dim, d=args.hidden_dim, num_classes=num_classes, num_steps=args.num_steps)
        et_complete_acc, et_complete_hist = train_eval_graph_classification(
            "ET-Complete",
            et_complete,
            dataset,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            lr=args.lr_pairwise,
            max_grad_norm=0.5,
            seed=seed + 11,
            compile_model=args.compile,
            use_amp=args.amp,
            amp_dtype=args.amp_dtype,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
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
            )
        else:
            gin_acc, gin_hist = None, None

        results.append(
            {
                "seed": seed,
                "pairwise_acc": float(pw_acc),
                "fullget_acc": float(fg_acc),
                "gin_acc": None if gin_acc is None else float(gin_acc),
                "et_local_acc": float(et_local_acc),
                "et_complete_acc": float(et_complete_acc),
                "histories": {
                    "pairwise": pw_hist,
                    "fullget": fg_hist,
                    "gin": gin_hist,
                    "et_local": et_local_hist,
                    "et_complete": et_complete_hist,
                },
            }
        )

    pw_vals = [r["pairwise_acc"] for r in results]
    fg_vals = [r["fullget_acc"] for r in results]
    et_local_vals = [r["et_local_acc"] for r in results]
    et_complete_vals = [r["et_complete_acc"] for r in results]
    out = {
        "task": "graph_classification",
        "dataset": args.dataset,
        "summary": {
            "pairwise_mean": mean_std(pw_vals)[0],
            "pairwise_std": mean_std(pw_vals)[1],
            "fullget_mean": mean_std(fg_vals)[0],
            "fullget_std": mean_std(fg_vals)[1],
            "et_local_mean": mean_std(et_local_vals)[0],
            "et_local_std": mean_std(et_local_vals)[1],
            "et_complete_mean": mean_std(et_complete_vals)[0],
            "et_complete_std": mean_std(et_complete_vals)[1],
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
        f"ET-Local acc: {out['summary']['et_local_mean']:.4f} ± {out['summary']['et_local_std']:.4f} | "
        f"ET-Complete acc: {out['summary']['et_complete_mean']:.4f} ± {out['summary']['et_complete_std']:.4f}"
    )


def run_stage2_graph_anomaly(args, device):
    if args.dataset == "synth":
        dataset = generate_synth_graph_anomaly(num_graphs=args.num_graphs, n_nodes=args.n_nodes, seed=args.seed)
    else:
        dataset = load_node_anomaly_dataset(
            dataset_name=args.dataset,
            root=args.data_root,
            num_hops=args.ego_hops,
            limit=args.limit,
        )
    in_dim = int(dataset[0]["x"].size(1))
    seeds = parse_seeds(args.seeds)
    label_rates = [float(x.strip()) for x in args.anomaly_label_rates.split(",") if x.strip()]
    results = {rate: [] for rate in label_rates}

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
            )

            et_local = ETLocal(in_dim=in_dim, d=args.hidden_dim, num_classes=1, num_steps=args.num_steps)
            et_local_auc, et_local_hist = train_eval_graph_anomaly(
                "ET-Local",
                et_local,
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
            )

            et_complete = ETComplete(in_dim=in_dim, d=args.hidden_dim, num_classes=1, num_steps=args.num_steps)
            et_complete_auc, et_complete_hist = train_eval_graph_anomaly(
                "ET-Complete",
                et_complete,
                dataset,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_pairwise,
                max_grad_norm=0.5,
                seed=seed + 11,
                compile_model=args.compile,
                split_data=split_data,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
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
                )
            else:
                gin_auc, gin_hist = None, None

            results[rate].append(
                {
                    "seed": seed,
                    "pairwise_auc": float(pw_auc),
                    "fullget_auc": float(fg_auc),
                    "gin_auc": None if gin_auc is None else float(gin_auc),
                    "et_local_auc": float(et_local_auc),
                    "et_complete_auc": float(et_complete_auc),
                    "histories": {
                        "pairwise": pw_hist,
                        "fullget": fg_hist,
                        "gin": gin_hist,
                        "et_local": et_local_hist,
                        "et_complete": et_complete_hist,
                    },
                }
            )

    summary = {}
    for rate in label_rates:
        runs = results[rate]
        pw_vals = [r["pairwise_auc"] for r in runs]
        fg_vals = [r["fullget_auc"] for r in runs]
        et_local_vals = [r["et_local_auc"] for r in runs]
        et_complete_vals = [r["et_complete_auc"] for r in runs]
        summary[str(rate)] = {
            "pairwise_mean": mean_std(pw_vals)[0],
            "pairwise_std": mean_std(pw_vals)[1],
            "fullget_mean": mean_std(fg_vals)[0],
            "fullget_std": mean_std(fg_vals)[1],
            "et_local_mean": mean_std(et_local_vals)[0],
            "et_local_std": mean_std(et_local_vals)[1],
            "et_complete_mean": mean_std(et_complete_vals)[0],
            "et_complete_std": mean_std(et_complete_vals)[1],
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
            f"ET-Local AUC: {s['et_local_mean']:.4f} ± {s['et_local_std']:.4f} | "
            f"ET-Complete AUC: {s['et_complete_mean']:.4f} ± {s['et_complete_std']:.4f}"
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
    parser.add_argument("--limit", type=int, default=None, help="Optional dataset size limit for TU datasets.")
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
    parser.add_argument("--lr_gin", type=float, default=2e-4)
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
