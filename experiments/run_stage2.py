import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get import ETFaithful, FullGET, PairwiseGET
from get.data import CachedGraphDataset
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

    dataset_other = CachedGraphDataset(dataset, name=f"{args.dataset}_other", max_motifs=motif_budget_other, pe_k=args.et_pe_k)
    dataset_full = CachedGraphDataset(dataset, name=f"{args.dataset}_full", max_motifs=motif_budget_full, pe_k=args.et_pe_k)

    labels = [int(g["y"].item()) for g in dataset]

    from sklearn.model_selection import StratifiedKFold
    for seed in seeds:
        set_seed(seed)
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        
        pw_accs, fg_accs, gin_accs, et_faithful_accs = [], [], [], []
        pw_hists, fg_hists, gin_hists, et_faithful_hists = [], [], [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
            split_data_other = {"train": [dataset_other[i] for i in train_idx], "test": [dataset_other[i] for i in test_idx]}
            split_data_full = {"train": [dataset_full[i] for i in train_idx], "test": [dataset_full[i] for i in test_idx]}

            pw = PairwiseGET(in_dim=in_dim, d=args.hidden_dim, num_classes=num_classes, num_steps=args.num_steps, pe_k=args.et_pe_k, noise_std=args.noise_std)
            pw_acc, pw_hist = train_eval_graph_classification(
                "PairwiseGET",
                pw,
                dataset_other,
                num_classes=num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_pairwise,
                max_grad_norm=0.5,
                seed=seed + fold,
                compile_model=args.compile,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_other,
                split_data=split_data_other,
            )
            pw_accs.append(pw_acc)
            pw_hists.append(pw_hist)

            fg = FullGET(
                in_dim=in_dim,
                d=args.hidden_dim,
                num_classes=num_classes,
                num_steps=args.num_steps,
                R=args.rank,
                lambda_3=args.lambda3,
                lambda_m=0.0,
                pe_k=args.et_pe_k,
                noise_std=args.noise_std,
            )
            fg_acc, fg_hist = train_eval_graph_classification(
                "FullGET",
                fg,
                dataset_full,
                num_classes=num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_full,
                max_grad_norm=0.5,
                seed=seed + 1 + fold,
                compile_model=args.compile,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_full,
                split_data=split_data_full,
            )
            fg_accs.append(fg_acc)
            fg_hists.append(fg_hist)

            et_faithful = ETFaithful(
                in_dim=in_dim,
                d=args.hidden_dim,
                num_classes=num_classes,
                num_steps=args.num_steps,
                num_heads=args.et_num_heads,
                head_dim=args.et_head_dim,
                pe_k=args.et_pe_k,
                K=args.et_memory_slots,
                noise_std=args.noise_std,
            )
            et_faithful_acc, et_faithful_hist = train_eval_graph_classification(
                "ETFaithful",
                et_faithful,
                dataset_other,
                num_classes=num_classes,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_et_faithful,
                max_grad_norm=0.5,
                seed=seed + 11 + fold,
                compile_model=args.compile,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_other,
                split_data=split_data_other,
            )
            et_faithful_accs.append(et_faithful_acc)
            et_faithful_hists.append(et_faithful_hist)

            gin = _try_build_gin(in_dim=in_dim, d=args.hidden_dim, num_classes=num_classes)
            if gin is not None:
                gin_acc, gin_hist = train_eval_graph_classification(
                    "GIN",
                    gin,
                    dataset_other,
                    num_classes=num_classes,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    lr=args.lr_gin,
                    max_grad_norm=1.0,
                    seed=seed + 2 + fold,
                    compile_model=args.compile,
                    use_amp=args.amp,
                    amp_dtype=args.amp_dtype,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    max_motifs=motif_budget_other,
                    split_data=split_data_other,
                )
                gin_accs.append(gin_acc)
                gin_hists.append(gin_hist)
            else:
                gin_accs.append(None)
                gin_hists.append(None)

        def avg_hist(hists):
            if not hists or hists[0] is None: return None
            keys = hists[0].keys()
            out = {k: [] for k in keys}
            for k in keys:
                min_len = min(len(h[k]) for h in hists)
                out[k] = [sum(h[k][i] for h in hists)/len(hists) for i in range(min_len)]
            return out

        results.append(
            {
                "seed": seed,
                "pairwise_acc": float(np.mean(pw_accs)),
                "fullget_acc": float(np.mean(fg_accs)),
                "gin_acc": None if gin_accs[0] is None else float(np.mean(gin_accs)),
                "et_faithful_acc": float(np.mean(et_faithful_accs)),
                "histories": {
                    "pairwise": avg_hist(pw_hists),
                    "fullget": avg_hist(fg_hists),
                    "gin": avg_hist(gin_hists),
                    "et_faithful": avg_hist(et_faithful_hists),
                },
            }
        )

    pw_vals = [r["pairwise_acc"] for r in results]
    fg_vals = [r["fullget_acc"] for r in results]
    et_faithful_vals = [r["et_faithful_acc"] for r in results]
    out = {
        "task": "graph_classification",
        "dataset": args.dataset,
        "summary": {
            "pairwise_mean": mean_std(pw_vals)[0],
            "pairwise_std": mean_std(pw_vals)[1],
            "fullget_mean": mean_std(fg_vals)[0],
            "fullget_std": mean_std(fg_vals)[1],
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

    dataset_other = CachedGraphDataset(dataset, name=f"{args.dataset}_other_anomaly", max_motifs=motif_budget_other, pe_k=args.et_pe_k)
    dataset_full = CachedGraphDataset(dataset, name=f"{args.dataset}_full_anomaly", max_motifs=motif_budget_full, pe_k=args.et_pe_k)

    for rate in label_rates:
        for seed in seeds:
            set_seed(seed)
            split_data_other = build_anomaly_protocol_split(
                dataset_other,
                seed=seed,
                labeled_rate=rate,
                val_ratio=args.anomaly_val_ratio,
                test_ratio=args.anomaly_test_ratio,
            )
            split_data_full = build_anomaly_protocol_split(
                dataset_full,
                seed=seed,
                labeled_rate=rate,
                val_ratio=args.anomaly_val_ratio,
                test_ratio=args.anomaly_test_ratio,
            )
            
            pw = PairwiseGET(in_dim=in_dim, d=args.hidden_dim, num_classes=1, num_steps=args.num_steps, pe_k=args.et_pe_k, noise_std=args.noise_std)
            pw_auc, pw_f1, pw_hist = train_eval_graph_anomaly(
                "PairwiseGET",
                pw,
                dataset_other,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_pairwise,
                max_grad_norm=0.5,
                seed=seed,
                compile_model=args.compile,
                split_data=split_data_other,
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
                pe_k=args.et_pe_k,
                noise_std=args.noise_std,
            )
            fg_auc, fg_f1, fg_hist = train_eval_graph_anomaly(
                "FullGET",
                fg,
                dataset_full,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_full,
                max_grad_norm=0.5,
                seed=seed + 1,
                compile_model=args.compile,
                split_data=split_data_full,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_full,
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
                noise_std=args.noise_std,
            )
            et_faithful_auc, et_faithful_f1, et_faithful_hist = train_eval_graph_anomaly(
                "ETFaithful",
                et_faithful,
                dataset_other,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                lr=args.lr_et_faithful,
                max_grad_norm=0.5,
                seed=seed + 11,
                compile_model=args.compile,
                split_data=split_data_other,
                use_amp=args.amp,
                amp_dtype=args.amp_dtype,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
                max_motifs=motif_budget_other,
            )

            gin = _try_build_gin(in_dim=in_dim, d=args.hidden_dim, num_classes=1)
            if gin is not None:
                gin_auc, gin_f1, gin_hist = train_eval_graph_anomaly(
                    "GIN",
                    gin,
                    dataset_other,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    device=device,
                    lr=args.lr_gin,
                    max_grad_norm=1.0,
                    seed=seed + 2,
                    compile_model=args.compile,
                    split_data=split_data_other,
                    use_amp=args.amp,
                    amp_dtype=args.amp_dtype,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    max_motifs=motif_budget_other,
                )
            else:
                gin_auc, gin_f1, gin_hist = None, None, None

            results[rate].append(
                {
                    "seed": seed,
                    "pairwise_auc": float(pw_auc),
                    "pairwise_f1": float(pw_f1),
                    "fullget_auc": float(fg_auc),
                    "fullget_f1": float(fg_f1),
                    "gin_auc": None if gin_auc is None else float(gin_auc),
                    "gin_f1": None if gin_f1 is None else float(gin_f1),
                    "et_faithful_auc": float(et_faithful_auc),
                    "et_faithful_f1": float(et_faithful_f1),
                    "histories": {
                        "pairwise": pw_hist,
                        "fullget": fg_hist,
                        "gin": gin_hist,
                        "et_faithful": et_faithful_hist,
                    },
                }
            )

    summary = {}
    for rate in label_rates:
        runs = results[rate]
        pw_vals = [r["pairwise_auc"] for r in runs]
        pw_f1_vals = [r["pairwise_f1"] for r in runs]
        fg_vals = [r["fullget_auc"] for r in runs]
        fg_f1_vals = [r["fullget_f1"] for r in runs]
        et_faithful_vals = [r["et_faithful_auc"] for r in runs]
        et_faithful_f1_vals = [r["et_faithful_f1"] for r in runs]
        summary[str(rate)] = {
            "pairwise_mean": mean_std(pw_vals)[0],
            "pairwise_std": mean_std(pw_vals)[1],
            "pairwise_f1_mean": mean_std(pw_f1_vals)[0],
            "pairwise_f1_std": mean_std(pw_f1_vals)[1],
            "fullget_mean": mean_std(fg_vals)[0],
            "fullget_std": mean_std(fg_vals)[1],
            "fullget_f1_mean": mean_std(fg_f1_vals)[0],
            "fullget_f1_std": mean_std(fg_f1_vals)[1],
            "et_faithful_mean": mean_std(et_faithful_vals)[0],
            "et_faithful_std": mean_std(et_faithful_vals)[1],
            "et_faithful_f1_mean": mean_std(et_faithful_f1_vals)[0],
            "et_faithful_f1_std": mean_std(et_faithful_f1_vals)[1],
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
            f"label_rate={rate:.2f} | Pairwise AUC: {s['pairwise_mean']:.4f} ± {s['pairwise_std']:.4f} (F1: {s['pairwise_f1_mean']:.4f}) | "
            f"FullGET AUC: {s['fullget_mean']:.4f} ± {s['fullget_std']:.4f} (F1: {s['fullget_f1_mean']:.4f}) | "
            f"ETFaithful AUC: {s['et_faithful_mean']:.4f} ± {s['et_faithful_std']:.4f} (F1: {s['et_faithful_f1_mean']:.4f})"
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
    parser.add_argument("--noise_std", type=float, default=0.0)
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
