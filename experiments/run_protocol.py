from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get.data.synthetic import collate_graph_samples
from get.utils.seed import seed_everything

from experiments.protocol.data import ListGraphDataset, build_dataset, summarize_splits
from experiments.protocol.registry import TASK_SPECS
from experiments.protocol.training import fit_once, make_loaders


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full protocol runner (Stage 1-4)")
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS.keys()))
    p.add_argument("--model_name", type=str, default="fullget", choices=["fullget", "pairwiseget", "quadratic_only", "external_baseline", "gin", "gcn", "gat"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dataset_root", type=str, default="data")
    p.add_argument("--brec_file", type=str, default="")
    p.add_argument("--tu_name", type=str, default="MUTAG")
    p.add_argument("--cv_folds", type=int, default=1)
    p.add_argument("--max_graphs", type=int, default=0)
    p.add_argument("--ego_hops", type=int, default=1)

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--min_nodes", type=int, default=10)
    p.add_argument("--max_nodes", type=int, default=20)
    p.add_argument("--edge_prob", type=float, default=0.2)
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--max_motifs_per_anchor", type=int, default=8)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--R", type=int, default=2)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--lambda_2", type=float, default=1.0)
    p.add_argument("--lambda_3", type=float, default=10.0)
    p.add_argument("--lambda_m", type=float, default=0.0)
    p.add_argument("--beta_2", type=float, default=1.0)
    p.add_argument("--beta_3", type=float, default=1.0)
    p.add_argument("--beta_m", type=float, default=1.0)
    p.add_argument("--update_damping", type=float, default=0.0)
    p.add_argument("--fixed_step_size", type=float, default=0.1)
    p.add_argument("--armijo_eta0", type=float, default=0.2)
    p.add_argument("--armijo_gamma", type=float, default=0.5)
    p.add_argument("--armijo_c", type=float, default=1e-4)
    p.add_argument("--armijo_max_backtracks", type=int, default=20)
    p.add_argument("--armijo_eval_max_backtracks", type=int, default=5)
    p.add_argument("--inference_mode_train", type=str, default="fixed", choices=["fixed", "armijo"])
    p.add_argument("--inference_mode_eval", type=str, default="armijo", choices=["fixed", "armijo"])
    
    # New flags for GET optimization
    p.add_argument("--use_energy_norm", action="store_true", default=True)
    p.add_argument("--no_energy_norm", action="store_false", dest="use_energy_norm")
    p.add_argument("--agg_mode", type=str, default="softmax", choices=["softmax", "sum"])

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    
    # AMP & Compile
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile_backend", type=str, default="inductor")
    p.add_argument("--compile_dynamic", action="store_true", default=True)
    p.add_argument("--compile_mode", type=str, default="default")
    p.add_argument("--compile_allow_double_backward", action="store_true")
    p.add_argument("--compile_scope", type=str, default="eval_only", choices=["eval_only", "all"])
    
    p.add_argument("--output", type=str, default="outputs/protocol/last_metrics.json")
    return p


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def main():
    args = build_arg_parser().parse_args()
    seed_everything(args.seed)
    args.eval_batch_size = args.eval_batch_size or args.batch_size
    device = resolve_device(args.device)

    spec = TASK_SPECS[args.task]
    items, num_classes = build_dataset(args.task, args)
    total_graphs = sum(len(v) for v in items.values()) if isinstance(items, dict) else len(items)
    if total_graphs < 4:
        raise RuntimeError("Dataset too small after loading; increase --max_graphs or fix dataset path")

    if args.cv_folds > 1 and args.task == "stage2_csl":
        if not isinstance(items, dict):
            raise RuntimeError("stage2_csl cross-validation requires split-preserving official splits")

        trainval_items = list(items["train"]) + list(items["val"])
        test_items = list(items["test"])
        if len(trainval_items) < args.cv_folds:
            raise RuntimeError(f"Not enough train/val samples ({len(trainval_items)}) for cv_folds={args.cv_folds}")

        trainval_labels = np.array([int(sample["y"].view(-1)[0].item()) for sample in trainval_items])
        try:
            splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
            fold_iter = splitter.split(np.arange(len(trainval_items)), trainval_labels)
        except ValueError:
            splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
            fold_iter = splitter.split(np.arange(len(trainval_items)))
        fold = []
        for i, (train_idx, val_idx) in enumerate(fold_iter, start=1):
            tr_items = [trainval_items[j] for j in train_idx.tolist()]
            va_items = [trainval_items[j] for j in val_idx.tolist()]
            tr = DataLoader(ListGraphDataset(tr_items), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_samples, num_workers=args.num_workers, pin_memory=args.pin_memory)
            va = DataLoader(ListGraphDataset(va_items), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples, num_workers=args.num_workers, pin_memory=args.pin_memory)
            te = DataLoader(ListGraphDataset(test_items), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples, num_workers=args.num_workers, pin_memory=args.pin_memory)
            m = fit_once(args, spec.task_type, num_classes, tr, va, te, device)
            m["fold"] = i
            m["split_stats"] = summarize_splits({"train": tr_items, "val": va_items, "test": test_items}, task_type=spec.task_type)
            fold.append(m)
        accs = [f["test"].get("acc", 0.0) for f in fold]
        losses = [f["test"].get("loss", 0.0) for f in fold]
        result = {
            "task": args.task,
            "cv_folds": args.cv_folds,
            "fold_metrics": fold,
            "split_stats": summarize_splits(items, task_type=spec.task_type),
            "summary": {
                "test_metric_mean": float(np.mean(accs) if accs else 0.0),
                "test_metric_std": float(np.std(accs) if accs else 0.0),
                "test_loss_mean": float(np.mean(losses) if losses else 0.0),
                "test_loss_std": float(np.std(losses) if losses else 0.0),
            },
        }
    else:
        if isinstance(items, dict):
            tr = DataLoader(
                ListGraphDataset(items["train"]),
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=collate_graph_samples,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            va = DataLoader(
                ListGraphDataset(items["val"]),
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=collate_graph_samples,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            te = DataLoader(
                ListGraphDataset(items["test"]),
                batch_size=args.eval_batch_size,
                shuffle=False,
                collate_fn=collate_graph_samples,
                num_workers=args.num_workers,
                pin_memory=args.pin_memory,
            )
            split_stats = summarize_splits(items, task_type=spec.task_type)
        else:
            tr, va, te, split_stats = make_loaders(items, args, task_type=spec.task_type, return_split_stats=True)
        m = fit_once(args, spec.task_type, num_classes, tr, va, te, device)
        result = {"task": args.task, "stage": spec.stage, "metrics": m, "split_stats": split_stats}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
