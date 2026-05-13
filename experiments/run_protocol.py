from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
try:
    from torch_geometric.loader import PrefetchLoader
    _has_prefetch_loader = True
except Exception:
    _has_prefetch_loader = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get.utils.seed import seed_everything
from get.data import (
    ListGraphDataset,
    build_dataset,
    infer_edge_attr_dim,
    summarize_splits,
    TASK_SPECS,
    get_k_fold_splits,
    collate_graph_samples,
    split_items,
)
from get.models import build_model
from get.trainers import UnifiedTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full protocol runner (Stage 1-4)")
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS.keys()))
    p.add_argument(
        "--model_name",
        type=str,
        default="fullget",
        choices=[
            "fullget", "pairwiseget", "quadratic_only",
            "get_ham_global", "get_ham_cls", "get_ham_full",
            "et", "etfaithful", "gt", "bwgnn", "external_baseline",
            "gin", "gcn", "gat",
        ],
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dataset_root", type=str, default="data")
    p.add_argument("--brec_file", type=str, default="")
    p.add_argument("--tu_name", type=str, default="MUTAG")
    p.add_argument("--cv_folds", type=int, default=1)
    p.add_argument("--num_runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--max_motifs_per_anchor", type=int, default=8)
    p.add_argument("--max_graphs", type=int, default=0)
    p.add_argument("--min_nodes", type=int, default=6)
    p.add_argument("--max_nodes", type=int, default=12)
    p.add_argument("--edge_prob", type=float, default=0.3)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)

    # Model parameters
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--num_blocks", type=int, default=1)

    # GET specific
    p.add_argument("--lambda_2", type=float, default=1.0)
    p.add_argument("--lambda_3", type=float, default=10.0)
    p.add_argument("--lambda_m", type=float, default=1.0)
    p.add_argument("--lambda_g", type=float, default=0.0)
    p.add_argument("--beta_2", type=float, default=1.0)
    p.add_argument("--beta_3", type=float, default=1.0)
    p.add_argument("--beta_m", type=float, default=1.0)
    p.add_argument("--beta_g", type=float, default=1.0)
    p.add_argument("--max_global_nodes", type=int, default=512)

    # ET specific
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--multiplier", type=float, default=4.0)
    p.add_argument("--pos_k", type=int, default=15)

    # Training parameters
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="use_amp")
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_pyg_prefetch_loader", action="store_true")
    p.add_argument("--output_file", type=str, default="")
    p.add_argument("--output", type=str, default="")
    p.add_argument("--output_dir", type=str, default="outputs/protocol")
    return p


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _get_peak_cuda_memory() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _score_key_for_task(task_type: str) -> str:
    if task_type == "regression":
        return "mae"
    if task_type in {"binary", "node_binary", "multilabel"}:
        return "auc"
    return "acc"


def _maybe_wrap_prefetch(loader, *, use_prefetch: bool, device: torch.device):
    if not use_prefetch:
        return loader
    if device.type != "cuda":
        return loader
    if not _has_prefetch_loader:
        return loader
    return PrefetchLoader(loader, device=device)


def _evaluate_brec_by_category(
    trainer: UnifiedTrainer,
    args: argparse.Namespace,
    te_items,
) -> Dict[str, Dict[str, float]]:
    if not te_items:
        return {}

    grouped: Dict[int, List] = {}
    for sample in te_items:
        if "brec_category_id" not in sample:
            continue
        cat_id = int(sample["brec_category_id"].view(-1)[0].item())
        grouped.setdefault(cat_id, []).append(sample)

    if not grouped:
        return {}

    id_to_name = getattr(args, "_brec_category_names", {}) or {}
    out: Dict[str, Dict[str, float]] = {}
    num_workers = int(getattr(args, "num_workers", 0))
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    for cat_id, samples in grouped.items():
        name = str(id_to_name.get(cat_id, f"category_{cat_id}"))
        loader = torch.utils.data.DataLoader(
            ListGraphDataset(samples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_graph_samples,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        stats = trainer._run_epoch(loader, train=False)
        out[name] = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}
    return out


def run_experiment(args: argparse.Namespace, tr_items, va_items, te_items, task_type: str, num_classes: int, device: torch.device) -> Dict:
    cfg_dict = dict(vars(args))
    cfg_dict["task_type"] = task_type
    cfg_dict["num_classes"] = num_classes
    cfg_dict["edge_attr_dim"] = infer_edge_attr_dim({"train": tr_items, "val": va_items, "test": te_items})

    from omegaconf import DictConfig
    cfg = DictConfig(cfg_dict)

    model = build_model(cfg).to(device)
    parameter_count = _count_params(model)

    trainer_cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "use_amp": args.use_amp,
        "task_type": task_type,
        "num_classes": num_classes,
    }

    num_workers = int(getattr(args, "num_workers", 0))
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None

    train_loader = torch.utils.data.DataLoader(
        ListGraphDataset(tr_items), batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_graph_samples, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = torch.utils.data.DataLoader(
        ListGraphDataset(va_items), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_graph_samples, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = torch.utils.data.DataLoader(
        ListGraphDataset(te_items), batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_graph_samples, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    use_prefetch = bool(getattr(args, "use_pyg_prefetch_loader", False))
    train_loader = _maybe_wrap_prefetch(train_loader, use_prefetch=use_prefetch, device=device)
    val_loader = _maybe_wrap_prefetch(val_loader, use_prefetch=use_prefetch, device=device)
    test_loader = _maybe_wrap_prefetch(test_loader, use_prefetch=use_prefetch, device=device)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer = UnifiedTrainer(model=model, device=device, trainer_cfg=trainer_cfg)
    fit_result = trainer.fit(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time

    history = fit_result.pop("history", {"train": [], "val": []})

    model_name = str(getattr(args, "model_name", "fullget")).lower()
    is_ham = model_name.startswith("get_ham_")
    enabled_branches = {
        "pairwise": True,
        "motif": model_name in ("fullget", "get_ham_global", "get_ham_cls", "get_ham_full"),
        "memory": model_name in ("fullget", "get_ham_global", "get_ham_cls", "get_ham_full"),
        "global_attention": model_name in ("get_ham_global", "get_ham_cls", "get_ham_full"),
        "cls_token": model_name in ("get_ham_cls", "get_ham_full"),
        "structural_memory": False,
        "dynamic_edges": False,
    }

    result = {
        "train": fit_result.get("final_train", {}),
        "val": fit_result.get("final_val", {}),
        "test": fit_result["test"],
        "best_val_score": fit_result["best_val_score"],
        "epochs_ran": fit_result["epochs_ran"],
        "history": history,
        "parameter_count": parameter_count,
        "runtime_seconds": elapsed,
        "peak_cuda_memory_mb": _get_peak_cuda_memory(),
        "enabled_branches": enabled_branches,
        "solver_state_keys": ["H"],
        "num_inference_steps": int(getattr(args, "num_steps", 8)),
        "compile_scope": "eval_only",
    }
    if str(getattr(args, "task", "")) == "stage2_brec":
        by_cat = _evaluate_brec_by_category(trainer, args, te_items)
        if by_cat:
            result["test_by_category"] = by_cat
    return result


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    spec = TASK_SPECS[args.task]
    task_type = spec.task_type

    items, num_classes = build_dataset(args.task, args)

    all_run_results = []
    base_seed = args.seed

    stage = spec.stage
    task_key = args.task

    for run_idx in range(args.num_runs):
        current_seed = base_seed + run_idx
        seed_everything(current_seed)

        if args.cv_folds > 1:
            if isinstance(items, dict):
                raise ValueError("k-fold CV is not supported for datasets with official fixed splits.")
            folds = get_k_fold_splits(items, num_folds=args.cv_folds, seed=current_seed, task_type=task_type)
            fold_results = []
            for fold_idx, (tr, va, te) in enumerate(folds):
                metrics = run_experiment(args, tr, va, te, task_type, num_classes, device)
                metrics["fold"] = fold_idx + 1
                fold_results.append(metrics)

            import statistics
            score_key = _score_key_for_task(task_type)
            scores = [r["test"][score_key] for r in fold_results]

            # Build split_stats for all folds
            split_stats = summarize_splits(
                {"train": tr, "val": va, "test": te}, task_type=task_type
            )

            run_result = {
                "task": task_key,
                "model_name": args.model_name,
                "seed": current_seed,
                "num_runs": args.num_runs,
                "run_idx": run_idx,
                "split_stats": split_stats,
                "runtime_config": {"device": str(device), **vars(args)},
                "parameter_count": fold_results[0]["parameter_count"],
                "fold_results": fold_results,
                "summary": {
                    f"test_{score_key}_mean": statistics.mean(scores),
                    f"test_{score_key}_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "num_folds": args.cv_folds,
                },
                "runtime_seconds": sum(r["runtime_seconds"] for r in fold_results),
                "peak_cuda_memory_mb": max(r["peak_cuda_memory_mb"] for r in fold_results),
            }
        else:
            if isinstance(items, dict):
                tr, va, te = items["train"], items["val"], items["test"]
            else:
                tr, va, te = split_items(items, seed=current_seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio, task_type=task_type)

            split_stats = summarize_splits(
                {"train": tr, "val": va, "test": te}, task_type=task_type
            )
            metrics = run_experiment(args, tr, va, te, task_type, num_classes, device)

            run_result = {
                "task": task_key,
                "model_name": args.model_name,
                "seed": current_seed,
                "num_runs": args.num_runs,
                "run_idx": run_idx,
                "split_stats": split_stats,
                "runtime_config": {"device": str(device), **vars(args)},
                "parameter_count": metrics["parameter_count"],
                "train": metrics["train"],
                "val": metrics["val"],
                "test": metrics["test"],
                "test_by_category": metrics.get("test_by_category", {}),
                "history": metrics["history"],
                "runtime_seconds": metrics["runtime_seconds"],
                "peak_cuda_memory_mb": metrics["peak_cuda_memory_mb"],
            }

        all_run_results.append(run_result)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    explicit_output = str(getattr(args, "output_file", "") or getattr(args, "output", "")).strip()
    if explicit_output:
        out_path = Path(explicit_output)
        if not out_path.is_absolute():
            out_path = out_dir / out_path
    else:
        out_path = out_dir / f"{stage}_{task_key}_{args.model_name}_seed{base_seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_runs > 1:
        import statistics
        score_key = _score_key_for_task(task_type)
        run_scores = []
        for r in all_run_results:
            if "fold_results" in r:
                run_scores.append(r["summary"][f"test_{score_key}_mean"])
            else:
                run_scores.append(r["test"][score_key])

        final_output = {
            "task": task_key,
            "model_name": args.model_name,
            "seed": base_seed,
            "num_runs": args.num_runs,
            "all_results": all_run_results,
            "summary": {
                f"test_{score_key}_mean": statistics.mean(run_scores),
                f"test_{score_key}_std": statistics.stdev(run_scores) if len(run_scores) > 1 else 0.0,
            },
            "runtime_config": {"device": str(device), "num_runs": args.num_runs, "cv_folds": args.cv_folds},
            "artifact_paths": {"results": str(out_path)},
        }
    else:
        final_output = all_run_results[0]

    out_path.write_text(json.dumps(final_output, indent=2))
    print(f"\nResults saved to {out_path}")

    # Update or create sweep manifest
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"experiments": []}
    manifest["experiments"].append({
        "file": str(out_path.relative_to(out_dir)) if out_path.is_relative_to(out_dir) else str(out_path),
        "task": task_key,
        "model": args.model_name,
        "seed": base_seed,
        "num_runs": args.num_runs,
        "num_folds": args.cv_folds,
        "timestamp": str(__import__("datetime").datetime.now()),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if "summary" in final_output:
        print(json.dumps(final_output["summary"], indent=2))
    else:
        print(json.dumps(final_output["test"], indent=2))


if __name__ == "__main__":
    main()
