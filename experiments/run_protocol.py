from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get.utils.seed import seed_everything
from get.data import (
    ListGraphDataset, 
    build_dataset, 
    summarize_splits, 
    TASK_SPECS, 
    get_k_fold_splits, 
    collate_graph_samples,
    split_items
)
from get.models import build_model
from get.trainers import UnifiedTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full protocol runner (Stage 1-4)")
    p.add_argument("--task", type=str, required=True, choices=sorted(TASK_SPECS.keys()))
    p.add_argument("--model_name", type=str, default="fullget", choices=["fullget", "pairwiseget", "quadratic_only", "et", "etfaithful", "gt", "bwgnn", "external_baseline", "gin", "gcn", "gat"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dataset_root", type=str, default="data")
    p.add_argument("--tu_name", type=str, default="MUTAG")
    p.add_argument("--cv_folds", type=int, default=1)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--max_motifs_per_anchor", type=int, default=8)
    p.add_argument("--max_graphs", type=int, default=0)
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
    p.add_argument("--beta_2", type=float, default=1.0)
    p.add_argument("--beta_3", type=float, default=1.0)
    p.add_argument("--beta_m", type=float, default=1.0)
    
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
    
    p.add_argument("--output", type=str, default="outputs/protocol/last_metrics.json")
    return p


def run_experiment(args: argparse.Namespace, tr_items, va_items, te_items, task_type: str, num_classes: int, device: torch.device) -> Dict:
    # Map argparse to the factory config format
    cfg_dict = vars(args)
    # Factory expects some specific keys
    cfg_dict["task_type"] = task_type
    cfg_dict["num_classes"] = num_classes
    
    # Convert to DictConfig for factory
    cfg = DictConfig(cfg_dict)
    
    model = build_model(cfg).to(device)
    
    trainer_cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "use_amp": args.use_amp,
        "task_type": task_type,
        "num_classes": num_classes
    }
    
    train_loader = DataLoader(ListGraphDataset(tr_items), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_samples)
    val_loader = DataLoader(ListGraphDataset(va_items), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graph_samples)
    test_loader = DataLoader(ListGraphDataset(te_items), batch_size=args.batch_size, shuffle=False, collate_fn=collate_graph_samples)
    
    trainer = UnifiedTrainer(model=model, device=device, trainer_cfg=trainer_cfg)
    return trainer.fit(train_loader, val_loader, test_loader)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    spec = TASK_SPECS[args.task]
    task_type = spec.task_type
    
    # Load dataset
    items, num_classes = build_dataset(args.task, args)
    
    all_results = []
    
    if args.cv_folds > 1:
        print(f"\n>>> Starting {args.cv_folds}-fold Cross-Validation for {args.task} (Model: {args.model_name})")
        # Ensure items is a list for k-fold
        if isinstance(items, dict):
            # If already split, we merge back if allowed or raise error
            if args.task in {"stage3_zinc", "stage3_molhiv"}:
                 raise ValueError("k-fold CV is not supported for datasets with official fixed splits.")
            items_list = items["train"] + items["val"] + items["test"]
        else:
            items_list = items
            
        folds = get_k_fold_splits(items_list, num_folds=args.cv_folds, seed=args.seed, task_type=task_type)
        for fold_idx, (tr, va, te) in enumerate(folds):
            print(f"\n>>> Fold {fold_idx + 1}/{args.cv_folds}")
            metrics = run_experiment(args, tr, va, te, task_type, num_classes, device)
            metrics["fold"] = fold_idx + 1
            all_results.append(metrics)
            
        # Summary
        import statistics
        score_key = "acc" if task_type != "regression" else "mae"
        scores = [r["test"][score_key] for r in all_results]
        summary = {
            "test_metric_mean": statistics.mean(scores),
            "test_metric_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "num_folds": args.cv_folds
        }
        final_output = {
            "task": args.task,
            "model": args.model_name,
            "summary": summary,
            "all_results": all_results
        }
    else:
        print(f"\n>>> Starting single run for {args.task} (Model: {args.model_name})")
        if isinstance(items, dict):
            tr, va, te = items["train"], items["val"], items["test"]
        else:
            tr, va, te = split_items(items, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio, task_type=task_type)
            
        metrics = run_experiment(args, tr, va, te, task_type, num_classes, device)
        final_output = {
            "task": args.task,
            "model": args.model_name,
            "metrics": metrics
        }
        
    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final_output, indent=2))
    print(f"\nResults saved to {args.output}")
    if "summary" in final_output:
        print(json.dumps(final_output["summary"], indent=2))
    else:
        print(json.dumps(final_output["metrics"]["test"], indent=2))


if __name__ == "__main__":
    main()
