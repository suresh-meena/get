from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Tuple, List, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from get.data import (
    SyntheticGraphDataset,
    collate_graph_samples,
    build_dataset,
    ListGraphDataset,
    split_items,
    summarize_splits,
    get_k_fold_splits,
)
from get.models import build_model
from get.trainers import UnifiedTrainer
from get.utils import maybe_compile_model, seed_everything


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders_from_items(
    train_items: List[Dict],
    val_items: List[Dict],
    test_items: List[Dict],
    tr_cfg: DictConfig,
    task_type: str,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    num_workers = int(tr_cfg.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()
    persistent_workers = num_workers > 0
    batch_size = int(tr_cfg.batch_size)
    eval_batch_size = int(tr_cfg.get("eval_batch_size", batch_size))

    train_ds = ListGraphDataset(train_items)
    val_ds = ListGraphDataset(val_items)
    test_ds = ListGraphDataset(test_items)
    split_stats = summarize_splits({"train": train_items, "val": val_items, "test": test_items}, task_type=task_type)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_graph_samples,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_graph_samples,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_graph_samples,
    )
    return train_loader, val_loader, test_loader, split_stats


def _run_single_experiment(
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str,
    num_classes: int,
    device: torch.device,
) -> Dict:
    # Inject task info into config for the model factory
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    run_cfg["task_type"] = task_type
    run_cfg["num_classes"] = num_classes
    run_cfg["in_dim"] = int(cfg.dataset.in_dim)
    
    # Build model using centralized factory
    model = build_model(DictConfig(run_cfg)).to(device)
    
    compile_cfg = cfg.experiment.get("compile", {"enabled": False})
    compile_scope = str(compile_cfg.get("scope", "eval_only")).lower()
    eval_model = model
    
    if bool(compile_cfg.get("enabled", False)):
        if compile_scope == "all":
            if getattr(model, "requires_double_backward", False):
                raise ValueError("compile.scope='all' is unsupported for GET/ET training because torch.compile does not currently support double backward. Use compile.scope='eval_only'.")
            model = maybe_compile_model(model, compile_cfg)
            eval_model = model
        elif compile_scope == "eval_only":
            eval_compile_cfg = dict(compile_cfg)
            eval_compile_cfg["allow_double_backward"] = True
            eval_model = maybe_compile_model(model, eval_compile_cfg)

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["task_type"] = task_type
    trainer_cfg["num_classes"] = num_classes
    trainer_cfg["epochs"] = int(cfg.trainer.epochs)

    trainer = UnifiedTrainer(model=model, eval_model=eval_model, device=device, trainer_cfg=trainer_cfg)
    return trainer.fit(train_loader, val_loader, test_loader)


def run_from_cfg(cfg: DictConfig) -> Dict:
    seed_everything(int(cfg.seed))
    device = _resolve_device(str(cfg.experiment.get("device", "auto")))
    base_seed = int(cfg.seed)
    ds = cfg.dataset
    dataset_name = str(ds.get("name", "synthetic")).lower()
    task_type = str(ds.get("task_type", "binary"))
    
    num_runs = int(cfg.get("num_runs", 1))
    num_folds = int(cfg.get("num_folds", 0))
    all_run_metrics = []

    if num_folds > 1:
        print(f"\n>>> Starting {num_folds}-fold Cross-Validation (Dataset: {dataset_name}, Seed: {base_seed})")
        # Real-world or protocol task items needed
        if dataset_name.startswith("synthetic"):
             # For synthetic, we generate a fixed set for CV
             full_ds = SyntheticGraphDataset(num_graphs=int(ds.num_train_graphs), seed=base_seed, **{k: v for k, v in ds.items() if k not in ["name", "num_train_graphs", "num_val_graphs", "num_test_graphs"]})
             items = full_ds.items
             num_classes = 1
        else:
             ds_run = OmegaConf.to_container(ds, resolve=True)
             ds_run["seed"] = base_seed
             res, num_classes = build_dataset(dataset_name, DictConfig(ds_run))
             if isinstance(res, dict):
                 raise ValueError("k-fold CV is not supported for datasets that are already split (e.g. ZINC, MolHIV).")
             items = res

        folds = get_k_fold_splits(items, num_folds=num_folds, seed=base_seed, task_type=task_type)
        for fold_idx, (tr_items, va_items, te_items) in enumerate(folds):
            print(f"\n>>> Fold {fold_idx + 1}/{num_folds}")
            tr_loader, va_loader, te_loader, split_stats = _build_loaders_from_items(tr_items, va_items, te_items, cfg.trainer, task_type)
            metrics = _run_single_experiment(cfg, tr_loader, va_loader, te_loader, task_type, num_classes, device)
            metrics["fold_idx"] = fold_idx
            metrics["split_stats"] = split_stats
            all_run_metrics.append(metrics)
    else:
        # Standard multi-seed runs
        for run_idx in range(num_runs):
            current_seed = base_seed + run_idx
            seed_everything(current_seed)
            print(f"\n>>> Starting Run {run_idx + 1}/{num_runs} (Seed: {current_seed})")

            if dataset_name.startswith("synthetic"):
                # ... same synthetic logic ...
                train_ds = SyntheticGraphDataset(num_graphs=int(ds.num_train_graphs), seed=current_seed, in_dim=int(ds.in_dim), min_nodes=int(ds.min_nodes), max_nodes=int(ds.max_nodes), edge_prob=float(ds.edge_prob), max_motifs_per_anchor=int(ds.max_motifs_per_anchor))
                val_ds = SyntheticGraphDataset(num_graphs=int(ds.num_val_graphs), seed=current_seed + 1, in_dim=int(ds.in_dim), min_nodes=int(ds.min_nodes), max_nodes=int(ds.max_nodes), edge_prob=float(ds.edge_prob), max_motifs_per_anchor=int(ds.max_motifs_per_anchor))
                test_ds = SyntheticGraphDataset(num_graphs=int(ds.num_test_graphs), seed=current_seed + 2, in_dim=int(ds.in_dim), min_nodes=int(ds.min_nodes), max_nodes=int(ds.max_nodes), edge_prob=float(ds.edge_prob), max_motifs_per_anchor=int(ds.max_motifs_per_anchor))
                tr_items, va_items, te_items = train_ds.items, val_ds.items, test_ds.items
                num_classes = 1
            else:
                ds_run = OmegaConf.to_container(ds, resolve=True)
                ds_run["seed"] = current_seed
                res, num_classes = build_dataset(dataset_name, DictConfig(ds_run))
                if isinstance(res, dict):
                    tr_items, va_items, te_items = res["train"], res["val"], res["test"]
                else:
                    tr_items, va_items, te_items = split_items(res, seed=current_seed, train_ratio=float(ds.get("train_ratio", 0.7)), val_ratio=float(ds.get("val_ratio", 0.15)), task_type=task_type)
            
            tr_loader, va_loader, te_loader, split_stats = _build_loaders_from_items(tr_items, va_items, te_items, cfg.trainer, task_type)
            metrics = _run_single_experiment(cfg, tr_loader, va_loader, te_loader, task_type, num_classes, device)
            metrics["run_idx"] = run_idx
            metrics["seed"] = current_seed
            metrics["split_stats"] = split_stats
            all_run_metrics.append(metrics)

    if len(all_run_metrics) > 1:
        score_key = "acc" if task_type != "regression" else "mae"
        run_scores = [m["test"][score_key] for m in all_run_metrics]
        final_metrics = {
            "num_runs_or_folds": len(all_run_metrics),
            "all_results": all_run_metrics,
            "final_summary": {
                f"test_{score_key}_mean": float(statistics.mean(run_scores)),
                f"test_{score_key}_std": float(statistics.pstdev(run_scores) if len(run_scores) > 1 else 0.0),
            }
        }
        return final_metrics
    else:
        return all_run_metrics[0]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_name = str(cfg.dataset.get("name", "synthetic")).lower()
    metrics = run_from_cfg(cfg)
    print(f"\nFinal Metrics for {dataset_name}:")
    if "final_summary" in metrics:
        print(json.dumps(metrics["final_summary"], indent=2))
    else:
        print(json.dumps(metrics["test"], indent=2))
    
    out_dir = Path(cfg.get("output_dir", "outputs/unified"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"metrics_{dataset_name}.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
