from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Dict, Tuple, List, Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    from torch_geometric.loader import PrefetchLoader
    _has_prefetch_loader = True
except ImportError:
    _has_prefetch_loader = False

from get.data import (
    SyntheticGraphDataset,
    collate_graph_samples,
    build_dataset,
    ListGraphDataset,
    infer_edge_attr_dim,
    split_items,
    summarize_splits,
    get_k_fold_splits,
)
from get.models import build_model
from get.trainers import UnifiedTrainer
from get.utils import seed_everything, move_batch_to_device
from experiments.common import resolve_device, score_key_for_task


def _build_loaders_from_items(
    train_items: List[Dict],
    val_items: List[Dict],
    test_items: List[Dict],
    tr_cfg: DictConfig,
    task_type: str,
    device: torch.device | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    num_workers = int(tr_cfg.get("num_workers", 0))
    if device is not None and device.type != "cuda":
        # CPU runs (including unit-test subprocess runs) are typically slower
        # and less stable with multiprocessing DataLoaders.
        num_workers = 0
    pin_memory = bool(device is not None and device.type == "cuda")
    persistent_workers = num_workers > 0
    prefetch_factor = 2 if num_workers > 0 else None
    batch_size = int(tr_cfg.batch_size)
    eval_batch_size = int(tr_cfg.get("eval_batch_size", batch_size))
    use_prefetch = bool(tr_cfg.get("use_pyg_prefetch_loader", False)) and _has_prefetch_loader and device is not None and device.type == "cuda"
    dataset_on_gpu = bool(tr_cfg.get("dataset_on_gpu", False)) and device is not None and device.type == "cuda"

    train_ds = ListGraphDataset(train_items)
    val_ds = ListGraphDataset(val_items)
    test_ds = ListGraphDataset(test_items)
    split_stats = summarize_splits({"train": train_items, "val": val_items, "test": test_items}, task_type=task_type)

    if dataset_on_gpu and device is not None:
        train_ds.samples = [move_batch_to_device(s, device) for s in train_ds.samples]
        val_ds.samples = [move_batch_to_device(s, device) for s in val_ds.samples]
        test_ds.samples = [move_batch_to_device(s, device) for s in test_ds.samples]
        pin_memory = False
        num_workers = 0
        persistent_workers = False
        prefetch_factor = None

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        collate_fn=collate_graph_samples,
    )
    val_loader = DataLoader(
        val_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        collate_fn=collate_graph_samples,
    )
    test_loader = DataLoader(
        test_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        collate_fn=collate_graph_samples,
    )

    if use_prefetch and _has_prefetch_loader:
        train_loader = PrefetchLoader(train_loader, device=device)
        val_loader = PrefetchLoader(val_loader, device=device)
        test_loader = PrefetchLoader(test_loader, device=device)

    return train_loader, val_loader, test_loader, split_stats


def _run_single_experiment(
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str,
    num_classes: int,
    device: torch.device,
    edge_attr_dim: int = 0,
) -> Dict:
    run_cfg = OmegaConf.to_container(cfg, resolve=True)
    run_cfg["task_type"] = task_type
    run_cfg["num_classes"] = num_classes
    run_cfg["in_dim"] = int(cfg.dataset.in_dim)
    run_cfg["edge_attr_dim"] = int(edge_attr_dim)

    model = build_model(DictConfig(run_cfg)).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    eval_model = model

    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg["task_type"] = task_type
    trainer_cfg["num_classes"] = num_classes
    trainer_cfg["epochs"] = int(cfg.trainer.epochs)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer = UnifiedTrainer(model=model, eval_model=eval_model, device=device, trainer_cfg=trainer_cfg)
    fit_result = trainer.fit(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time

    history = fit_result.pop("history", {"train": [], "val": []})

    peak_memory = 0.0
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    model_name = str(run_cfg.get("model_name", "fullget")).lower()
    is_ham = model_name.startswith("get_ham_")
    enabled_branches = {
        "pairwise": True,
        "motif": model_name in ("fullget", "get_ham_global"),
        "memory": model_name in ("fullget", "get_ham_global"),
        "global_attention": model_name in ("get_ham_global"),
    }

    return {
        "train": fit_result.get("final_train", {}),
        "val": fit_result.get("final_val", {}),
        "test": fit_result["test"],
        "best_val_score": fit_result["best_val_score"],
        "epochs_ran": fit_result["epochs_ran"],
        "history": history,
        "parameter_count": parameter_count,
        "runtime_seconds": elapsed,
        "peak_cuda_memory_mb": peak_memory,
        "enabled_branches": enabled_branches,
        "solver_state_keys": ["H"],
        "num_inference_steps": int(run_cfg.get("num_steps", 8)),
    }


def run_from_cfg(cfg: DictConfig) -> Dict:
    seed_everything(int(cfg.seed))
    device = resolve_device(str(cfg.experiment.get("device", "auto")))
    base_seed = int(cfg.seed)
    ds = cfg.dataset
    dataset_name = str(ds.get("name", "synthetic")).lower()
    task_type = str(ds.get("task_type", "binary"))

    num_runs = int(cfg.get("num_runs", 1))
    num_folds = int(cfg.get("num_folds", 0))
    all_run_metrics = []

    if num_folds > 1:
        if dataset_name.startswith("synthetic"):
            full_ds = SyntheticGraphDataset(num_graphs=int(ds.num_train_graphs), seed=base_seed, **{k: v for k, v in ds.items() if k not in ["name", "num_train_graphs", "num_val_graphs", "num_test_graphs"]})
            items = full_ds.items
            num_classes = 1
        else:
            ds_run = OmegaConf.to_container(ds, resolve=True)
            ds_run["seed"] = base_seed
            res, num_classes = build_dataset(dataset_name, DictConfig(ds_run))
            if isinstance(res, dict):
                raise ValueError("k-fold CV is not supported for datasets that are already split.")
            items = res

        folds = get_k_fold_splits(items, num_folds=num_folds, seed=base_seed, task_type=task_type)
        for fold_idx, (tr_items, va_items, te_items) in enumerate(folds):
            tr_loader, va_loader, te_loader, split_stats = _build_loaders_from_items(tr_items, va_items, te_items, cfg.trainer, task_type, device=device)
            edge_attr_dim = infer_edge_attr_dim({"train": tr_items, "val": va_items, "test": te_items})
            metrics = _run_single_experiment(cfg, tr_loader, va_loader, te_loader, task_type, num_classes, device, edge_attr_dim=edge_attr_dim)
            metrics["fold_idx"] = fold_idx
            metrics["split_stats"] = split_stats
            all_run_metrics.append(metrics)
    else:
        for run_idx in range(num_runs):
            current_seed = base_seed + run_idx
            seed_everything(current_seed)

            if dataset_name.startswith("synthetic"):
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

            tr_loader, va_loader, te_loader, split_stats = _build_loaders_from_items(tr_items, va_items, te_items, cfg.trainer, task_type, device=device)
            edge_attr_dim = infer_edge_attr_dim({"train": tr_items, "val": va_items, "test": te_items})
            metrics = _run_single_experiment(cfg, tr_loader, va_loader, te_loader, task_type, num_classes, device, edge_attr_dim=edge_attr_dim)
            metrics["run_idx"] = run_idx
            metrics["seed"] = current_seed
            metrics["split_stats"] = split_stats
            all_run_metrics.append(metrics)

    if len(all_run_metrics) > 1:
        score_key = score_key_for_task(task_type)
        run_scores = [m["test"][score_key] for m in all_run_metrics]
        final_metrics = {
            "num_runs_or_folds": len(all_run_metrics),
            "all_results": all_run_metrics,
            "final_summary": {
                f"test_{score_key}_mean": float(statistics.mean(run_scores)),
                f"test_{score_key}_std": float(statistics.pstdev(run_scores) if len(run_scores) > 1 else 0.0),
            },
        }
        return final_metrics
    else:
        return all_run_metrics[0]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_name = str(cfg.dataset.get("name", "synthetic")).lower()
    metrics = run_from_cfg(cfg)

    out_dir = Path(cfg.get("output_dir", "outputs/unified"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"metrics_{dataset_name}.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if "final_summary" in metrics:
        print(f"\nFinal Metrics for {dataset_name}:")
        print(json.dumps(metrics["final_summary"], indent=2))
    elif "test" in metrics:
        print(f"\nResults saved to {out_file}")
        print(json.dumps(metrics["test"], indent=2))
    else:
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
