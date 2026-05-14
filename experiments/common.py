from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from get.data import ListGraphDataset, collate_graph_samples, summarize_splits
from get.trainers import UnifiedTrainer


def score_key_for_task(task_type: str) -> str:
    if task_type == "regression":
        return "mae"
    if task_type in {"binary", "node_binary", "multilabel"}:
        return "auc"
    return "acc"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but not available")
    return torch.device(device_name)


def make_loader_kwargs(num_workers: int, device: torch.device | None = None) -> Dict[str, object]:
    num_workers = int(num_workers)
    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda" if device is not None else torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


def build_data_loaders(
    train_items: List[Any],
    val_items: List[Any],
    test_items: List[Any],
    batch_size: int,
    eval_batch_size: Optional[int] = None,
    num_workers: int = 8,
    device: torch.device | None = None,
    task_type: str | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    train_ds = ListGraphDataset(train_items)
    val_ds = ListGraphDataset(val_items)
    test_ds = ListGraphDataset(test_items)
    eval_bs = eval_batch_size or batch_size
    loader_kwargs = make_loader_kwargs(num_workers, device=device)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=len(train_items) > 0, collate_fn=collate_graph_samples, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_graph_samples, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_graph_samples, **loader_kwargs)
    split_stats = summarize_splits({"train": train_items, "val": val_items, "test": test_items}, task_type=task_type)
    return train_loader, val_loader, test_loader, split_stats


def fit_unified_trainer(
    *,
    model: torch.nn.Module,
    device: torch.device,
    trainer_cfg: Dict[str, Any],
    train_loader,
    val_loader,
    test_loader,
    eval_model: torch.nn.Module | None = None,
) -> Tuple[UnifiedTrainer, Dict[str, Any], Dict[str, Any], float, float]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    trainer = UnifiedTrainer(model=model, eval_model=eval_model, device=device, trainer_cfg=trainer_cfg)
    fit_result = trainer.fit(train_loader, val_loader, test_loader)
    elapsed = time.time() - start_time
    history = fit_result.pop("history", {"train": [], "val": []})
    peak_memory_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    return trainer, fit_result, history, elapsed, peak_memory_mb
