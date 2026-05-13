from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import torch

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


def make_loader_kwargs(num_workers: int) -> Dict[str, object]:
    num_workers = int(num_workers)
    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return kwargs


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
