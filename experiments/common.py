from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from get.data import ListGraphDataset, collate_graph_samples, summarize_splits
from get.trainers import UnifiedTrainer
from get.utils.device import move_batch_to_device


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


def _to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(val) for val in value]
    return value


def collect_energy_diagnostics(
    model: torch.nn.Module,
    loaders: Dict[str, Any],
    *,
    device: torch.device | None = None,
    max_batches: int | None = None,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    orig_model = getattr(model, "_orig_mod", None)
    models_to_restore = [model]
    if orig_model is not None and orig_model is not model:
        models_to_restore.append(orig_model)

    previous_training = {id(module): bool(module.training) for module in models_to_restore if hasattr(module, "training")}
    for module in models_to_restore:
        if hasattr(module, "eval"):
            module.eval()

    try:
        for split_name, loader in loaders.items():
            batches: List[Dict[str, Any]] = []
            for batch_index, batch in enumerate(loader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                if device is not None:
                    batch = move_batch_to_device(batch, device)
                try:
                    output = model(batch, return_solver_stats=True)
                except TypeError:
                    if orig_model is None:
                        raise
                    output = orig_model(batch, return_solver_stats=True)
                if not isinstance(output, tuple) or len(output) != 3:
                    continue
                logits, energy_trace, solver_stats = output
                record: Dict[str, Any] = {
                    "batch_index": batch_index,
                    "logit_shape": list(logits.shape) if torch.is_tensor(logits) else None,
                    "energy_trace": _to_jsonable(energy_trace),
                    "solver_stats": _to_jsonable(solver_stats),
                }
                try:
                    num_graphs_value = batch["num_graphs"]
                except Exception:
                    num_graphs_value = getattr(batch, "num_graphs", None)
                if torch.is_tensor(num_graphs_value):
                    record["num_graphs"] = int(num_graphs_value.detach().item())
                elif num_graphs_value is not None:
                    record["num_graphs"] = int(num_graphs_value)
                if record["energy_trace"]:
                    record["energy_initial"] = float(record["energy_trace"][0])
                    record["energy_final"] = float(record["energy_trace"][-1])
                    record["energy_drop"] = float(record["energy_initial"] - record["energy_final"])
                if isinstance(record.get("solver_stats"), dict):
                    solver_stats_dict = record["solver_stats"]
                    if "energy_initial" not in record and solver_stats_dict.get("energy_initial") is not None:
                        record["energy_initial"] = float(solver_stats_dict["energy_initial"])
                    if "energy_final" not in record and solver_stats_dict.get("energy_final") is not None:
                        record["energy_final"] = float(solver_stats_dict["energy_final"])
                    if "energy_drop" not in record and solver_stats_dict.get("energy_drop") is not None:
                        record["energy_drop"] = float(solver_stats_dict["energy_drop"])
                    if solver_stats_dict.get("latent_displacement") is not None:
                        record["latent_displacement"] = float(solver_stats_dict["latent_displacement"])
                    if solver_stats_dict.get("branch_energies_initial") is not None:
                        record["branch_energies_initial"] = _to_jsonable(solver_stats_dict["branch_energies_initial"])
                    if solver_stats_dict.get("branch_energies_final") is not None:
                        record["branch_energies_final"] = _to_jsonable(solver_stats_dict["branch_energies_final"])
                batches.append(record)
            diagnostics[split_name] = {
                "num_batches": len(batches),
                "batches": batches,
            }
    finally:
        for module in models_to_restore:
            if hasattr(module, "train"):
                module.train(previous_training.get(id(module), False))

    return diagnostics
