from __future__ import annotations

import copy
import os
from typing import Dict, Iterable

# Avoid matplotlib/fontconfig cache warnings in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.amp import GradScaler, autocast
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _set_global_avg_degree(model: torch.nn.Module, avg_degree: float | None) -> None:
    if avg_degree is None:
        return
    target = model
    if hasattr(target, "set_global_avg_degree"):
        target.set_global_avg_degree(avg_degree)
        return
    orig = getattr(target, "_orig_mod", None)
    if orig is not None and hasattr(orig, "set_global_avg_degree"):
        orig.set_global_avg_degree(avg_degree)


class UnifiedTrainer:
    """Single trainer path with centralized AMP and torchmetrics usage."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        trainer_cfg: Dict,
        eval_model: torch.nn.Module | None = None,
    ) -> None:
        self.model = model.to(device)
        self.eval_model = (eval_model if eval_model is not None else self.model).to(device)
        self.device = device
        self.epochs = int(trainer_cfg["epochs"])
        self.max_grad_norm = float(trainer_cfg.get("max_grad_norm", 1.0))
        self.log_every_steps = int(trainer_cfg.get("log_every_steps", 10))
        self.patience = int(trainer_cfg.get("patience", 10))
        self.use_amp = bool(trainer_cfg.get("use_amp", False)) and device.type == "cuda"
        self.amp_dtype = str(trainer_cfg.get("amp_dtype", "fp16")).lower()
        self.task_type = str(trainer_cfg.get("task_type", "binary")).lower()
        self.num_classes = int(trainer_cfg.get("num_classes", 1))

        if self.task_type == "multiclass":
            self.criterion = torch.nn.CrossEntropyLoss()
            self.metric = MulticlassAccuracy(num_classes=max(self.num_classes, 2)).to(device)
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.metric = BinaryAccuracy().to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(trainer_cfg["lr"]),
            weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    def _autocast_dtype(self) -> torch.dtype:
        if self.amp_dtype == "bf16":
            return torch.bfloat16
        return torch.float16

    def _run_epoch(self, loader: Iterable[Dict[str, torch.Tensor]], train: bool) -> Dict[str, float]:
        active_model = self.model if train else self.eval_model
        if train:
            self.model.train()
        else:
            self.model.eval()
            self.eval_model.eval()

        self.metric.reset()
        loss_sum = 0.0
        seen = 0
        y_true: list[float] = []
        y_score: list[float] = []

        for batch in loader:
            batch = _move_batch_to_device(batch, self.device)
            if self.task_type == "multiclass":
                targets = batch["y"].view(-1).long()
            else:
                targets = batch["y"].view(-1).float()
            bsz = int(targets.shape[0])

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            # Solver-based inference needs input-state gradients even in eval mode.
            # We keep autograd enabled for the forward pass and simply skip backward/optimizer steps.
            with torch.enable_grad():
                with autocast(
                    device_type=self.device.type,
                    dtype=self._autocast_dtype(),
                    enabled=self.use_amp,
                ):
                    logits = active_model(batch)
                    if self.task_type == "multiclass":
                        loss = self.criterion(logits, targets)
                    else:
                        logits = logits.view(-1)
                        loss = self.criterion(logits, targets)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if self.task_type == "multiclass":
                preds = logits.detach()
                self.metric.update(preds, targets)
                y_true.extend(targets.detach().cpu().tolist())
                y_score.extend(preds.argmax(dim=-1).detach().cpu().tolist())
            else:
                probs = torch.sigmoid(logits.detach())
                self.metric.update(probs, targets.int())
                y_true.extend(targets.detach().cpu().tolist())
                y_score.extend(probs.detach().cpu().tolist())
            loss_sum += loss.detach().item() * bsz
            seen += bsz

        mean_loss = loss_sum / max(seen, 1)
        acc = self.metric.compute().item()

        metrics: Dict[str, float] = {"loss": float(mean_loss), "acc": float(acc)}
        if self.task_type == "multiclass":
            metrics["binary_ranking_available"] = 0.0
            metrics["auc"] = 0.0
            metrics["pr_auc"] = 0.0
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
        else:
            binary_ranking_available = len(set(y_true)) > 1
            metrics["binary_ranking_available"] = float(binary_ranking_available)
            if binary_ranking_available:
                try:
                    metrics["auc"] = float(roc_auc_score(y_true, y_score))
                    metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
                    y_pred = [1 if s >= 0.5 else 0 for s in y_score]
                    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
                    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
                except Exception:
                    metrics["binary_ranking_available"] = 0.0
                    metrics["auc"] = 0.5
                    metrics["pr_auc"] = 0.0
                    metrics["precision"] = 0.0
                    metrics["recall"] = 0.0
                    metrics["f1"] = 0.0
            else:
                metrics["auc"] = 0.5
                metrics["pr_auc"] = 0.0
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1"] = 0.0
        return metrics

    def _collect_train_stats(self, train_loader: Iterable[Dict[str, torch.Tensor]]) -> tuple[torch.Tensor | None, float | None]:
        pos = 0.0
        total = 0.0
        saw_targets = False
        degree_sum = 0.0
        node_sum = 0

        for batch in train_loader:
            if self.task_type == "multiclass":
                saw_targets = True
            else:
                y = batch["y"].float().view(batch["y"].size(0), -1)
                pos += float(y.sum().item())
                total += float(y.numel())
                saw_targets = True

            if "c_2" in batch and "x" in batch:
                degree_sum += float(batch["c_2"].numel())
                node_sum += int(batch["x"].size(0))

        pos_weight = None
        if self.task_type != "multiclass" and saw_targets and pos > 0.0 and pos < total:
            neg = total - pos
            pos_weight = torch.tensor([neg / pos], device=self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        avg_degree = (degree_sum / node_sum) if node_sum > 0 else None
        return pos_weight, avg_degree

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        val_loader: Iterable[Dict[str, torch.Tensor]],
        test_loader: Iterable[Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, float]]:
        _, avg_degree = self._collect_train_stats(train_loader)
        _set_global_avg_degree(self.model, avg_degree)

        best_score = -float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        bad_epochs = 0

        history = {"train": [], "val": []}
        for _ in range(self.epochs):
            train_stats = self._run_epoch(train_loader, train=True)
            val_stats = self._run_epoch(val_loader, train=False)
            history["train"].append(train_stats)
            history["val"].append(val_stats)

            if self.task_type == "multiclass":
                score = val_stats["acc"]
            elif val_stats.get("binary_ranking_available", 0.0) > 0.5:
                score = val_stats["auc"]
            else:
                score = -val_stats["loss"]

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        test_stats = self._run_epoch(test_loader, train=False)

        return {
            "best_val_score": float(best_score),
            "final_train": history["train"][-1],
            "final_val": history["val"][-1],
            "test": test_stats,
            "epochs_ran": len(history["train"]),
        }
