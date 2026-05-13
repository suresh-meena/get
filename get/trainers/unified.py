from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List
import numpy as np

# Avoid matplotlib/fontconfig cache warnings in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".matplotlib_cache")

import torch
from torch.amp import GradScaler, autocast
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
)
from torchmetrics.regression import MeanAbsoluteError
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from get.utils.device import move_batch_to_device


def _iter_stat_samples(loader: Iterable[Dict[str, torch.Tensor]]):
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        for idx in range(len(dataset)):
            yield dataset[idx]
        return
    yield from loader


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
        self.use_tqdm = bool(trainer_cfg.get("use_tqdm", False))
        self.use_amp = bool(trainer_cfg.get("use_amp", True)) and device.type == "cuda"
        self.amp_dtype = str(trainer_cfg.get("amp_dtype", "bf16")).lower()
        if self.amp_dtype == "auto" and device.type == "cuda":
            self.amp_dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        self.task_type = str(trainer_cfg.get("task_type", "binary")).lower()
        self.num_classes = int(trainer_cfg.get("num_classes", 1))

        if self.task_type == "multiclass":
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=trainer_cfg.get("label_smoothing", 0.05))
            self.metric = MulticlassAccuracy(num_classes=max(self.num_classes, 2)).to(device)
        elif self.task_type == "regression":
            self.criterion = torch.nn.MSELoss()
            self.metric = MeanAbsoluteError().to(device)
        elif self.task_type in {"multilabel", "node_binary"}:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
            self.metric = BinaryAccuracy().to(device)
        else:
            # Covers binary
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.metric = BinaryAccuracy().to(device)
        if self.task_type in {"binary", "node_binary", "multilabel"}:
            self.metric_auc = BinaryAUROC().to(device)
            self.metric_pr_auc = BinaryAveragePrecision().to(device)
            self.metric_precision = BinaryPrecision().to(device)
            self.metric_recall = BinaryRecall().to(device)
            self.metric_f1 = BinaryF1Score().to(device)
        else:
            self.metric_auc = None
            self.metric_pr_auc = None
            self.metric_precision = None
            self.metric_recall = None
            self.metric_f1 = None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(trainer_cfg["lr"]),
            weight_decay=float(trainer_cfg.get("weight_decay", 0.05)),
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        # Warmup Cosine Scheduler
        warmup_epochs = int(trainer_cfg.get("warmup_epochs", 50))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch) / float(max(1, warmup_epochs))
            return 1.0

        self.warmup_scheduler = LambdaLR(self.optimizer, lr_lambda)
        self.cosine_scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=max(1, self.epochs - warmup_epochs), 
            eta_min=float(trainer_cfg.get("min_lr", 5e-6))
        )

    def _binary_targets_and_mask(self, batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.task_type == "multilabel":
            targets = batch["y"].float().reshape(-1, self.num_classes)
            label_mask = batch.get("y_mask")
            if label_mask is None:
                return targets, torch.ones_like(targets, dtype=torch.bool)
            return targets, label_mask.to(device=targets.device).reshape_as(targets).bool()
        if self.task_type == "node_binary":
            targets = batch["y"].view(-1).float()
            node_mask = batch.get("mask")
            if node_mask is None:
                return targets, torch.ones_like(targets, dtype=torch.bool)
            return targets, node_mask.to(device=targets.device).reshape_as(targets).bool()
        targets = batch["y"].view(-1).float()
        return targets, None

    def _masked_bce_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, int]:
        raw_loss = self.criterion(logits, targets)
        if mask is None:
            return raw_loss, int(targets.numel())
        mask = mask.to(device=raw_loss.device).bool().reshape_as(raw_loss)
        supervised = int(mask.sum().item())
        if supervised == 0:
            return raw_loss.new_zeros(()), 0
        loss = raw_loss.masked_select(mask).sum() / float(supervised)
        return loss, supervised

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
        if self.metric_auc is not None:
            self.metric_auc.reset()
            self.metric_pr_auc.reset()
            self.metric_precision.reset()
            self.metric_recall.reset()
            self.metric_f1.reset()
        loss_accum = torch.zeros(1, device=self.device)
        seen = 0
        cls_pos = torch.zeros(1, device=self.device)
        cls_total = torch.zeros(1, device=self.device)

        if self.use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(loader, desc=f"{'Train' if train else 'Val'}", leave=False)
        else:
            iterator = loader

        def _update_binary_family_metrics(probs: torch.Tensor, targets01: torch.Tensor) -> None:
            if probs.numel() == 0:
                return
            probs = probs.reshape(-1)
            targets01 = targets01.reshape(-1).int()
            self.metric.update(probs, targets01)
            if self.metric_auc is not None:
                self.metric_auc.update(probs, targets01)
                self.metric_pr_auc.update(probs, targets01)
                self.metric_precision.update(probs, targets01)
                self.metric_recall.update(probs, targets01)
                self.metric_f1.update(probs, targets01)
            cls_pos.add_(targets01.float().sum())
            cls_total.add_(targets01.numel())

        for batch in iterator:
            batch = move_batch_to_device(batch, self.device)
            mask = None
            if self.task_type == "multiclass":
                targets = batch["y"].view(-1).long()
            elif self.task_type in {"multilabel", "node_binary"}:
                targets, mask = self._binary_targets_and_mask(batch)
            else:
                targets = batch["y"].view(-1).float()
            loss_weight = int(targets.numel())

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.enable_grad():
                with autocast(
                    device_type=self.device.type,
                    dtype=self._autocast_dtype(),
                    enabled=self.use_amp,
                ):
                    logits = active_model(batch)
                    if self.task_type == "multiclass":
                        loss = self.criterion(logits, targets)
                    elif self.task_type == "regression":
                        logits = logits.reshape_as(targets)
                        loss = self.criterion(logits, targets)
                    elif self.task_type in {"node_binary", "multilabel"}:
                        logits = logits.reshape_as(targets)
                        loss, loss_weight = self._masked_bce_loss(logits, targets, mask)
                    else:
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
            elif self.task_type == "regression":
                preds = logits.detach()
                self.metric.update(preds, targets)
            elif self.task_type == "node_binary":
                preds = torch.sigmoid(logits.detach())
                if mask is not None and mask.any():
                    _update_binary_family_metrics(preds[mask], targets[mask])
            elif self.task_type == "multilabel":
                probs = torch.sigmoid(logits.detach())
                if mask is not None and mask.any():
                    _update_binary_family_metrics(probs[mask], targets[mask])
                else:
                    _update_binary_family_metrics(probs, targets)
            else:
                # binary
                probs = torch.sigmoid(logits.detach())
                _update_binary_family_metrics(probs, targets)
            
            loss_accum += loss.detach() * loss_weight
            seen += loss_weight

        if seen == 0:
            return {"loss": 0.0, "acc": 0.0}

        mean_loss = (loss_accum / seen).item()
        acc = self.metric.compute().item()
        metrics: Dict[str, float] = {"loss": float(mean_loss), "acc" if self.task_type != "regression" else "mae": float(acc)}
        if self.task_type in {"multiclass", "regression"}:
            metrics["binary_ranking_available"] = 0.0
            metrics["auc"] = 0.0
            metrics["pr_auc"] = 0.0
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
        else:
            pos_val = float(cls_pos.item())
            total_val = float(cls_total.item())
            binary_ranking_available = total_val > 0.0 and pos_val > 0.0 and pos_val < total_val
            metrics["binary_ranking_available"] = float(binary_ranking_available)
            if binary_ranking_available:
                auc = float(self.metric_auc.compute().item()) if self.metric_auc is not None else 0.5
                pr_auc = float(self.metric_pr_auc.compute().item()) if self.metric_pr_auc is not None else 0.0
                precision = float(self.metric_precision.compute().item()) if self.metric_precision is not None else 0.0
                recall = float(self.metric_recall.compute().item()) if self.metric_recall is not None else 0.0
                f1 = float(self.metric_f1.compute().item()) if self.metric_f1 is not None else 0.0
                metrics["auc"] = 0.5 if not np.isfinite(auc) else auc
                metrics["pr_auc"] = 0.0 if not np.isfinite(pr_auc) else pr_auc
                metrics["precision"] = 0.0 if not np.isfinite(precision) else precision
                metrics["recall"] = 0.0 if not np.isfinite(recall) else recall
                metrics["f1"] = 0.0 if not np.isfinite(f1) else f1
            else:
                metrics["auc"] = 0.5
                metrics["pr_auc"] = 0.0
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1"] = 0.0
        return metrics

    def _collect_train_stats(self, train_loader: Iterable[Dict[str, torch.Tensor]]) -> tuple[torch.Tensor | None, float | None]:
        pos = torch.zeros(1, device=self.device)
        total = torch.zeros(1, device=self.device)
        saw_targets = False
        degree_sum = torch.zeros(1, device=self.device)
        node_sum = torch.zeros(1, device=self.device)

        for batch in _iter_stat_samples(train_loader):
            if self.task_type == "multiclass":
                saw_targets = True
            elif self.task_type in {"multilabel", "node_binary"}:
                targets, mask = self._binary_targets_and_mask(batch)
                pos += targets.masked_select(mask).sum()
                total += mask.sum()
                saw_targets = True
            else:
                y = batch["y"].float().view(batch["y"].size(0), -1)
                pos += y.sum()
                total += y.numel()
                saw_targets = True

            if "c_2" in batch and "x" in batch:
                degree_sum += batch["c_2"].numel()
                node_sum += batch["x"].size(0)

        pos_weight = None
        pos_val = float(pos.item())
        total_val = float(total.item())
        if self.task_type in {"binary", "node_binary"} and saw_targets and pos_val > 0.0 and pos_val < total_val:
            neg_val = total_val - pos_val
            pos_weight = torch.tensor([neg_val / pos_val], device=self.device)
            reduction = "none" if self.task_type == "node_binary" else "mean"
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

        node_sum_val = float(node_sum.item())
        avg_degree = (float(degree_sum.item()) / node_sum_val) if node_sum_val > 0 else None
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
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        bad_epochs = 0

        history = {"train": [], "val": []}
        warmup_epochs = int(self.warmup_scheduler.lr_lambdas[0].__code__.co_consts[1]) if hasattr(self.warmup_scheduler, "lr_lambdas") else 50
        
        if self.use_tqdm:
            from tqdm import tqdm
            epoch_iter = tqdm(range(self.epochs), desc="Training")
        else:
            epoch_iter = range(self.epochs)
        for epoch in epoch_iter:
            train_stats = self._run_epoch(train_loader, train=True)
            val_stats = self._run_epoch(val_loader, train=False)
            
            if epoch < warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.cosine_scheduler.step()

            history["train"].append(train_stats)
            history["val"].append(val_stats)

            if self.task_type == "multiclass":
                score = val_stats["acc"]
            elif self.task_type == "node_binary":
                # Official ET anomaly detection protocol uses Macro-F1
                score = val_stats.get("f1", 0.0)
            elif val_stats.get("binary_ranking_available", 0.0) > 0.5:
                score = val_stats["auc"]
            else:
                score = -val_stats["loss"]

            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    if self.use_tqdm and hasattr(epoch_iter, "write"):
                        epoch_iter.write(f"Early stopping at epoch {epoch}")
                    break
            
            if self.use_tqdm and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix({
                    "tr_loss": f"{train_stats['loss']:.3f}",
                    "val_auc": f"{val_stats.get('auc', 0.0):.3f}",
                    "val_acc": f"{val_stats['acc']:.3f}"
                })

        self.model.load_state_dict(best_state)
        test_stats = self._run_epoch(test_loader, train=False)

        return {
            "best_val_score": float(best_score),
            "final_train": history["train"][-1],
            "final_val": history["val"][-1],
            "test": test_stats,
            "epochs_ran": len(history["train"]),
            "history": history,
        }
