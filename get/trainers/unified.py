from __future__ import annotations

import copy
import os
from typing import Any, Dict, Iterable, List

# Avoid matplotlib/fontconfig cache warnings in restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".matplotlib_cache")

import torch
from torch.amp import GradScaler, autocast
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torchmetrics.classification import BinaryAccuracy, MulticlassAccuracy
from torchmetrics.regression import MeanAbsoluteError
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def _move_batch_to_device(batch: Any, device: torch.device) -> Any:
    """Recursive move of tensors to device, handling dicts, lists, and objects with .to()."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_move_batch_to_device(v, device) for v in batch]
    if hasattr(batch, "to") and callable(batch.to):
        return batch.to(device)
    return batch


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
            self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=trainer_cfg.get("label_smoothing", 0.05))
            self.metric = MulticlassAccuracy(num_classes=max(self.num_classes, 2)).to(device)
        elif self.task_type == "regression":
            self.criterion = torch.nn.MSELoss()
            self.metric = MeanAbsoluteError().to(device)
        else:
            # Covers binary, multilabel, node_binary
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.metric = BinaryAccuracy().to(device)
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
            T_max=self.epochs - warmup_epochs, 
            eta_min=float(trainer_cfg.get("min_lr", 5e-6))
        )

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
        loss_accum = torch.zeros(1, device=self.device)
        seen = 0
        all_y_true: list[torch.Tensor] = []
        all_y_score: list[torch.Tensor] = []

        from tqdm import tqdm
        
        pbar = tqdm(loader, desc=f"{'Train' if train else 'Val'}", leave=False)
        for batch in pbar:
            batch = _move_batch_to_device(batch, self.device)
            if self.task_type == "multiclass":
                targets = batch["y"].view(-1).long()
            else:
                targets = batch["y"].view(-1).float()
            bsz = int(targets.shape[0])

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
                    elif self.task_type == "node_binary":
                        logits = logits.reshape_as(targets)
                        loss = self.criterion(logits, targets)
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
                all_y_true.append(targets.detach())
                all_y_score.append(preds.argmax(dim=-1).detach())
            elif self.task_type == "regression":
                preds = logits.detach()
                self.metric.update(preds, targets)
                all_y_true.append(targets.detach())
                all_y_score.append(preds.detach())
            elif self.task_type == "node_binary":
                preds = torch.sigmoid(logits.detach())
                mask = batch.get("mask", torch.ones_like(targets, dtype=torch.bool)).reshape(-1)
                if mask.sum() > 0:
                    self.metric.update(preds[mask], targets[mask].int())
                    all_y_true.append(targets[mask].detach())
                    all_y_score.append(preds[mask].detach())
            else:
                # binary or multilabel
                probs = torch.sigmoid(logits.detach())
                self.metric.update(probs.reshape(-1), targets.reshape(-1).int())
                all_y_true.append(targets.detach())
                all_y_score.append(probs.detach())
            
            loss_val = loss.detach()
            loss_accum += loss_val * bsz
            seen += bsz
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_val.item():.4f}",
                "acc": f"{self.metric.compute().item():.4f}"
            })

        if seen == 0:
            return {"loss": 0.0, "acc": 0.0}

        mean_loss = (loss_accum / seen).item()
        acc = self.metric.compute().item()

        y_true_ts = torch.cat(all_y_true).cpu()
        y_score_ts = torch.cat(all_y_score).cpu()
        y_true = y_true_ts.tolist()
        y_score = y_score_ts.tolist()

        metrics: Dict[str, float] = {"loss": float(mean_loss), "acc" if self.task_type != "regression" else "mae": float(acc)}
        if self.task_type in {"multiclass", "regression"}:
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
        pos = torch.zeros(1, device=self.device)
        total = torch.zeros(1, device=self.device)
        saw_targets = False
        degree_sum = torch.zeros(1, device=self.device)
        node_sum = torch.zeros(1, device=self.device)

        for batch in train_loader:
            batch = _move_batch_to_device(batch, self.device)
            if self.task_type == "multiclass":
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
        if self.task_type != "multiclass" and saw_targets and pos_val > 0.0 and pos_val < total_val:
            neg_val = total_val - pos_val
            pos_weight = torch.tensor([neg_val / pos_val], device=self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        node_sum_val = float(node_sum.item())
        avg_degree = (float(degree_sum.item()) / node_sum_val) if node_sum_val > 0 else None
        return pos_weight, avg_degree

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        val_loader: Iterable[Dict[str, torch.Tensor]],
        test_loader: Iterable[Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, float]]:
        from tqdm import tqdm
        
        # Stats collection with pbar
        degree_sum = torch.zeros(1, device=self.device)
        node_sum = torch.zeros(1, device=self.device)
        pos = torch.zeros(1, device=self.device)
        total = torch.zeros(1, device=self.device)
        
        pbar_stats = tqdm(train_loader, desc="Collecting Stats", leave=False)
        for batch in pbar_stats:
            batch = _move_batch_to_device(batch, self.device)
            if self.task_type != "multiclass":
                y = batch["y"].float().view(batch["y"].size(0), -1)
                pos += y.sum()
                total += y.numel()
            if "c_2" in batch and "x" in batch:
                degree_sum += batch["c_2"].numel()
                node_sum += batch["x"].size(0)
        
        pos_val = float(pos.item())
        total_val = float(total.item())
        if self.task_type != "multiclass" and pos_val > 0.0 and pos_val < total_val:
            neg_val = total_val - pos_val
            pos_weight = torch.tensor([neg_val / pos_val], device=self.device)
            self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        node_sum_val = float(node_sum.item())
        avg_degree = (float(degree_sum.item()) / node_sum_val) if node_sum_val > 0 else None
        _set_global_avg_degree(self.model, avg_degree)

        best_score = -float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        bad_epochs = 0

        history = {"train": [], "val": []}
        warmup_epochs = int(self.warmup_scheduler.lr_lambdas[0].__code__.co_consts[1]) if hasattr(self.warmup_scheduler, "lr_lambdas") else 50
        
        epoch_pbar = tqdm(range(self.epochs), desc="Training")
        for epoch in epoch_pbar:
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
                    epoch_pbar.write(f"Early stopping at epoch {epoch}")
                    break
            
            epoch_pbar.set_postfix({
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
        }
