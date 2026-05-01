from __future__ import annotations

import copy
import os
from typing import Dict, Iterable

# Avoid matplotlib/fontconfig cache warnings in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.amp import GradScaler, autocast
from torchmetrics.classification import BinaryAccuracy


def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


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

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(trainer_cfg["lr"]),
            weight_decay=float(trainer_cfg.get("weight_decay", 0.0)),
        )
        self.scaler = GradScaler("cuda", enabled=self.use_amp)
        self.metric = BinaryAccuracy().to(device)

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

        for batch in loader:
            batch = _move_batch_to_device(batch, self.device)
            targets = batch["y"].float()
            bsz = targets.shape[0]

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
                    loss = self.criterion(logits, targets)

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            probs = torch.sigmoid(logits.detach())
            self.metric.update(probs, targets.int())
            loss_sum += loss.detach().item() * bsz
            seen += bsz

        mean_loss = loss_sum / max(seen, 1)
        acc = self.metric.compute().item()
        return {"loss": float(mean_loss), "acc": float(acc)}

    def fit(
        self,
        train_loader: Iterable[Dict[str, torch.Tensor]],
        val_loader: Iterable[Dict[str, torch.Tensor]],
        test_loader: Iterable[Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, float]]:
        best_val = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        bad_epochs = 0

        history = {"train": [], "val": []}
        for _ in range(self.epochs):
            train_stats = self._run_epoch(train_loader, train=True)
            val_stats = self._run_epoch(val_loader, train=False)
            history["train"].append(train_stats)
            history["val"].append(val_stats)

            if val_stats["loss"] < best_val:
                best_val = val_stats["loss"]
                best_state = copy.deepcopy(self.model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        self.model.load_state_dict(best_state)
        test_stats = self._run_epoch(test_loader, train=False)

        return {
            "best_val_loss": float(best_val),
            "final_train": history["train"][-1],
            "final_val": history["val"][-1],
            "test": test_stats,
            "epochs_ran": len(history["train"]),
        }
