from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.regression import MeanAbsoluteError

from get.data.synthetic import collate_graph_samples
from get.utils.compile import maybe_compile_model
from torch.amp import GradScaler, autocast

from .data import ListGraphDataset, split_items, summarize_splits
from .modeling import build_model


def run_epoch(model, loader, device, task_type: str, optimizer=None, pos_weight=None, use_amp=False, amp_dtype=torch.float16, scaler=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    is_classification = task_type in {"binary", "multilabel"}
    bacc = BinaryAccuracy().to(device) if is_classification else None
    mae = MeanAbsoluteError().to(device) if task_type == "regression" else None

    losses: List[float] = []
    y_true: List[float] = []
    y_score: List[float] = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        y = batch["y"]
        if train:
            optimizer.zero_grad(set_to_none=True)
        # GET solver-based forward uses autograd even in eval mode.
        with torch.set_grad_enabled(True):
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(batch)
                if task_type == "binary":
                    target = y.view(-1).float()
                    out = out.view(-1)
                    loss = F.binary_cross_entropy_with_logits(out, target, pos_weight=pos_weight)
                    prob = torch.sigmoid(out)
                    bacc.update(prob, target.long())
                    y_true.extend(target.detach().cpu().reshape(-1).tolist())
                    y_score.extend(prob.detach().cpu().reshape(-1).tolist())
                elif task_type == "multilabel":
                    target = y.float()
                    loss = F.binary_cross_entropy_with_logits(out, target, pos_weight=pos_weight)
                    prob = torch.sigmoid(out)
                    flat_prob = prob.reshape(-1)
                    flat_target = target.reshape(-1)
                    bacc.update(flat_prob, flat_target.long())
                    y_true.extend(flat_target.detach().cpu().tolist())
                    y_score.extend(flat_prob.detach().cpu().tolist())
                elif task_type == "multiclass":
                    target = y.view(-1).long()
                    loss = F.cross_entropy(out, target)
                    pred = torch.softmax(out, -1)
                    y_true.extend(target.detach().cpu().tolist())
                    y_score.extend(pred.argmax(-1).detach().cpu().tolist())
                else:
                    target = y.float().contiguous()
                    out = out.reshape_as(target).contiguous()
                    loss = F.mse_loss(out, target)
                    mae.update(out, target)
                    y_true.extend(target.detach().cpu().reshape(-1).tolist())
                    y_score.extend(out.detach().cpu().reshape(-1).tolist())
            if train:
                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    metrics: Dict[str, float] = {"loss": float(np.mean(losses)) if losses else 0.0}
    if is_classification:
        metrics["acc"] = float(bacc.compute().item())
        ranking_available = len(set(y_true)) > 1
        metrics["binary_ranking_available"] = float(ranking_available)
        try:
            if ranking_available:
                metrics["auc"] = float(roc_auc_score(y_true, y_score))
                metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
                
                # Threshold for discrete metrics
                y_pred = [1 if s >= 0.5 else 0 for s in y_score]
                metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
                metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
                metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
            else:
                metrics["auc"] = 0.5
                metrics["pr_auc"] = 0.0
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1"] = 0.0
        except Exception:
            metrics["binary_ranking_available"] = 0.0
            metrics["auc"] = 0.5
            metrics["pr_auc"] = 0.0
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
    elif task_type == "multiclass":
        yt = np.array(y_true)
        yp = np.array(y_score)
        metrics["acc"] = float((yt == yp).mean()) if yt.size > 0 else 0.0
    else:
        metrics["mae"] = float(mae.compute().item())
    
    mode = "Train" if train else "Val"
    msg = f"[{mode}] Loss: {metrics['loss']:.4f}"
    if "acc" in metrics: msg += f" Acc: {metrics['acc']:.4f}"
    if "auc" in metrics: msg += f" AUC: {metrics['auc']:.4f}"
    if "mae" in metrics: msg += f" MAE: {metrics['mae']:.4f}"
    print(msg)
    
    return metrics


def _collect_train_stats(loader, device, task_type: str):
    pos = None
    total = 0.0
    degree_sum = 0.0
    node_sum = 0

    for batch in loader:
        if task_type in {"binary", "multilabel"}:
            y = batch["y"].float().view(batch["y"].size(0), -1)
            batch_pos = y.sum(dim=0)
            pos = batch_pos if pos is None else pos + batch_pos
            total += float(y.size(0))

        if "c_2" in batch and "x" in batch:
            degree_sum += float(batch["c_2"].numel())
            node_sum += int(batch["x"].size(0))

    pos_weight = None
    if pos is not None:
        neg = total - pos
        pos_weight = torch.ones_like(pos, device=device)
        valid = pos > 0
        pos_weight[valid] = neg[valid] / pos[valid]

    avg_degree = (degree_sum / node_sum) if node_sum > 0 else None
    return pos_weight, avg_degree


def _set_global_avg_degree(model, avg_degree: float | None) -> None:
    if avg_degree is None:
        return
    target = model
    if hasattr(target, "set_global_avg_degree"):
        target.set_global_avg_degree(avg_degree)
        return
    orig = getattr(target, "_orig_mod", None)
    if orig is not None and hasattr(orig, "set_global_avg_degree"):
        orig.set_global_avg_degree(avg_degree)


def fit_once(args, task_type: str, num_classes: int, tr, va, te, device):
    model = build_model(args, task_type=task_type, num_classes=num_classes).to(device)
    
    use_amp = getattr(args, "use_amp", False) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if getattr(args, "amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=use_amp) if use_amp else None

    compile_cfg = {
        "enabled": getattr(args, "compile", False),
        "backend": getattr(args, "compile_backend", "inductor"),
        "dynamic": getattr(args, "compile_dynamic", True),
        "mode": getattr(args, "compile_mode", "default") if getattr(args, "compile_mode", "default") != "default" else None,
        "allow_double_backward": getattr(args, "compile_allow_double_backward", False),
    }
    compile_scope = str(getattr(args, "compile_scope", "eval_only")).lower()
    if compile_cfg["enabled"]:
        if compile_scope == "all":
            if getattr(model, "requires_double_backward", False):
                raise ValueError(
                    "compile_scope='all' is unsupported for GET training because torch.compile "
                    "does not currently support double backward. Use compile_scope='eval_only'."
                )
            model = maybe_compile_model(model, compile_cfg)
            eval_model = model
        elif compile_scope == "eval_only":
            eval_compile_cfg = dict(compile_cfg)
            eval_compile_cfg["allow_double_backward"] = True
            eval_model = maybe_compile_model(model, eval_compile_cfg)
        else:
            raise ValueError(f"Unsupported compile_scope '{compile_scope}'. Use 'eval_only' or 'all'.")
    else:
        eval_model = model

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = -1e18 if task_type in {"binary", "multilabel"} else 1e18
    best_state = None
    final_train = final_val = None

    pos_weight, avg_degree = _collect_train_stats(tr, device, task_type)
    _set_global_avg_degree(model, avg_degree)
    if eval_model is not model:
        _set_global_avg_degree(eval_model, avg_degree)

    for epoch in range(args.epochs):
        final_train = run_epoch(model, tr, device, task_type=task_type, optimizer=optim, pos_weight=pos_weight, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler)
        final_val = run_epoch(eval_model, va, device, task_type=task_type, optimizer=None, pos_weight=pos_weight, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler)
        print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {final_val['loss']:.4f}")
        
        is_best = False
        if task_type in {"binary", "multilabel"}:
            if final_val.get("binary_ranking_available", 0.0) > 0.5:
                score = final_val["auc"]
            else:
                score = -final_val["loss"]
            if score > best:
                is_best = True
                best = score
        else:
            if final_val["loss"] < best:
                is_best = True
                best = final_val["loss"]
                
        if is_best:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test = run_epoch(eval_model, te, device, task_type=task_type, optimizer=None, pos_weight=pos_weight, use_amp=use_amp, amp_dtype=amp_dtype, scaler=scaler)
    result = {"best_val_score": float(best), "final_train": final_train, "final_val": final_val, "test": test}
    if avg_degree is not None:
        result["train_avg_degree"] = float(avg_degree)
    return result


def make_loaders(items, args, task_type: str | None = None, return_split_stats: bool = False):
    tr_i, va_i, te_i = split_items(items, seed=args.seed, task_type=task_type)
    nw = getattr(args, "num_workers", 0)
    pm = getattr(args, "pin_memory", False)
    tr = DataLoader(ListGraphDataset(tr_i), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_samples, num_workers=nw, pin_memory=pm)
    va = DataLoader(ListGraphDataset(va_i), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples, num_workers=nw, pin_memory=pm)
    te = DataLoader(ListGraphDataset(te_i), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples, num_workers=nw, pin_memory=pm)
    if return_split_stats:
        split_stats = summarize_splits({"train": tr_i, "val": va_i, "test": te_i}, task_type=task_type)
        return tr, va, te, split_stats
    return tr, va, te
