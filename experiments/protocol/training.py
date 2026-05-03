from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.regression import MeanAbsoluteError

from get.data.synthetic import collate_graph_samples

from .data import ListGraphDataset, split_items
from .modeling import build_model


def run_epoch(model, loader, device, task_type: str, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    bacc = BinaryAccuracy().to(device) if task_type == "binary" else None
    mae = MeanAbsoluteError().to(device) if task_type == "regression" else None

    losses: List[float] = []
    y_true: List[float] = []
    y_score: List[float] = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        y = batch["y"].view(-1)
        if train:
            optimizer.zero_grad(set_to_none=True)
        # GET solver-based forward uses autograd even in eval mode.
        with torch.set_grad_enabled(True):
            out = model(batch)
            if task_type == "binary":
                out = out.view(-1)
                loss = F.binary_cross_entropy_with_logits(out, y)
                prob = torch.sigmoid(out)
                bacc.update(prob, y.long())
                y_true.extend(y.detach().cpu().tolist())
                y_score.extend(prob.detach().cpu().tolist())
            elif task_type == "multiclass":
                target = y.long()
                loss = F.cross_entropy(out, target)
                pred = torch.softmax(out, -1)
                y_true.extend(target.detach().cpu().tolist())
                y_score.extend(pred.argmax(-1).detach().cpu().tolist())
            else:
                out = out.view(-1)
                loss = F.mse_loss(out, y)
                mae.update(out, y)
                y_true.extend(y.detach().cpu().tolist())
                y_score.extend(out.detach().cpu().tolist())
            if train:
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    metrics: Dict[str, float] = {"loss": float(np.mean(losses)) if losses else 0.0}
    if task_type == "binary":
        metrics["acc"] = float(bacc.compute().item())
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_score)) if len(set(y_true)) > 1 else 0.5
        except Exception:
            metrics["auc"] = 0.5
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


def fit_once(args, task_type: str, num_classes: int, tr, va, te, device):
    model = build_model(args, task_type=task_type, num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = 1e18
    best_state = None
    final_train = final_val = None
    for epoch in range(args.epochs):
        final_train = run_epoch(model, tr, device, task_type=task_type, optimizer=optim)
        final_val = run_epoch(model, va, device, task_type=task_type, optimizer=None)
        print(f"Epoch {epoch+1}/{args.epochs} - Val Loss: {final_val['loss']:.4f}")
        if final_val["loss"] < best:
            best = final_val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test = run_epoch(model, te, device, task_type=task_type, optimizer=None)
    return {"best_val_loss": float(best), "final_train": final_train, "final_val": final_val, "test": test}


def make_loaders(items, args):
    tr_i, va_i, te_i = split_items(items, seed=args.seed)
    tr = DataLoader(ListGraphDataset(tr_i), batch_size=args.batch_size, shuffle=True, collate_fn=collate_graph_samples)
    va = DataLoader(ListGraphDataset(va_i), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples)
    te = DataLoader(ListGraphDataset(te_i), batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_graph_samples)
    return tr, va, te
