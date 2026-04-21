import math
import random
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm

from get import build_adamw_optimizer
from get.compile_utils import maybe_compile_model
from get.data import collate_get_batch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(seeds_str):
    seeds = []
    for item in seeds_str.split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    if not seeds:
        raise ValueError("No valid seeds provided.")
    return seeds


def history_mean_std(histories, key):
    arr = np.array([h[key] for h in histories], dtype=np.float64)
    return arr.mean(axis=0).tolist(), arr.std(axis=0).tolist()


def mean_std(xs):
    arr = np.array(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _make_loader(data, batch_size, shuffle, num_workers, pin_memory):
    kwargs = {
        "dataset": data,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_get_batch,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def _partition_ids(ids, rng, train_ratio, val_ratio):
    ids = list(ids)
    rng.shuffle(ids)
    if len(ids) < 3:
        return ids, [], []

    train_cut = max(1, int(train_ratio * len(ids)))
    val_count = max(1, int(val_ratio * len(ids)))
    if train_cut + val_count >= len(ids):
        train_cut = max(1, len(ids) - 2)
        val_count = 1
    val_cut = train_cut + val_count

    return ids[:train_cut], ids[train_cut:val_cut], ids[val_cut:]


def _split_grouped_dataset(dataset, split_key, seed, train_ratio=0.70, val_ratio=0.15):
    ids = sorted({g[split_key] for g in dataset})
    if len(ids) < 3:
        raise ValueError(f"Need at least 3 unique {split_key} groups for train/val/test split.")

    labels_by_id = {}
    for g in dataset:
        if "y" not in g:
            continue
        labels_by_id.setdefault(g[split_key], set()).add(float(g["y"].reshape(-1)[0].item()))

    homogeneous_binary = all(
        len(labels_by_id.get(group_id, set())) == 1
        and next(iter(labels_by_id[group_id])) in (0.0, 1.0)
        for group_id in ids
    )

    split_rng = random.Random(seed)
    if homogeneous_binary:
        train_parts, val_parts, test_parts = [], [], []
        for label in (0.0, 1.0):
            class_ids = [group_id for group_id in ids if next(iter(labels_by_id[group_id])) == label]
            train_i, val_i, test_i = _partition_ids(class_ids, split_rng, train_ratio, val_ratio)
            train_parts.extend(train_i)
            val_parts.extend(val_i)
            test_parts.extend(test_i)
        split_rng.shuffle(train_parts)
        split_rng.shuffle(val_parts)
        split_rng.shuffle(test_parts)
        train_ids = set(train_parts)
        val_ids = set(val_parts)
        test_ids = set(test_parts)
    else:
        train_i, val_i, test_i = _partition_ids(ids, split_rng, train_ratio, val_ratio)
        train_ids = set(train_i)
        val_ids = set(val_i)
        test_ids = set(test_i)

    train_data = [g for g in dataset if g[split_key] in train_ids]
    val_data = [g for g in dataset if g[split_key] in val_ids]
    test_data = [g for g in dataset if g[split_key] in test_ids]
    return train_data, val_data, test_data


def _safe_auc(labels, preds):
    try:
        auc = float(roc_auc_score(labels, preds))
    except ValueError:
        return 0.5
    if not math.isfinite(auc):
        return 0.5
    return auc


def _signed_margin_loss(logits, target, margin):
    signed_target = target.mul(2.0).sub(1.0)
    return torch.relu(float(margin) - signed_target * logits).mean()


def _logit_diagnostics(labels, logits):
    labels_arr = np.array(labels, dtype=np.float64)
    logits_arr = np.array(logits, dtype=np.float64)
    if logits_arr.size == 0:
        return {
            "pos_logit_mean": 0.0,
            "neg_logit_mean": 0.0,
            "logit_margin": 0.0,
            "signed_accuracy": 0.0,
        }

    pos = logits_arr[labels_arr >= 0.5]
    neg = logits_arr[labels_arr < 0.5]
    pos_mean = float(pos.mean()) if pos.size else 0.0
    neg_mean = float(neg.mean()) if neg.size else 0.0
    signed_pred = logits_arr >= 0.0
    signed_acc = float((signed_pred == (labels_arr >= 0.5)).mean())
    return {
        "pos_logit_mean": pos_mean,
        "neg_logit_mean": neg_mean,
        "logit_margin": pos_mean - neg_mean,
        "signed_accuracy": signed_acc,
    }


def _evaluate_binary_loader(model, loader, device, pin_memory, amp_enabled, amp_torch_dtype, criterion, apply_sigmoid_eval):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=pin_memory)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_torch_dtype,
                enabled=amp_enabled,
            ):
                out, _ = model(batch, task_level="graph")
                logits = torch.nan_to_num(out.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
                target = batch.y.reshape(-1).float()
                loss = criterion(logits, target)

            batch_size = int(target.numel())
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size

            preds = torch.sigmoid(logits) if apply_sigmoid_eval else logits
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(target.cpu().numpy().tolist())
            all_logits.extend(logits.cpu().numpy().tolist())

    auc = _safe_auc(all_labels, all_preds)
    avg_loss = total_loss / max(total_examples, 1)
    diagnostics = _logit_diagnostics(all_labels, all_logits)
    return auc, avg_loss, diagnostics


def train_and_eval_binary(
    model_name,
    model,
    dataset,
    split_key,
    epochs,
    batch_size,
    device,
    lr,
    max_grad_norm,
    weight_decay=1e-4,
    seed=0,
    compile_model=False,
    apply_sigmoid_eval=False,
    track_grad_norm=False,
    use_amp=True,
    amp_dtype="float16",
    num_workers=0,
    pin_memory=None,
    margin_loss_weight=0.0,
    logit_margin=1.0,
):
    model = model.to(device)
    model = maybe_compile_model(model, compile_model, model_name=model_name)
    optimizer = build_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss()

    train_data, val_data, test_data = _split_grouped_dataset(dataset, split_key, seed)

    if pin_memory is None:
        pin_memory = bool(device.type == "cuda")

    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_torch_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    best_val_auc = float("-inf")
    best_ckpt_path = None
    history = {
        "train_loss": [],
        "train_bce_loss": [],
        "train_margin_loss": [],
        "val_loss": [],
        "val_auc": [],
        "val_logit_margin": [],
        "val_signed_accuracy": [],
        "bad_batches": [],
        "split_sizes": {
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
    }
    if track_grad_norm:
        history["grad_norm"] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        best_ckpt_path = f"{tmpdir}/best_{model_name.replace('/', '_')}.pt"
        train_loader = _make_loader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = _make_loader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = _make_loader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        pbar = tqdm(range(epochs), desc=f"Training {model_name}")
        for _ in pbar:
            model.train()
            total_loss = 0.0
            total_bce_loss = 0.0
            total_margin_loss = 0.0
            total_grad_norm = 0.0
            grad_steps = 0
            bad_batches = 0

            for batch in train_loader:
                batch = batch.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()

                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_torch_dtype,
                    enabled=amp_enabled,
                ):
                    out, _ = model(batch, task_level="graph")
                    out = out.reshape(-1)
                    target = batch.y.reshape(-1).float()
                    if not torch.isfinite(out).all():
                        bad_batches += 1
                        continue
                    bce_loss = criterion(out, target)
                    margin_loss = _signed_margin_loss(out, target, logit_margin)
                    loss = bce_loss + float(margin_loss_weight) * margin_loss
                    if not torch.isfinite(loss):
                        bad_batches += 1
                        continue

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not math.isfinite(float(grad_norm)):
                    optimizer.zero_grad(set_to_none=True)
                    if scaler.is_enabled():
                        scaler.update()
                    bad_batches += 1
                    continue

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                total_loss += float(loss.item())
                total_bce_loss += float(bce_loss.item())
                total_margin_loss += float(margin_loss.item())
                total_grad_norm += float(grad_norm)
                grad_steps += 1

            val_auc, val_loss, val_diag = _evaluate_binary_loader(
                model,
                val_loader,
                device,
                pin_memory,
                amp_enabled,
                amp_torch_dtype,
                criterion,
                apply_sigmoid_eval,
            )

            scheduler.step(val_auc)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_ckpt_path)

            avg_loss = total_loss / max(grad_steps, 1)
            avg_bce_loss = total_bce_loss / max(grad_steps, 1)
            avg_margin_loss = total_margin_loss / max(grad_steps, 1)
            history["train_loss"].append(avg_loss)
            history["train_bce_loss"].append(avg_bce_loss)
            history["train_margin_loss"].append(avg_margin_loss)
            history["val_loss"].append(float(val_loss))
            history["val_auc"].append(float(val_auc))
            history["val_logit_margin"].append(float(val_diag["logit_margin"]))
            history["val_signed_accuracy"].append(float(val_diag["signed_accuracy"]))
            history["bad_batches"].append(int(bad_batches))

            if track_grad_norm:
                avg_grad_norm = total_grad_norm / max(grad_steps, 1)
                history["grad_norm"].append(avg_grad_norm)
                pbar.set_postfix(
                    {
                        "loss": avg_loss,
                        "bce": avg_bce_loss,
                        "val_auc": val_auc,
                        "val_margin": val_diag["logit_margin"],
                        "grad": avg_grad_norm,
                        "bad": bad_batches,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": avg_loss,
                        "bce": avg_bce_loss,
                        "val_auc": val_auc,
                        "val_margin": val_diag["logit_margin"],
                        "bad": bad_batches,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

        model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu", weights_only=True))
        test_auc, test_loss, test_diag = _evaluate_binary_loader(
            model,
            test_loader,
            device,
            pin_memory,
            amp_enabled,
            amp_torch_dtype,
            criterion,
            apply_sigmoid_eval,
        )
        history["selected_val_auc"] = float(best_val_auc)
        history["selected_test_auc"] = float(test_auc)
        history["selected_test_loss"] = float(test_loss)
        history["selected_test_logit_margin"] = float(test_diag["logit_margin"])
        history["selected_test_signed_accuracy"] = float(test_diag["signed_accuracy"])
    return test_auc, history, model
