import math
import random

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
):
    model = model.to(device)
    model = maybe_compile_model(model, compile_model, model_name=model_name)
    optimizer = build_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss()

    ids = sorted({g[split_key] for g in dataset})
    split_rng = random.Random(seed)
    split_rng.shuffle(ids)
    train_cut = int(0.8 * len(ids))
    train_ids = set(ids[:train_cut])
    train_data = [g for g in dataset if g[split_key] in train_ids]
    test_data = [g for g in dataset if g[split_key] not in train_ids]

    if pin_memory is None:
        pin_memory = bool(device.type == "cuda")

    amp_enabled = bool(use_amp and device.type in {"cuda", "cpu"})
    amp_torch_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")

    best_auc = 0.0
    best_state = None
    history = {"train_loss": [], "test_auc": [], "bad_batches": []}
    if track_grad_norm:
        history["grad_norm"] = []

    pbar = tqdm(range(epochs), desc=f"Training {model_name}")
    for _ in pbar:
        model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        grad_steps = 0
        bad_batches = 0

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
                loss = criterion(out, target)
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
            total_grad_norm += float(grad_norm)
            grad_steps += 1

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device, non_blocking=pin_memory)
                with torch.autocast(
                    device_type=device.type,
                    dtype=amp_torch_dtype,
                    enabled=amp_enabled,
                ):
                    out, _ = model(batch, task_level="graph")
                out = torch.nan_to_num(out.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
                if apply_sigmoid_eval:
                    out = torch.sigmoid(out)
                all_preds.extend(out.cpu().numpy().tolist())
                all_labels.extend(batch.y.reshape(-1).cpu().numpy().tolist())

        try:
            auc = float(roc_auc_score(all_labels, all_preds))
        except ValueError:
            auc = 0.5

        scheduler.step(auc)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        avg_loss = total_loss / max(grad_steps, 1)
        history["train_loss"].append(avg_loss)
        history["test_auc"].append(float(auc))
        history["bad_batches"].append(int(bad_batches))

        if track_grad_norm:
            avg_grad_norm = total_grad_norm / max(grad_steps, 1)
            history["grad_norm"].append(avg_grad_norm)
            pbar.set_postfix(
                {
                    "loss": avg_loss,
                    "test_auc": auc,
                    "grad": avg_grad_norm,
                    "bad": bad_batches,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
        else:
            pbar.set_postfix(
                {
                    "loss": avg_loss,
                    "test_auc": auc,
                    "bad": bad_batches,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_auc, history, model
