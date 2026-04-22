import math
import random
import tempfile
import json
import os
from pathlib import Path
from functools import partial
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error
from tqdm.auto import tqdm

from get import build_adamw_optimizer, collate_get_batch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def mean_std(xs):
    arr = np.array(xs, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


def _signed_margin_loss(logits, target, margin):
    signed_target = torch.where(target.view(-1) >= 0.5, 1.0, -1.0)
    return torch.relu(float(margin) - signed_target * logits.view(-1)).mean()


def split_grouped_dataset(dataset, split_key, seed, train_ratio=0.70, val_ratio=0.15):
    ids = sorted({g[split_key] for g in dataset})
    if len(ids) == 0:
        return [], [], []
    if len(ids) == 1:
        return list(dataset), list(dataset[:1]), list(dataset[:1])

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

    rng = random.Random(int(seed))

    def _partition_ids(pool, rng_obj, train_r, val_r):
        pool = list(pool)
        rng_obj.shuffle(pool)
        if len(pool) < 3:
            if len(pool) == 2:
                return [pool[0]], [pool[1]], [pool[1]]
            return pool, [], []

        train_cut = max(1, int(train_r * len(pool)))
        val_count = max(1, int(val_r * len(pool)))
        if train_cut + val_count >= len(pool):
            train_cut = max(1, len(pool) - 2)
            val_count = 1
        val_cut = train_cut + val_count
        return pool[:train_cut], pool[train_cut:val_cut], pool[val_cut:]

    if homogeneous_binary:
        train_parts, val_parts, test_parts = [], [], []
        for label in (0.0, 1.0):
            class_ids = [group_id for group_id in ids if next(iter(labels_by_id[group_id])) == label]
            train_i, val_i, test_i = _partition_ids(class_ids, rng, train_ratio, val_ratio)
            train_parts.extend(train_i)
            val_parts.extend(val_i)
            test_parts.extend(test_i)
        rng.shuffle(train_parts)
        rng.shuffle(val_parts)
        rng.shuffle(test_parts)
        train_ids = set(train_parts)
        val_ids = set(val_parts)
        test_ids = set(test_parts)
    else:
        train_i, val_i, test_i = _partition_ids(ids, rng, train_ratio, val_ratio)
        train_ids = set(train_i)
        val_ids = set(val_i)
        test_ids = set(test_i)

    train_data = [g for g in dataset if g[split_key] in train_ids]
    val_data = [g for g in dataset if g[split_key] in val_ids]
    test_data = [g for g in dataset if g[split_key] in test_ids]
    return train_data, val_data, test_data

def save_results(name, payload):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    path = Path("outputs") / f"{name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {path}")

def load_tu_dataset(dataset_name, limit=None):
    from torch_geometric.datasets import TUDataset
    ds = TUDataset(root="data/TU", name=dataset_name)
    if limit is not None: ds = ds[:int(limit)]
    out = []
    for idx, data in enumerate(ds):
        x = data.x.float() if data.x is not None else torch.ones(data.num_nodes, 1)
        edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        y = data.y.view(-1)[0].long()
        out.append({"x": x, "edges": edges, "y": y, "graph_id": idx})
    return out


def build_dataloader_kwargs(device, num_workers=None, prefetch_factor=2):
    """Return DataLoader kwargs tuned for the target device."""
    is_cuda = str(device).startswith("cuda")
    if num_workers is None:
        num_workers = 2 if is_cuda else 0
    num_workers = max(0, int(num_workers))

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": is_cuda,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return kwargs


def _build_undirected_adj(num_nodes: int, edge_index: torch.Tensor) -> list[set[int]]:
    adj = [set() for _ in range(int(num_nodes))]
    if edge_index.numel() == 0:
        return adj
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        ui = int(u)
        vi = int(v)
        if ui == vi:
            continue
        adj[ui].add(vi)
        adj[vi].add(ui)
    return adj


def _k_hop_nodes(adj: list[set[int]], center: int, num_hops: int) -> list[int]:
    seen = {int(center)}
    q = deque([(int(center), 0)])
    while q:
        node, depth = q.popleft()
        if depth >= num_hops:
            continue
        for nbr in adj[node]:
            if nbr not in seen:
                seen.add(nbr)
                q.append((nbr, depth + 1))
    return sorted(seen)


def _build_ego_graph_dataset_dgl(data, centers: list[int], num_hops: int) -> list[dict] | None:
    try:
        import dgl
    except Exception:
        return None

    n = int(data.num_nodes)
    edge_index = data.edge_index
    x_all = data.x
    y_all = data.y
    src = edge_index[0].to(dtype=torch.long)
    dst = edge_index[1].to(dtype=torch.long)
    g = dgl.graph((src, dst), num_nodes=n)

    samples: list[dict] = []
    for center in centers:
        sg, _inv = dgl.khop_in_subgraph(g, int(center), k=int(num_hops), relabel_nodes=True)
        global_nodes = sg.ndata[dgl.NID].to(dtype=torch.long)
        sg_src, sg_dst = sg.edges()
        local_edges = []
        for u, v in zip(sg_src.tolist(), sg_dst.tolist()):
            ui = int(u)
            vi = int(v)
            if ui <= vi:
                local_edges.append((ui, vi))

        x_local = x_all[global_nodes].float()
        y_local = torch.tensor([float(y_all[int(center)].item())], dtype=torch.float32)
        samples.append(
            {
                "x": x_local,
                "edges": local_edges,
                "y": y_local,
                "graph_id": int(center),
            }
        )
    return samples


def build_ego_graph_dataset(data, num_hops: int = 1, limit: int | None = None, use_dgl_if_available: bool = True) -> list[dict]:
    """
    Convert a node-labeled single graph into graph samples via ego subgraphs.

    Expected fields on `data`: num_nodes, edge_index [2, E], x [N, F], y [N].
    Output label is binary float tensor [1] copied from center-node anomaly label.
    """
    n = int(data.num_nodes)
    centers = list(range(n))
    if limit is not None:
        centers = centers[: int(limit)]

    if use_dgl_if_available:
        fast = _build_ego_graph_dataset_dgl(data, centers=centers, num_hops=int(num_hops))
        if fast is not None:
            return fast

    edge_index = data.edge_index
    x_all = data.x
    y_all = data.y
    adj = _build_undirected_adj(n, edge_index)

    samples: list[dict] = []
    for center in centers:
        nodes = _k_hop_nodes(adj, center=center, num_hops=int(num_hops))
        local_index = {node: i for i, node in enumerate(nodes)}

        local_edges = []
        for u in nodes:
            for v in adj[u]:
                if v in local_index and u <= v:
                    local_edges.append((local_index[u], local_index[v]))

        x_local = x_all[nodes].float()
        y_local = torch.tensor([float(y_all[center].item())], dtype=torch.float32)
        samples.append(
            {
                "x": x_local,
                "edges": local_edges,
                "y": y_local,
                "graph_id": int(center),
            }
        )
    return samples


def _label_from_item(item: dict) -> int:
    y = item["y"]
    if isinstance(y, torch.Tensor):
        if y.numel() == 0:
            return 0
        return int(float(y.view(-1)[0].item()) >= 0.5)
    return int(float(y) >= 0.5)


def build_anomaly_protocol_split(
    dataset: list[dict],
    seed: int = 42,
    labeled_rate: float = 0.4,
    val_ratio: int = 1,
    test_ratio: int = 2,
) -> dict[str, list[dict]]:
    """
    Split per ET-style anomaly protocol:
    - train: labeled subset according to `labeled_rate`
    - remaining -> val:test by `val_ratio:test_ratio`
    """
    if len(dataset) == 0:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(int(seed))
    by_label: dict[int, list[dict]] = {0: [], 1: []}
    for item in dataset:
        by_label[_label_from_item(item)].append(item)
    for xs in by_label.values():
        rng.shuffle(xs)

    train: list[dict] = []
    remainder: list[dict] = []

    for label in (0, 1):
        xs = by_label[label]
        if not xs:
            continue
        n_lab = max(1, int(round(float(labeled_rate) * len(xs))))
        n_lab = min(n_lab, len(xs) - 1) if len(xs) > 1 else 1
        train.extend(xs[:n_lab])
        remainder.extend(xs[n_lab:])

    if len(train) == 0:
        train = [dataset[0]]
        remainder = dataset[1:]

    rng.shuffle(remainder)
    total_ratio = int(val_ratio) + int(test_ratio)
    if total_ratio <= 0:
        total_ratio = 3
        val_ratio = 1
        test_ratio = 2

    if len(remainder) == 0:
        return {"train": train, "val": train[:1], "test": train[1:] or train[:1]}

    n_val = max(1, int(round(len(remainder) * (float(val_ratio) / float(total_ratio)))))
    n_val = min(n_val, len(remainder) - 1) if len(remainder) > 1 else 1
    val = remainder[:n_val]
    test = remainder[n_val:]
    if len(test) == 0:
        test = val[-1:]
        val = val[:-1] or test

    return {"train": train, "val": val, "test": test}

class GETTrainer:
    """Unified trainer for GET experiments supporting Classification and Regression."""
    def __init__(self, model, task_type='binary', device='cuda', **kwargs):
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device
        self.lr = kwargs.get('lr', 1e-4)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.margin_loss_weight = float(kwargs.get('margin_loss_weight', 0.0))
        self.logit_margin = float(kwargs.get('logit_margin', 1.0))
        self.use_amp = (device == 'cuda' or 'cuda' in str(device))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        
        self.optimizer = build_adamw_optimizer(self.model, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min' if task_type == 'regression' else 'max', 
            factor=0.5, patience=10
        )
        self.loader_num_workers = kwargs.get('num_workers', None)
        self.loader_prefetch_factor = kwargs.get('prefetch_factor', 2)
        
        # Loss selection
        if task_type == 'regression':
            self.criterion = nn.L1Loss()
        elif task_type == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        if len(loader) == 0:
            return 0.0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=torch.device(self.device).type, enabled=self.use_amp):
                out, _ = self.model(batch, task_level='graph')
                # Align shapes
                if self.task_type in ['binary', 'regression']:
                    loss = self.criterion(out.view(-1), batch.y.view(-1).float())
                    if self.task_type == 'binary' and self.margin_loss_weight > 0.0:
                        loss = loss + self.margin_loss_weight * _signed_margin_loss(out.view(-1), batch.y.view(-1).float(), self.logit_margin)
                else:
                    loss = self.criterion(out, batch.y.view(-1).long())
            
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_y = [], []
        total_loss = 0
        if len(loader) == 0:
            metrics = {'loss': 0.0}
            metrics['metric'] = 0.5 if self.task_type == 'binary' else 0.0
            return metrics
        for batch in loader:
            batch = batch.to(self.device)
            with torch.autocast(device_type=torch.device(self.device).type, enabled=self.use_amp):
                out, _ = self.model(batch, task_level='graph')
                if self.task_type in ['binary', 'regression']:
                    total_loss += self.criterion(out.view(-1), batch.y.view(-1).float()).item()
                    pred = torch.sigmoid(out).view(-1) if self.task_type == 'binary' else out.view(-1)
                else:
                    total_loss += self.criterion(out, batch.y.view(-1).long()).item()
                    pred = out.argmax(dim=-1)
            all_preds.extend(pred.cpu().numpy().tolist())
            all_y.extend(batch.y.view(-1).cpu().numpy().tolist())
        
        metrics = {'loss': total_loss / max(len(loader), 1)}
        if self.task_type == 'binary':
            metrics['metric'] = roc_auc_score(all_y, all_preds) if len(set(all_y)) > 1 else 0.5
        elif self.task_type == 'regression':
            metrics['metric'] = mean_absolute_error(all_y, all_preds)
        else:
            metrics['metric'] = accuracy_score(all_y, all_preds)
        return metrics

    def run(self, train_ds, val_ds, test_ds, epochs, batch_size, collate_fn=collate_get_batch):
        loader_kwargs = build_dataloader_kwargs(
            self.device,
            num_workers=self.loader_num_workers,
            prefetch_factor=self.loader_prefetch_factor,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **loader_kwargs)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, **loader_kwargs)
        
        best_metric = float('inf') if self.task_type == 'regression' else -float('inf')
        best_state = None
        
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            train_loss = self.train_epoch(train_loader)
            val_res = self.evaluate(val_loader)
            self.scheduler.step(val_res['loss'] if self.task_type == 'regression' else val_res['metric'])
            
            is_best = (val_res['metric'] < best_metric) if self.task_type == 'regression' else (val_res['metric'] > best_metric)
            if is_best:
                best_metric = val_res['metric']
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            pbar.set_postfix({'t_loss': f"{train_loss:.3f}", 'v_metric': f"{val_res['metric']:.3f}"})
        
        if best_state:
            self.model.load_state_dict(best_state)
        return self.evaluate(test_loader)
