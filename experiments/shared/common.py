import random
import json
import time
import inspect
from pathlib import Path

import numpy as np
from numba import njit
import torch
from torch.nn.parameter import UninitializedParameter
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from tqdm.auto import tqdm

from get import build_adamw_optimizer, collate_get_batch, validate_get_batch

def get_num_params(model):
    """Return formatted string of trainable parameter count."""
    num_params = sum(
        p.numel() for p in model.parameters() 
        if p.requires_grad and not isinstance(p, UninitializedParameter)
    )
    if num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    if num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    return str(num_params)

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

def save_results(name, payload, metadata=None):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    path = Path("outputs") / f"{name}.json"
    
    data = payload
    if metadata is not None:
        data = {"results": payload, "metadata": metadata}
        
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")

def load_tu_dataset(dataset_name, limit=None):
    from torch_geometric.datasets import TUDataset
    tu_root = Path("data") / "TU"
    root = tu_root
    name = dataset_name
    if str(dataset_name).upper() == "MUTAGENICITY":
        root = tu_root / "MUTAGENICITY"
        name = "Mutagenicity"
    ds = TUDataset(root=str(root), name=name)
    if limit is not None:
        ds = ds[:int(limit)]
    out = []
    for idx, data in enumerate(ds):
        x = data.x.float() if data.x is not None else torch.ones(data.num_nodes, 1)
        edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        y = data.y.view(-1)[0].long()
        out.append({"x": x, "edges": edges, "y": y, "graph_id": idx})
    return out


def build_dataloader_kwargs(device, num_workers=None, prefetch_factor=4):
    """Return DataLoader kwargs tuned for aggressive asynchronous prefetching."""
    is_cuda = str(device).startswith("cuda")
    if num_workers is None:
        # Scale workers with CPU count, capped for stability
        import os
        num_workers = min(os.cpu_count() or 4, 8) if is_cuda else 0
    num_workers = max(0, int(num_workers))

    kwargs = {
        "num_workers": num_workers,
        "pin_memory": is_cuda,
    }
    if num_workers > 0:
        # persistent_workers=True keeps motifs/PEs in worker memory across epochs
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
    return kwargs


@njit
def _numba_build_csr_adj(num_nodes, edge_index):
    """Build CSR adjacency representation directly in Numba."""
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    E = edge_index.shape[1]
    
    # Count degrees
    for i in range(E):
        u, v = edge_index[0, i], edge_index[1, i]
        if u == v:
            continue
        indptr[u + 1] += 1
        indptr[v + 1] += 1
        
    for i in range(num_nodes):
        indptr[i + 1] += indptr[i]
        
    indices = np.empty(indptr[num_nodes], dtype=np.int64)
    curr_idx = indptr[:-1].copy()
    for i in range(E):
        u, v = edge_index[0, i], edge_index[1, i]
        if u == v:
            continue
        indices[curr_idx[u]] = v
        curr_idx[u] += 1
        indices[curr_idx[v]] = u
        curr_idx[v] += 1
        
    return indptr, indices


@njit
def _numba_k_hop_nodes(indptr, indices, center, num_hops):
    """Numba-accelerated BFS for k-hop neighborhood."""
    num_nodes = len(indptr) - 1
    # Use a boolean array for 'seen' instead of a set
    visited = np.zeros(num_nodes, dtype=np.bool_)
    
    # Simple queue
    q = np.empty(num_nodes, dtype=np.int64)
    dist = np.empty(num_nodes, dtype=np.int32)
    
    head = 0
    tail = 0
    
    q[tail] = center
    dist[tail] = 0
    visited[center] = True
    tail += 1
    
    while head < tail:
        u = q[head]
        d = dist[head]
        head += 1
        
        if d >= num_hops:
            continue
            
        for i in range(indptr[u], indptr[u+1]):
            v = indices[i]
            if not visited[v]:
                visited[v] = True
                q[tail] = v
                dist[tail] = d + 1
                tail += 1
                
    # Extract only visited indices
    res = np.empty(tail, dtype=np.int64)
    for i in range(tail):
        res[i] = q[i]
    res.sort()
    return res


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


@njit
def _numba_extract_subgraph_edges(indptr, indices, nodes):
    """Numba-accelerated extraction of edges within a subset of nodes."""
    # Build a local mapping array (faster than a dict in Numba)
    # Map global_id -> local_id
    max_global = -1
    for n in nodes:
        if n > max_global:
            max_global = n
            
    mapping = np.full(max_global + 1, -1, dtype=np.int64)
    for i in range(len(nodes)):
        mapping[nodes[i]] = i
        
    # Estimate max possible edges (worst case: complete graph)
    # We use a dynamic list growth pattern in logic, but here we'll pre-count or use a buffer
    num_nodes = len(nodes)
    
    # Pass 1: count edges
    count = 0
    for i in range(num_nodes):
        u_global = nodes[i]
        for idx in range(indptr[u_global], indptr[u_global+1]):
            v_global = indices[idx]
            if v_global <= max_global:
                v_local = mapping[v_global]
                if v_local != -1 and i <= v_local:
                    count += 1
                    
    # Pass 2: fill edges
    local_src = np.empty(count, dtype=np.int64)
    local_dst = np.empty(count, dtype=np.int64)
    curr = 0
    for i in range(num_nodes):
        u_global = nodes[i]
        for idx in range(indptr[u_global], indptr[u_global+1]):
            v_global = indices[idx]
            if v_global <= max_global:
                v_local = mapping[v_global]
                if v_local != -1 and i <= v_local:
                    local_src[curr] = i
                    local_dst[curr] = v_local
                    curr += 1
    return local_src, local_dst


def build_ego_graph_dataset(data, num_hops: int = 1, limit: int | None = None, use_dgl_if_available: bool = True) -> list[dict]:
    """
    Convert a node-labeled single graph into graph samples via ego subgraphs.
    """
    n = int(data.num_nodes)
    centers = list(range(n))
    if limit is not None:
        centers = centers[: int(limit)]

    if use_dgl_if_available:
        fast = _build_ego_graph_dataset_dgl(data, centers=centers, num_hops=int(num_hops))
        if fast is not None:
            return fast

    edge_index_np = data.edge_index.detach().cpu().numpy().astype(np.int64)
    indptr, indices = _numba_build_csr_adj(n, edge_index_np)
    
    x_all = data.x
    y_all = data.y

    samples: list[dict] = []
    for center in tqdm(centers, desc=f"Building ego graphs (h={int(num_hops)})", leave=False):
        # Numba-accelerated BFS
        nodes = _numba_k_hop_nodes(indptr, indices, int(center), int(num_hops))
        
        # Numba-accelerated subgraph extraction
        l_src, l_dst = _numba_extract_subgraph_edges(indptr, indices, nodes)
        local_edges = list(zip(l_src.tolist(), l_dst.tolist()))

        x_local = x_all[nodes].float()
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
    def __init__(self, model, task_type='binary', device='cuda', model_name=None, **kwargs):
        self.model = model.to(device)
        self.task_type = task_type
        self.device = device
        self.model_name = model_name or model.__class__.__name__
        
        self.lr = kwargs.get('lr', 1e-3)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.max_grad_norm = kwargs.get('max_grad_norm', 1.0)
        self.max_grad_val = kwargs.get('max_grad_val', 0.01) # Hard value clip
        self.opt_type = kwargs.get('opt_type', 'adam')
        self.margin_loss_weight = float(kwargs.get('margin_loss_weight', 0.0))
        self.logit_margin = float(kwargs.get('logit_margin', 1.0))
        
        self.use_amp = (device == 'cuda' or 'cuda' in str(device))
        self.autocast_device_type = torch.device(self.device).type
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.supports_armijo = 'inference_mode' in inspect.signature(self.model.forward).parameters
        self.validate_batches = bool(kwargs.get('validate_batches', True))
        
        if self.opt_type == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            self.optimizer = build_adamw_optimizer(self.model, lr=self.lr, weight_decay=self.weight_decay)
            
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min' if task_type == 'regression' else 'max', 
            factor=0.5, patience=10
        )
        self.loader_num_workers = kwargs.get('num_workers', None)
        self.loader_prefetch_factor = kwargs.get('prefetch_factor', 2)
        
        # Loss selection (placeholder, can be updated in run() with data-dependent weights)
        if task_type == 'regression':
            self.criterion = nn.L1Loss()
        elif task_type in ['binary', 'multilabel']:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

    def _prepare_weighted_criterion(self, dataset):
        """Compute pos_weight for imbalanced binary/multilabel datasets."""
        if self.task_type not in ['binary', 'multilabel']:
            return
        
        ys = []
        for item in dataset:
            y = item['y']
            ys.append(y.view(-1).float())
        ys = torch.stack(ys, dim=0)
        
        pos = (ys > 0.5).sum(dim=0).float()
        neg = (ys <= 0.5).sum(dim=0).float()
        
        # pos_weight = neg / pos
        weights = torch.ones_like(pos)
        mask = (pos > 0) & (neg > 0)
        weights[mask] = neg[mask] / pos[mask]
        
        print(f"INFO:    Applied class weights: {weights.cpu().numpy()}")
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weights.to(self.device))

    def log_info(self):
        """Print a summary of the model and training configuration."""
        print("-" * 50)
        print(f"EXPERIMENT: {self.model_name}")
        print(f"TASK:       {self.task_type.upper()}")
        print(f"DEVICE:     {self.device}")
        print(f"PARAMS:     {get_num_params(self.model)}")
        
        # Detect GET-specific config
        target = self.model
        if hasattr(target, 'get_layer'):
            layer = target.get_layer
            steps = getattr(target, 'num_steps', '?')
            print(f"CONFIG:     d={layer.d}, H={layer.num_heads}, steps={steps}")
            if hasattr(layer, 'R'):
                print(f"            R={layer.R}, K={layer.K}")
            print(f"COUPLINGS:  λ2={torch.nn.functional.softplus(layer.lambda_2).item():.3f}, "
                  f"λ3={torch.nn.functional.softplus(layer.lambda_3).item():.3f}, "
                  f"λm={torch.nn.functional.softplus(layer.lambda_m).item():.3f}")
        
        print(f"HYPER-P:    lr={self.lr}, wd={self.weight_decay}, clip={self.max_grad_norm}")
        print("-" * 50)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        if len(loader) == 0:
            return 0.0
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            if self.validate_batches:
                validate_get_batch(batch)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.autocast_device_type, enabled=self.use_amp):
                out, _ = self.model(batch, task_level='graph')
                # Align shapes
                if self.task_type in ['binary', 'regression']:
                    loss = self.criterion(out.view(-1), batch.y.view(-1).float())
                    if self.task_type == 'binary' and self.margin_loss_weight > 0.0:
                        loss = loss + self.margin_loss_weight * _signed_margin_loss(out.view(-1), batch.y.view(-1).float(), self.logit_margin)
                elif self.task_type == 'multilabel':
                    loss = self.criterion(out, batch.y.view(out.shape).float())
                else:
                    loss = self.criterion(out, batch.y.view(-1).long())
            
            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_val)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_value_(self.model.parameters(), self.max_grad_val)
                self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader, inference_mode='fixed'):
        self.model.eval()
        all_preds, all_y = [], []
        total_loss = 0
        if len(loader) == 0:
            metrics = {'loss': 0.0}
            metrics['metric'] = 0.5 if self.task_type == 'binary' else 0.0
            return metrics
        for batch in loader:
            batch = batch.to(self.device, non_blocking=True)
            if self.validate_batches:
                validate_get_batch(batch)
            use_autocast = self.use_amp and not (inference_mode == 'armijo' and self.supports_armijo)
            with torch.autocast(device_type=self.autocast_device_type, enabled=use_autocast):
                # Use Armijo if requested and supported by the model
                forward_kwargs = {'task_level': 'graph'}
                if inference_mode == 'armijo' and self.supports_armijo:
                    forward_kwargs['inference_mode'] = 'armijo'
                
                out, _ = self.model(batch, **forward_kwargs)
                if self.task_type in ['binary', 'regression']:
                    total_loss += self.criterion(out.view(-1), batch.y.view(-1).float()).item()
                    pred = torch.sigmoid(out).view(-1) if self.task_type == 'binary' else out.view(-1)
                    all_preds.extend(pred.cpu().numpy().tolist())
                    all_y.extend(batch.y.view(-1).cpu().numpy().tolist())
                elif self.task_type == 'multilabel':
                    total_loss += self.criterion(out, batch.y.view(out.shape).float()).item()
                    pred = torch.sigmoid(out)
                    all_preds.append(pred.cpu().numpy())
                    all_y.append(batch.y.view(out.shape).cpu().numpy())
                else:
                    total_loss += self.criterion(out, batch.y.view(-1).long()).item()
                    pred = out.argmax(dim=-1)
                    all_preds.extend(pred.cpu().numpy().tolist())
                    all_y.extend(batch.y.view(-1).cpu().numpy().tolist())
        
        metrics = {'loss': total_loss / max(len(loader), 1)}
        if self.task_type == 'binary':
            metrics['metric'] = roc_auc_score(all_y, all_preds) if len(set(all_y)) > 1 else 0.5
            y_pred_bin = [int(p >= 0.5) for p in all_preds]
            metrics['accuracy'] = accuracy_score(all_y, y_pred_bin)
        elif self.task_type == 'regression':
            metrics['metric'] = mean_absolute_error(all_y, all_preds)
        elif self.task_type == 'multilabel':
            y_true_arr = np.concatenate(all_y, axis=0)
            y_pred_arr = np.concatenate(all_preds, axis=0)
            from sklearn.metrics import average_precision_score
            aps = []
            for c in range(y_true_arr.shape[1]):
                if len(set(y_true_arr[:, c].tolist())) < 2:
                    continue
                aps.append(average_precision_score(y_true_arr[:, c], y_pred_arr[:, c]))
            metrics['metric'] = float(np.mean(aps)) if aps else 0.0
        else:
            metrics['metric'] = accuracy_score(all_y, all_preds)
        return metrics

    def run(self, train_ds, val_ds, test_ds, epochs, batch_size, collate_fn=collate_get_batch, use_armijo_at_test=True, use_weighted_loss=False):
        if use_weighted_loss:
            self._prepare_weighted_criterion(train_ds)
        self.log_info()
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
        
        param_cnt = get_num_params(self.model)
        pbar = tqdm(range(epochs), desc=f"Train {self.model_name} [{param_cnt}]", bar_format='{l_bar}{bar:20}{r_bar}')
        
        history = {'train_loss': [], 'val_metric': []}
        for epoch in pbar:
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            # Diagnostic: Print motif gradient norms if they exist
            if epoch % 5 == 0 or epoch == epochs - 1:
                motif_grad_norm = 0.0
                count = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        # Try to identify motif parameters by their shape or if they are in the GETLayer
                        if p.shape == torch.Size([]) or (p.dim() > 1 and p.shape[-1] == self.model.d if hasattr(self.model, 'd') else False):
                            motif_grad_norm += p.grad.norm().item()
                            count += 1
                if count > 0:
                    print(f"Epoch {epoch} | Avg Param Grad Norm: {motif_grad_norm/count:.6f}")
            # Use fixed during validation for speed
            val_res = self.evaluate(val_loader, inference_mode='fixed')
            epoch_time = time.time() - t0
            
            history['train_loss'].append(train_loss)
            history['val_metric'].append(val_res['metric'])
            
            self.scheduler.step(val_res['loss'] if self.task_type == 'regression' else val_res['metric'])
            
            is_best = (val_res['metric'] < best_metric) if self.task_type == 'regression' else (val_res['metric'] > best_metric)
            if is_best:
                best_metric = val_res['metric']
                best_state = {}
                for k, v in self.model.state_dict().items():
                    if isinstance(v, UninitializedParameter):
                        continue
                    best_state[k] = v.cpu().clone()
            
            metrics_str = f"L: {train_loss:.3f} | V: {val_res['metric']:.3f} | B: {best_metric:.3f} | {epoch_time:.1f}s/ep"
            pbar.set_postfix_str(metrics_str)
        
        if best_state:
            self.model.load_state_dict(best_state, strict=False)
        
        # Use Armijo for the final test evaluation as per paper
        test_mode = 'armijo' if use_armijo_at_test else 'fixed'
        test_res = self.evaluate(test_loader, inference_mode=test_mode)
        test_res['history'] = history
        return test_res
