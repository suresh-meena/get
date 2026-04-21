import random
import tempfile
from collections import deque

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm

from get import build_adamw_optimizer
from get.compile_utils import maybe_compile_model
from get.data import collate_get_batch


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


def train_eval_graph_classification(
    model_name,
    model,
    dataset,
    num_classes,
    epochs,
    batch_size,
    device,
    lr,
    max_grad_norm,
    weight_decay=1e-4,
    seed=0,
    compile_model=False,
    use_amp=True,
    amp_dtype="float16",
    num_workers=0,
    pin_memory=None,
):
    model = model.to(device)
    model = maybe_compile_model(model, compile_model, model_name=model_name)
    optimizer = build_adamw_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()

    indices = list(range(len(dataset)))
    split_rng = random.Random(seed)
    split_rng.shuffle(indices)
    cut = int(0.8 * len(indices))
    train_idx = set(indices[:cut])
    train_data = [dataset[i] for i in range(len(dataset)) if i in train_idx]
    test_data = [dataset[i] for i in range(len(dataset)) if i not in train_idx]

    if pin_memory is None:
        pin_memory = bool(device.type == "cuda")
    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_torch_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")

    best_acc = 0.0
    best_ckpt_path = None
    history = {"train_loss": [], "test_acc": [], "bad_batches": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        best_ckpt_path = f"{tmpdir}/best_{model_name.replace('/', '_')}.pt"
        train_loader = _make_loader(train_data, batch_size, True, num_workers, pin_memory)
        test_loader = _make_loader(test_data, batch_size, False, num_workers, pin_memory)
        pbar = tqdm(range(epochs), desc=f"Training {model_name}")
        for _ in pbar:
            model.train()
            total_loss = 0.0
            grad_steps = 0
            bad_batches = 0

            for batch in train_loader:
                batch = batch.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()

                with torch.autocast(device_type=device.type, dtype=amp_torch_dtype, enabled=amp_enabled):
                    out, _ = model(batch, task_level="graph")
                    y = batch.y.reshape(-1).long()
                    if not torch.isfinite(out).all():
                        bad_batches += 1
                        continue
                    loss = criterion(out, y)
                    if not torch.isfinite(loss):
                        bad_batches += 1
                        continue

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not np.isfinite(float(grad_norm)):
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
                grad_steps += 1

            model.eval()
            all_pred = []
            all_y = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device, non_blocking=pin_memory)
                    with torch.autocast(device_type=device.type, dtype=amp_torch_dtype, enabled=amp_enabled):
                        out, _ = model(batch, task_level="graph")
                    pred = out.argmax(dim=-1)
                    all_pred.extend(pred.cpu().numpy().tolist())
                    all_y.extend(batch.y.reshape(-1).cpu().numpy().tolist())

            acc = float(accuracy_score(all_y, all_pred)) if all_y else 0.0
            scheduler.step(acc)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), best_ckpt_path)

            avg_loss = total_loss / max(grad_steps, 1)
            history["train_loss"].append(avg_loss)
            history["test_acc"].append(acc)
            history["bad_batches"].append(int(bad_batches))
            pbar.set_postfix({"loss": avg_loss, "test_acc": acc, "bad": bad_batches})

        if best_acc > 0.0:
            model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
    return best_acc, history


def train_eval_graph_anomaly(
    model_name,
    model,
    dataset,
    epochs,
    batch_size,
    device,
    lr,
    max_grad_norm,
    weight_decay=1e-4,
    seed=0,
    compile_model=False,
    split_data=None,
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

    if split_data is None:
        indices = list(range(len(dataset)))
        split_rng = random.Random(seed)
        split_rng.shuffle(indices)
        cut = int(0.8 * len(indices))
        train_idx = set(indices[:cut])
        train_data = [dataset[i] for i in range(len(dataset)) if i in train_idx]
        val_data = []
        test_data = [dataset[i] for i in range(len(dataset)) if i not in train_idx]
    else:
        train_data = split_data["train"]
        val_data = split_data["val"]
        test_data = split_data["test"]

    if pin_memory is None:
        pin_memory = bool(device.type == "cuda")
    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_torch_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device.type == "cuda")

    best_auc = 0.0
    best_ckpt_path = None
    history = {"train_loss": [], "val_auc": [], "test_auc": [], "bad_batches": []}

    with tempfile.TemporaryDirectory() as tmpdir:
        best_ckpt_path = f"{tmpdir}/best_{model_name.replace('/', '_')}.pt"
        train_loader = _make_loader(train_data, batch_size, True, num_workers, pin_memory)
        val_loader = _make_loader(val_data, batch_size, False, num_workers, pin_memory) if len(val_data) > 0 else None
        test_loader = _make_loader(test_data, batch_size, False, num_workers, pin_memory)
        pbar = tqdm(range(epochs), desc=f"Training {model_name}")
        for _ in pbar:
            model.train()
            total_loss = 0.0
            grad_steps = 0
            bad_batches = 0

            for batch in train_loader:
                batch = batch.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, dtype=amp_torch_dtype, enabled=amp_enabled):
                    out, _ = model(batch, task_level="graph")
                    out = out.reshape(-1)
                    y = batch.y.reshape(-1).float()
                    if not torch.isfinite(out).all():
                        bad_batches += 1
                        continue
                    loss = criterion(out, y)
                    if not torch.isfinite(loss):
                        bad_batches += 1
                        continue

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if not np.isfinite(float(grad_norm)):
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
                grad_steps += 1

            model.eval()

            def eval_auc(eval_loader):
                all_score = []
                all_y = []
                if eval_loader is None:
                    return 0.5
                with torch.no_grad():
                    for batch in eval_loader:
                        batch = batch.to(device, non_blocking=pin_memory)
                        with torch.autocast(device_type=device.type, dtype=amp_torch_dtype, enabled=amp_enabled):
                            out, _ = model(batch, task_level="graph")
                        out = torch.nan_to_num(out.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
                        all_score.extend(torch.sigmoid(out).cpu().numpy().tolist())
                        all_y.extend(batch.y.reshape(-1).cpu().numpy().tolist())
                try:
                    return float(roc_auc_score(all_y, all_score))
                except ValueError:
                    return 0.5

            val_auc = eval_auc(val_loader) if val_loader is not None else eval_auc(test_loader)
            test_auc = eval_auc(test_loader)

            scheduler.step(val_auc)
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_ckpt_path)

            avg_loss = total_loss / max(grad_steps, 1)
            history["train_loss"].append(avg_loss)
            history["val_auc"].append(val_auc)
            history["test_auc"].append(test_auc)
            history["bad_batches"].append(int(bad_batches))
            pbar.set_postfix({"loss": avg_loss, "val_auc": val_auc, "test_auc": test_auc, "bad": bad_batches})

        if best_auc > 0.0:
            model.load_state_dict(torch.load(best_ckpt_path, map_location="cpu"))
    final_test_auc = history["test_auc"][-1] if history["test_auc"] else 0.5
    return final_test_auc, history


def generate_synth_graph_classification(num_graphs=300, n_nodes=24, seed=0, num_classes=3):
    rng = random.Random(seed)
    dataset = []
    for graph_id in range(num_graphs):
        p = 0.1 + 0.4 * rng.random()
        G = nx.erdos_renyi_graph(n_nodes, p, seed=rng.randint(0, 10**9))
        tri = sum(nx.triangles(G).values()) // 3
        label = int(np.clip((tri / max(1, n_nodes)) * 1.5, 0, num_classes - 1))
        dataset.append(
            {
                "x": torch.ones(n_nodes, 1, dtype=torch.float32),
                "edges": list(G.edges()),
                "y": torch.tensor([label], dtype=torch.long),
                "graph_id": graph_id,
            }
        )
    return dataset


def generate_synth_graph_anomaly(num_graphs=300, n_nodes=24, seed=0):
    rng = random.Random(seed)
    dataset = []
    for graph_id in range(num_graphs):
        is_anomaly = 1 if rng.random() < 0.25 else 0
        p = 0.08 if is_anomaly else 0.22
        G = nx.erdos_renyi_graph(n_nodes, p, seed=rng.randint(0, 10**9))
        dataset.append(
            {
                "x": torch.ones(n_nodes, 1, dtype=torch.float32),
                "edges": list(G.edges()),
                "y": torch.tensor([float(is_anomaly)], dtype=torch.float32),
                "graph_id": graph_id,
            }
        )
    return dataset


def load_tu_dataset(dataset_name, limit=None):
    try:
        from torch_geometric.datasets import TUDataset
    except Exception as exc:
        raise RuntimeError("torch_geometric is required for TU datasets") from exc

    ds = TUDataset(root="data/TU", name=dataset_name)
    if limit is not None:
        ds = ds[: int(limit)]

    out = []
    for idx, data in enumerate(ds):
        if data.x is None:
            x = torch.ones(data.num_nodes, 1, dtype=torch.float32)
        else:
            x = data.x.float()
        edges = list(zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()))
        y = data.y.view(-1)[0].long()
        out.append({"x": x, "edges": edges, "y": y.view(1), "graph_id": idx})
    return out


def _adj_from_edge_index(num_nodes, edge_index):
    adj = [set() for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for u, v in zip(src, dst):
        u_i = int(u)
        v_i = int(v)
        adj[u_i].add(v_i)
        adj[v_i].add(u_i)
    return adj


def _k_hop_nodes(center, adj, num_hops):
    seen = {int(center)}
    q = deque([(int(center), 0)])
    while q:
        node, dist = q.popleft()
        if dist >= num_hops:
            continue
        for nb in adj[node]:
            if nb not in seen:
                seen.add(nb)
                q.append((nb, dist + 1))
    return sorted(seen)


def build_ego_graph_dataset(data, num_hops=2, limit=None):
    """Convert a single node-labeled graph into graph-level ego samples.

    Each node becomes one sample whose label is derived from the node label.
    Non-binary labels are mapped to {0,1} by checking label > 0.
    """
    num_nodes = int(data.num_nodes)
    x_all = data.x.float() if data.x is not None else torch.ones(num_nodes, 1, dtype=torch.float32)
    y_all = data.y.view(-1)
    if y_all.numel() != num_nodes:
        raise ValueError("Expected node-level labels with length equal to num_nodes.")

    adj = _adj_from_edge_index(num_nodes, data.edge_index)
    samples = []
    total = num_nodes if limit is None else min(int(limit), num_nodes)

    for center in range(total):
        keep = _k_hop_nodes(center, adj, num_hops=num_hops)
        old_to_new = {old: new for new, old in enumerate(keep)}

        edges = []
        for old_u in keep:
            for old_v in adj[old_u]:
                if old_v in old_to_new:
                    edges.append((old_to_new[old_u], old_to_new[old_v]))

        x = x_all[torch.tensor(keep, dtype=torch.long)]
        y = torch.tensor([1.0 if int(y_all[center].item()) > 0 else 0.0], dtype=torch.float32)
        samples.append({"x": x, "edges": edges, "y": y, "graph_id": center})
    return samples


def load_node_anomaly_dataset(dataset_name, root="data/node", num_hops=2, limit=None):
    """Load node-labeled datasets and convert them to ego-graph anomaly samples."""
    try:
        from torch_geometric.datasets import AmazonProducts, Yelp
    except Exception as exc:
        raise RuntimeError("torch_geometric is required for node anomaly datasets") from exc

    name = dataset_name.lower()
    if name == "yelp":
        ds = Yelp(root=f"{root}/Yelp")
    elif name in {"amazon", "amazonproducts", "amazon_products"}:
        ds = AmazonProducts(root=f"{root}/AmazonProducts")
    else:
        raise ValueError(f"Unsupported node anomaly dataset: {dataset_name}")

    data = ds[0]
    return build_ego_graph_dataset(data, num_hops=num_hops, limit=limit)


def build_anomaly_protocol_split(dataset, seed=0, labeled_rate=0.01, val_ratio=1, test_ratio=2):
    """Build protocol-like anomaly splits: train uses a small labeled fraction, rest split into val/test."""
    if len(dataset) == 0:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    labels = [int(float(g["y"].item()) > 0.0) for g in dataset]
    idx_pos = [i for i, y in enumerate(labels) if y == 1]
    idx_neg = [i for i, y in enumerate(labels) if y == 0]
    rng.shuffle(idx_pos)
    rng.shuffle(idx_neg)

    def take_count(n):
        return max(1, int(round(labeled_rate * n)))

    n_pos_train = min(len(idx_pos), take_count(len(idx_pos)))
    n_neg_train = min(len(idx_neg), take_count(len(idx_neg)))
    train_idx = set(idx_pos[:n_pos_train] + idx_neg[:n_neg_train])

    remaining = [i for i in range(len(dataset)) if i not in train_idx]
    rng.shuffle(remaining)
    denom = max(1, val_ratio + test_ratio)
    val_cut = int(len(remaining) * (val_ratio / denom))
    val_idx = set(remaining[:val_cut])
    test_idx = set(remaining[val_cut:])

    return {
        "train": [dataset[i] for i in sorted(train_idx)],
        "val": [dataset[i] for i in sorted(val_idx)],
        "test": [dataset[i] for i in sorted(test_idx)],
    }
