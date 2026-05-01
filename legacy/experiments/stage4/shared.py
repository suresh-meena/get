from __future__ import annotations

import copy
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from experiments.shared.common import build_ego_graph_dataset, get_num_params, _normalize_amp_dtype, _should_enable_amp
from experiments.shared.model_config import instantiate_models_from_catalog
from get import collate_get_batch, CachedGraphDataset
from get.data import _graph_dataset_cache_fingerprint

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
except Exception:
    accuracy_score = None
    f1_score = None
    roc_auc_score = None

try:
    from get import GINBaseline
except Exception:
    GINBaseline = None

try:
    from sklearn.model_selection import StratifiedKFold, train_test_split
except Exception:
    StratifiedKFold = None
    train_test_split = None


TU8_DATASETS = ["PROTEINS", "NCI1", "NCI109", "DD", "ENZYMES", "MUTAG", "MUTAGENICITY", "FRANKENSTEIN"]


def _safe_auc(y_true: list[float], y_score: list[float]) -> float:
    n = min(len(y_true), len(y_score))
    if n == 0:
        return 0.5
    if len(y_true) != len(y_score):
        y_true = y_true[:n]
        y_score = y_score[:n]
    if roc_auc_score is None:
        return 0.5
    y_bin = [int(float(v) >= 0.5) for v in y_true]
    if len(set(y_bin)) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_bin, y_score))
    except Exception:
        return 0.5


def _safe_f1(y_true: list[float], y_score: list[float], threshold: float = 0.5) -> float:
    n = min(len(y_true), len(y_score))
    if n == 0:
        return 0.0
    if len(y_true) != len(y_score):
        y_true = y_true[:n]
        y_score = y_score[:n]
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    y_bin = [1 if y >= 0.5 else 0 for y in y_true]
    if f1_score is None:
        def _f1_for(label: int) -> float:
            tp = sum(1 for a, b in zip(y_bin, y_pred) if a == label and b == label)
            fp = sum(1 for a, b in zip(y_bin, y_pred) if a != label and b == label)
            fn = sum(1 for a, b in zip(y_bin, y_pred) if a == label and b != label)
            if tp == 0 and (fp > 0 or fn > 0):
                return 0.0
            denom = (2 * tp + fp + fn)
            return 0.0 if denom == 0 else (2 * tp) / denom
        return float((_f1_for(0) + _f1_for(1)) / 2.0)
    return float(f1_score(y_bin, y_pred, average="macro"))


def _is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return ("out of memory" in msg) and ("cuda" in msg)


def _synthetic_graph(rng: random.Random, num_nodes: int, in_dim: int, motif_boost: bool = False) -> dict:
    x = torch.randn(num_nodes, in_dim)
    edges = set()
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng.random() < 0.1:
                edges.add((u, v))
    for u in range(num_nodes - 1):
        edges.add((u, u + 1))
    if motif_boost:
        for u in range(0, max(0, num_nodes - 2), 3):
            edges.add((u, u + 1))
            edges.add((u + 1, u + 2))
            edges.add((u, u + 2))
    else:
        for u in range(0, max(0, num_nodes - 2), 3):
            if (u, u + 2) in edges:
                edges.remove((u, u + 2))
    return {"x": x, "edges": sorted(edges)}


def make_synth_classification_dataset(num_graphs: int, in_dim: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ds = []
    for i in range(int(num_graphs)):
        label = 1 if i % 2 == 0 else 0
        g = _synthetic_graph(rng, rng.randint(14, 24), in_dim, motif_boost=(label == 1))
        g["y"] = torch.tensor([label], dtype=torch.long)
        g["graph_id"] = i
        ds.append(g)
    return ds


def make_synth_anomaly_dataset(num_graphs: int, in_dim: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ds = []
    for i in range(int(num_graphs)):
        label = 1 if rng.random() < 0.2 else 0
        g = _synthetic_graph(rng, rng.randint(18, 30), in_dim, motif_boost=(label == 1))
        if label == 1:
            g["x"][: min(3, g["x"].size(0))] += 1.0
        g["y"] = torch.tensor([float(label)], dtype=torch.float32)
        g["graph_id"] = i
        ds.append(g)
    return ds


@dataclass
class FitResult:
    metric: float
    history: dict[str, list[float]]
    extra: dict[str, float | list[float]]


class _NodeBudgetBatchSampler:
    """Pack variable-size graphs into batches under a node budget."""

    def __init__(self, dataset: list[dict], node_budget: int, hard_cap: int, shuffle: bool, seed: int):
        self.dataset = dataset
        self.node_budget = max(1, int(node_budget))
        self.hard_cap = max(1, int(hard_cap))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self.sizes = [max(1, int(item["x"].size(0))) for item in dataset]

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(indices)
            self._epoch += 1
        indices.sort(key=lambda idx: self.sizes[idx])

        batch: list[int] = []
        nodes = 0
        for idx in indices:
            size = self.sizes[idx]
            if batch and (nodes + size > self.node_budget or len(batch) >= self.hard_cap):
                yield batch
                batch = []
                nodes = 0
            batch.append(idx)
            nodes += size
        if batch:
            yield batch

    def __len__(self):
        return len(self.dataset)


def _normalized_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _resolve_stage4_num_workers(device: str, requested: int) -> int:
    if str(device).startswith("cuda"):
        return max(0, int(requested))
    return 0


def _local_anomaly_path(dataset_name: str, data_root: str) -> Path | None:
    norm = _normalized_name(dataset_name)
    candidates = [
        Path(data_root) / f"{dataset_name}.pt",
        Path(data_root) / f"{norm}.pt",
        Path("data") / "anomaly" / f"{dataset_name}.pt",
        Path("data") / "anomaly" / f"{norm}.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _to_simple_graph_from_obj(obj) -> SimpleNamespace:
    if isinstance(obj, dict):
        x = obj["x"]
        edge_index = obj["edge_index"]
        y = obj["y"]
        num_nodes = int(obj.get("num_nodes", x.size(0)))
        if isinstance(y, torch.Tensor) and y.dim() > 1:
            y = y.argmax(dim=-1).to(dtype=torch.long)
        return SimpleNamespace(num_nodes=num_nodes, edge_index=edge_index, x=x, y=y)
    if hasattr(obj, "edge_index") and hasattr(obj, "x") and hasattr(obj, "y"):
        num_nodes = int(getattr(obj, "num_nodes", obj.x.size(0)))
        y = obj.y
        if isinstance(y, torch.Tensor) and y.dim() > 1:
            y = y.argmax(dim=-1).to(dtype=torch.long)
        return SimpleNamespace(num_nodes=num_nodes, edge_index=obj.edge_index, x=obj.x, y=y)
    raise ValueError("Unsupported anomaly graph object format.")


def _load_anomaly_graph(dataset_name: str, data_root: str) -> SimpleNamespace:
    norm = _normalized_name(dataset_name)
    if norm in {"yelp", "yelpchi", "amazon"}:
        try:
            import dgl  # noqa: F401
            from dgl.data import FraudAmazonDataset, FraudYelpDataset

            ds = FraudYelpDataset(raw_dir=data_root) if norm in {"yelp", "yelpchi"} else FraudAmazonDataset(raw_dir=data_root)
            g = ds[0]
            if getattr(g, "is_homogeneous", False):
                src, dst = g.edges()
            else:
                src_parts = []
                dst_parts = []
                for etype in g.canonical_etypes:
                    s, d = g.edges(etype=etype)
                    src_parts.append(s.long())
                    dst_parts.append(d.long())
                if src_parts:
                    src = torch.cat(src_parts, dim=0)
                    dst = torch.cat(dst_parts, dim=0)
                else:
                    src = torch.empty(0, dtype=torch.long)
                    dst = torch.empty(0, dtype=torch.long)
            edge_index = torch.stack([src, dst], dim=0)
            x = g.ndata.get("feature", g.ndata.get("feat")).float()
            y = g.ndata.get("label").view(-1).long()
            return SimpleNamespace(num_nodes=int(g.num_nodes()), edge_index=edge_index, x=x, y=y)
        except Exception:
            pass

    if norm in {"yelp", "yelpchi"}:
        local_yelp = Path("data") / "Yelp" / "processed" / "data.pt"
        if local_yelp.exists():
            try:
                obj = torch.load(local_yelp, weights_only=False)
            except Exception:
                pass
            else:
                candidate = obj[0] if isinstance(obj, tuple) and obj else obj
                y = candidate.get("y") if isinstance(candidate, dict) else getattr(candidate, "y", None)
                if isinstance(y, torch.Tensor) and y.dim() > 1:
                    print("Warning: Using compatibility conversion for Yelp fallback labels: multi-dimensional y -> binary (any positive class).")
                    y_bin = (y.sum(dim=-1) > 0).to(dtype=torch.long)
                    if isinstance(candidate, dict):
                        candidate = dict(candidate)
                        candidate["y"] = y_bin
                    else:
                        candidate.y = y_bin
                return _to_simple_graph_from_obj(candidate)

    if norm in {"tfinance", "tsocial"}:
        try:
            from pygod.utils import load_data

            pyg_name = "tfinance" if norm == "tfinance" else "tsocial"
            data = load_data(pyg_name)
            return _to_simple_graph_from_obj(data)
        except Exception:
            pass

    local = _local_anomaly_path(dataset_name, data_root=data_root)
    if local is not None:
        obj = torch.load(local, weights_only=False)
        return _to_simple_graph_from_obj(obj)

    raise RuntimeError(
        f"Could not load anomaly dataset '{dataset_name}'. "
        f"Supported: YelpChi/Amazon via DGL, YelpChi via data/Yelp/processed/data.pt "
        f"(only if labels are already 1D binary), "
        f"T-Finance/T-Social via pygod, or local .pt in {data_root}."
    )


def _maybe_cache_ego_dataset(
    base_graph: SimpleNamespace,
    dataset_name: str,
    num_hops: int,
    limit: int | None,
    cache_dir: str,
    num_workers: int = 8,
    max_nodes: int | None = None,
) -> list[dict]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    lim = "all" if limit is None else str(int(limit))
    cap = "all" if max_nodes is None or int(max_nodes) <= 0 else str(int(max_nodes))
    tag = f"{_normalized_name(dataset_name)}_ego_h{int(num_hops)}_l{lim}_n{int(base_graph.num_nodes)}_c{cap}.pt"
    cache_path = Path(cache_dir) / tag
    if cache_path.exists():
        print(f"Loading cached ego dataset from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    print(f"Building ego dataset cache at {cache_path}")
    ds = build_ego_graph_dataset(base_graph, num_hops=num_hops, limit=limit, num_workers=num_workers, max_nodes=max_nodes)
    torch.save(ds, cache_path)
    return ds


def _prepare_get_cached_dataset(
    dataset: list[dict],
    name: str,
    cache_dir: str,
    max_motifs: int | None,
    pe_k: int,
    rwse_k: int,
    enable_cache: bool,
) -> list[dict]:
    if not enable_cache:
        return dataset
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _graph_dataset_cache_fingerprint(dataset, name, max_motifs, pe_k, rwse_k)
    cache_path = cache_root / f"{name}_{fingerprint}.pt"
    if cache_path.exists():
        print(f"Loading cached processed dataset from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    wrapped = CachedGraphDataset(dataset=dataset, name=name, max_motifs=max_motifs, pe_k=pe_k, rwse_k=rwse_k)
    cached = wrapped.cached_data
    torch.save(cached, cache_path)
    print(f"Saved cached processed dataset to {cache_path}")
    return cached


def _build_optimizer(model: torch.nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))


def _stage4_autocast_enabled(
    model: torch.nn.Module,
    device: str,
    inference_mode: str | None = None,
    use_amp: bool | None = None,
) -> bool:
    if inference_mode == "armijo":
        return False
    if use_amp is not None:
        return bool(use_amp) and str(device).startswith("cuda")
    return _should_enable_amp(model, device)


def _stage4_amp_dtype(device: str, amp_dtype=None):
    return _normalize_amp_dtype(amp_dtype, device)


def _train_graph_classification(
    model: torch.nn.Module,
    train_ds: list[dict],
    val_ds: list[dict],
    test_ds: list[dict],
    epochs: int,
    batch_size: int,
    device: str,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    loader_kwargs: dict | None = None,
    model_name: str | None = None,
    use_amp: bool | None = None,
    amp_dtype=None,
) -> FitResult:
    import time

    model = model.to(device)
    model_name = model_name or model.__class__.__name__
    opt = _build_optimizer(model, lr=lr, wd=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = _stage4_autocast_enabled(model, device, use_amp=use_amp)
    autocast_dtype = _stage4_amp_dtype(device, amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and str(device).startswith("cuda") and autocast_dtype == torch.float16))
    loader_kwargs = loader_kwargs or {}
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_get_batch, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)

    history = {"train_loss": [], "val_acc": [], "test_acc": []}
    param_cnt = get_num_params(model)
    best_val = -1.0
    best_state = None

    print("-" * 50)
    print(f"EXPERIMENT: {model_name}")
    print(f"DEVICE:     {device}")
    print(f"PARAMS:     {param_cnt}")
    if hasattr(model, "get_layer"):
        layer = model.get_layer
        steps = getattr(model, "num_steps", "?")
        print(f"CONFIG:     d={layer.d}, H={layer.num_heads}, steps={steps}")
    print("-" * 50)

    epoch_bar = tqdm(range(int(epochs)), desc=f"Train {model_name} [{param_cnt}]", bar_format="{l_bar}{bar:20}{r_bar}", leave=False)
    for _ in epoch_bar:
        t0 = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype, enabled=use_amp):
                logits, _ = model(batch, task_level="graph")
                loss = criterion(logits, batch.y.view(-1).long())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            train_losses.append(float(loss.detach().item()))

        train_loss = float(statistics.fmean(train_losses) if train_losses else 0.0)
        history["train_loss"].append(train_loss)

        model.eval()

        def _eval_acc(loader: DataLoader) -> float:
            y_true_chunks: list[torch.Tensor] = []
            y_pred_chunks: list[torch.Tensor] = []
            import inspect

            forward_kwargs = {"task_level": "graph"}
            if "inference_mode" in inspect.signature(model.forward).parameters:
                forward_kwargs["inference_mode"] = "armijo"
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    with torch.autocast(
                        device_type=torch.device(device).type,
                        dtype=autocast_dtype,
                        enabled=_stage4_autocast_enabled(model, device, forward_kwargs.get("inference_mode"), use_amp=use_amp),
                    ):
                        logits, _ = model(batch, **forward_kwargs)
                    y_pred_chunks.append(logits.argmax(dim=-1).detach())
                    y_true_chunks.append(batch.y.view(-1).long().detach())
            if not y_true_chunks:
                return 0.0
            y_true = torch.cat(y_true_chunks, dim=0).cpu().tolist()
            y_pred = torch.cat(y_pred_chunks, dim=0).cpu().tolist()
            if accuracy_score is not None:
                return float(accuracy_score(y_true, y_pred))
            return float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true))

        val_acc = _eval_acc(val_loader)
        test_acc = _eval_acc(test_loader)
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not isinstance(v, UninitializedParameter)}

        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        scheduler.step(val_acc)
        epoch_time = time.time() - t0
        epoch_bar.set_postfix_str(f"L: {train_loss:.3f} | V: {val_acc:.3f} | T: {test_acc:.3f} | Bv: {best_val:.3f} | {epoch_time:.1f}s/ep")

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    best_test = _eval_acc(test_loader)
    return FitResult(metric=best_test, history=history, extra={"best_val_acc": best_val})


def _select_threshold(y_true: list[float], y_score: list[float]) -> float:
    n = min(len(y_true), len(y_score))
    if n == 0:
        return 0.5
    if len(y_true) != len(y_score):
        y_true = y_true[:n]
        y_score = y_score[:n]
    best_t = 0.5
    best_f1 = -1.0
    for t in [0.05 * i for i in range(1, 20)]:
        f1 = _safe_f1(y_true, y_score, threshold=float(t))
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _train_graph_binary_with_val(
    model: torch.nn.Module,
    train_ds: list[dict],
    val_ds: list[dict],
    test_ds: list[dict],
    epochs: int,
    batch_size: int,
    device: str,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    loader_kwargs: dict | None = None,
    use_weighted_bce: bool = True,
    eval_batch_size: int | None = None,
    model_name: str | None = None,
    train_batch_sampler=None,
    eval_batch_sampler=None,
    use_amp: bool | None = None,
    amp_dtype=None,
) -> FitResult:
    import time

    model = model.to(device)
    model_name = model_name or model.__class__.__name__
    opt = _build_optimizer(model, lr=lr, wd=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    if use_weighted_bce and len(train_ds) > 0:
        y = torch.tensor([g["y"].view(-1)[0] for g in train_ds], dtype=torch.float32)
        pos = float((y > 0.5).sum().item())
        neg = float((y <= 0.5).sum().item())
        if pos > 0 and neg > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    use_amp = _stage4_autocast_enabled(model, device, use_amp=use_amp)
    autocast_dtype = _stage4_amp_dtype(device, amp_dtype)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(use_amp and str(device).startswith("cuda") and autocast_dtype == torch.float16))
    loader_kwargs = loader_kwargs or {}
    if train_batch_sampler is not None:
        train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler, collate_fn=collate_get_batch, **loader_kwargs)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_get_batch, **loader_kwargs)
    if eval_batch_sampler is not None:
        val_loader = DataLoader(val_ds, batch_sampler=eval_batch_sampler(val_ds), collate_fn=collate_get_batch, **loader_kwargs)
        test_loader = DataLoader(test_ds, batch_sampler=eval_batch_sampler(test_ds), collate_fn=collate_get_batch, **loader_kwargs)
    else:
        eval_bs = int(eval_batch_size) if eval_batch_size is not None else int(batch_size)
        eval_bs = max(1, eval_bs)
        val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)
        test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, collate_fn=collate_get_batch, **loader_kwargs)

    history = {"train_loss": [], "val_auc": [], "val_f1": [], "test_auc": [], "test_f1": []}
    best = {"val_auc": -1.0, "test_auc": 0.5, "test_f1": 0.0}
    param_cnt = get_num_params(model)

    print("-" * 50)
    print(f"EXPERIMENT: {model_name}")
    print(f"DEVICE:     {device}")
    print(f"PARAMS:     {param_cnt}")
    if hasattr(model, "get_layer"):
        layer = model.get_layer
        steps = getattr(model, "num_steps", "?")
        print(f"CONFIG:     d={layer.d}, H={layer.num_heads}, steps={steps}")
    print("-" * 50)

    epoch_bar = tqdm(range(int(epochs)), desc=f"Train {model_name} [{param_cnt}]", bar_format="{l_bar}{bar:20}{r_bar}", leave=False)
    for _ in epoch_bar:
        t0 = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=torch.device(device).type, dtype=autocast_dtype, enabled=use_amp):
                logits, _ = model(batch, task_level="graph")
                loss = criterion(logits.view(-1), batch.y.view(-1).float())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(float(loss.detach().item()))

        def _collect(loader: DataLoader, capture_trace: bool = False) -> tuple[list[float], list[float], list[float]]:
            y_chunks: list[torch.Tensor] = []
            p_chunks: list[torch.Tensor] = []
            energy_trace: list[float] = []
            model.eval()
            run_on_cpu = False
            cpu_model = None
            captured_trace = False

            import inspect

            base_forward_kwargs = {"task_level": "graph"}
            if "inference_mode" in inspect.signature(model.forward).parameters:
                base_forward_kwargs["inference_mode"] = "armijo"

            with torch.no_grad():
                for b in loader:
                    if run_on_cpu:
                        b_cpu = b.to("cpu")
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                            logits, batch_energy_trace = cpu_model(b_cpu, **base_forward_kwargs)
                        y_chunks.append(b_cpu.y.view(-1).float().detach())
                        p_chunks.append(torch.sigmoid(logits.view(-1)).detach())
                        if capture_trace and not captured_trace and batch_energy_trace is not None:
                            captured_trace = True
                            for step_energy in batch_energy_trace:
                                if isinstance(step_energy, torch.Tensor):
                                    energy_trace.append(float(step_energy.detach().float().mean().cpu().item()))
                                else:
                                    energy_trace.append(float(np.asarray(step_energy, dtype=np.float64).mean()))
                        continue
                    try:
                        b_dev = b.to(device)
                        with torch.autocast(
                            device_type=torch.device(device).type,
                            dtype=autocast_dtype,
                            enabled=_stage4_autocast_enabled(model, device, base_forward_kwargs.get("inference_mode"), use_amp=use_amp),
                        ):
                            logits, batch_energy_trace = model(b_dev, **base_forward_kwargs)
                        y_chunks.append(b_dev.y.view(-1).float().detach())
                        p_chunks.append(torch.sigmoid(logits.view(-1)).detach())
                        if capture_trace and not captured_trace and batch_energy_trace is not None:
                            captured_trace = True
                            for step_energy in batch_energy_trace:
                                if isinstance(step_energy, torch.Tensor):
                                    energy_trace.append(float(step_energy.detach().float().mean().cpu().item()))
                                else:
                                    energy_trace.append(float(np.asarray(step_energy, dtype=np.float64).mean()))
                    except Exception as exc:
                        if not (str(device).startswith("cuda") and _is_cuda_oom(exc)):
                            raise
                        torch.cuda.empty_cache()
                        if cpu_model is None:
                            cpu_model = copy.deepcopy(model).to("cpu")
                            cpu_model.eval()
                            print("Warning: CUDA OOM in eval; switching remaining eval batches to CPU.")
                            if y_chunks:
                                y_chunks = [chunk.cpu() for chunk in y_chunks]
                            if p_chunks:
                                p_chunks = [chunk.cpu() for chunk in p_chunks]
                        run_on_cpu = True
                        b_cpu = b.to("cpu")
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False):
                            logits, batch_energy_trace = cpu_model(b_cpu, **base_forward_kwargs)
                        y_chunks.append(b_cpu.y.view(-1).float().detach())
                        p_chunks.append(torch.sigmoid(logits.view(-1)).detach())
                        if capture_trace and not captured_trace and batch_energy_trace is not None:
                            captured_trace = True
                            for step_energy in batch_energy_trace:
                                if isinstance(step_energy, torch.Tensor):
                                    energy_trace.append(float(step_energy.detach().float().mean().cpu().item()))
                                else:
                                    energy_trace.append(float(np.asarray(step_energy, dtype=np.float64).mean()))
            if not y_chunks:
                return [], [], []
            y_tensor = torch.cat(y_chunks, dim=0)
            p_tensor = torch.cat(p_chunks, dim=0)
            return y_tensor.cpu().tolist(), p_tensor.cpu().tolist(), energy_trace

        yv, pv, _ = _collect(val_loader)
        yt, pt, energy_trace = _collect(test_loader, capture_trace=True)
        threshold = _select_threshold(yv, pv) if yv else 0.5
        val_auc = _safe_auc(yv, pv) if yv else 0.5
        val_f1 = _safe_f1(yv, pv, threshold=threshold) if yv else 0.0
        test_auc = _safe_auc(yt, pt) if yt else 0.5
        test_f1 = _safe_f1(yt, pt, threshold=threshold) if yt else 0.0

        train_loss = float(statistics.fmean(losses) if losses else 0.0)
        history["train_loss"].append(train_loss)
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["test_auc"].append(test_auc)
        history["test_f1"].append(test_f1)
        scheduler.step(val_auc)

        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "test_auc": test_auc, "test_f1": test_f1}

        epoch_time = time.time() - t0
        epoch_bar.set_postfix_str(f"L: {train_loss:.3f} | V: {val_auc:.3f} | B: {best['test_auc']:.3f} | {epoch_time:.1f}s/ep")

    extra: dict[str, float | list[float]] = {"best_test_f1": best["test_f1"]}
    if energy_trace:
        extra["energy_trace"] = energy_trace
    return FitResult(metric=best["test_auc"], history=history, extra=extra)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if len(xs) == 0:
        return 0.0, 0.0
    return float(statistics.fmean(xs)), float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def _recommend_anomaly_batch_size(dataset: list[dict], requested: int, node_budget: int, hard_cap: int) -> int:
    if len(dataset) == 0:
        return 1
    max_nodes = max(int(item["x"].size(0)) for item in dataset)
    if max_nodes <= 0:
        return 1
    dynamic_cap = max(1, node_budget // max_nodes)
    safe_batch = min(int(requested), dynamic_cap, hard_cap)
    return max(1, int(safe_batch))


def _recommend_anomaly_motif_cap(dataset: list[dict], requested: int, motif_budget: int, hard_cap: int) -> int:
    if len(dataset) == 0:
        return max(1, int(requested))
    max_nodes = max(int(item["x"].size(0)) for item in dataset)
    if max_nodes <= 0:
        return max(1, int(requested))
    dynamic_cap = max(1, motif_budget // max_nodes)
    safe_cap = min(int(requested), dynamic_cap, hard_cap)
    return max(1, int(safe_cap))


def _capture_energy_trace(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    task_level: str = "graph",
    use_amp: bool | None = None,
    amp_dtype=None,
) -> list[float]:
    try:
        batch = next(iter(loader))
    except StopIteration:
        return []
    model.eval()
    forward_kwargs = {"task_level": task_level}
    import inspect
    signature = inspect.signature(model.forward)
    if "inference_mode" in signature.parameters:
        forward_kwargs["inference_mode"] = "armijo"
    if "return_solver_stats" in signature.parameters:
        forward_kwargs["return_solver_stats"] = False
    try:
        with torch.no_grad():
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=_stage4_amp_dtype(device, amp_dtype),
                enabled=_stage4_autocast_enabled(model, device, forward_kwargs.get("inference_mode"), use_amp=use_amp),
            ):
                outputs = model(batch.to(device), **forward_kwargs)
    except Exception:
        return []
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        return []
    energy_trace = outputs[1]
    if energy_trace is None:
        return []
    trace: list[float] = []
    for step_energy in energy_trace:
        if isinstance(step_energy, torch.Tensor):
            trace.append(float(step_energy.detach().float().mean().cpu().item()))
        else:
            trace.append(float(np.asarray(step_energy, dtype=np.float64).mean()))
    return trace


def _build_classification_folds(labels: list[int], requested_folds: int, seed: int) -> list[tuple[list[int], list[int]]]:
    class_counts = [labels.count(label) for label in sorted(set(labels))]
    max_stratified_folds = min(class_counts) if class_counts else 0
    if (
        requested_folds > 1
        and StratifiedKFold is not None
        and len(class_counts) > 1
        and max_stratified_folds >= 2
    ):
        effective_folds = min(int(requested_folds), max_stratified_folds, len(labels))
        if effective_folds >= 2:
            splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=int(seed))
            return [(list(train_idx), list(test_idx)) for train_idx, test_idx in splitter.split([0] * len(labels), labels)]
    split = int(0.8 * len(labels))
    return [(list(range(split)), list(range(split, len(labels))))]


def _split_train_val_indices(train_idx: list[int], labels: list[int], seed: int, fold_id: int) -> tuple[list[int], list[int]]:
    train_labels = [labels[i] for i in train_idx]
    if (
        len(train_idx) >= 5
        and train_test_split is not None
        and len(set(train_labels)) > 1
        and min(train_labels.count(label) for label in set(train_labels)) >= 2
    ):
        train_sub_idx, val_sub_idx = train_test_split(
            train_idx,
            test_size=0.20,
            random_state=int(seed) + int(fold_id),
            stratify=train_labels,
        )
        return list(train_sub_idx), list(val_sub_idx)

    val_count = max(1, int(round(0.20 * len(train_idx))))
    val_sub_idx = train_idx[-val_count:]
    train_sub_idx = train_idx[:-val_count] or train_idx
    return list(train_sub_idx), list(val_sub_idx)


def _build_graph_classification_models(
    in_dim: int,
    num_classes: int,
    args: SimpleNamespace,
    get_pe_k: int,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    context = {
        "in_dim": in_dim,
        "num_classes": num_classes,
        "hidden_dim": int(args.hidden_dim),
        "pairwise_hidden_dim": int(args.hidden_dim * 1.73),
        "num_steps": int(args.num_steps),
        "get_num_heads": int(args.get_num_heads),
        "get_num_blocks": int(args.get_num_blocks),
        "lambda_3": float(args.lambda_3),
        "get_norm_style": str(args.get_norm_style),
        "get_pairwise_et_mask": bool(args.get_pairwise_et_mask),
        "get_pe_k": int(get_pe_k),
        "rwse_k": int(args.rwse_k),
        "et_num_blocks": int(args.et_num_blocks),
        "et_num_heads": int(args.et_num_heads),
        "et_head_dim_or_none": int(args.et_head_dim) if int(args.et_head_dim) > 0 else None,
        "et_pe_k": int(args.et_pe_k),
        "et_mask_mode": str(args.et_mask_mode),
        "et_official_mode": bool(args.et_mask_mode == "official_dense"),
        "et_node_cap": args.et_node_cap,
    }
    built = instantiate_models_from_catalog(
        args.model_config,
        context=context,
        names=["PairwiseGET", "FullGET", "ETFaithful"],
    )
    return built["PairwiseGET"], built["FullGET"], built["ETFaithful"]


def _build_graph_anomaly_models(
    in_dim: int,
    args: SimpleNamespace,
    get_pe_k: int,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    context = {
        "in_dim": in_dim,
        "num_classes": 1,
        "hidden_dim": int(args.hidden_dim),
        "pairwise_hidden_dim": int(args.hidden_dim * 1.73),
        "num_steps": int(args.num_steps),
        "get_num_heads": int(args.get_num_heads),
        "get_num_blocks": int(args.get_num_blocks),
        "lambda_3": float(args.lambda_3),
        "get_norm_style": str(args.get_norm_style),
        "get_pairwise_et_mask": bool(args.get_pairwise_et_mask),
        "get_pe_k": int(get_pe_k),
        "rwse_k": int(args.rwse_k),
        "et_num_blocks": int(args.et_num_blocks),
        "et_num_heads": int(args.et_num_heads),
        "et_head_dim_or_none": int(args.et_head_dim) if int(args.et_head_dim) > 0 else None,
        "et_pe_k": int(args.et_pe_k),
        "et_mask_mode": str(args.et_mask_mode),
        "et_official_mode": bool(args.et_mask_mode == "official_dense"),
        "et_node_cap": args.et_node_cap,
    }
    built = instantiate_models_from_catalog(
        args.model_config,
        context=context,
        names=["PairwiseGET", "FullGET", "ETFaithful"],
    )
    return built["PairwiseGET"], built["FullGET"], built["ETFaithful"]


def _make_anomaly_batch_samplers(
    train_ds: list[dict],
    seed: int,
    node_budget: int,
    hard_cap: int,
):
    def _make_train_batch_sampler() -> _NodeBudgetBatchSampler:
        return _NodeBudgetBatchSampler(
            train_ds,
            node_budget=int(node_budget),
            hard_cap=int(hard_cap),
            shuffle=True,
            seed=int(seed),
        )

    def _make_eval_batch_sampler(split_ds: list[dict]) -> _NodeBudgetBatchSampler:
        return _NodeBudgetBatchSampler(
            split_ds,
            node_budget=int(node_budget),
            hard_cap=int(hard_cap),
            shuffle=False,
            seed=int(seed),
        )

    return _make_train_batch_sampler, _make_eval_batch_sampler
