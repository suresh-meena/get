from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
import statistics
from types import SimpleNamespace
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import (
    build_dataloader_kwargs,
    load_tu_dataset,
    set_seed,
    build_anomaly_protocol_split,
    build_ego_graph_dataset,
)
from get import ETFaithful, FullGET, PairwiseGET, collate_get_batch, CachedGraphDataset

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
    from sklearn.model_selection import StratifiedKFold
except Exception:
    StratifiedKFold = None


TU8_DATASETS = [
    "PROTEINS",
    "NCI1",
    "NCI109",
    "DD",
    "ENZYMES",
    "MUTAG",
    "MUTAGENICITY",
    "FRANKENSTEIN",
]


def _safe_auc(y_true: list[float], y_score: list[float]) -> float:
    if roc_auc_score is None:
        # Fallback proxy when sklearn is unavailable.
        return 0.5
    if len(set(int(v >= 0.5) for v in y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def _safe_f1(y_true: list[float], y_score: list[float], threshold: float = 0.5) -> float:
    y_pred = [1 if s >= threshold else 0 for s in y_score]
    y_bin = [1 if y >= 0.5 else 0 for y in y_true]
    if f1_score is None:
        # Macro-F1 fallback for binary labels.
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


def _synthetic_graph(
    rng: random.Random,
    num_nodes: int,
    in_dim: int,
    motif_boost: bool = False,
) -> dict:
    x = torch.randn(num_nodes, in_dim)
    edges = set()
    # Sparse random base.
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if rng.random() < 0.1:
                edges.add((u, v))

    # Keep graph connected enough.
    for u in range(num_nodes - 1):
        edges.add((u, u + 1))

    # Inject/avoid triangles for class signal.
    if motif_boost:
        for u in range(0, max(0, num_nodes - 2), 3):
            edges.add((u, u + 1))
            edges.add((u + 1, u + 2))
            edges.add((u, u + 2))
    else:
        # Break some potential triangles.
        for u in range(0, max(0, num_nodes - 2), 3):
            if (u, u + 2) in edges:
                edges.remove((u, u + 2))

    edge_list = sorted(edges)
    return {"x": x, "edges": edge_list}


def make_synth_classification_dataset(num_graphs: int, in_dim: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ds = []
    for i in range(int(num_graphs)):
        label = 1 if i % 2 == 0 else 0
        g = _synthetic_graph(
            rng,
            num_nodes=rng.randint(14, 24),
            in_dim=in_dim,
            motif_boost=(label == 1),
        )
        g["y"] = torch.tensor([label], dtype=torch.long)
        g["graph_id"] = i
        ds.append(g)
    return ds


def make_synth_anomaly_dataset(num_graphs: int, in_dim: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ds = []
    for i in range(int(num_graphs)):
        label = 1 if rng.random() < 0.2 else 0
        g = _synthetic_graph(
            rng,
            num_nodes=rng.randint(18, 30),
            in_dim=in_dim,
            motif_boost=(label == 1),
        )
        # Add a local signal for positive samples.
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
    extra: dict[str, float]


def _normalized_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


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
    # Supports torch_geometric Data-like objects and dict payloads.
    if isinstance(obj, dict):
        x = obj["x"]
        edge_index = obj["edge_index"]
        y = obj["y"]
        num_nodes = int(obj.get("num_nodes", x.size(0)))
        return SimpleNamespace(num_nodes=num_nodes, edge_index=edge_index, x=x, y=y)
    if hasattr(obj, "edge_index") and hasattr(obj, "x") and hasattr(obj, "y"):
        num_nodes = int(getattr(obj, "num_nodes", obj.x.size(0)))
        return SimpleNamespace(num_nodes=num_nodes, edge_index=obj.edge_index, x=obj.x, y=obj.y)
    raise ValueError("Unsupported anomaly graph object format.")


def _load_anomaly_graph(dataset_name: str, data_root: str) -> SimpleNamespace:
    norm = _normalized_name(dataset_name)

    # DGL fraud datasets (YelpChi / Amazon)
    if norm in {"yelp", "yelpchi", "amazon"}:
        try:
            import dgl  # noqa: F401
            from dgl.data import FraudAmazonDataset, FraudYelpDataset

            if norm in {"yelp", "yelpchi"}:
                ds = FraudYelpDataset(raw_dir=data_root)
            else:
                ds = FraudAmazonDataset(raw_dir=data_root)
            g = ds[0]
            src, dst = g.edges()
            edge_index = torch.stack([src.long(), dst.long()], dim=0)
            x = g.ndata.get("feature", g.ndata.get("feat")).float()
            y = g.ndata.get("label").view(-1).long()
            return SimpleNamespace(num_nodes=int(g.num_nodes()), edge_index=edge_index, x=x, y=y)
        except Exception:
            pass

    # PyGOD loader for tfinance / tsocial.
    if norm in {"tfinance", "tsocial"}:
        try:
            from pygod.utils import load_data

            pyg_name = "tfinance" if norm == "tfinance" else "tsocial"
            data = load_data(pyg_name)
            return _to_simple_graph_from_obj(data)
        except Exception:
            pass

    # Local fallback.
    local = _local_anomaly_path(dataset_name, data_root=data_root)
    if local is not None:
        obj = torch.load(local, weights_only=False)
        return _to_simple_graph_from_obj(obj)

    raise RuntimeError(
        f"Could not load anomaly dataset '{dataset_name}'. "
        f"Supported: YelpChi/Amazon via DGL, T-Finance/T-Social via pygod, or local .pt in {data_root}."
    )


def _maybe_cache_ego_dataset(
    base_graph: SimpleNamespace,
    dataset_name: str,
    num_hops: int,
    limit: int | None,
    cache_dir: str,
) -> list[dict]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    lim = "all" if limit is None else str(int(limit))
    tag = f"{_normalized_name(dataset_name)}_ego_h{int(num_hops)}_l{lim}_n{int(base_graph.num_nodes)}.pt"
    cache_path = Path(cache_dir) / tag
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    ds = build_ego_graph_dataset(base_graph, num_hops=num_hops, limit=limit)
    torch.save(ds, cache_path)
    return ds


def _prepare_get_cached_dataset(
    dataset: list[dict],
    name: str,
    cache_dir: str,
    max_motifs: int | None,
    pe_k: int,
    enable_cache: bool,
) -> list[dict]:
    if not enable_cache:
        return dataset
    wrapped = CachedGraphDataset(
        dataset=dataset,
        cache_dir=cache_dir,
        name=name,
        max_motifs=max_motifs,
        pe_k=pe_k,
    )
    return wrapped.cached_data


def _build_optimizer(model: torch.nn.Module, lr: float, wd: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(wd))


def _train_graph_classification(
    model: torch.nn.Module,
    train_ds: list[dict],
    test_ds: list[dict],
    epochs: int,
    batch_size: int,
    device: str,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loader_kwargs: dict | None = None,
) -> FitResult:
    model = model.to(device)
    opt = _build_optimizer(model, lr=lr, wd=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    loader_kwargs = loader_kwargs or {}
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_get_batch,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_get_batch,
        **loader_kwargs,
    )

    history = {"train_loss": [], "test_acc": [], "bad_batches": []}
    for _ in range(int(epochs)):
        model.train()
        train_losses = []
        bad = 0
        for batch in train_loader:
            try:
                batch = batch.to(device)
                opt.zero_grad()
                logits, _ = model(batch, task_level="graph")
                loss = criterion(logits, batch.y.view(-1).long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_losses.append(float(loss.detach().item()))
            except Exception:
                bad += 1
        history["train_loss"].append(float(statistics.fmean(train_losses) if train_losses else 0.0))
        history["bad_batches"].append(float(bad))

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits, _ = model(batch, task_level="graph")
                preds = logits.argmax(dim=-1)
                y_pred.extend(preds.cpu().tolist())
                y_true.extend(batch.y.view(-1).long().cpu().tolist())
        if y_true:
            if accuracy_score is not None:
                acc = float(accuracy_score(y_true, y_pred))
            else:
                acc = float(sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true))
        else:
            acc = 0.0
        history["test_acc"].append(acc)

    return FitResult(metric=history["test_acc"][-1], history=history, extra={})


def _select_threshold(y_true: list[float], y_score: list[float]) -> float:
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
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    loader_kwargs: dict | None = None,
    use_weighted_bce: bool = True,
) -> FitResult:
    model = model.to(device)
    opt = _build_optimizer(model, lr=lr, wd=weight_decay)
    if use_weighted_bce and len(train_ds) > 0:
        y = torch.tensor([float(g["y"].view(-1)[0].item()) for g in train_ds], dtype=torch.float32)
        pos = float((y > 0.5).sum().item())
        neg = float((y <= 0.5).sum().item())
        if pos > 0 and neg > 0:
            pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
    loader_kwargs = loader_kwargs or {}
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_get_batch,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_get_batch,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_get_batch,
        **loader_kwargs,
    )

    history = {
        "train_loss": [],
        "val_auc": [],
        "val_f1": [],
        "test_auc": [],
        "test_f1": [],
        "bad_batches": [],
    }
    best = {"val_auc": -1.0, "test_auc": 0.5, "test_f1": 0.0}

    for _ in range(int(epochs)):
        model.train()
        losses = []
        bad = 0
        for batch in train_loader:
            try:
                batch = batch.to(device)
                opt.zero_grad()
                logits, _ = model(batch, task_level="graph")
                loss = criterion(logits.view(-1), batch.y.view(-1).float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.detach().item()))
            except Exception:
                bad += 1

        def _collect(loader: DataLoader) -> tuple[list[float], list[float]]:
            ys, ps = [], []
            model.eval()
            with torch.no_grad():
                for b in loader:
                    b = b.to(device)
                    logits, _ = model(b, task_level="graph")
                    prob = torch.sigmoid(logits.view(-1))
                    ys.extend(b.y.view(-1).float().cpu().tolist())
                    ps.extend(prob.cpu().tolist())
            return ys, ps

        yv, pv = _collect(val_loader)
        yt, pt = _collect(test_loader)
        threshold = _select_threshold(yv, pv) if yv else 0.5
        val_auc = _safe_auc(yv, pv) if yv else 0.5
        val_f1 = _safe_f1(yv, pv, threshold=threshold) if yv else 0.0
        test_auc = _safe_auc(yt, pt) if yt else 0.5
        test_f1 = _safe_f1(yt, pt, threshold=threshold) if yt else 0.0

        history["train_loss"].append(float(statistics.fmean(losses) if losses else 0.0))
        history["val_auc"].append(val_auc)
        history["val_f1"].append(val_f1)
        history["test_auc"].append(test_auc)
        history["test_f1"].append(test_f1)
        history["bad_batches"].append(float(bad))

        if val_auc > best["val_auc"]:
            best = {"val_auc": val_auc, "test_auc": test_auc, "test_f1": test_f1}

    return FitResult(metric=best["test_auc"], history=history, extra={"best_test_f1": best["test_f1"]})


def _mean_std(xs: list[float]) -> tuple[float, float]:
    if len(xs) == 0:
        return 0.0, 0.0
    mean = float(statistics.fmean(xs))
    std = float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0
    return mean, std


def run_graph_classification(args: argparse.Namespace) -> dict:
    if _normalized_name(args.dataset) in {"tu8", "alltu"}:
        target_datasets = list(TU8_DATASETS)
    else:
        target_datasets = [args.dataset]

    all_payloads = {}
    for dataset_name in target_datasets:
        runs = []
        for seed in args.seeds:
            set_seed(int(seed))
            if _normalized_name(dataset_name) == "synth":
                raw_ds = make_synth_classification_dataset(args.num_graphs, args.in_dim, seed=int(seed))
            else:
                raw_ds = load_tu_dataset(dataset_name, limit=args.limit_graphs if args.limit_graphs > 0 else None)
                if len(raw_ds) == 0:
                    raise RuntimeError(f"No samples loaded for classification dataset '{dataset_name}'.")

            proc_ds = _prepare_get_cached_dataset(
                dataset=raw_ds,
                name=f"stage4_cls_{dataset_name}",
                cache_dir=args.cache_dir,
                max_motifs=args.max_motifs if args.max_motifs > 0 else None,
                pe_k=args.get_pe_k,
                enable_cache=args.cache_processed,
            )
            labels = [int(g["y"].view(-1)[0].item()) for g in proc_ds]
            unique_labels = sorted(set(labels))
            num_classes = int(max(unique_labels)) + 1 if unique_labels else 2
            in_dim = int(proc_ds[0]["x"].size(1))

            if args.cv_folds > 1 and StratifiedKFold is not None and len(proc_ds) >= args.cv_folds:
                splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=int(seed))
                split_indices = list(splitter.split([0] * len(labels), labels))
            else:
                split = int(0.8 * len(proc_ds))
                idx_train = list(range(split))
                idx_test = list(range(split, len(proc_ds)))
                split_indices = [(idx_train, idx_test)]

            fold_runs = []
            loader_kwargs = build_dataloader_kwargs(
                args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            )
            for fold_id, (train_idx, test_idx) in enumerate(split_indices):
                train_ds = [proc_ds[i] for i in train_idx]
                test_ds = [proc_ds[i] for i in test_idx]

                common_kwargs = {
                    "in_dim": in_dim,
                    "d": args.hidden_dim,
                    "num_classes": num_classes,
                    "num_steps": args.num_steps,
                }
                pairwise = PairwiseGET(
                    **common_kwargs,
                    norm_style=args.get_norm_style,
                    pairwise_et_mask=args.get_pairwise_et_mask,
                    use_cls_token=False,
                )
                fullget = FullGET(
                    **common_kwargs,
                    lambda_3=args.lambda_3,
                    norm_style=args.get_norm_style,
                    pairwise_et_mask=args.get_pairwise_et_mask,
                    use_cls_token=False,
                )
                et_faithful = ETFaithful(
                    in_dim=in_dim,
                    d=args.hidden_dim,
                    num_classes=num_classes,
                    num_steps=args.num_steps,
                    num_heads=args.et_num_heads,
                    head_dim=args.et_head_dim if args.et_head_dim > 0 else None,
                    pe_k=args.et_pe_k,
                    mask_mode=args.et_mask_mode,
                    et_official_mode=(args.et_mask_mode == "official_dense"),
                    node_cap=args.et_node_cap,
                )
                gin = None
                if GINBaseline is not None:
                    try:
                        gin = GINBaseline(in_dim, args.hidden_dim, num_classes)
                    except Exception:
                        gin = None

                pair_res = _train_graph_classification(
                    pairwise, train_ds, test_ds, args.epochs, args.batch_size, args.device, loader_kwargs=loader_kwargs
                )
                full_res = _train_graph_classification(
                    fullget, train_ds, test_ds, args.epochs, args.batch_size, args.device, loader_kwargs=loader_kwargs
                )
                et_res = _train_graph_classification(
                    et_faithful, train_ds, test_ds, args.epochs, args.batch_size, args.device, loader_kwargs=loader_kwargs
                )

                fold_run = {
                    "fold": int(fold_id),
                    "pairwise_acc": pair_res.metric,
                    "fullget_acc": full_res.metric,
                    "et_faithful_acc": et_res.metric,
                    "histories": {
                        "pairwise": pair_res.history,
                        "fullget": full_res.history,
                        "et_faithful": et_res.history,
                    },
                }
                if gin is not None:
                    gin_res = _train_graph_classification(
                        gin, train_ds, test_ds, args.epochs, args.batch_size, args.device, loader_kwargs=loader_kwargs
                    )
                    fold_run["gin_acc"] = gin_res.metric
                    fold_run["histories"]["gin"] = gin_res.history
                fold_runs.append(fold_run)

            runs.append(
                {
                    "seed": int(seed),
                    "folds": fold_runs,
                    "pairwise_acc": float(statistics.fmean([fr["pairwise_acc"] for fr in fold_runs])),
                    "fullget_acc": float(statistics.fmean([fr["fullget_acc"] for fr in fold_runs])),
                    "et_faithful_acc": float(statistics.fmean([fr["et_faithful_acc"] for fr in fold_runs])),
                }
            )

        pair_mean, pair_std = _mean_std([r["pairwise_acc"] for r in runs])
        full_mean, full_std = _mean_std([r["fullget_acc"] for r in runs])
        et_mean, et_std = _mean_std([r["et_faithful_acc"] for r in runs])
        all_payloads[dataset_name] = {
            "dataset": dataset_name,
            "summary": {
                "pairwise_mean": pair_mean,
                "pairwise_std": pair_std,
                "fullget_mean": full_mean,
                "fullget_std": full_std,
                "et_faithful_mean": et_mean,
                "et_faithful_std": et_std,
            },
            "runs": runs,
        }

    if len(all_payloads) == 1:
        only = next(iter(all_payloads.values()))
        return {"task": "graph_classification", **only}
    return {"task": "graph_classification", "dataset": "tu8", "per_dataset": all_payloads}


def run_graph_anomaly(args: argparse.Namespace) -> dict:
    by_rate: dict[str, list[dict]] = {}
    for rate in args.anomaly_label_rates:
        runs = []
        for seed in args.seeds:
            set_seed(int(seed))
            if _normalized_name(args.dataset) == "synth":
                ds = make_synth_anomaly_dataset(args.num_graphs, args.in_dim, seed=int(seed))
            else:
                base_graph = _load_anomaly_graph(args.dataset, data_root=args.data_root)
                limit = args.ego_limit if args.ego_limit > 0 else None
                if args.cache_processed:
                    ds = _maybe_cache_ego_dataset(
                        base_graph=base_graph,
                        dataset_name=args.dataset,
                        num_hops=args.ego_hops,
                        limit=limit,
                        cache_dir=args.cache_dir,
                    )
                else:
                    ds = build_ego_graph_dataset(base_graph, num_hops=args.ego_hops, limit=limit)
                ds = _prepare_get_cached_dataset(
                    dataset=ds,
                    name=f"stage4_anom_{args.dataset}_h{args.ego_hops}",
                    cache_dir=args.cache_dir,
                    max_motifs=args.max_motifs if args.max_motifs > 0 else None,
                    pe_k=args.get_pe_k,
                    enable_cache=args.cache_processed,
                )

            split = build_anomaly_protocol_split(
                ds,
                seed=int(seed),
                labeled_rate=float(rate),
                val_ratio=1,
                test_ratio=2,
            )
            train_ds, val_ds, test_ds = split["train"], split["val"], split["test"]
            in_dim = int(train_ds[0]["x"].size(1))

            fullget = FullGET(
                in_dim=in_dim,
                d=args.hidden_dim,
                num_classes=1,
                num_steps=args.num_steps,
                lambda_3=args.lambda_3,
                norm_style=args.get_norm_style,
                pairwise_et_mask=args.get_pairwise_et_mask,
                use_cls_token=False,
            )
            et_faithful = ETFaithful(
                in_dim=in_dim,
                d=args.hidden_dim,
                num_classes=1,
                num_steps=args.num_steps,
                num_heads=args.et_num_heads,
                head_dim=args.et_head_dim if args.et_head_dim > 0 else None,
                pe_k=args.et_pe_k,
                mask_mode=args.et_mask_mode,
                et_official_mode=(args.et_mask_mode == "official_dense"),
                node_cap=args.et_node_cap,
            )
            loader_kwargs = build_dataloader_kwargs(
                args.device,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
            )
            full_res = _train_graph_binary_with_val(
                fullget,
                train_ds,
                val_ds,
                test_ds,
                args.epochs,
                args.batch_size,
                args.device,
                loader_kwargs=loader_kwargs,
                use_weighted_bce=args.weighted_bce,
            )
            et_res = _train_graph_binary_with_val(
                et_faithful,
                train_ds,
                val_ds,
                test_ds,
                args.epochs,
                args.batch_size,
                args.device,
                loader_kwargs=loader_kwargs,
                use_weighted_bce=args.weighted_bce,
            )
            runs.append(
                {
                    "seed": int(seed),
                    "fullget_auc": float(full_res.metric),
                    "fullget_f1": float(full_res.extra["best_test_f1"]),
                    "et_faithful_auc": float(et_res.metric),
                    "et_faithful_f1": float(et_res.extra["best_test_f1"]),
                    "histories": {
                        "fullget": full_res.history,
                        "et_faithful": et_res.history,
                    },
                }
            )
        by_rate[str(float(rate))] = runs

    summary = {}
    for rate_key, runs in by_rate.items():
        fg_auc_mean, fg_auc_std = _mean_std([r["fullget_auc"] for r in runs])
        fg_f1_mean, fg_f1_std = _mean_std([r["fullget_f1"] for r in runs])
        et_auc_mean, et_auc_std = _mean_std([r["et_faithful_auc"] for r in runs])
        et_f1_mean, et_f1_std = _mean_std([r["et_faithful_f1"] for r in runs])
        summary[rate_key] = {
            "fullget_mean": fg_auc_mean,
            "fullget_std": fg_auc_std,
            "fullget_f1_mean": fg_f1_mean,
            "fullget_f1_std": fg_f1_std,
            "et_faithful_mean": et_auc_mean,
            "et_faithful_std": et_auc_std,
            "et_faithful_f1_mean": et_f1_mean,
            "et_faithful_f1_std": et_f1_std,
        }
    return {"task": "graph_anomaly", "dataset": args.dataset, "summary": summary, "runs": by_rate}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Stage-2 runner for ET-style transfer tasks.")
    parser.add_argument("--task", choices=["graph_classification", "graph_anomaly"], required=True)
    parser.add_argument("--dataset", default="synth")
    parser.add_argument("--data_root", default="data/stage4")
    parser.add_argument("--num_graphs", type=int, default=120)
    parser.add_argument("--limit_graphs", type=int, default=0)
    parser.add_argument("--in_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--cv_folds", type=int, default=1)
    parser.add_argument("--lambda_3", type=float, default=0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--cache_processed", action="store_true", default=False)
    parser.add_argument("--cache_dir", default=".cache/get_data")
    parser.add_argument("--max_motifs", type=int, default=16)
    parser.add_argument("--get_pe_k", type=int, default=0)
    parser.add_argument("--ego_hops", type=int, default=1)
    parser.add_argument("--ego_limit", type=int, default=0)

    # Required by tests and ET-vs-GET controls.
    parser.add_argument("--get_pairwise_et_mask", action="store_true", default=False)
    parser.add_argument("--get_norm_style", choices=["standard", "et"], default="et")
    parser.add_argument("--anomaly_label_rates", nargs="+", type=float, default=[0.01, 0.4])
    parser.add_argument("--weighted_bce", action="store_true", default=False)

    # ET-faithful knobs.
    parser.add_argument("--et_num_heads", type=int, default=1)
    parser.add_argument("--et_head_dim", type=int, default=0)
    parser.add_argument("--et_pe_k", type=int, default=8)
    parser.add_argument("--et_mask_mode", choices=["sparse", "official_dense"], default="sparse")
    parser.add_argument("--et_node_cap", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    # Keep CLI contract simple and robust for tests.
    if args.et_node_cap <= 0:
        args.et_node_cap = None
    # ET anomaly protocol uses weighted BCE by default.
    if args.task == "graph_anomaly" and not args.weighted_bce:
        args.weighted_bce = True
    # ET TU graph classification protocol uses 10-fold CV.
    if args.task == "graph_classification" and args.cv_folds <= 1 and _normalized_name(args.dataset) != "synth":
        args.cv_folds = 10

    # Current runner focuses on synthetic ET-style smoke/ablation diagnostics.
    if args.dataset != "synth" and args.task == "graph_classification":
        # Fallback to TU where available, otherwise still synthetic.
        try:
            _ = load_tu_dataset("MUTAG", limit=2)
        except Exception:
            pass

    random.seed(args.seeds[0])
    torch.manual_seed(args.seeds[0])

    if args.task == "graph_classification":
        payload = run_graph_classification(args)
        out_path = Path("outputs/stage2_graph_classification.json")
    else:
        payload = run_graph_anomaly(args)
        out_path = Path("outputs/stage2_graph_anomaly.json")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
