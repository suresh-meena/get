from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path
from typing import Dict
from typing import Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from external.graph_baselines.torch_baselines import ExternalGraphBaseline
from get.data import SyntheticGraphDataset, collate_graph_samples
from get.models import EnergyGraphClassifier
from get.trainers import UnifiedTrainer
from get.utils.seed import seed_everything


def _apply_task_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = args.task_preset.lower()
    if preset == "none":
        return args
    if preset == "csl":
        args.min_nodes = 16
        args.max_nodes = 24
        args.edge_prob = 0.18
        args.max_motifs_per_anchor = 10
        return args
    if preset == "brec":
        args.min_nodes = 20
        args.max_nodes = 28
        args.edge_prob = 0.16
        args.max_motifs_per_anchor = 12
        return args
    if preset == "graph_classification":
        args.min_nodes = 14
        args.max_nodes = 26
        args.edge_prob = 0.22
        args.max_motifs_per_anchor = 8
        return args
    if preset == "graph_anomaly":
        args.min_nodes = 12
        args.max_nodes = 22
        args.edge_prob = 0.20
        args.max_motifs_per_anchor = 6
        return args
    raise ValueError(f"Unknown task_preset: {args.task_preset}")


def _build_loaders(args: argparse.Namespace):
    if args.dataset_name.lower() in {"csl", "brec"}:
        return _build_real_stage2_loaders(args)

    common = dict(
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        edge_prob=args.edge_prob,
        in_dim=args.in_dim,
        max_motifs_per_anchor=args.max_motifs_per_anchor,
    )
    train_ds = SyntheticGraphDataset(num_graphs=args.num_train_graphs, seed=args.seed, **common)
    val_ds = SyntheticGraphDataset(num_graphs=args.num_val_graphs, seed=args.seed + 1, **common)
    test_ds = SyntheticGraphDataset(num_graphs=args.num_test_graphs, seed=args.seed + 2, **common)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    eval_bs = args.eval_batch_size or args.batch_size
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    return train_loader, val_loader, test_loader


def _extract_motifs_from_adj(adj: torch.Tensor, max_motifs_per_anchor: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = adj.size(0)
    c3_list = []
    u3_list = []
    v3_list = []
    tt_list = []
    
    for c in range(n):
        neigh = torch.nonzero(adj[c], as_tuple=False).flatten()
        n_neigh = neigh.numel()
        if n_neigh < 2:
            continue
            
        idx_j, idx_k = torch.triu_indices(n_neigh, n_neigh, offset=1)
        
        if idx_j.numel() > max_motifs_per_anchor:
            idx_j = idx_j[:max_motifs_per_anchor]
            idx_k = idx_k[:max_motifs_per_anchor]
            
        u = neigh[idx_j]
        v = neigh[idx_k]
        
        c3_list.append(torch.full((u.numel(),), c, dtype=torch.long))
        u3_list.append(u)
        v3_list.append(v)
        tt_list.append(adj[u, v].long())
        
    if not c3_list:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty, empty, empty
        
    return (
        torch.cat(c3_list),
        torch.cat(u3_list),
        torch.cat(v3_list),
        torch.cat(tt_list),
    )


def _pyg_data_to_sample(data, in_dim: int, max_motifs_per_anchor: int) -> Dict[str, torch.Tensor]:
    n = int(data.num_nodes)
    edge_index = data.edge_index.long()
    x = data.x
    if x is None:
        x = torch.ones((n, 1), dtype=torch.float32)
    x = x.float()
    if x.size(1) < in_dim:
        pad = torch.zeros((n, in_dim - x.size(1)), dtype=x.dtype)
        x = torch.cat([x, pad], dim=1)
    elif x.size(1) > in_dim:
        x = x[:, :in_dim]

    adj = torch.zeros((n, n), dtype=torch.bool)
    if edge_index.numel() > 0:
        adj[edge_index[0], edge_index[1]] = True
    adj.fill_diagonal_(False)

    c2 = edge_index[0].contiguous() if edge_index.numel() > 0 else torch.empty(0, dtype=torch.long)
    u2 = edge_index[1].contiguous() if edge_index.numel() > 0 else torch.empty(0, dtype=torch.long)
    c3, u3, v3, tt = _extract_motifs_from_adj(adj=adj, max_motifs_per_anchor=max_motifs_per_anchor)

    y = data.y
    if y is None:
        y = torch.tensor(0.0)
    y = y.view(-1).float()
    y = y[0] if y.numel() > 0 else torch.tensor(0.0)
    # Binary simplification for unified BCE trainer.
    y_bin = torch.tensor(1.0 if float(y.item()) > 0 else 0.0, dtype=torch.float32)

    return {"x": x, "y": y_bin, "c_2": c2, "u_2": u2, "c_3": c3, "u_3": u3, "v_3": v3, "t_tau": tt}


class _ListGraphDataset(torch.utils.data.Dataset):
    def __init__(self, samples: List[Dict[str, torch.Tensor]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def _split_list(items: List[Dict[str, torch.Tensor]], train_ratio: float, val_ratio: float):
    n = len(items)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    if len(test) == 0:
        test = val[-1:]
    return train, val, test


def _load_csl_samples(args: argparse.Namespace) -> List[Dict[str, torch.Tensor]]:
    from torch_geometric.datasets import GNNBenchmarkDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    ds = GNNBenchmarkDataset(root=str(root), name="CSL", split="train")
    dsv = GNNBenchmarkDataset(root=str(root), name="CSL", split="val")
    dst = GNNBenchmarkDataset(root=str(root), name="CSL", split="test")
    items = list(ds) + list(dsv) + list(dst)
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    return [_pyg_data_to_sample(d, in_dim=args.in_dim, max_motifs_per_anchor=args.max_motifs_per_anchor) for d in items]


def _load_brec_samples(args: argparse.Namespace) -> List[Dict[str, torch.Tensor]]:
    # Expected local format: a torch-saved list of torch_geometric Data objects.
    if not args.brec_file:
        raise ValueError("--brec_file is required when --dataset_name brec")
    path = Path(args.brec_file).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"BREC file not found: {path}")
    data_list = torch.load(path, map_location="cpu")
    if not isinstance(data_list, list):
        raise ValueError("BREC file must contain a list of torch_geometric Data objects")
    if args.max_graphs > 0:
        data_list = data_list[: args.max_graphs]
    return [_pyg_data_to_sample(d, in_dim=args.in_dim, max_motifs_per_anchor=args.max_motifs_per_anchor) for d in data_list]


def _build_real_stage2_loaders(args: argparse.Namespace):
    name = args.dataset_name.lower()
    if name == "csl":
        samples = _load_csl_samples(args)
    elif name == "brec":
        samples = _load_brec_samples(args)
    else:
        raise ValueError(f"Unsupported real Stage-2 dataset: {args.dataset_name}")

    rng = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(samples), generator=rng).tolist()
    samples = [samples[i] for i in perm]
    train_items, val_items, test_items = _split_list(samples, train_ratio=0.70, val_ratio=0.15)

    train_ds = _ListGraphDataset(train_items)
    val_ds = _ListGraphDataset(val_items)
    test_ds = _ListGraphDataset(test_items)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    eval_bs = args.eval_batch_size or args.batch_size
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    return train_loader, val_loader, test_loader


def _build_loaders_from_samples(
    train_items: List[Dict[str, torch.Tensor]],
    val_items: List[Dict[str, torch.Tensor]],
    test_items: List[Dict[str, torch.Tensor]],
    args: argparse.Namespace,
):
    train_ds = _ListGraphDataset(train_items)
    val_ds = _ListGraphDataset(val_items)
    test_ds = _ListGraphDataset(test_items)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    eval_bs = args.eval_batch_size or args.batch_size
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_graph_samples)
    return train_loader, val_loader, test_loader


def _run_single_fit(
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
) -> Dict[str, Dict[str, float]]:
    model = _build_model(args)
    trainer_cfg: Dict[str, object] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "patience": args.patience,
        "use_amp": bool(args.use_amp),
        "amp_dtype": args.amp_dtype,
    }
    trainer = UnifiedTrainer(model=model, device=device, trainer_cfg=trainer_cfg)
    return trainer.fit(train_loader, val_loader, test_loader)


def _build_model(args: argparse.Namespace) -> torch.nn.Module:
    name = args.model_name.lower()
    if name == "external_baseline":
        return ExternalGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim)
    if name == "gcn":
        from external.graph_baselines.torch_baselines import GCNGraphBaseline
        return GCNGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim)
    if name == "gat":
        from external.graph_baselines.torch_baselines import GATGraphBaseline
        return GATGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim)
    if name == "gin":
        from external.graph_baselines.torch_baselines import GINGraphBaseline
        return GINGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim)

    if name == "pairwiseget":
        energy_name = "pairwise_only"
    elif name == "et":
        energy_name = "quadratic_only"
    elif name == "fullget":
        energy_name = "get_full"
    else:
        raise ValueError(f"Unknown model_name: {args.model_name}")

    return EnergyGraphClassifier(
        in_dim=args.in_dim,
        hidden_dim=args.hidden_dim,
        num_classes=1,
        num_steps=args.num_steps,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        R=args.R,
        K=args.K,
        num_motif_types=2,
        lambda_2=args.lambda_2,
        lambda_3=args.lambda_3 if energy_name == "get_full" else 0.0,
        lambda_m=args.lambda_m,
        beta_2=args.beta_2,
        beta_3=args.beta_3,
        beta_m=args.beta_m,
        update_damping=args.update_damping,
        fixed_step_size=args.fixed_step_size,
        armijo_eta0=args.armijo_eta0,
        armijo_gamma=args.armijo_gamma,
        armijo_c=args.armijo_c,
        armijo_max_backtracks=args.armijo_max_backtracks,
        inference_mode_train=args.inference_mode_train,
        inference_mode_eval=args.inference_mode_eval,
        energy_name=energy_name,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Graph-task experiment runner with external baseline support")
    p.add_argument(
        "--task_preset",
        type=str,
        default="none",
        choices=["none", "csl", "brec", "graph_classification", "graph_anomaly"],
    )
    p.add_argument("--dataset_name", type=str, default="synthetic", choices=["synthetic", "csl", "brec"])
    p.add_argument("--dataset_root", type=str, default="data")
    p.add_argument("--brec_file", type=str, default="")
    p.add_argument("--max_graphs", type=int, default=0)
    p.add_argument("--cv_folds", type=int, default=1)
    p.add_argument("--model_name", type=str, default="fullget", choices=["fullget", "pairwiseget", "et", "gcn", "gat", "gin", "external_baseline"])
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--num_train_graphs", type=int, default=96)
    p.add_argument("--num_val_graphs", type=int, default=32)
    p.add_argument("--num_test_graphs", type=int, default=32)
    p.add_argument("--min_nodes", type=int, default=10)
    p.add_argument("--max_nodes", type=int, default=20)
    p.add_argument("--edge_prob", type=float, default=0.20)
    p.add_argument("--in_dim", type=int, default=32)
    p.add_argument("--max_motifs_per_anchor", type=int, default=8)

    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=4)
    p.add_argument("--R", type=int, default=2)
    p.add_argument("--K", type=int, default=8)

    p.add_argument("--lambda_2", type=float, default=1.0)
    p.add_argument("--lambda_3", type=float, default=10.0)
    p.add_argument("--lambda_m", type=float, default=0.0)
    p.add_argument("--beta_2", type=float, default=1.0)
    p.add_argument("--beta_3", type=float, default=1.0)
    p.add_argument("--beta_m", type=float, default=1.0)
    p.add_argument("--update_damping", type=float, default=0.0)

    p.add_argument("--fixed_step_size", type=float, default=0.1)
    p.add_argument("--armijo_eta0", type=float, default=0.2)
    p.add_argument("--armijo_gamma", type=float, default=0.5)
    p.add_argument("--armijo_c", type=float, default=1e-4)
    p.add_argument("--armijo_max_backtracks", type=int, default=20)
    p.add_argument("--inference_mode_train", type=str, default="armijo", choices=["fixed", "armijo"])
    p.add_argument("--inference_mode_eval", type=str, default="armijo", choices=["fixed", "armijo"])

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--amp_dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    p.add_argument("--output", type=str, default="outputs/graph_tasks/last_metrics.json")
    args = p.parse_args()
    args = _apply_task_preset(args)

    seed_everything(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but not available")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cv_folds > 1:
        if args.dataset_name.lower() != "csl":
            raise ValueError("Cross-validation is currently supported for --dataset_name csl only.")
        all_samples = _load_csl_samples(args)
        if len(all_samples) < args.cv_folds:
            raise ValueError(f"Not enough samples ({len(all_samples)}) for cv_folds={args.cv_folds}")

        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        fold_metrics: List[Dict[str, Dict[str, float]]] = []
        idx_all = list(range(len(all_samples)))
        for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(idx_all), start=1):
            trainval_items = [all_samples[int(i)] for i in trainval_idx.tolist()]
            test_items = [all_samples[int(i)] for i in test_idx.tolist()]

            n_tv = len(trainval_items)
            n_val = max(1, int(0.15 * n_tv))
            val_items = trainval_items[:n_val]
            train_items = trainval_items[n_val:]
            if len(train_items) == 0:
                train_items = val_items

            train_loader, val_loader, test_loader = _build_loaders_from_samples(
                train_items=train_items, val_items=val_items, test_items=test_items, args=args
            )
            m = _run_single_fit(args=args, device=device, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
            m["fold"] = fold_idx
            fold_metrics.append(m)

        test_accs = [float(m["test"]["acc"]) for m in fold_metrics]
        test_losses = [float(m["test"]["loss"]) for m in fold_metrics]
        metrics = {
            "cv_folds": int(args.cv_folds),
            "fold_metrics": fold_metrics,
            "summary": {
                "test_acc_mean": float(statistics.mean(test_accs)),
                "test_acc_std": float(statistics.pstdev(test_accs) if len(test_accs) > 1 else 0.0),
                "test_loss_mean": float(statistics.mean(test_losses)),
                "test_loss_std": float(statistics.pstdev(test_losses) if len(test_losses) > 1 else 0.0),
            },
        }
    else:
        train_loader, val_loader, test_loader = _build_loaders(args)
        metrics = _run_single_fit(args=args, device=device, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
