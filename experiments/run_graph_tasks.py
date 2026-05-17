from __future__ import annotations

import argparse
import json
import sys
import statistics
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.loader import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from get.data import SyntheticGraphDataset, ListGraphDataset  # noqa: E402
from get.models import build_model  # noqa: E402
from get.models.factory import canonicalize_model_name  # noqa: E402
from get.utils.seed import seed_everything  # noqa: E402
from get.utils.compile import maybe_compile_model  # noqa: E402
from get.data import infer_edge_attr_dim, split_items, summarize_splits  # noqa: E402
from experiments.common import collect_energy_diagnostics, fit_unified_trainer, make_loader_kwargs, resolve_device, build_data_loaders  # noqa: E402


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


def _apply_benchmark_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = args.benchmark_preset.lower()
    if preset == "none":
        return args

    args.model_name = "etfaithful"
    args.et_hidden_dim = 128
    args.et_num_heads = 12
    args.et_head_dim = 64
    args.et_num_steps = 1
    args.et_num_blocks = 4
    args.et_alpha = 0.1
    args.et_multiplier = 4.0
    args.et_chn_type = "relu"
    args.et_use_bias_attn = False
    args.et_use_bias_chn = False
    args.et_use_bias_norm = True
    args.et_use_cls_token = True
    args.et_pos_k = 15
    args.et_embed_type = "eigen"
    args.et_flip_sign = False
    args.et_compute_corr = True
    args.et_noise_std = 0.02
    args.et_vary_noise = False
    args.et_readout_mode = "cls"
    args.lambda_m = 1.0
    args.use_amp = True
    args.num_workers = 8

    if preset == "et_zinc":
        args.dataset_name = "ZINC"
        args.batch_size = max(1, 128 * (torch.cuda.device_count() if torch.cuda.is_available() else 1))
        args.eval_batch_size = args.batch_size
        args.epochs = 50
        args.patience = 50
    elif preset == "et_dd":
        args.dataset_name = "DD"
        args.batch_size = 64
        args.eval_batch_size = 64
        args.epochs = 300
        args.patience = 300
    elif preset == "et_tu":
        args.batch_size = 64
        args.eval_batch_size = 64
        args.epochs = 50
        args.patience = 50
    else:
        raise ValueError(f"Unknown benchmark_preset: {args.benchmark_preset}")
    return args


def _runtime_config(args: argparse.Namespace) -> dict:
    model_name = str(args.model_name).lower()
    canonical_model_name = canonicalize_model_name(model_name)
    return {
        "task_preset": args.task_preset,
        "benchmark_preset": args.benchmark_preset,
        "dataset_name": args.dataset_name,
        "model_name": args.model_name,
        "canonical_model_name": canonical_model_name,
        "model_alias": model_name,
        "use_amp": bool(args.use_amp),
        "amp_dtype": args.amp_dtype,
        "lambda_m": float(args.lambda_m),
        "raw_lambda_m": float(args.lambda_m),
        "lambda_2": float(args.lambda_2),
        "lambda_3": float(args.lambda_3),
        "inference_mode_train": args.inference_mode_train,
        "inference_mode_eval": args.inference_mode_eval,
    }


def _dataset_task_type(dataset_name: str) -> str:
    return "multiclass" if dataset_name.lower() == "csl" else "binary"


def _remap_multiclass_labels(samples: List[Dict[str, torch.Tensor]]) -> tuple[List[Dict[str, torch.Tensor]], int]:
    labels = sorted({int(sample["y"].view(-1)[0].item()) for sample in samples})
    if not labels:
        return samples, 1
    if labels == list(range(len(labels))):
        return samples, len(labels)

    mapping = {label: idx for idx, label in enumerate(labels)}
    remapped: List[Dict[str, torch.Tensor]] = []
    for sample in samples:
        sample_copy = sample.clone() if hasattr(sample, "clone") else dict(sample)
        label = int(sample_copy["y"].view(-1)[0].item())
        sample_copy["y"] = torch.tensor([float(mapping[label])], dtype=torch.float32, device=sample_copy["y"].device)
        remapped.append(sample_copy)
    return remapped, len(labels)


def _build_tudataset_loaders(args: argparse.Namespace):
    from get.data import RealWorldGraphDataset
    dataset = RealWorldGraphDataset(
        name=args.dataset_name,
        root=args.dataset_root,
        in_dim=args.in_dim,
        max_motifs_per_anchor=args.max_motifs_per_anchor,
        pos_k=getattr(args, "et_pos_k", 0),
        task_type="auto"
    )
    all_samples = dataset.items if hasattr(dataset, "items") else [dataset[i] for i in range(len(dataset))]
    if args.max_graphs > 0:
        all_samples = all_samples[:args.max_graphs]

    num_classes = dataset.num_classes
    task_type = dataset.task_type

    # Default splitting logic if not doing CV
    train_items, val_items, test_items = split_items(
        all_samples, seed=args.seed, train_ratio=0.7, val_ratio=0.1, task_type=task_type
    )

    train_loader, val_loader, test_loader, split_stats = _build_loaders_from_samples(
        train_items, val_items, test_items, args, task_type
    )
    return train_loader, val_loader, test_loader, split_stats, task_type, num_classes


def _build_loaders_from_samples(train_items, val_items, test_items, args, task_type):
    return build_data_loaders(
        train_items, val_items, test_items,
        batch_size=args.batch_size, eval_batch_size=getattr(args, "eval_batch_size", args.batch_size),
        num_workers=args.num_workers, task_type=task_type,
    )


def _build_loaders(args: argparse.Namespace):
    if args.dataset_name.lower() in {"csl", "brec"}:
        return _build_real_stage2_loaders(args)

    real_datasets = {"proteins", "nci1", "nci109", "dd", "enzymes", "mutag", "mutagenicity", "frankenstein"}
    if args.dataset_name.lower() in real_datasets:
        return _build_tudataset_loaders(args)

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

    loader_kwargs = make_loader_kwargs(getattr(args, "num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    eval_bs = args.eval_batch_size or args.batch_size
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    split_stats = summarize_splits({"train": train_ds.items, "val": val_ds.items, "test": test_ds.items}, task_type="binary")
    return train_loader, val_loader, test_loader, split_stats, "binary", 1


def _pyg_data_to_sample(data, in_dim: int, max_motifs_per_anchor: int, task_type: str = "binary") -> Dict[str, torch.Tensor]:
    from get.data.protocol import graph_to_sample
    y_mode = "multiclass" if task_type == "multiclass" else "binary"
    return graph_to_sample(data, in_dim=in_dim, max_motifs_per_anchor=max_motifs_per_anchor, y_mode=y_mode)


def _load_csl_samples(args: argparse.Namespace) -> tuple[List[Dict[str, torch.Tensor]], int]:
    from torch_geometric.datasets import GNNBenchmarkDataset

    root = Path(args.dataset_root).expanduser() / "pyg"
    ds = GNNBenchmarkDataset(root=str(root), name="CSL", split="train")
    dsv = GNNBenchmarkDataset(root=str(root), name="CSL", split="val")
    dst = GNNBenchmarkDataset(root=str(root), name="CSL", split="test")
    items = list(ds) + list(dsv) + list(dst)
    if args.max_graphs > 0:
        items = items[: args.max_graphs]
    samples = [_pyg_data_to_sample(d, in_dim=args.in_dim, max_motifs_per_anchor=args.max_motifs_per_anchor, task_type="multiclass") for d in items]
    return _remap_multiclass_labels(samples)


def _load_brec_samples(args: argparse.Namespace) -> tuple[List[Dict[str, torch.Tensor]], int]:
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
    return [_pyg_data_to_sample(d, in_dim=args.in_dim, max_motifs_per_anchor=args.max_motifs_per_anchor, task_type="binary") for d in data_list], 1


def _build_real_stage2_loaders(args: argparse.Namespace):
    name = args.dataset_name.lower()
    if name == "csl":
        samples, num_classes = _load_csl_samples(args)
        task_type = "multiclass"
    elif name == "brec":
        samples, num_classes = _load_brec_samples(args)
        task_type = "binary"
    else:
        raise ValueError(f"Unsupported real Stage-2 dataset: {args.dataset_name}")

    train_items, val_items, test_items = split_items(samples, seed=args.seed, train_ratio=0.70, val_ratio=0.15, task_type=task_type)

    train_ds = ListGraphDataset(train_items)
    val_ds = ListGraphDataset(val_items)
    test_ds = ListGraphDataset(test_items)

    loader_kwargs = make_loader_kwargs(getattr(args, "num_workers", 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    eval_bs = args.eval_batch_size or args.batch_size
    val_loader = DataLoader(val_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_bs, shuffle=False, **loader_kwargs)
    split_stats = summarize_splits({"train": train_items, "val": val_items, "test": test_items}, task_type=task_type)
    return train_loader, val_loader, test_loader, split_stats, task_type, num_classes




def _run_single_fit(
    args: argparse.Namespace,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    task_type: str,
    num_classes: int,
) -> Dict[str, Dict[str, float]]:
    cfg_dict = dict(vars(args))
    cfg_dict["task_type"] = task_type
    cfg_dict["num_classes"] = num_classes
    cfg_dict["edge_attr_dim"] = getattr(args, "edge_attr_dim", 0)
    from omegaconf import DictConfig
    model = build_model(DictConfig(cfg_dict)).to(device)
    energy_metadata = model.energy_metadata() if hasattr(model, "energy_metadata") else {}
    compile_cfg = {
        "enabled": bool(getattr(args, "compile", True)),
        "scope": str(getattr(args, "compile_scope", "all")),
        "backend": str(getattr(args, "compile_backend", "inductor")),
        "dynamic": bool(getattr(args, "compile_dynamic", True)),
        "fullgraph": bool(getattr(args, "compile_fullgraph", False)),
        "mode": getattr(args, "compile_mode", None),
    }
    scope = compile_cfg.get("scope", "all")
    if scope == "all":
        model = maybe_compile_model(model, compile_cfg)
        eval_model = model
    else:
        eval_model = maybe_compile_model(model, compile_cfg)
    parameter_count = sum(p.numel() for p in model.parameters())

    trainer_cfg: Dict[str, object] = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "patience": args.patience,
        "use_amp": bool(args.use_amp),
        "amp_dtype": args.amp_dtype,
        "task_type": task_type,
        "num_classes": num_classes,
    }
    _, fit_result, _, _, peak_memory_mb = fit_unified_trainer(
        model=model,
        eval_model=eval_model,
        device=device,
        trainer_cfg=trainer_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
    energy_diagnostics = collect_energy_diagnostics(
        model if eval_model is None else eval_model,
        {"val": val_loader, "test": test_loader},
        device=device,
    )
    fit_result["parameter_count"] = parameter_count
    fit_result["peak_cuda_memory_mb"] = peak_memory_mb
    fit_result["energy_metadata"] = energy_metadata
    fit_result["energy_diagnostics"] = energy_diagnostics
    return fit_result





def main() -> None:
    p = argparse.ArgumentParser(description="Graph-task experiment runner with external baseline support")
    p.add_argument(
        "--task_preset",
        type=str,
        default="none",
        choices=["none", "csl", "brec", "graph_classification", "graph_anomaly"],
    )
    p.add_argument("--benchmark_preset", type=str, default="none", choices=["none", "et_tu", "et_zinc", "et_dd"])
    p.add_argument("--dataset_name", type=str, default="synthetic")
    p.add_argument("--dataset_root", type=str, default="data")
    p.add_argument("--brec_file", type=str, default="")
    p.add_argument("--max_graphs", type=int, default=0)
    p.add_argument("--cv_folds", type=int, default=1)
    p.add_argument("--model_name", type=str, default="fullget", choices=["quadratic_only", "pairwise_only", "pairwiseget", "memory_only", "motif_only", "nomotif_local", "no_memory_local", "fullget_local", "fullget", "nomotif_global", "no_memory_global", "fullget_global", "get_ham_global", "get_ham_full", "et", "etfaithful", "bwgnn", "graphtransformer", "gcn", "gat", "gin", "external_baseline"])
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

    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--head_dim", type=int, default=32)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--num_blocks", type=int, default=1)
    p.add_argument("--et_num_blocks", type=int, default=4)
    p.add_argument("--et_hidden_dim", type=int, default=128)
    p.add_argument("--et_num_heads", type=int, default=12)
    p.add_argument("--et_head_dim", type=int, default=64)
    p.add_argument("--et_num_steps", type=int, default=1)
    p.add_argument("--inference_mode_train", type=str, default="fixed", choices=["fixed", "armijo"])
    p.add_argument("--inference_mode_eval", type=str, default="armijo", choices=["fixed", "armijo"])
    p.add_argument("--lambda_2", type=float, default=1.0)
    p.add_argument("--lambda_3", type=float, default=10.0)
    p.add_argument("--lambda_m", type=float, default=1.0)
    p.add_argument("--lambda_g", type=float, default=0.0)
    p.add_argument("--et_alpha", type=float, default=0.1)
    p.add_argument("--et_multiplier", type=float, default=4.0)
    p.add_argument("--et_chn_type", type=str, default="relu", choices=["relu", "gelu", "lse"])
    p.add_argument("--et_use_bias_attn", action="store_true", default=False)
    p.add_argument("--et_use_bias_chn", action="store_true", default=False)
    p.add_argument("--et_use_bias_norm", action="store_true", default=True)
    p.add_argument("--et_use_cls_token", action="store_true", default=True)
    p.add_argument("--et_pos_k", type=int, default=15)
    p.add_argument("--et_embed_type", type=str, default="eigen", choices=["eigen", "svd"])
    p.add_argument("--et_flip_sign", action="store_true", default=False)
    p.add_argument("--et_compute_corr", action="store_true", default=True)
    p.add_argument("--et_noise_std", type=float, default=0.02)
    p.add_argument("--et_vary_noise", action="store_true", default=False)
    p.add_argument("--et_readout_mode", type=str, default="cls", choices=["cls", "mean"])

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=3)
    
    # BWGNN-specific parameters
    p.add_argument("--bwgnn_order", type=int, default=3, help="Order for BWGNN polynomial filter")
    
    # GraphTransformer-specific parameters
    p.add_argument("--gt_num_heads", type=int, default=4, help="Number of heads for Graph Transformer")
    p.add_argument("--gt_num_layers", type=int, default=3, help="Number of layers for Graph Transformer")
    p.add_argument("--gt_dropout", type=float, default=0.2, help="Dropout for Graph Transformer")
    p.add_argument("--gt_ffn_ratio", type=int, default=4, help="FFN expansion ratio for Graph Transformer")
    p.add_argument("--gt_layer_norm", action="store_true", default=True, help="Use layer norm in Graph Transformer")
    p.add_argument("--gt_residual", action="store_true", default=True, help="Use residual in Graph Transformer")

    # AMP & Compile
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="use_amp")
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", action="store_false", dest="compile")
    p.add_argument("--compile_backend", type=str, default="inductor")
    p.add_argument("--compile_dynamic", action="store_true", default=True)
    p.add_argument("--compile_fullgraph", action="store_true", default=False)
    p.add_argument("--compile_scope", type=str, default="all", choices=["all", "eval_only"])
    p.add_argument("--compile_mode", type=str, default=None)
    p.add_argument("--compile_allow_double_backward", action="store_true")

    p.add_argument("--output", type=str, default="outputs/graph_tasks/last_metrics.json")
    p.add_argument("--num_runs", type=int, default=1, help="Number of independent runs with different seeds")
    args = p.parse_args()
    args = _apply_task_preset(args)
    args = _apply_benchmark_preset(args)

    seed_everything(args.seed)
    device = resolve_device(args.device)

    all_run_metrics = []
    base_seed = args.seed

    for run_idx in range(args.num_runs):
        current_seed = base_seed + run_idx
        seed_everything(current_seed)
        print(f"\n>>> Starting Run {run_idx + 1}/{args.num_runs} (Seed: {current_seed})")

        if args.cv_folds > 1:
            real_datasets = {"proteins", "nci1", "nci109", "dd", "enzymes", "mutag", "mutagenicity", "frankenstein"}
            if args.dataset_name.lower() == "csl":
                all_samples, num_classes = _load_csl_samples(args)
                task_type = "multiclass"
            elif args.dataset_name.lower() in real_datasets:
                from get.data import RealWorldGraphDataset
                dataset = RealWorldGraphDataset(name=args.dataset_name, root=args.dataset_root, in_dim=args.in_dim, max_motifs_per_anchor=args.max_motifs_per_anchor, task_type="auto")
                all_samples = dataset.items if hasattr(dataset, "items") else [dataset[i] for i in range(len(dataset))]
                num_classes = dataset.num_classes
                task_type = dataset.task_type
            else:
                raise ValueError(f"Cross-validation is not supported for {args.dataset_name}.")
            
            if args.max_graphs > 0:
                all_samples = all_samples[:args.max_graphs]

            labels = np.array([int(sample["y"].view(-1)[0].item()) for sample in all_samples])
            try:
                splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=current_seed)
                fold_iter = splitter.split(np.arange(len(all_samples)), labels)
            except ValueError:
                splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=current_seed)
                fold_iter = splitter.split(np.arange(len(all_samples)))
            
            fold_metrics: List[Dict[str, Dict[str, float]]] = []
            for fold_idx, (trainval_idx, test_idx) in enumerate(fold_iter, start=1):
                trainval_items = [all_samples[int(i)] for i in trainval_idx.tolist()]
                test_items = [all_samples[int(i)] for i in test_idx.tolist()]

                train_items, val_items, _ = split_items(trainval_items, seed=current_seed + fold_idx, train_ratio=0.85, val_ratio=0.15, task_type=task_type)
                args.edge_attr_dim = infer_edge_attr_dim({"train": train_items, "val": val_items, "test": test_items})

                train_loader, val_loader, test_loader, fold_split_stats = _build_loaders_from_samples(
                    train_items=train_items, val_items=val_items, test_items=test_items, args=args, task_type=task_type
                )
                m = _run_single_fit(args=args, device=device, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, task_type=task_type, num_classes=num_classes)
                m["fold"] = fold_idx
                m["split_stats"] = fold_split_stats
                m["runtime_config"] = _runtime_config(args)
                fold_metrics.append(m)

            test_accs = [float(m["test"]["acc"]) for m in fold_metrics]
            test_losses = [float(m["test"]["loss"]) for m in fold_metrics]
            metrics = {
                "run_idx": run_idx,
                "seed": current_seed,
                "cv_folds": int(args.cv_folds),
                "task_type": task_type,
                "num_classes": int(num_classes),
                "fold_metrics": fold_metrics,
                "split_stats": summarize_splits({"all": all_samples}, task_type=task_type),
                "runtime_config": _runtime_config(args),
                "summary": {
                    "test_acc_mean": float(statistics.mean(test_accs)),
                    "test_acc_std": float(statistics.pstdev(test_accs) if len(test_accs) > 1 else 0.0),
                    "test_loss_mean": float(statistics.mean(test_losses)),
                    "test_loss_std": float(statistics.pstdev(test_losses) if len(test_losses) > 1 else 0.0),
                },
            }
        else:
            train_loader, val_loader, test_loader, split_stats, task_type, num_classes = _build_loaders(args)
            if hasattr(train_loader, "dataset") and len(train_loader.dataset) > 0:
                probe_items = {
                    "train": [train_loader.dataset[i] for i in range(min(3, len(train_loader.dataset)))],
                    "val": [val_loader.dataset[i] for i in range(min(3, len(val_loader.dataset)))],
                    "test": [test_loader.dataset[i] for i in range(min(3, len(test_loader.dataset)))],
                }
                args.edge_attr_dim = infer_edge_attr_dim(probe_items)
            metrics = _run_single_fit(
                args=args,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                task_type=task_type,
                num_classes=num_classes,
            )
            metrics["run_idx"] = run_idx
            metrics["seed"] = current_seed
            metrics["task_type"] = task_type
            metrics["num_classes"] = int(num_classes)
            metrics["split_stats"] = split_stats
            metrics["runtime_config"] = _runtime_config(args)
        
        all_run_metrics.append(metrics)

    # Aggregate across runs
    if args.num_runs > 1:
        run_accs = []
        for m in all_run_metrics:
            if "summary" in m:
                run_accs.append(m["summary"]["test_acc_mean"])
            else:
                run_accs.append(m["test"]["acc"])
        
        metrics = {
            "num_runs": args.num_runs,
            "all_runs": all_run_metrics,
            "final_summary": {
                "test_acc_mean": float(statistics.mean(run_accs)),
                "test_acc_std": float(statistics.pstdev(run_accs) if len(run_accs) > 1 else 0.0),
            }
        }
    else:
        metrics = all_run_metrics[0]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
