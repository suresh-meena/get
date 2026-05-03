from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Dict, Tuple

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from get.data import SyntheticGraphDataset, collate_graph_samples
from get.models import EnergyGraphClassifier
from get.trainers import UnifiedTrainer
from get.utils import maybe_compile_model, seed_everything


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders(cfg: DictConfig, seed: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    ds = cfg.dataset
    tr = cfg.trainer
    train_ds = SyntheticGraphDataset(
        num_graphs=int(ds.num_train_graphs),
        min_nodes=int(ds.min_nodes),
        max_nodes=int(ds.max_nodes),
        edge_prob=float(ds.edge_prob),
        in_dim=int(ds.in_dim),
        max_motifs_per_anchor=int(ds.max_motifs_per_anchor),
        seed=seed,
    )
    val_ds = SyntheticGraphDataset(
        num_graphs=int(ds.num_val_graphs),
        min_nodes=int(ds.min_nodes),
        max_nodes=int(ds.max_nodes),
        edge_prob=float(ds.edge_prob),
        in_dim=int(ds.in_dim),
        max_motifs_per_anchor=int(ds.max_motifs_per_anchor),
        seed=seed + 1,
    )
    test_ds = SyntheticGraphDataset(
        num_graphs=int(ds.num_test_graphs),
        min_nodes=int(ds.min_nodes),
        max_nodes=int(ds.max_nodes),
        edge_prob=float(ds.edge_prob),
        in_dim=int(ds.in_dim),
        max_motifs_per_anchor=int(ds.max_motifs_per_anchor),
        seed=seed + 2,
    )
    num_workers = int(tr.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(tr.batch_size),
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_graph_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(tr.get("eval_batch_size", tr.batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_samples,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(tr.get("eval_batch_size", tr.batch_size)),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_graph_samples,
    )
    return train_loader, val_loader, test_loader


def _build_model(cfg: DictConfig) -> torch.nn.Module:
    m = cfg.model
    # Handle both flat and nested 'params' style configs
    p = m.get("params", m)
    factory = m.get("factory", "fullget").lower()

    if factory == "etfaithful" or factory == "et":
        from get.models import ETGraphClassifier
        # Map 'd' to 'hidden_dim' if present (etfaithful style)
        h_dim = p.get("d", p.get("hidden_dim", 128))
        return ETGraphClassifier(
            in_dim=int(cfg.dataset.in_dim),
            hidden_dim=int(h_dim),
            num_classes=int(p.num_classes),
            num_heads=int(p.num_heads),
            head_dim=int(p.head_dim),
            num_steps=int(p.num_steps),
            num_blocks=int(p.get("num_blocks", 1)),
            alpha=float(p.get("fixed_step_size", 0.1)),
            multiplier=float(p.get("multiplier", 4.0)),
            chn_type=str(p.get("chn_type", "relu")),
            use_bias_attn=bool(p.get("use_bias_attn", False)),
            update_damping=float(p.get("update_damping", 0.0)),
        )

    if factory == "bwgnn":
        from get.models.baselines import BWGNNBaseline
        return BWGNNBaseline(
            in_dim=int(cfg.dataset.in_dim),
            hidden_dim=int(p.get("hidden_dim", 64)),
            num_classes=int(p.num_classes),
            d=int(p.get("bwgnn_order", 2))
        )

    if factory in {"graphtransformer", "gt"}:
        from get.models.baselines import GraphTransformerBaseline
        return GraphTransformerBaseline(
            in_dim=int(cfg.dataset.in_dim),
            hidden_dim=int(p.get("hidden_dim", 64)),
            num_classes=int(p.num_classes),
            num_heads=int(p.get("gt_num_heads", p.get("num_heads", 4))),
            n_layers=int(p.get("gt_num_layers", p.get("num_steps", 6))),
            dropout=float(p.get("gt_dropout", 0.0)),
            ffn_ratio=int(p.get("gt_ffn_ratio", 2)),
            layer_norm=bool(p.get("gt_layer_norm", True)),
            residual=bool(p.get("gt_residual", True))
        )


    # Default to GET
    return EnergyGraphClassifier(
        in_dim=int(cfg.dataset.in_dim),
        hidden_dim=int(p.hidden_dim),
        num_classes=int(p.num_classes),
        num_steps=int(p.num_steps),
        num_heads=int(p.num_heads),
        head_dim=int(p.head_dim),
        R=int(p.R),
        K=int(p.K),
        num_motif_types=int(p.num_motif_types),
        lambda_2=float(p.lambda_2),
        lambda_3=float(p.lambda_3),
        lambda_m=float(p.lambda_m),
        beta_2=float(p.beta_2),
        beta_3=float(p.beta_3),
        beta_m=float(p.beta_m),
        update_damping=float(p.update_damping),
        fixed_step_size=float(p.get("fixed_step_size", 0.1)),
        armijo_eta0=float(p.get("armijo_eta0", 0.2)),
        armijo_gamma=float(p.get("armijo_gamma", 0.5)),
        armijo_c=float(p.get("armijo_c", 1e-4)),
        armijo_max_backtracks=int(p.get("armijo_max_backtracks", 20)),
        inference_mode_train=str(cfg.experiment.get("inference_mode_train", "fixed")),
        inference_mode_eval=str(cfg.experiment.get("inference_mode_eval", "armijo")),
        energy_name=str(p.get("energy_name", "get_full")),
    )


def run_from_cfg(cfg: DictConfig) -> Dict:
    seed_everything(int(cfg.seed))
    device = _resolve_device(str(cfg.experiment.get("device", "auto")))

    train_loader, val_loader, test_loader = _build_loaders(cfg, seed=int(cfg.seed))
    model = _build_model(cfg)

    compile_cfg = OmegaConf.to_container(cfg.experiment.get("compile", {}), resolve=True)
    compile_scope = str(compile_cfg.get("scope", "eval_only")).lower()
    eval_model = model
    if bool(compile_cfg.get("enabled", False)):
        if compile_scope == "all":
            if getattr(model, "requires_double_backward", False):
                raise ValueError(
                    "compile.scope='all' is unsupported for GET training because torch.compile "
                    "does not currently support double backward. Use compile.scope='eval_only'."
                )
            model = maybe_compile_model(model, compile_cfg)
            eval_model = model
        elif compile_scope == "eval_only":
            eval_compile_cfg = dict(compile_cfg)
            # Safe default for GET: compile eval path while keeping train path uncompiled.
            eval_compile_cfg["allow_double_backward"] = True
            eval_model = maybe_compile_model(model, eval_compile_cfg)
        else:
            raise ValueError(f"Unsupported compile.scope '{compile_scope}'. Use 'eval_only' or 'all'.")

    num_runs = int(cfg.get("num_runs", 3))
    all_run_metrics = []
    base_seed = int(cfg.seed)

    for run_idx in range(num_runs):
        current_seed = base_seed + run_idx
        seed_everything(current_seed)
        print(f"\n>>> Starting Run {run_idx + 1}/{num_runs} (Seed: {current_seed})")

        trainer = UnifiedTrainer(
            model=model,
            eval_model=eval_model,
            device=device,
            trainer_cfg=OmegaConf.to_container(cfg.trainer, resolve=True),
        )
        metrics = trainer.fit(train_loader, val_loader, test_loader)
        metrics["run_idx"] = run_idx
        metrics["seed"] = current_seed
        all_run_metrics.append(metrics)

    if num_runs > 1:
        run_accs = [m["test"]["acc"] for m in all_run_metrics]
        final_metrics = {
            "num_runs": num_runs,
            "all_runs": all_run_metrics,
            "final_summary": {
                "test_acc_mean": float(statistics.mean(run_accs)),
                "test_acc_std": float(statistics.pstdev(run_accs) if len(run_accs) > 1 else 0.0),
            }
        }
        metrics = final_metrics
    else:
        metrics = all_run_metrics[0]

    out_dir = Path("outputs") / "refactor"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "last_run_metrics.json"
    out_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    metrics = run_from_cfg(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
