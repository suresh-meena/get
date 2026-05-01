from __future__ import annotations

import json
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


def _build_model(cfg: DictConfig) -> EnergyGraphClassifier:
    m = cfg.model
    exp = cfg.experiment
    return EnergyGraphClassifier(
        in_dim=int(cfg.dataset.in_dim),
        hidden_dim=int(m.hidden_dim),
        num_classes=int(m.num_classes),
        num_steps=int(m.num_steps),
        num_heads=int(m.num_heads),
        head_dim=int(m.head_dim),
        R=int(m.R),
        K=int(m.K),
        num_motif_types=int(m.num_motif_types),
        lambda_2=float(m.lambda_2),
        lambda_3=float(m.lambda_3),
        lambda_m=float(m.lambda_m),
        beta_2=float(m.beta_2),
        beta_3=float(m.beta_3),
        beta_m=float(m.beta_m),
        update_damping=float(m.update_damping),
        fixed_step_size=float(m.get("fixed_step_size", 0.1)),
        armijo_eta0=float(m.get("armijo_eta0", 0.2)),
        armijo_gamma=float(m.get("armijo_gamma", 0.5)),
        armijo_c=float(m.get("armijo_c", 1e-4)),
        armijo_max_backtracks=int(m.get("armijo_max_backtracks", 20)),
        inference_mode_train=str(exp.get("inference_mode_train", "fixed")),
        inference_mode_eval=str(exp.get("inference_mode_eval", "armijo")),
        energy_name=str(m.get("energy_name", "get_full")),
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
            model = maybe_compile_model(model, compile_cfg)
            eval_model = model
        elif compile_scope == "eval_only":
            eval_compile_cfg = dict(compile_cfg)
            # Safe default for GET: compile eval path while keeping train path uncompiled.
            eval_compile_cfg["allow_double_backward"] = True
            eval_model = maybe_compile_model(model, eval_compile_cfg)
        else:
            raise ValueError(f"Unsupported compile.scope '{compile_scope}'. Use 'eval_only' or 'all'.")

    trainer = UnifiedTrainer(
        model=model,
        eval_model=eval_model,
        device=device,
        trainer_cfg=OmegaConf.to_container(cfg.trainer, resolve=True),
    )
    metrics = trainer.fit(train_loader, val_loader, test_loader)

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
