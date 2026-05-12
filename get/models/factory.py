from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf
from typing import Any

from external.graph_baselines.torch_baselines import (
    ExternalGraphBaseline,
    GATGraphBaseline,
    GCNGraphBaseline,
    GINGraphBaseline,
)
from get.models.energy_classifier import EnergyGraphClassifier
from get.models.et_classifier import ETGraphClassifier
from get.models.baselines import BWGNNBaseline, GraphTransformerBaseline


def build_model(cfg: Any) -> torch.nn.Module:
    """
    Universal model factory for GET and baselines.
    Supports both Hydra DictConfig and argparse Namespace.
    """
    # Helper to get values from either DictConfig or Namespace, with fallback to sub-configs
    def get_val(key: str, default: Any = None) -> Any:
        res = None
        if isinstance(cfg, DictConfig):
            res = cfg.get(key)
            if res is None and "model" in cfg:
                res = cfg.model.get(key)
        else:
            res = getattr(cfg, key, None)
            if res is None and hasattr(cfg, "model"):
                model_attr = getattr(cfg, "model")
                res = getattr(model_attr, key, None) if not isinstance(model_attr, dict) else model_attr.get(key)
        
        return res if res is not None else default

    model_name = str(get_val("model_name", "fullget")).lower()
    task_type = str(get_val("task_type", "binary")).lower()
    num_classes = int(get_val("num_classes", 1))
    in_dim = int(get_val("in_dim", 32))
    hidden_dim = int(get_val("hidden_dim", 128))
    
    out_dim = 1 if task_type in {"binary", "node_binary", "multilabel"} else num_classes
    if task_type == "multiclass":
        out_dim = num_classes

    # GET Variants
    if model_name in {"fullget", "pairwiseget", "quadratic_only"}:
        energy_name = "get_full"
        if model_name == "pairwiseget":
            energy_name = "pairwise_only"
        elif model_name == "quadratic_only":
            energy_name = "quadratic_only"
            
        return EnergyGraphClassifier(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=out_dim,
            num_steps=int(get_val("num_steps", 8)),
            num_heads=int(get_val("num_heads", 4)),
            head_dim=int(get_val("head_dim", 32)),
            R=int(get_val("R", 3)),
            K=int(get_val("K", 48)),
            num_motif_types=int(get_val("num_motif_types", 2)),
            lambda_2=float(get_val("lambda_2", 1.0)),
            lambda_3=float(get_val("lambda_3", 10.0)) if energy_name == "get_full" else 0.0,
            lambda_m=float(get_val("lambda_m", 1.0)) if energy_name == "get_full" else 0.0,
            beta_2=float(get_val("beta_2", 1.0)),
            beta_3=float(get_val("beta_3", 1.0)),
            beta_m=float(get_val("beta_m", 1.0)),
            update_damping=float(get_val("update_damping", 0.0)),
            fixed_step_size=float(get_val("fixed_step_size", 0.1)),
            armijo_eta0=float(get_val("armijo_eta0", 0.2)),
            armijo_gamma=float(get_val("armijo_gamma", 0.5)),
            armijo_c=float(get_val("armijo_c", 1e-4)),
            armijo_max_backtracks=int(get_val("armijo_max_backtracks", 20)),
            armijo_eval_max_backtracks=int(get_val("armijo_eval_max_backtracks", 5)),
            inference_mode_train=str(get_val("inference_mode_train", "fixed")),
            inference_mode_eval=str(get_val("inference_mode_eval", "armijo")),
            energy_name=energy_name,
            use_energy_norm=bool(get_val("use_energy_norm", True)),
            agg_mode=str(get_val("agg_mode", "softmax")),
            readout_mode="node" if task_type.startswith("node_") else "graph",
            num_blocks=int(get_val("num_blocks", 1)),
        )

    # Energy Transformer (ET)
    if model_name in {"et", "etfaithful"}:
        et_readout = str(get_val("readout_mode", "cls"))
        if task_type.startswith("node_"):
            et_readout = "node"
            
        return ETGraphClassifier(
            in_dim=in_dim,
            hidden_dim=int(get_val("hidden_dim", hidden_dim)),
            num_classes=out_dim,
            num_steps=int(get_val("num_steps", 1)),
            num_heads=int(get_val("num_heads", 12)),
            head_dim=int(get_val("head_dim", 64)),
            num_blocks=int(get_val("num_blocks", 4)),
            alpha=float(get_val("alpha", 0.1)),
            multiplier=float(get_val("multiplier", 4.0)),
            chn_type=str(get_val("chn_type", "relu")),
            use_bias_attn=bool(get_val("use_bias_attn", False)),
            use_bias_chn=bool(get_val("use_bias_chn", False)),
            use_bias_norm=bool(get_val("use_bias_norm", True)),
            use_cls_token=bool(get_val("use_cls_token", True)),
            pos_k=int(get_val("pos_k", 15)),
            embed_type=str(get_val("embed_type", "eigen")),
            flip_sign=bool(get_val("flip_sign", False)),
            compute_corr=bool(get_val("compute_corr", True)),
            noise_std=float(get_val("noise_std", 0.02)),
            vary_noise=bool(get_val("vary_noise", False)),
            readout_mode=et_readout,
            update_damping=float(get_val("update_damping", 0.0)),
        )

    # Baselines
    if model_name == "bwgnn":
        return BWGNNBaseline(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=out_dim,
            d=int(get_val("bwgnn_order", 3))
        )
        
    if model_name in {"graphtransformer", "gt"}:
        return GraphTransformerBaseline(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_classes=out_dim,
            num_heads=int(get_val("gt_num_heads", 4)),
            n_layers=int(get_val("gt_num_layers", 8)),
            dropout=float(get_val("gt_dropout", 0.2)),
            ffn_ratio=int(get_val("gt_ffn_ratio", 4)),
            layer_norm=bool(get_val("gt_layer_norm", True)),
            residual=bool(get_val("gt_residual", True))
        )

    if model_name == "external_baseline":
        return ExternalGraphBaseline(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "gin":
        return GINGraphBaseline(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "gcn":
        return GCNGraphBaseline(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
    if model_name == "gat":
        return GATGraphBaseline(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)

    raise ValueError(f"Unknown model_name: {model_name}")
