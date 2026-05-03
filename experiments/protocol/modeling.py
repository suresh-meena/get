from __future__ import annotations

from external.graph_baselines.torch_baselines import (
    ExternalGraphBaseline,
    GATGraphBaseline,
    GCNGraphBaseline,
    GINGraphBaseline,
)
from get.models import EnergyGraphClassifier


def build_model(args, task_type: str, num_classes: int):
    out_dim = 1 if task_type == "binary" else num_classes
    
    # Extract optional flags if present in args, otherwise use defaults
    use_energy_norm = getattr(args, "use_energy_norm", True)
    agg_mode = getattr(args, "agg_mode", "softmax")

    if args.model_name == "fullget":
        return EnergyGraphClassifier(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            num_steps=args.num_steps,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            R=args.R,
            K=args.K,
            num_motif_types=2,
            lambda_2=args.lambda_2,
            lambda_3=args.lambda_3,
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
            armijo_eval_max_backtracks=getattr(args, "armijo_eval_max_backtracks", 5),
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
            energy_name="get_full",
            use_energy_norm=use_energy_norm,
            agg_mode=agg_mode,
        )
    if args.model_name == "pairwiseget":
        return EnergyGraphClassifier(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            num_steps=args.num_steps,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            R=args.R,
            K=args.K,
            num_motif_types=2,
            lambda_2=args.lambda_2,
            lambda_3=0.0,
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
            armijo_eval_max_backtracks=getattr(args, "armijo_eval_max_backtracks", 5),
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
            energy_name="pairwise_only",
            use_energy_norm=use_energy_norm,
            agg_mode=agg_mode,
        )
    if args.model_name in {"et", "etfaithful"}:
        from get.models import ETGraphClassifier
        return ETGraphClassifier(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            num_steps=args.num_steps,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            num_blocks=getattr(args, "et_num_blocks", 1),
            alpha=args.fixed_step_size,
            multiplier=getattr(args, "et_multiplier", 4.0),
            chn_type=getattr(args, "et_chn_type", "relu"),
            use_bias_attn=getattr(args, "et_use_bias_attn", False),
            update_damping=args.update_damping,
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
        )
    if args.model_name == "quadratic_only":
        return EnergyGraphClassifier(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            num_steps=args.num_steps,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            R=args.R,
            K=args.K,
            num_motif_types=2,
            lambda_2=0.0,
            lambda_3=0.0,
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
            armijo_eval_max_backtracks=getattr(args, "armijo_eval_max_backtracks", 5),
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
            energy_name="quadratic_only",
            use_energy_norm=use_energy_norm,
            agg_mode=agg_mode,
        )
    if args.model_name == "bwgnn":
        from get.models.baselines import BWGNNBaseline
        return BWGNNBaseline(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            d=getattr(args, "bwgnn_order", 2)
        )
    if args.model_name in {"graphtransformer", "gt"}:
        from get.models.baselines import GraphTransformerBaseline
        return GraphTransformerBaseline(
            in_dim=args.in_dim,
            hidden_dim=args.hidden_dim,
            num_classes=out_dim,
            num_heads=getattr(args, "gt_num_heads", getattr(args, "num_heads", 4)),
            n_layers=getattr(args, "gt_num_layers", getattr(args, "num_steps", 6)),
            dropout=getattr(args, "gt_dropout", 0.0),
            ffn_ratio=getattr(args, "gt_ffn_ratio", 2),
            layer_norm=getattr(args, "gt_layer_norm", True),
            residual=getattr(args, "gt_residual", True)
        )
    if args.model_name == "external_baseline":
        return ExternalGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    if args.model_name == "gin":
        return GINGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    if args.model_name == "gcn":
        return GCNGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    if args.model_name == "gat":
        return GATGraphBaseline(in_dim=args.in_dim, hidden_dim=args.hidden_dim, out_dim=out_dim)
    raise ValueError(args.model_name)
