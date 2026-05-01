from __future__ import annotations

from external.graph_baselines.torch_baselines import (
    ExternalGraphBaseline,
    GATGraphBaseline,
    GCNGraphBaseline,
    GINGraphBaseline,
)
from get.models import EnergyGraphClassifier


def build_model(args, task_type: str, num_classes: int):
    out_dim = 1 if task_type in {"binary", "regression"} else num_classes
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
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
            energy_name="get_full",
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
            inference_mode_train=args.inference_mode_train,
            inference_mode_eval=args.inference_mode_eval,
            energy_name="pairwise_only",
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

