"""Baseline models and factory functions."""
from get.utils.registry import register_model
from get.models.get_model import GETModel
from get.models.et_core import ETFaithfulGraphModel
from get.models.gnn_baselines import GINBaseline, GCNBaseline, GATBaseline


@register_model("pairwise")
@register_model("pairwiseget")
def PairwiseGET(in_dim, d=256, num_classes=1, **kwargs):
    """Pairwise-only baseline (lambda_3=0, lambda_m=0)"""
    return GETModel(in_dim, d, num_classes, lambda_3=0.0, lambda_m=0.0,
                    use_motif=False, use_memory=False, **kwargs)


@register_model("full")
@register_model("fullget")
def FullGET(in_dim, d=256, num_classes=1,
            lambda_2=1.0, lambda_3=0.5, lambda_m=1.0,
            beta_2=1.0, beta_3=1.0, beta_m=1.0, num_motif_types=2, **kwargs):
    """Full GET model with motif and memory branches active"""
    return GETModel(in_dim, d, num_classes,
                    lambda_2=lambda_2, lambda_3=lambda_3, lambda_m=lambda_m,
                    beta_2=beta_2, beta_3=beta_3, beta_m=beta_m,
                    num_motif_types=num_motif_types, **kwargs)


@register_model("etfaithful")
@register_model("etfaithfulgraphmodel")
def ETFaithful(in_dim, d, num_classes, **kwargs):
    """Paper-inspired ET with CLS token, Laplacian PE, masked energy attention, and HN memory."""
    return ETFaithfulGraphModel(in_dim, d, num_classes, **kwargs)


@register_model("gin")
@register_model("ginbaseline")
def _build_gin(in_dim, d, num_classes, **kwargs):
    return GINBaseline(in_dim, d, num_classes, **kwargs)


@register_model("gcn")
@register_model("gcnbaseline")
def _build_gcn(in_dim, d, num_classes, **kwargs):
    return GCNBaseline(in_dim, d, num_classes, **kwargs)


@register_model("gat")
@register_model("gatbaseline")
def _build_gat(in_dim, d, num_classes, **kwargs):
    return GATBaseline(in_dim, d, num_classes, **kwargs)
