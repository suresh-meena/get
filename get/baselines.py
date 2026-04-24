from .model import GETModel
from .et_faithful import ETFaithfulGraphModel
from .registry import register_model
import torch
import torch.nn as nn

class GINBaseline(nn.Module):
    def __init__(self, in_dim, d, num_classes, num_layers=3):
        super().__init__()
        try:
            from torch_geometric.nn import GINConv, global_add_pool
        except ImportError as exc:
            raise ImportError("torch_geometric is required to use GINBaseline") from exc
        self.encoder = nn.Linear(in_dim, d)
        self.global_add_pool = global_add_pool
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, d)
            )
            self.convs.append(GINConv(mlp))
        
        self.readout = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, num_classes)
        )
        
    def forward(self, batch_data, task_level='graph'):
        x = self.encoder(batch_data.x)
        edge_index = torch.stack([batch_data.c_2, batch_data.u_2], dim=0)
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
            
        if task_level == 'graph':
            x = self.global_add_pool(x, batch_data.batch)
            out = self.readout(x)
            return out, None
        else:
            out = self.readout(x)
            return out, None


@register_model("pairwise")
@register_model("pairwiseget")
def PairwiseGET(in_dim, d=256, num_classes=1, **kwargs):
    """Pairwise-only baseline (lambda_3=0, lambda_m=0)"""
    return GETModel(
        in_dim,
        d,
        num_classes,
        lambda_3=0.0,
        lambda_m=0.0,
        use_motif=False,
        use_memory=False,
        **kwargs,
    )


@register_model("full")
@register_model("fullget")
def FullGET(in_dim, d=256, num_classes=1,
            lambda_2=1.0, lambda_3=0.5, lambda_m=1.0,
            beta_2=1.0, beta_3=1.0, beta_m=1.0, num_motif_types=2, **kwargs):
    """Full GET model with motif and memory branches active"""
    return GETModel(in_dim, d, num_classes, 
                    lambda_2=lambda_2, lambda_3=lambda_3, lambda_m=lambda_m, 
                    beta_2=beta_2, beta_3=beta_3, beta_m=beta_m, num_motif_types=num_motif_types, **kwargs)


@register_model("etfaithful")
@register_model("etfaithfulgraphmodel")
def ETFaithful(in_dim, d, num_classes, **kwargs):
    """Paper-inspired ET with CLS token, Laplacian PE, masked energy attention, and HN memory."""
    return ETFaithfulGraphModel(in_dim, d, num_classes, **kwargs)


@register_model("gin")
@register_model("ginbaseline")
def _build_gin(in_dim, d, num_classes, **kwargs):
    return GINBaseline(in_dim, d, num_classes, **kwargs)
