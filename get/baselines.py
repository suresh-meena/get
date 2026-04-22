from .model import GETModel
from .et_faithful import ETFaithfulGraphModel
import torch
import torch.nn as nn

try:
    from torch_geometric.nn import GINConv, global_add_pool
except ImportError:
    GINConv = None
    global_add_pool = None

class GINBaseline(nn.Module):
    def __init__(self, in_dim, d, num_classes, num_layers=3):
        super().__init__()
        if GINConv is None or global_add_pool is None:
            raise ImportError("torch_geometric is required to use GINBaseline")
        self.encoder = nn.Linear(in_dim, d)
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
            x = global_add_pool(x, batch_data.batch)
            out = self.readout(x)
            return out, None
        else:
            out = self.readout(x)
            return out, None


def PairwiseGET(in_dim, d, num_classes, **kwargs):
    """Pairwise-only baseline (lambda_3=0, lambda_m=0)"""
    beta_2 = kwargs.pop('beta_2', 5.0)
    return GETModel(
        in_dim,
        d,
        num_classes,
        lambda_3=0.0,
        lambda_m=0.0,
        beta_2=beta_2,
        use_motif=False,
        use_memory=False,
        **kwargs,
    )


def MotifOnlyGET(in_dim, d, num_classes, **kwargs):
    """Motif-only ablation (pairwise and memory branches disabled)."""
    return GETModel(
        in_dim,
        d,
        num_classes,
        lambda_2=0.0,
        lambda_m=0.0,
        use_pairwise=False,
        use_motif=True,
        use_memory=False,
        **kwargs,
    )


def FullGET(in_dim, d, num_classes, 
            lambda_2=1.0, lambda_3=0.5, lambda_m=1.0, 
            beta_2=1.0, beta_3=1.0, beta_m=1.0, **kwargs):
    """Full GET model with motif and memory branches active"""
    return GETModel(in_dim, d, num_classes, 
                    lambda_2=lambda_2, lambda_3=lambda_3, lambda_m=lambda_m, 
                    beta_2=beta_2, beta_3=beta_3, beta_m=beta_m, **kwargs)


def MemoryOnlyGET(in_dim, d, num_classes, **kwargs):
    """Memory-only baseline (pairwise and motif branches disabled)."""
    return GETModel(
        in_dim,
        d,
        num_classes,
        lambda_2=0.0,
        lambda_3=0.0,
        use_pairwise=False,
        use_motif=False,
        use_memory=True,
        **kwargs,
    )


def ETFaithful(in_dim, d, num_classes, **kwargs):
    """Paper-inspired ET with CLS token, Laplacian PE, masked energy attention, and HN memory."""
    return ETFaithfulGraphModel(in_dim, d, num_classes, **kwargs)
