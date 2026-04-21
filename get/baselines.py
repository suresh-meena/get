from .model import GETModel
from .et_faithful import ETFaithfulGraphModel
from .data import GETBatch
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


def _build_complete_graph_batch(batch_data):
    ptr = batch_data.ptr
    device = batch_data.x.device
    c_2_parts = []
    u_2_parts = []
    for g_idx in range(ptr.numel() - 1):
        start = int(ptr[g_idx].item())
        end = int(ptr[g_idx + 1].item())
        n = end - start
        if n <= 0:
            continue
        nodes = torch.arange(start, end, device=device, dtype=torch.long)
        row = nodes.repeat_interleave(n)
        col = nodes.repeat(n)
        c_2_parts.append(row)
        u_2_parts.append(col)

    if c_2_parts:
        c_2 = torch.cat(c_2_parts, dim=0)
        u_2 = torch.cat(u_2_parts, dim=0)
    else:
        c_2 = torch.empty(0, dtype=torch.long, device=device)
        u_2 = torch.empty(0, dtype=torch.long, device=device)

    c_3 = torch.empty(0, dtype=torch.long, device=device)
    u_3 = torch.empty(0, dtype=torch.long, device=device)
    v_3 = torch.empty(0, dtype=torch.long, device=device)
    t_tau = torch.empty(0, dtype=torch.long, device=device)
    edge_attr = None

    out = GETBatch(
        batch_data.x,
        c_2,
        u_2,
        c_3,
        u_3,
        v_3,
        t_tau,
        batch_data.batch,
        batch_data.ptr,
        y=batch_data.y,
        edge_attr=edge_attr,
    )
    return out


class ETLocalBaseline(nn.Module):
    """ET-local baseline: pairwise GET on provided sparse graph support."""

    def __init__(self, in_dim, d, num_classes, **kwargs):
        super().__init__()
        beta_2 = kwargs.pop("beta_2", 1.0)
        self.model = GETModel(
            in_dim,
            d,
            num_classes,
            lambda_3=0.0,
            lambda_m=0.0,
            beta_2=beta_2,
            **kwargs,
        )

    def forward(self, batch_data, task_level="graph"):
        return self.model(batch_data, task_level=task_level)


class ETCompleteBaseline(nn.Module):
    """ET-complete baseline: pairwise GET on complete graph support per graph."""

    def __init__(self, in_dim, d, num_classes, **kwargs):
        super().__init__()
        beta_2 = kwargs.pop("beta_2", 1.0)
        self.model = GETModel(
            in_dim,
            d,
            num_classes,
            lambda_3=0.0,
            lambda_m=0.0,
            beta_2=beta_2,
            **kwargs,
        )

    def forward(self, batch_data, task_level="graph"):
        complete_batch = _build_complete_graph_batch(batch_data)
        return self.model(complete_batch, task_level=task_level)

def PairwiseGET(in_dim, d, num_classes, **kwargs):
    """Pairwise-only baseline (lambda_3=0, lambda_m=0)"""
    beta_2 = kwargs.pop('beta_2', 5.0)
    return GETModel(in_dim, d, num_classes, lambda_3=0.0, lambda_m=0.0, beta_2=beta_2, **kwargs)

def FullGET(in_dim, d, num_classes, 
            lambda_2=1.0, lambda_3=0.5, lambda_m=1.0, 
            beta_2=1.0, beta_3=1.0, beta_m=1.0, **kwargs):
    """Full GET model with motif and memory branches active"""
    return GETModel(in_dim, d, num_classes, 
                    lambda_2=lambda_2, lambda_3=lambda_3, lambda_m=lambda_m, 
                    beta_2=beta_2, beta_3=beta_3, beta_m=beta_m, **kwargs)

def GETWithMemory(in_dim, d, num_classes, lambda_3=0.0, **kwargs):
    """Pairwise + memory baseline (lambda_3=0)"""
    return GETModel(in_dim, d, num_classes, lambda_3=lambda_3, **kwargs)

def GETWithMotif(in_dim, d, num_classes, lambda_m=0.0, **kwargs):
    """Pairwise + motif baseline (lambda_m=0)"""
    return GETModel(in_dim, d, num_classes, lambda_m=lambda_m, **kwargs)


def ETLocal(in_dim, d, num_classes, **kwargs):
    return ETLocalBaseline(in_dim, d, num_classes, **kwargs)


def ETComplete(in_dim, d, num_classes, **kwargs):
    return ETCompleteBaseline(in_dim, d, num_classes, **kwargs)


def ETFaithful(in_dim, d, num_classes, **kwargs):
    """Paper-inspired ET with CLS token, Laplacian PE, masked energy attention, and HN memory."""
    return ETFaithfulGraphModel(in_dim, d, num_classes, **kwargs)


def ETInspiredGET(in_dim, d, num_classes, **kwargs):
    """
    GET variant inspired by ET equations:
    - symmetric pairwise energy term q_i*k_j + q_j*k_i
    - keeps GET memory block energy active by default
    - motif branch disabled to stay close to ET attention+memory design
    """
    beta_2 = kwargs.pop("beta_2", 1.0)
    beta_m = kwargs.pop("beta_m", 1.0)
    lambda_m = kwargs.pop("lambda_m", 1.0)
    return GETModel(
        in_dim,
        d,
        num_classes,
        lambda_3=0.0,
        lambda_m=lambda_m,
        beta_2=beta_2,
        beta_m=beta_m,
        pairwise_symmetric=True,
        **kwargs,
    )
