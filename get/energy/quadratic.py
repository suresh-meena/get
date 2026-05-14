import torch
import torch.nn as nn
from torch_geometric.utils import scatter

class QuadraticEnergy(nn.Module):
    def forward(self, X: torch.Tensor, batch: torch.Tensor, num_graphs: int):
        node_energies = 0.5 * (X ** 2).sum(dim=-1)
        graph_energy = scatter(node_energies, batch, dim=0, dim_size=num_graphs, reduce="sum")
        counts = torch.bincount(batch, minlength=num_graphs).to(dtype=X.dtype, device=X.device)
        return graph_energy / counts.clamp_min(1.0)
