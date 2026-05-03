from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset

from .synthetic import sample_from_adj


class RealWorldGraphDataset(Dataset):
    """
    Wrapper for real-world graph datasets (TUDataset, etc.) 
    to match the unified sample format.
    """

    def __init__(
        self,
        name: str,
        root: str = "data/real",
        in_dim: int = 32,
        max_motifs_per_anchor: int = 8,
        task_type: str = "auto",
    ) -> None:
        super().__init__()
        self.name = name
        self.root = root
        self.in_dim = in_dim
        self.max_motifs_per_anchor = max_motifs_per_anchor
        self.task_type = task_type

        if name.upper() in ["PROTEINS", "NCI1", "NCI109", "DD", "ENZYMES", "MUTAG", "MUTAGENICITY", "FRANKENSTEIN"]:
            self.pyg_ds = TUDataset(root=root, name=name.upper())
        elif name.upper() == "CSL":
            self.pyg_ds = GNNBenchmarkDataset(root=root, name="CSL")
        else:
            # Fallback for other TUDatasets
            try:
                self.pyg_ds = TUDataset(root=root, name=name)
            except:
                raise ValueError(f"Unknown dataset: {name}")

        self.num_classes = self.pyg_ds.num_classes
        raw_labels = []
        for item in self.pyg_ds:
            y = getattr(item, "y", None)
            if y is None:
                continue
            raw_labels.append(int(y.view(-1)[0].item()))
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(raw_labels)))}
        task_type = task_type.lower()
        if task_type == "auto":
            task_type = "binary" if len(self.label_map) <= 2 else "multiclass"
        self.task_type = task_type

    def __len__(self) -> int:
        return len(self.pyg_ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.pyg_ds[idx]
        n = data.num_nodes
        
        # Node features
        x = data.x
        if x is None:
            # Use constant if no features
            x = torch.ones((n, 1), dtype=torch.float32)
        x = x.float()
        
        # Feature padding/truncation
        if x.size(1) < self.in_dim:
            pad = torch.zeros((n, self.in_dim - x.size(1)), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif x.size(1) > self.in_dim:
            x = x[:, :self.in_dim]

        # Adjacency
        adj = torch.zeros((n, n), dtype=torch.bool)
        if data.edge_index.numel() > 0:
            adj[data.edge_index[0], data.edge_index[1]] = True
        adj.fill_diagonal_(False)

        # Label
        y = data.y
        if y is None:
            y = torch.tensor(0)
        y = y.view(-1).long()
        y_val = int(y[0].item())
        y_val = self.label_map.get(y_val, y_val)

        if self.task_type == "binary":
            label = torch.tensor([float(y_val)], dtype=torch.float32)
        elif self.task_type == "multiclass":
            label = torch.tensor([float(y_val)], dtype=torch.float32)
        else:
            label = torch.tensor([float(y_val)], dtype=torch.float32)

        return sample_from_adj(adj, x, label, self.max_motifs_per_anchor)
