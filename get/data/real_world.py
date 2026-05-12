from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset
from torch_geometric.datasets import TUDataset, GNNBenchmarkDataset
from torch_geometric.utils import remove_self_loops

from .synthetic import sample_from_edge_index


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
        cache_enabled: bool = True,
        cache_root: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.root = root
        self.in_dim = in_dim
        self.max_motifs_per_anchor = max_motifs_per_anchor
        self.cache_enabled = cache_enabled
        self.cache_root = Path(cache_root).expanduser() if cache_root is not None else Path(root).expanduser() / "processed_cache" / "real_world"
        self.cache_version = "v2" # Bump version for lazy loading
        
        self.processed_dir = self.cache_root / self.name.upper() / f"dim_{in_dim}_motifs_{max_motifs_per_anchor}"

        metadata = self._load_metadata()
        if metadata is not None:
            self.label_map = metadata["label_map"]
            self.num_classes = int(metadata["num_classes"])
            self.task_type = task_type if task_type != "auto" else metadata["task_type"]
            self._len = metadata["len"]
            return

        if name.upper() in ["PROTEINS", "NCI1", "NCI109", "DD", "ENZYMES", "MUTAG", "MUTAGENICITY", "FRANKENSTEIN"]:
            pyg_ds = TUDataset(root=root, name=name.upper(), use_node_attr=True)
        elif name.upper() == "CSL":
            pyg_ds = GNNBenchmarkDataset(root=root, name="CSL")
        else:
            try:
                pyg_ds = TUDataset(root=root, name=name, use_node_attr=True)
            except Exception:
                raise ValueError(f"Unknown dataset: {name}")

        raw_labels = []
        for item in pyg_ds:
            y = getattr(item, "y", None)
            if y is not None:
                raw_labels.append(int(y.view(-1)[0].item()))
        self.label_map = {label: idx for idx, label in enumerate(sorted(set(raw_labels)))}
        self.num_classes = max(1, len(self.label_map))
        
        if task_type == "auto":
            task_type = "binary" if len(self.label_map) <= 2 else "multiclass"
        self.task_type = task_type
        
        from tqdm import tqdm
        
        # Process and save samples to disk
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._len = len(pyg_ds)
        
        pbar = tqdm(total=self._len, desc=f"Processing {self.name}", unit="graph")
        for i, item in enumerate(pyg_ds):
            sample = self._to_sample(item)
            torch.save(sample, self.processed_dir / f"sample_{i}.pt")
            pbar.update(1)
        pbar.close()
            
        self._save_metadata()

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return torch.load(self.processed_dir / f"sample_{idx}.pt", map_location="cpu", weights_only=False)

    @property
    def items(self):
        # Warning: this will still load everything into RAM if called.
        # Keeping it for compatibility but adding a warning.
        import warnings
        warnings.warn("Accessing .items on a lazy dataset will load everything into RAM and may cause OOM.")
        return [self[i] for i in range(self._len)]

    def _metadata_path(self) -> Path:
        return self.processed_dir / "metadata.pt"

    def _load_metadata(self):
        if not self.cache_enabled:
            return None
        meta_path = self._metadata_path()
        if not meta_path.exists():
            return None
        return torch.load(meta_path, map_location="cpu", weights_only=False)

    def _save_metadata(self) -> None:
        if not self.cache_enabled:
            return
        payload = {
            "cache_version": self.cache_version,
            "name": self.name,
            "in_dim": self.in_dim,
            "max_motifs_per_anchor": self.max_motifs_per_anchor,
            "task_type": self.task_type,
            "label_map": self.label_map,
            "num_classes": self.num_classes,
            "len": self._len,
        }
        torch.save(payload, self._metadata_path())

    def _to_sample(self, data) -> Dict[str, torch.Tensor]:
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
        edge_index, _ = remove_self_loops(data.edge_index.long())

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

        return sample_from_edge_index(edge_index, n, x, label, self.max_motifs_per_anchor)
