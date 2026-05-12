from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from get.trainers import UnifiedTrainer
from get.data.synthetic import sample_from_adj, collate_graph_samples


class _TinyGraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class _BiasHead(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, batch):
        # Handle both dict-based and object-based batches
        if isinstance(batch, dict) and "batch" in batch:
            num_graphs = int(batch["batch"].max().item() + 1)
        else:
            num_graphs = int(getattr(batch, "num_graphs", 1))
        return self.bias.unsqueeze(0).expand(num_graphs, -1)


def _make_sample(y):
    adj = torch.zeros((1, 1), dtype=torch.bool)
    x = torch.zeros((1, 2), dtype=torch.float32)
    return sample_from_adj(adj, x, torch.tensor(y, dtype=torch.float32), max_motifs_per_anchor=0)


def test_unified_trainer_supports_multilabel_targets():
    samples = [_make_sample([0.0, 1.0]), _make_sample([1.0, 0.0])]
    loader = DataLoader(_TinyGraphDataset(samples), batch_size=2, shuffle=False, collate_fn=collate_graph_samples)
    model = _BiasHead(out_dim=2)
    
    trainer_cfg = {
        "task_type": "multilabel",
        "num_classes": 2,
        "lr": 1e-2,
        "weight_decay": 0.01,
        "epochs": 1,
    }
    
    trainer = UnifiedTrainer(
        model=model,
        device=torch.device("cpu"),
        trainer_cfg=trainer_cfg
    )

    metrics = trainer._run_epoch(loader, train=True)

    assert "auc" in metrics
    assert metrics["binary_ranking_available"] == 1.0
    assert "acc" in metrics


def test_unified_trainer_supports_vector_regression_targets():
    samples = [_make_sample([0.5, 1.5, 2.5]), _make_sample([3.5, 4.5, 5.5])]
    loader = DataLoader(_TinyGraphDataset(samples), batch_size=2, shuffle=False, collate_fn=collate_graph_samples)
    model = _BiasHead(out_dim=3)
    
    trainer_cfg = {
        "task_type": "regression",
        "num_classes": 3,
        "lr": 1e-2,
        "weight_decay": 0.01,
        "epochs": 1,
    }
    
    trainer = UnifiedTrainer(
        model=model,
        device=torch.device("cpu"),
        trainer_cfg=trainer_cfg
    )

    metrics = trainer._run_epoch(loader, train=True)

    assert "mae" in metrics
    assert metrics["mae"] >= 0.0
