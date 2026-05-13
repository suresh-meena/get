from __future__ import annotations

import torch
import pytest
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


class _FixedNodeLogitHead(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits.clone().float())
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        num_nodes = int(batch["y"].numel())
        return self.fixed_logits[:num_nodes].to(batch["y"].device) + self.bias * 0.0


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


def test_unified_trainer_masks_node_binary_loss_and_pos_weight():
    masked_sample = _make_sample([1.0, 1.0, 0.0, 0.0])
    masked_sample.mask = torch.tensor([True, True, False, False])
    plain_sample = _make_sample([1.0, 1.0, 0.0, 0.0])

    masked_loader = DataLoader(_TinyGraphDataset([masked_sample]), batch_size=1, shuffle=False, collate_fn=collate_graph_samples)
    plain_loader = DataLoader(_TinyGraphDataset([plain_sample]), batch_size=1, shuffle=False, collate_fn=collate_graph_samples)

    trainer_cfg = {
        "task_type": "node_binary",
        "num_classes": 1,
        "lr": 1e-2,
        "weight_decay": 0.01,
        "epochs": 1,
    }

    masked_trainer = UnifiedTrainer(
        model=_FixedNodeLogitHead(torch.tensor([5.0, 5.0, 5.0, 5.0])),
        device=torch.device("cpu"),
        trainer_cfg=trainer_cfg,
    )
    masked_pos_weight, _ = masked_trainer._collect_train_stats(masked_loader)
    masked_metrics = masked_trainer._run_epoch(masked_loader, train=False)

    plain_trainer = UnifiedTrainer(
        model=_FixedNodeLogitHead(torch.tensor([5.0, 5.0, 5.0, 5.0])),
        device=torch.device("cpu"),
        trainer_cfg=trainer_cfg,
    )
    plain_pos_weight, _ = plain_trainer._collect_train_stats(plain_loader)
    plain_metrics = plain_trainer._run_epoch(plain_loader, train=False)

    assert masked_pos_weight is None
    assert masked_metrics["loss"] < 0.05
    assert plain_pos_weight is not None
    assert plain_pos_weight.item() == pytest.approx(1.0)
    assert plain_metrics["loss"] > 1.0
