from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.protocol.training import run_epoch
from get.data.synthetic import collate_graph_samples


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
        num_graphs = int(batch["num_graphs"].item())
        return self.bias.unsqueeze(0).expand(num_graphs, -1)


def _make_sample(y):
    return {
        "x": torch.zeros((1, 2), dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.float32),
        "c_2": torch.empty(0, dtype=torch.long),
        "u_2": torch.empty(0, dtype=torch.long),
        "c_3": torch.empty(0, dtype=torch.long),
        "u_3": torch.empty(0, dtype=torch.long),
        "v_3": torch.empty(0, dtype=torch.long),
        "t_tau": torch.empty(0, dtype=torch.long),
    }


def test_run_epoch_supports_multilabel_targets():
    samples = [_make_sample([0.0, 1.0]), _make_sample([1.0, 0.0])]
    loader = DataLoader(_TinyGraphDataset(samples), batch_size=2, shuffle=False, collate_fn=collate_graph_samples)
    model = _BiasHead(out_dim=2)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    pos_weight = torch.ones(2)

    metrics = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        task_type="multilabel",
        optimizer=optim,
        pos_weight=pos_weight,
        use_amp=False,
    )

    assert "auc" in metrics
    assert metrics["binary_ranking_available"] == 1.0
    assert "acc" in metrics


def test_run_epoch_supports_vector_regression_targets():
    samples = [_make_sample([0.5, 1.5, 2.5]), _make_sample([3.5, 4.5, 5.5])]
    loader = DataLoader(_TinyGraphDataset(samples), batch_size=2, shuffle=False, collate_fn=collate_graph_samples)
    model = _BiasHead(out_dim=3)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)

    metrics = run_epoch(
        model,
        loader,
        torch.device("cpu"),
        task_type="regression",
        optimizer=optim,
        use_amp=False,
    )

    assert "mae" in metrics
    assert metrics["mae"] >= 0.0
