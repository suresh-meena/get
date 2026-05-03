from __future__ import annotations

import sys
import types

import torch

from get.data import RealWorldGraphDataset


class _FakeData:
    def __init__(self, y: torch.Tensor):
        self.num_nodes = 3
        self.x = torch.arange(6, dtype=torch.float32).view(3, 2)
        self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.y = y


def _install_fake_tu(monkeypatch, samples: list[_FakeData], num_classes: int):
    class _FakeTUDataset(list):
        def __init__(self, root=None, name=None):
            del root, name
            super().__init__(samples)
            self.num_classes = num_classes

    datasets = types.ModuleType("torch_geometric.datasets")
    datasets.TUDataset = _FakeTUDataset
    datasets.GNNBenchmarkDataset = _FakeTUDataset

    tg_root = types.ModuleType("torch_geometric")
    tg_root.__path__ = []  # type: ignore[attr-defined]
    tg_root.datasets = datasets

    monkeypatch.setitem(sys.modules, "torch_geometric", tg_root)
    monkeypatch.setitem(sys.modules, "torch_geometric.datasets", datasets)


def test_real_world_dataset_auto_preserves_multiclass_labels(monkeypatch):
    _install_fake_tu(
        monkeypatch,
        [_FakeData(torch.tensor([0])), _FakeData(torch.tensor([1])), _FakeData(torch.tensor([2]))],
        num_classes=3,
    )

    ds = RealWorldGraphDataset(name="ENZYMES", root="data", in_dim=2, max_motifs_per_anchor=2, task_type="auto")
    sample = ds[2]

    assert ds.task_type == "multiclass"
    assert torch.equal(sample["y"], torch.tensor([2.0]))


def test_real_world_dataset_auto_preserves_binary_labels(monkeypatch):
    _install_fake_tu(
        monkeypatch,
        [_FakeData(torch.tensor([1])), _FakeData(torch.tensor([2]))],
        num_classes=2,
    )

    ds = RealWorldGraphDataset(name="MUTAG", root="data", in_dim=2, max_motifs_per_anchor=2, task_type="auto")
    sample = ds[1]

    assert ds.task_type == "binary"
    assert torch.equal(sample["y"], torch.tensor([1.0]))
