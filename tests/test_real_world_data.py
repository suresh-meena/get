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
        def __init__(self, root=None, name=None, use_node_attr=True):
            del root, name, use_node_attr
            super().__init__(samples)
            self.num_classes = num_classes

    monkeypatch.setattr("get.data.real_world.TUDataset", _FakeTUDataset)
    monkeypatch.setattr("get.data.real_world.GNNBenchmarkDataset", _FakeTUDataset)


def test_real_world_dataset_auto_preserves_multiclass_labels(monkeypatch):
    _install_fake_tu(
        monkeypatch,
        [_FakeData(torch.tensor([0])), _FakeData(torch.tensor([1])), _FakeData(torch.tensor([2]))],
        num_classes=3,
    )

    ds = RealWorldGraphDataset(name="ENZYMES", root="data", in_dim=2, max_motifs_per_anchor=2, task_type="auto", cache_enabled=False)
    sample = ds[2]

    assert ds.task_type == "multiclass"
    assert torch.equal(sample["y"], torch.tensor([2.0]))


def test_real_world_dataset_auto_preserves_binary_labels(monkeypatch):
    _install_fake_tu(
        monkeypatch,
        [_FakeData(torch.tensor([1])), _FakeData(torch.tensor([2]))],
        num_classes=2,
    )

    ds = RealWorldGraphDataset(name="MUTAG", root="data", in_dim=2, max_motifs_per_anchor=2, task_type="auto", cache_enabled=False)
    sample = ds[1]

    assert ds.task_type == "binary"
    assert torch.equal(sample["y"], torch.tensor([1.0]))


def test_real_world_dataset_cache_roundtrip(tmp_path, monkeypatch):
    calls = {"count": 0}

    class _CachingTUDataset(list):
        def __init__(self, root=None, name=None, use_node_attr=True):
            del root, name, use_node_attr
            calls["count"] += 1
            super().__init__([_FakeData(torch.tensor([0])), _FakeData(torch.tensor([1]))])
            self.num_classes = 2

    monkeypatch.setattr("get.data.real_world.TUDataset", _CachingTUDataset)
    monkeypatch.setattr("get.data.real_world.GNNBenchmarkDataset", _CachingTUDataset)

    root = tmp_path / "real"
    ds1 = RealWorldGraphDataset(name="MUTAG", root=str(root), in_dim=2, max_motifs_per_anchor=2, task_type="auto", cache_enabled=True)
    ds2 = RealWorldGraphDataset(name="MUTAG", root=str(root), in_dim=2, max_motifs_per_anchor=2, task_type="auto", cache_enabled=True)

    assert calls["count"] == 1
    assert len(ds1) == len(ds2) == 2
    assert torch.equal(ds1[0]["y"], ds2[0]["y"])
