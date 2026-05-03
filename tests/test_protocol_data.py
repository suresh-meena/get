from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import torch

from experiments.protocol import data as pdata


class _FakeData:
    def __init__(self, y: torch.Tensor, num_nodes: int = 3):
        self.num_nodes = num_nodes
        self.x = torch.arange(num_nodes * 2, dtype=torch.float32).view(num_nodes, 2)
        self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.y = y


def _make_split_dataset(split_map: dict[str, list[torch.Tensor]]):
    class _SplitDataset(list):
        def __init__(self, root=None, name=None, split=None, subset=None):
            del root, name, subset
            super().__init__([_FakeData(y.clone()) for y in split_map[split]])

    return _SplitDataset


class _FakeMolHivDataset:
    def __init__(self, name=None, root=None):
        del name, root
        self.samples = [
            _FakeData(torch.tensor([0.0])),
            _FakeData(torch.tensor([1.0])),
            _FakeData(torch.tensor([0.0])),
            _FakeData(torch.tensor([1.0])),
        ]
        self._idx_split = {
            "train": torch.tensor([0, 1], dtype=torch.long),
            "valid": torch.tensor([2], dtype=torch.long),
            "test": torch.tensor([3], dtype=torch.long),
        }

    def get_idx_split(self):
        return self._idx_split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _install_fake_graph_modules(monkeypatch, *, split_map: dict[str, list[torch.Tensor]]):
    tg_datasets = types.ModuleType("torch_geometric.datasets")
    tg_datasets.GNNBenchmarkDataset = _make_split_dataset(split_map)
    tg_datasets.ZINC = _make_split_dataset(split_map)
    tg_datasets.LRGBDataset = _make_split_dataset(split_map)

    tg_root = types.ModuleType("torch_geometric")
    tg_root.__path__ = []  # type: ignore[attr-defined]
    tg_root.datasets = tg_datasets

    ogb_graphproppred = types.ModuleType("ogb.graphproppred")
    ogb_graphproppred.PygGraphPropPredDataset = _FakeMolHivDataset

    ogb_root = types.ModuleType("ogb")
    ogb_root.__path__ = []  # type: ignore[attr-defined]
    ogb_root.graphproppred = ogb_graphproppred

    monkeypatch.setitem(sys.modules, "torch_geometric", tg_root)
    monkeypatch.setitem(sys.modules, "torch_geometric.datasets", tg_datasets)
    monkeypatch.setitem(sys.modules, "ogb", ogb_root)
    monkeypatch.setitem(sys.modules, "ogb.graphproppred", ogb_graphproppred)


def test_stage2_csl_preserves_official_split_dict(monkeypatch):
    split_map = {
        "train": [torch.tensor([0]), torch.tensor([1])],
        "val": [torch.tensor([2])],
        "test": [torch.tensor([3])],
    }
    _install_fake_graph_modules(monkeypatch, split_map=split_map)
    args = SimpleNamespace(dataset_root="data", max_graphs=1, in_dim=4, max_motifs_per_anchor=2)

    splits, nclass = pdata._load_stage2_csl(args)

    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(len(v) == 1 for v in splits.values())
    assert nclass == 4


def test_stage3_zinc_preserves_official_split_dict(monkeypatch):
    split_map = {
        "train": [torch.tensor([0.0]), torch.tensor([1.0])],
        "val": [torch.tensor([2.0])],
        "test": [torch.tensor([3.0])],
    }
    _install_fake_graph_modules(monkeypatch, split_map=split_map)
    args = SimpleNamespace(dataset_root="data", max_graphs=1, in_dim=4, max_motifs_per_anchor=2)

    splits, num_classes = pdata._load_stage3_zinc(args)

    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(len(v) == 1 for v in splits.values())
    assert num_classes == 1


def test_stage3_molhiv_preserves_official_split_dict(monkeypatch):
    split_map = {
        "train": [torch.tensor([0.0]), torch.tensor([1.0])],
        "val": [torch.tensor([2.0])],
        "test": [torch.tensor([3.0])],
    }
    _install_fake_graph_modules(monkeypatch, split_map=split_map)
    args = SimpleNamespace(dataset_root="data", max_graphs=1, in_dim=4, max_motifs_per_anchor=2)

    splits, num_classes = pdata._load_stage3_molhiv(args)

    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(len(v) == 1 for v in splits.values())
    assert num_classes == 2


def test_stage3_peptides_func_does_not_mutate_input_targets(monkeypatch):
    split_map = {
        "train": [torch.tensor([0.0, 1.0]), torch.tensor([1.0, 0.0])],
        "val": [torch.tensor([0.0, 1.0])],
        "test": [torch.tensor([1.0, 0.0])],
    }
    created: dict[str, list[_FakeData]] = {}

    class _TrackingDataset(list):
        def __init__(self, root=None, name=None, split=None, subset=None):
            del root, name, subset
            items = [_FakeData(y.clone()) for y in split_map[split]]
            created[split] = items
            super().__init__(items)

    tg_datasets = types.ModuleType("torch_geometric.datasets")
    tg_datasets.GNNBenchmarkDataset = _TrackingDataset
    tg_datasets.ZINC = _TrackingDataset
    tg_datasets.LRGBDataset = _TrackingDataset

    tg_root = types.ModuleType("torch_geometric")
    tg_root.__path__ = []  # type: ignore[attr-defined]
    tg_root.datasets = tg_datasets

    monkeypatch.setitem(sys.modules, "torch_geometric", tg_root)
    monkeypatch.setitem(sys.modules, "torch_geometric.datasets", tg_datasets)

    args = SimpleNamespace(dataset_root="data", max_graphs=1, in_dim=4, max_motifs_per_anchor=2)

    splits, num_classes = pdata._load_stage3_peptides_func(args)

    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(len(v) == 1 for v in splits.values())
    assert num_classes == 2
    assert torch.equal(created["train"][0].y, torch.tensor([0.0, 1.0]))
    assert torch.equal(splits["train"][0]["y"], torch.tensor([0.0, 1.0]))


def test_stage3_peptides_struct_preserves_full_targets(monkeypatch):
    split_map = {
        "train": [torch.tensor([0.5, 1.5, 2.5]), torch.tensor([3.5, 4.5, 5.5])],
        "val": [torch.tensor([6.5, 7.5, 8.5])],
        "test": [torch.tensor([9.5, 10.5, 11.5])],
    }

    tg_datasets = types.ModuleType("torch_geometric.datasets")
    tg_datasets.GNNBenchmarkDataset = _make_split_dataset(split_map)
    tg_datasets.ZINC = _make_split_dataset(split_map)
    tg_datasets.LRGBDataset = _make_split_dataset(split_map)

    tg_root = types.ModuleType("torch_geometric")
    tg_root.__path__ = []  # type: ignore[attr-defined]
    tg_root.datasets = tg_datasets

    monkeypatch.setitem(sys.modules, "torch_geometric", tg_root)
    monkeypatch.setitem(sys.modules, "torch_geometric.datasets", tg_datasets)

    args = SimpleNamespace(dataset_root="data", max_graphs=1, in_dim=4, max_motifs_per_anchor=2)

    splits, num_targets = pdata._load_stage3_peptides(args)

    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(len(v) == 1 for v in splits.values())
    assert num_targets == 3
    assert torch.equal(splits["train"][0]["y"], torch.tensor([0.5, 1.5, 2.5]))


def test_stage1_max3sat_is_balanced():
    args = SimpleNamespace(
        seed=123,
        max_graphs=8,
        min_nodes=10,
        max_nodes=20,
        edge_prob=0.2,
        in_dim=4,
        max_motifs_per_anchor=2,
    )

    samples = pdata._make_stage1_max3sat(args)
    labels = [int(sample["y"].item()) for sample in samples]

    assert len(samples) == 8
    assert labels.count(1) == 4
    assert labels.count(0) == 4


def test_split_items_stratifies_binary_labels():
    items = [{"y": torch.tensor([float(i % 2)])} for i in range(30)]

    train, val, test = pdata.split_items(items, seed=7, task_type="binary")
    stats = pdata.summarize_splits({"train": train, "val": val, "test": test}, task_type="binary")

    assert stats["train"]["single_class"] is False
    assert stats["val"]["single_class"] is False
    assert stats["test"]["single_class"] is False
