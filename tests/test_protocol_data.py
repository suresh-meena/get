from __future__ import annotations

from types import SimpleNamespace

import torch

from get.data import protocol as pdata


class _FakeData:
    def __init__(self, y: torch.Tensor, num_nodes: int = 3):
        self.num_nodes = num_nodes
        self.x = torch.arange(num_nodes * 2, dtype=torch.float32).view(num_nodes, 2)
        self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.y = y


def test_task_specs_include_stage3_molpcba() -> None:
    assert "stage3_molpcba" in pdata.TASK_SPECS
    assert pdata.TASK_SPECS["stage3_molpcba"].task_type == "multilabel"
    assert pdata.TASK_SPECS["stage3_molpcba"].stage == "3"


def test_graph_to_sample_preserves_nan_mask_for_multilabel() -> None:
    data = _FakeData(y=torch.tensor([[1.0, float("nan"), 0.0]], dtype=torch.float32))
    sample = pdata.graph_to_sample(
        data,
        in_dim=2,
        max_motifs_per_anchor=2,
        y_mode="multilabel",
        preserve_nan_mask=True,
    )
    assert "y_mask" in sample
    assert sample["y"].shape[0] == 3
    assert sample["y_mask"].dtype == torch.bool
    assert sample["y_mask"].tolist() == [True, False, True]
    assert sample["y"][1].item() == 0.0


def test_load_stage2_brec_attaches_category_ids(tmp_path) -> None:
    payload = {
        "data_list": [
            _FakeData(y=torch.tensor([1.0])),
            _FakeData(y=torch.tensor([0.0])),
        ],
        "categories": ["Basic", "CFI"],
    }
    brec_path = tmp_path / "brec.pt"
    torch.save(payload, brec_path)

    args = SimpleNamespace(
        brec_file=str(brec_path),
        in_dim=2,
        max_motifs_per_anchor=2,
        pos_k=0,
        max_graphs=0,
    )

    samples, num_classes = pdata._load_stage2_brec(args)
    assert num_classes == 2
    assert len(samples) == 2
    assert "brec_category_id" in samples[0]
    assert "brec_category_id" in samples[1]
    names = getattr(args, "_brec_category_names", {})
    assert isinstance(names, dict)
    assert set(names.values()) == {"Basic", "CFI"}


def test_list_graph_dataset_loads_cached_sample_ref(tmp_path) -> None:
    sample = {
        "x": torch.randn(3, 2),
        "y": torch.tensor([1.0]),
        "c_2": torch.tensor([0, 1], dtype=torch.long),
        "u_2": torch.tensor([1, 0], dtype=torch.long),
        "c_3": torch.tensor([], dtype=torch.long),
        "u_3": torch.tensor([], dtype=torch.long),
        "v_3": torch.tensor([], dtype=torch.long),
        "t_tau": torch.tensor([], dtype=torch.long),
    }
    sample_path = tmp_path / "sample.pt"
    torch.save(sample, sample_path)

    ref = pdata.CachedSampleRef(path=str(sample_path), label0=1.0)
    ds = pdata.ListGraphDataset([ref])
    loaded = ds[0]
    assert torch.equal(loaded["y"], torch.tensor([1.0]))
    assert loaded["x"].shape == (3, 2)


def test_infer_edge_attr_dim_handles_cached_refs_and_normalizes_vectors(tmp_path) -> None:
    sample = {
        "x": torch.randn(3, 2),
        "y": torch.tensor([1.0]),
        "c_2": torch.tensor([0, 1], dtype=torch.long),
        "u_2": torch.tensor([1, 0], dtype=torch.long),
        "c_3": torch.tensor([], dtype=torch.long),
        "u_3": torch.tensor([], dtype=torch.long),
        "v_3": torch.tensor([], dtype=torch.long),
        "t_tau": torch.tensor([], dtype=torch.long),
        "edge_attr": torch.tensor([0.5, 1.5], dtype=torch.float32),
    }
    sample_path = tmp_path / "sample.pt"
    torch.save(sample, sample_path)

    ref = pdata.CachedSampleRef(path=str(sample_path), label0=1.0)
    dim = pdata.infer_edge_attr_dim({"train": [ref], "val": [], "test": []})
    assert dim == 1

    loaded = pdata.ListGraphDataset([ref])[0]
    assert "edge_attr" in loaded
    assert loaded["edge_attr"].shape == (2, 1)


def test_scalar_labels_from_cached_refs() -> None:
    items = [
        pdata.CachedSampleRef(path="/tmp/a.pt", label0=0.0),
        pdata.CachedSampleRef(path="/tmp/b.pt", label0=1.0),
    ]
    labels = pdata._scalar_labels(items)
    assert labels == [0, 1]


class _TinyDataset:
    def __init__(self):
        self._items = [
            _FakeData(y=torch.tensor([0.0])),
            _FakeData(y=torch.tensor([1.0])),
        ]

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def test_cached_transform_split_reuses_disk_cache(tmp_path) -> None:
    args = SimpleNamespace(
        dataset_root=str(tmp_path),
        in_dim=2,
        max_motifs_per_anchor=2,
        pos_k=0,
    )
    ds = _TinyDataset()
    refs_first = pdata._cached_transform_split(
        args=args,
        cache_tag="unit_cache",
        split_name="train",
        dataset=ds,
        indices=[0, 1],
        y_mode="binary",
    )
    refs_second = pdata._cached_transform_split(
        args=args,
        cache_tag="unit_cache",
        split_name="train",
        dataset=ds,
        indices=[0, 1],
        y_mode="binary",
    )
    assert len(refs_first) == 2
    assert len(refs_second) == 2
    assert refs_first[0].path == refs_second[0].path
    assert refs_second[1].label0 == 1.0
