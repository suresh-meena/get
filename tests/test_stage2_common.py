from types import SimpleNamespace

import torch

from experiments.common import build_anomaly_protocol_split, build_dataloader_kwargs, build_ego_graph_dataset


def test_build_ego_graph_dataset_shapes_and_labels():
    # Chain graph: 0-1-2-3 with binary node labels.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    x = torch.randn(4, 5)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    data = SimpleNamespace(num_nodes=4, edge_index=edge_index, x=x, y=y)

    samples = build_ego_graph_dataset(data, num_hops=1, limit=3)
    assert len(samples) == 3
    assert all("x" in s and "edges" in s and "y" in s for s in samples)
    assert all(s["x"].dim() == 2 for s in samples)
    # Labels are converted to float binary graph labels.
    assert samples[0]["y"].shape == (1,)
    assert float(samples[0]["y"].item()) in {0.0, 1.0}


def test_build_anomaly_protocol_split_non_empty_partitions():
    dataset = []
    for i in range(20):
        label = 1.0 if i % 5 == 0 else 0.0
        dataset.append(
            {
                "x": torch.ones(4, 2),
                "edges": [(0, 1), (1, 2)],
                "y": torch.tensor([label], dtype=torch.float32),
                "graph_id": i,
            }
        )

    split = build_anomaly_protocol_split(dataset, seed=123, labeled_rate=0.1, val_ratio=1, test_ratio=2)
    assert len(split["train"]) > 0
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0
    union_size = len(split["train"]) + len(split["val"]) + len(split["test"])
    assert union_size == len(dataset)


def test_build_dataloader_kwargs_device_aware_defaults():
    cpu_kwargs = build_dataloader_kwargs("cpu")
    cuda_kwargs = build_dataloader_kwargs("cuda:0")

    assert cpu_kwargs["num_workers"] == 0
    assert cpu_kwargs["pin_memory"] is False
    assert "persistent_workers" not in cpu_kwargs
    assert "prefetch_factor" not in cpu_kwargs

    import os
    expected_workers = min(os.cpu_count() or 4, 8)
    assert cuda_kwargs["num_workers"] == expected_workers
    assert cuda_kwargs["pin_memory"] is True
    assert cuda_kwargs["persistent_workers"] is True
    assert cuda_kwargs["prefetch_factor"] == 4
