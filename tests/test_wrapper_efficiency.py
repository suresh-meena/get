import torch

from get import GraphormerAdapter, collate_get_batch


def _make_graph(num_nodes: int, feature_dim: int):
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    return {
        "x": torch.randn(num_nodes, feature_dim),
        "edges": edges,
        "y": torch.tensor([0]),
    }


def test_graphormer_bias_is_computed_once_per_graph(monkeypatch):
    batch = collate_get_batch([_make_graph(4, 3), _make_graph(5, 3)])
    model = GraphormerAdapter(
        in_dim=3,
        d=8,
        num_classes=2,
        num_layers=3,
        num_heads=2,
        use_local=False,
        use_attention=True,
        use_spatial_bias=True,
    )

    calls = {"count": 0}

    def fake_graph_bias(chunk, n, device, dtype):
        calls["count"] += 1
        return torch.zeros(model.attn_blocks[0].num_heads, n, n, device=device, dtype=dtype)

    monkeypatch.setattr(model, "_graph_bias", fake_graph_bias)

    out, _ = model(batch, task_level="graph")

    assert out.shape == (2, 2)
    assert calls["count"] == len(batch.ptr) - 1