import torch

from get import ETAttentionCore, ETGraphMaskModulator, ETFaithful, FullGET
from get.data import collate_get_batch


def _tiny_graph_with_edge_attr(dtype=torch.float32):
    return {
        "x": torch.randn(5, 4, dtype=dtype),
        "edges": [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)],
        "edge_attr": torch.tensor(
            [
                [1.0, 0.0, 0.5],
                [0.1, 0.2, 0.3],
                [0.0, 1.0, 0.2],
                [0.4, 0.4, 0.4],
                [0.7, 0.1, 0.0],
            ],
            dtype=dtype,
        ),
        "y": torch.tensor([1.0], dtype=dtype),
    }


def test_et_graph_mask_modulator_handles_adjacency_and_edge_features():
    d = 6
    heads = 3
    n = 4
    x_local = torch.randn(n, d)
    adj = torch.randint(0, 2, (n, n), dtype=torch.float32)
    edge_features = torch.randn(n, n, 5)

    mod = ETGraphMaskModulator(d=d, num_heads=heads, edge_feat_dim=None, kernel_size=3)
    out_adj = mod.dense_modulation(x_local, adj)
    out_feat = mod.dense_modulation(x_local, edge_features)

    assert out_adj.shape == (heads, n, n)
    assert out_feat.shape == (heads, n, n)
    assert torch.isfinite(out_adj).all()
    assert torch.isfinite(out_feat).all()


def test_et_attention_dense_modulation_changes_energy():
    g = torch.randn(6, 8)
    # One graph chunk that is fully connected for deterministic behavior.
    adj = torch.ones(6, 6, dtype=torch.bool)
    graph_chunks = [{"start": 0, "size": 6, "adj": adj}]
    c_aug, u_aug = torch.nonzero(adj, as_tuple=True)
    attn = ETAttentionCore(d=8, num_heads=2, head_dim=4)

    e_nomod = attn.energy(g, c_aug, u_aug, graph_chunks, mask_mode="official_dense", dense_modulation=None)
    dense_mod = [torch.randn(2, 6, 6)]
    e_mod = attn.energy(g, c_aug, u_aug, graph_chunks, mask_mode="official_dense", dense_modulation=dense_mod)

    assert torch.isfinite(e_nomod)
    assert torch.isfinite(e_mod)
    assert not torch.allclose(e_nomod, e_mod)


def test_get_pairwise_et_mask_runs_with_and_without_edge_features():
    graph = _tiny_graph_with_edge_attr(dtype=torch.float32)
    batch_with = collate_get_batch([graph])

    model = FullGET(
        in_dim=4,
        d=8,
        num_classes=1,
        num_steps=1,
        pairwise_et_mask=True,
        pairwise_et_kernel_size=3,
        norm_style="et",
    )
    model.eval()
    out_with, trace_with = model(batch_with, task_level="graph")
    assert out_with.shape == (1, 1)
    assert len(trace_with) == 1

    graph_no_edge = dict(graph)
    graph_no_edge.pop("edge_attr")
    batch_no = collate_get_batch([graph_no_edge])
    out_no, trace_no = model(batch_no, task_level="graph")
    assert out_no.shape == (1, 1)
    assert len(trace_no) == 1


def test_et_faithful_official_dense_accepts_edge_features():
    graph = _tiny_graph_with_edge_attr(dtype=torch.float32)
    batch = collate_get_batch([graph])

    model = ETFaithful(
        in_dim=4,
        d=8,
        num_classes=1,
        num_steps=1,
        num_heads=2,
        head_dim=4,
        pe_k=2,
        et_official_mode=True,
        mask_mode="official_dense",
    )
    model.eval()
    out, energy_trace, stats = model(batch, task_level="graph", return_solver_stats=True)
    assert out.shape == (1, 1)
    assert len(energy_trace) == 1
    assert "memory_entropy" in stats
    assert torch.isfinite(torch.tensor(energy_trace[0]))
