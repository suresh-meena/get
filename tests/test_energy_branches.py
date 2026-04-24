import torch

from get import FullGET
from get.model import GETModel
from get.data import collate_get_batch
from get.energy import (
    compute_memory_entropy,
    compute_memory_energy,
    compute_motif_energy,
    compute_pairwise_energy,
    compute_quadratic_energy,
    segment_logsumexp,
)
import get.fused_ops as fused_ops_module
from get.fused_ops import segment_reduce_1d


def _make_tiny_batch(dtype=torch.float64):
    graph = {
        "x": torch.randn(4, 6, dtype=dtype),
        "edges": [(0, 1), (1, 2), (2, 3), (0, 2)],
    }
    return collate_get_batch([graph])


def test_segment_logsumexp_empty_segment_behavior():
    x = torch.tensor([1.0, -0.5, 0.2], dtype=torch.float64)
    segment_ids = torch.tensor([0, 0, 2], dtype=torch.long)
    out = segment_logsumexp(x, segment_ids, num_segments=4)

    assert out.shape == (4,)
    assert torch.isclose(out[1], torch.tensor(0.0, dtype=out.dtype))
    assert torch.isclose(out[3], torch.tensor(0.0, dtype=out.dtype))


def test_segment_reduce_backend_matches_reference():
    x = torch.tensor([1.2, -0.5, 0.7, 2.1, -1.0], dtype=torch.float64)
    segment_ids = torch.tensor([0, 2, 2, 4, 0], dtype=torch.long)
    num_segments = 6

    sum_out, counts = segment_reduce_1d(x, segment_ids, num_segments, reduce="sum")
    max_out, _ = segment_reduce_1d(x, segment_ids, num_segments, reduce="max")

    ref_sum = torch.zeros(num_segments, dtype=x.dtype)
    ref_sum.scatter_add_(0, segment_ids, x)
    ref_max = torch.full((num_segments,), float("-inf"), dtype=x.dtype)
    ref_max.scatter_reduce_(0, segment_ids, x, reduce="amax", include_self=False)
    ref_counts = torch.bincount(segment_ids, minlength=num_segments)

    assert torch.allclose(sum_out, ref_sum)
    assert torch.allclose(max_out, ref_max)
    assert torch.equal(counts, ref_counts)


def test_fused_motif_dot_matches_reference_cpu():
    q_batched = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    k_u_batched = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    k_v_batched = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    t_tau = torch.randn(3, 4, 5, dtype=torch.float64)

    out_batched = fused_ops_module.fused_motif_dot(q_batched, k_u_batched, k_v_batched, t_tau)
    ref_batched = torch.einsum("bmrd,bmrd->bm", q_batched, k_u_batched * k_v_batched + t_tau)

    q_unbatched = q_batched[0]
    k_u_unbatched = k_u_batched[0]
    k_v_unbatched = k_v_batched[0]
    t_unbatched = t_tau[0]

    out_unbatched = fused_ops_module.fused_motif_dot(q_unbatched, k_u_unbatched, k_v_unbatched, t_unbatched)
    ref_unbatched = torch.einsum("mrd,mrd->m", q_unbatched, k_u_unbatched * k_v_unbatched + t_unbatched)

    assert torch.allclose(out_batched, ref_batched)
    assert torch.allclose(out_unbatched, ref_unbatched)


def test_fused_motif_dot_gradcheck_cpu():
    q = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    k_u = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    k_v = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    t_tau = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(
        fused_ops_module.fused_motif_dot,
        (q, k_u, k_v, t_tau),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-4,
    )


def test_energy_decomposition_matches_total():
    batch = _make_tiny_batch(dtype=torch.float64)
    # Use num_heads=1 for small d
    model = FullGET(in_dim=6, d=8, num_classes=1, num_steps=1, num_heads=1).to(torch.float64)
    model.eval()

    X = model.node_encoder(batch.x).detach().requires_grad_(True)
    G = model.get_layers[0].layernorm(X)
    static_projections = model._build_static_projections(batch)
    projections = model.get_layers[0]._build_projections(G, static_projections=static_projections[0])

    e_total = model.get_layers[0].compute_energy(X, batch, static_projections=static_projections[0])

    num_graphs = 1
    e_quad = compute_quadratic_energy(X, batch.batch, num_graphs)
    
    params = model.get_layers[0].get_params_dict()
    params['num_heads'] = 1
    e_att2 = compute_pairwise_energy(G, batch.c_2, batch.u_2, batch.batch, num_graphs, params, projections, X.size(0))
    e_att3 = compute_motif_energy(G, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, batch.batch, num_graphs, params, projections, X.size(0))
    e_mem = compute_memory_energy(G, batch.batch, num_graphs, params, projections)
    
    e_recomposed = e_quad - e_att2 - e_att3 - e_mem

    assert torch.allclose(e_total, e_recomposed, atol=1e-9, rtol=1e-6)


def test_motif_and_memory_disable_to_zero():
    batch = _make_tiny_batch(dtype=torch.float64)
    model = FullGET(in_dim=6, d=8, num_classes=1, num_steps=1, num_heads=1).to(torch.float64)
    model.eval()

    X = model.node_encoder(batch.x).detach().requires_grad_(True)
    G = model.get_layers[0].layernorm(X)
    params = model.get_layers[0].get_params_dict()
    params_local = dict(params)
    params_local["use_pairwise"] = False
    params_local["use_motif"] = False
    params_local["use_memory"] = False
    params_local['num_heads'] = 1
    
    static_projections = model._build_static_projections(batch)
    projections = model.get_layers[0]._build_projections(G, static_projections=static_projections[0])

    num_graphs = 1
    e_att2 = compute_pairwise_energy(G, batch.c_2, batch.u_2, batch.batch, num_graphs, params_local, projections, X.size(0))
    e_att3 = compute_motif_energy(G, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, batch.batch, num_graphs, params_local, projections, X.size(0))
    e_mem = compute_memory_energy(G, batch.batch, num_graphs, params_local, projections)
    e_entropy = compute_memory_entropy(G, params_local, projections)

    assert torch.allclose(e_att2, torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(e_att3, torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(e_mem, torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(e_entropy, torch.zeros((), dtype=torch.float64), atol=1e-12)


def test_motif_only_ablation_disables_pairwise_and_memory():
    model = GETModel(
        in_dim=6,
        d=8,
        num_classes=1,
        num_steps=1,
        num_heads=1,
        use_pairwise=False,
        use_motif=True,
        use_memory=False,
        lambda_2=0.0,
        lambda_m=0.0,
    )
    params = model.get_layers[0].get_params_dict()

    assert params["use_pairwise"] is False
    assert params["use_motif"] is True
    assert params["use_memory"] is False


def test_scatter_add_nd_fallback_handles_mixed_dtypes(monkeypatch):
    import get.energy as energy_mod

    monkeypatch.setattr(energy_mod, "pyg_scatter", None)

    grad_buffer = torch.zeros(3, 2, dtype=torch.float16)
    indices = torch.tensor([0, 2], dtype=torch.long)
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

    out = energy_mod._scatter_add_nd(grad_buffer, indices, src, dim=0)

    expected = torch.tensor([[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], dtype=torch.float16)
    assert torch.allclose(out, expected)


if __name__ == "__main__":
    test_segment_logsumexp_empty_segment_behavior()
    test_energy_decomposition_matches_total()
    test_motif_and_memory_disable_to_zero()
    test_motif_only_ablation_disables_pairwise_and_memory()
    test_fused_motif_dot_matches_reference_cpu()
    print("Energy branch tests passed.")
