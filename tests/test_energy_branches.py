import torch

from get import FullGET, MotifOnlyGET
from get.data import collate_get_batch
from get.energy import (
    compute_energy_GET,
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


def test_fused_motif_dot_cuda_launch_uses_distinct_inputs_and_preserves_dtype(monkeypatch):
    class FakeCudaTensor:
        def __init__(self, shape, strides, dtype):
            self.shape = shape
            self._strides = strides
            self.dtype = dtype
            self.device = torch.device("cuda")
            self.is_cuda = True

        def dim(self):
            return len(self.shape)

        def stride(self, dim):
            return self._strides[dim]

    class DummyKernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return launch

    dummy_kernel = DummyKernel()
    captured_empty_kwargs = {}
    real_empty = torch.empty

    def fake_empty(*args, **kwargs):
        captured_empty_kwargs.clear()
        captured_empty_kwargs.update(kwargs)
        kwargs = dict(kwargs)
        kwargs["device"] = torch.device("cpu")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(fused_ops_module, "_fused_motif_dot_kernel", dummy_kernel)
    monkeypatch.setattr(fused_ops_module.torch, "empty", fake_empty)

    q = FakeCudaTensor((2, 3, 4, 5), (60, 20, 5, 1), torch.float16)
    k_u = FakeCudaTensor((2, 3, 4, 5), (60, 20, 5, 1), torch.float16)
    k_v = FakeCudaTensor((2, 3, 4, 5), (60, 20, 5, 1), torch.float16)
    t_tau = FakeCudaTensor((3, 4, 5), (20, 5, 1), torch.float16)

    out = fused_ops_module.fused_motif_dot(q, k_u, k_v, t_tau)

    assert out.shape == (2, 3)
    assert out.dtype == torch.float16
    assert captured_empty_kwargs["dtype"] == torch.float16
    assert len(dummy_kernel.calls) == 1

    grid, args, kwargs = dummy_kernel.calls[0]
    assert grid == (1, 2)
    assert args[0] is q
    assert args[1] is k_u
    assert args[2] is k_v
    assert args[3] is t_tau
    assert kwargs["BLOCK_SIZE_R"] == 4


def test_energy_decomposition_matches_total():
    batch = _make_tiny_batch(dtype=torch.float64)
    model = FullGET(in_dim=6, d=8, num_classes=1, num_steps=1).to(torch.float64)
    model.eval()

    X = model.node_encoder(batch.x).detach().requires_grad_(True)
    G = model.get_layer.layernorm(X)
    params = model.get_layer.get_params_dict()
    static_projections = model._build_static_projections(batch)
    projections = model.get_layer._build_projections(G, static_projections=static_projections)

    e_total = compute_energy_GET(
        X,
        G,
        batch.c_2,
        batch.u_2,
        batch.c_3,
        batch.u_3,
        batch.v_3,
        batch.t_tau,
        params,
        projections,
    )

    e_quad = compute_quadratic_energy(X)
    e_att2 = compute_pairwise_energy(G, batch.c_2, batch.u_2, params, projections, X.size(0))
    e_att3 = compute_motif_energy(
        G, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, params, projections, X.size(0)
    )
    e_mem = compute_memory_energy(G, params, projections)
    e_recomposed = e_quad - e_att2 - e_att3 - e_mem

    assert torch.allclose(e_total, e_recomposed, atol=1e-9, rtol=1e-6)


def test_motif_and_memory_disable_to_zero():
    batch = _make_tiny_batch(dtype=torch.float64)
    model = FullGET(in_dim=6, d=8, num_classes=1, num_steps=1).to(torch.float64)
    model.eval()

    X = model.node_encoder(batch.x).detach().requires_grad_(True)
    G = model.get_layer.layernorm(X)
    params = model.get_layer.get_params_dict()
    params_local = dict(params)
    params_local["use_pairwise"] = False
    params_local["use_motif"] = False
    params_local["use_memory"] = False
    static_projections = model._build_static_projections(batch)
    projections = model.get_layer._build_projections(G, static_projections=static_projections)

    e_att2 = compute_pairwise_energy(G, batch.c_2, batch.u_2, params_local, projections, X.size(0))
    e_att3 = compute_motif_energy(
        G, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, params_local, projections, X.size(0)
    )
    e_mem = compute_memory_energy(G, params_local, projections)
    e_entropy = compute_memory_entropy(G, params_local, projections)

    assert torch.allclose(e_att2, torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(e_att3, torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(e_mem, torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(e_entropy, torch.zeros((), dtype=torch.float64), atol=1e-12)


def test_motif_only_ablation_disables_pairwise_and_memory():
    model = MotifOnlyGET(in_dim=6, d=8, num_classes=1, num_steps=1)
    params = model.get_layer.get_params_dict()

    assert params["use_pairwise"] is False
    assert params["use_motif"] is True
    assert params["use_memory"] is False


if __name__ == "__main__":
    test_segment_logsumexp_empty_segment_behavior()
    test_energy_decomposition_matches_total()
    test_motif_and_memory_disable_to_zero()
    test_motif_only_ablation_disables_pairwise_and_memory()
    test_fused_motif_dot_matches_reference_cpu()
    print("Energy branch tests passed.")
