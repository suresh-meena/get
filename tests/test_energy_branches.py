import torch

from get import FullGET
from get.data import collate_get_batch
from get.energy import (
    compute_energy_GET,
    compute_memory_energy,
    compute_motif_energy,
    compute_pairwise_energy,
    compute_quadratic_energy,
    segment_logsumexp,
)


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
    params_local["lambda_3"] = torch.tensor(-30.0, dtype=torch.float64)
    params_local["lambda_m"] = torch.tensor(-30.0, dtype=torch.float64)
    static_projections = model._build_static_projections(batch)
    projections = model.get_layer._build_projections(G, static_projections=static_projections)

    e_att3 = compute_motif_energy(
        G, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, params_local, projections, X.size(0)
    )
    e_mem = compute_memory_energy(G, params_local, projections)

    assert torch.allclose(e_att3, torch.zeros((), dtype=torch.float64), atol=1e-12)
    assert torch.allclose(e_mem, torch.zeros((), dtype=torch.float64), atol=1e-12)


if __name__ == "__main__":
    test_segment_logsumexp_empty_segment_behavior()
    test_energy_decomposition_matches_total()
    test_motif_and_memory_disable_to_zero()
    print("Energy branch tests passed.")
