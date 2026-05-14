from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from get.data import SyntheticGraphDataset, collate_graph_samples
from get.data.synthetic import sample_from_edge_index
from get.models import EnergyGraphClassifier
from get.solvers import FixedStepSolver


def _tiny_batch():
    ds = SyntheticGraphDataset(
        num_graphs=4,
        min_nodes=5,
        max_nodes=6,
        edge_prob=0.2,
        in_dim=8,
        max_motifs_per_anchor=4,
        seed=3,
    )
    samples = [ds[i] for i in range(4)]
    return collate_graph_samples(samples)


def _make_model(
    energy_name: str,
    inference_mode_train: str = "fixed",
    inference_mode_eval: str = "fixed",
    armijo_eval_max_backtracks: int = 5,
) -> EnergyGraphClassifier:
    return EnergyGraphClassifier(
        in_dim=8,
        hidden_dim=32,
        num_classes=1,
        num_steps=2,
        num_heads=4,
        head_dim=8,
        R=2,
        K=8,
        num_motif_types=24,
        lambda_2=1.0,
        lambda_3=0.5,
        lambda_m=1.0,
        update_damping=0.05,
        armijo_eval_max_backtracks=armijo_eval_max_backtracks,
        inference_mode_train=inference_mode_train,
        inference_mode_eval=inference_mode_eval,
        energy_name=energy_name,
    )


@pytest.mark.parametrize("energy_name", ["get_full", "pairwise_only", "quadratic_only"])
def test_model_forward_supports_swappable_energy_functions(energy_name: str):
    batch = _tiny_batch()
    model = _make_model(energy_name).eval()
    logits = model(batch, inference_mode="fixed")
    assert logits.shape == (int(batch["num_graphs"].item()),)
    assert torch.isfinite(logits).all()


def test_full_get_gradient_combines_multiple_energy_terms():
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 2, 3], [1, 2, 0, 2, 3, 0]],
        dtype=torch.long,
    )
    sample = sample_from_edge_index(
        edge_index=edge_index,
        num_nodes=4,
        x=torch.randn(4, 8),
        y=torch.tensor([1.0]),
        max_motifs_per_anchor=8,
    )
    assert sample.c_3.numel() > 0
    batch = collate_graph_samples([sample])
    model = _make_model("get_full").train()
    x = model.encoder(batch["x"]).detach().requires_grad_(True)
    block = model.blocks[0]
    num_graphs = int(batch["num_graphs"].item())

    def grad_vec(lambda_2: float, lambda_3: float, lambda_m: float) -> torch.Tensor:
        cfg = model._build_params()
        cfg["lambda_2"] = lambda_2
        cfg["lambda_3"] = lambda_3
        cfg["lambda_m"] = lambda_m
        energy = block.energy_from_g(x, batch, cfg, scaler=None, num_graphs=num_graphs)
        grad, = torch.autograd.grad(energy, x, retain_graph=True)
        return grad.detach()

    grad_quad = grad_vec(0.0, 0.0, 0.0)
    grad_pair = grad_vec(1.0, 0.0, 0.0)
    grad_pair_motif = grad_vec(1.0, 0.5, 0.0)
    grad_full = grad_vec(1.0, 0.5, 1.0)

    assert not torch.allclose(grad_quad, grad_pair)
    assert not torch.allclose(grad_pair, grad_pair_motif)
    assert not torch.allclose(grad_pair_motif, grad_full)


def test_fixed_step_solver_applies_update_damping():
    solver = FixedStepSolver(num_steps=1, step_size=1.0, update_damping=0.25)
    x0 = torch.tensor([[2.0]], dtype=torch.float32)

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x ** 2).sum()

    x_final, _, stats = solver.run(x0, energy_fn)
    assert torch.allclose(x_final, torch.tensor([[0.5]], dtype=torch.float32))
    assert stats["update_damping"] == 0.25
    assert stats["step_sizes"] == [0.75]


def test_fixed_step_solver_accepts_explicit_gradient_closure():
    solver = FixedStepSolver(num_steps=1, step_size=1.0, update_damping=0.0)
    x0 = torch.tensor([[2.0]], dtype=torch.float32)

    def energy_fn(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x ** 2).sum()

    def energy_and_grad_fn(x: torch.Tensor, create_graph: bool = False):
        del create_graph
        return energy_fn(x), torch.full_like(x, 3.0)

    x_final, _, _ = solver.run(x0, energy_fn, energy_and_grad_fn)
    assert torch.allclose(x_final, torch.tensor([[-1.0]], dtype=torch.float32))


def test_collate_graph_samples_preserves_offsets_and_pos():
    samples = [
        {
            "x": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "y": torch.tensor([1.0]),
            "c_2": torch.tensor([0], dtype=torch.long),
            "u_2": torch.tensor([1], dtype=torch.long),
            "c_3": torch.tensor([0], dtype=torch.long),
            "u_3": torch.tensor([0], dtype=torch.long),
            "v_3": torch.tensor([1], dtype=torch.long),
            "t_tau": torch.tensor([1], dtype=torch.long),
            "pos": torch.tensor([[0.1], [0.2]]),
        },
        {
            "x": torch.tensor([[5.0, 6.0]]),
            "y": torch.tensor([0.0]),
            "c_2": torch.tensor([], dtype=torch.long),
            "u_2": torch.tensor([], dtype=torch.long),
            "c_3": torch.tensor([], dtype=torch.long),
            "u_3": torch.tensor([], dtype=torch.long),
            "v_3": torch.tensor([], dtype=torch.long),
            "t_tau": torch.tensor([], dtype=torch.long),
            "pos": torch.tensor([[0.3]]),
        },
    ]

    out = collate_graph_samples(samples)

    assert isinstance(out, Batch)
    assert out["x"].shape == (3, 2)
    assert torch.equal(out["x"], torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(out["batch"], torch.tensor([0, 0, 1], dtype=torch.long))
    assert torch.equal(out["num_graphs"], torch.tensor(2, dtype=torch.long))
    assert torch.equal(out["c_2"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(out["u_2"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(out["c_3"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(out["u_3"], torch.tensor([0], dtype=torch.long))
    assert torch.equal(out["v_3"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(out["t_tau"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(out["pos"], torch.tensor([[0.1], [0.2], [0.3]]))


def test_armijo_eval_backtracks_are_capped():
    batch = _tiny_batch()
    model = _make_model(
        "quadratic_only",
        inference_mode_train="fixed",
        inference_mode_eval="armijo",
        armijo_eval_max_backtracks=1,
    ).eval()

    logits, energy_trace, solver_stats = model(batch, inference_mode="armijo", return_solver_stats=True)
    assert logits.shape == (int(batch["num_graphs"].item()),)
    assert energy_trace
    assert solver_stats["mode"] == "armijo"
    assert solver_stats["max_backtracks"] == 1


