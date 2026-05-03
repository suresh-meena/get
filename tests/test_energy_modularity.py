from __future__ import annotations

import pytest
import torch

from get.data import SyntheticGraphDataset, collate_graph_samples
from get.energy import ENERGY_SPECS, build_energy
from get.models import EnergyGraphClassifier
from get.models.energy_norm import EnergyLayerNorm
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
        beta_2=1.0,
        beta_3=1.0,
        beta_m=1.0,
        update_damping=0.05,
        armijo_eval_max_backtracks=armijo_eval_max_backtracks,
        inference_mode_train=inference_mode_train,
        inference_mode_eval=inference_mode_eval,
        energy_name=energy_name,
    )


def test_energy_factory_contains_expected_names():
    names = {spec.name for spec in ENERGY_SPECS}
    assert {"get_full", "pairwise_only", "quadratic_only"}.issubset(names)


def test_energy_factory_unknown_name_fails():
    with pytest.raises(ValueError, match="Unknown energy function"):
        build_energy("does_not_exist")


@pytest.mark.parametrize("energy_name", ["get_full", "pairwise_only", "quadratic_only"])
def test_model_forward_supports_swappable_energy_functions(energy_name: str):
    batch = _tiny_batch()
    model = _make_model(energy_name).eval()
    logits = model(batch, inference_mode="fixed")
    assert logits.shape == (int(batch["num_graphs"].item()),)
    assert torch.isfinite(logits).all()


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


def test_energy_layer_norm_matches_et_formula():
    layer = EnergyLayerNorm(3, use_bias=True, eps=1e-5)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
        layer.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))

    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    out = layer(x)
    centered = x - x.mean(dim=-1, keepdim=True)
    expected = 2.0 * centered / torch.sqrt((centered ** 2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = expected + torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    assert torch.allclose(out, expected)
