from __future__ import annotations

import pytest
import torch

from get.data import SyntheticGraphDataset, collate_graph_samples
from get.energy import ENERGY_SPECS, build_energy
from get.models import EnergyGraphClassifier


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


def _make_model(energy_name: str) -> EnergyGraphClassifier:
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
        inference_mode_train="fixed",
        inference_mode_eval="fixed",
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
