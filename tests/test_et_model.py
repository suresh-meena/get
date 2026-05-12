from __future__ import annotations

import pytest
import torch
import argparse
from pathlib import Path

from get.data import SyntheticGraphDataset, collate_graph_samples
from get.models import ETGraphClassifier
from get.models.energy_norm import EnergyLayerNorm
from get.energy.ops import segment_logsumexp
from get.models.et_utils import build_et_batch_context
from get.models import build_model


def _tiny_batch():
    ds = SyntheticGraphDataset(
        num_graphs=4,
        min_nodes=5,
        max_nodes=6,
        edge_prob=0.5,
        in_dim=8,
        max_motifs_per_anchor=4,
        seed=42,
        pos_k=4,
    )
    samples = [ds[i] for i in range(4)]
    return collate_graph_samples(samples)


def test_et_forward_pass():
    batch = _tiny_batch()
    model = ETGraphClassifier(
        in_dim=8,
        hidden_dim=32,
        num_classes=1,
        num_heads=4,
        head_dim=8,
        num_steps=2,
        alpha=0.1,
        pos_k=4,
    )
    model.eval()
    
    # Simple forward
    logits = model(batch)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()
    assert model.readout_mode == "cls"


def test_et_bias_parameters():
    # Verify use_bias_attn creates params
    model = ETGraphClassifier(
        in_dim=8, hidden_dim=32, num_classes=1, num_heads=4, head_dim=8,
        use_bias_attn=True
    )
    # Check first block
    attn = model.blocks[0].attn
    assert attn.Bk is not None
    assert attn.Bq is not None
    assert attn.Bk.shape == (4, 8)

    model_no_bias = ETGraphClassifier(
        in_dim=8, hidden_dim=32, num_classes=1, num_heads=4, head_dim=8,
        use_bias_attn=False
    )
    assert model_no_bias.blocks[0].attn.Bk is None


def test_et_batch_context_adds_cls_token_and_positional_embeddings():
    batch = _tiny_batch()
    ctx = build_et_batch_context(batch, use_cls_token=True, pos_k=4, embed_type="eigen", flip_sign=False)
    
    # num_nodes + num_graphs
    expected_nodes = batch["x"].shape[0] + 4
    assert ctx.batch.shape[0] == expected_nodes
    assert ctx.pos.shape[0] == expected_nodes
    assert ctx.pos.shape[1] == 4
    assert ctx.num_graphs == 4


def test_energy_layer_norm_matches_reference_formula():
    layer = EnergyLayerNorm(4, use_bias=True, eps=1e-5)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
        layer.bias.copy_(torch.tensor([0.1, -0.2, 0.3, -0.4]))

    x = torch.tensor(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=torch.float32,
    )

    centered = x - x.mean(dim=-1, keepdim=True)
    expected = 2.0 * centered / torch.sqrt(centered.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
    expected = expected + torch.tensor([0.1, -0.2, 0.3, -0.4])
    actual = layer(x)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_segment_logsumexp_matches_naive_reference():
    src = torch.tensor(
        [
            [0.0, 1.0],
            [2.0, -1.0],
            [3.0, 4.0],
        ],
        dtype=torch.float32,
    )
    segment_ids = torch.tensor([0, 0, 2], dtype=torch.long)
    out = segment_logsumexp(src, segment_ids, num_segments=4, dim=0)

    expected = torch.full((4, 2), float("-inf"), dtype=torch.float32)
    for seg in [0, 2]:
        mask = segment_ids == seg
        expected[seg] = torch.logsumexp(src[mask], dim=0)

    assert torch.allclose(out[:3], expected[:3], atol=1e-6, equal_nan=True)
    assert torch.isneginf(out[1]).all()
    assert torch.isneginf(out[3]).all()


def test_et_update_semantics_fidelity():
    # Verify that the update is x - alpha * grad_g, NOT x - alpha * grad_x
    model = ETGraphClassifier(
        in_dim=8, hidden_dim=32, num_classes=1, num_heads=4, head_dim=8,
        num_steps=1, alpha=0.1, pos_k=4
    )
    model.eval()
    batch = _tiny_batch()
    x_enc = model.encoder(batch["x"]).detach().clone()
    
    # Manual one step
    block = model.blocks[0]
    g = block.norm(x_enc).detach().clone().requires_grad_(True)
    e = block.energy_from_g(g, batch["c_2"], batch["u_2"])
    grad_g, = torch.autograd.grad(e, g)
    expected_x1 = x_enc - 0.1 * grad_g
    
    # Model one step
    x_next, _ = block.step(
        x_enc,
        batch["c_2"],
        batch["u_2"],
        None,
        alpha=0.1,
        compute_corr=False,
        noise_std=0.0,
        vary_noise=False,
    )
    
    assert torch.allclose(x_next, expected_x1, atol=1e-6)


def test_graph_tasks_build_model_accepts_etfaithful_alias():
    args = argparse.Namespace(
        model_name="etfaithful",
        in_dim=8,
        hidden_dim=32,
        num_steps=5,
        num_heads=4,
        head_dim=8,
        fixed_step_size=0.1,
        update_damping=0.0,
        inference_mode_train="fixed",
        inference_mode_eval="fixed",
        et_num_blocks=1,
        et_multiplier=4.0,
        et_chn_type="relu",
        et_use_bias_attn=False,
        R=2,
        K=8,
        lambda_2=1.0,
        lambda_3=0.5,
        lambda_m=0.0,
        beta_2=1.0,
        beta_3=1.0,
        beta_m=1.0,
        armijo_eta0=0.2,
        armijo_gamma=0.5,
        armijo_c=1e-4,
        armijo_max_backtracks=20,
        armijo_eval_max_backtracks=5,
        readout_mode="cls",
        use_cls_token=True,
        pos_k=15,
        embed_type="eigen",
        flip_sign=False,
        compute_corr=True,
        noise_std=0.02,
        vary_noise=False,
    )

    model = build_model(args)
    assert isinstance(model, ETGraphClassifier)
