from __future__ import annotations

import pytest
import torch
import argparse

from get.data import SyntheticGraphDataset, collate_graph_samples
from get.models import ETGraphClassifier
from experiments.protocol.modeling import build_model
from experiments.run_graph_tasks import _build_model as build_graph_tasks_model


def _tiny_batch():
    ds = SyntheticGraphDataset(
        num_graphs=4,
        min_nodes=5,
        max_nodes=6,
        edge_prob=0.5,
        in_dim=8,
        max_motifs_per_anchor=4,
        seed=42,
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
    )
    model.eval()
    
    # Simple forward
    logits = model(batch)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


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


def test_et_update_semantics_fidelity():
    # Verify that the update is x - alpha * grad_g, NOT x - alpha * grad_x
    model = ETGraphClassifier(
        in_dim=8, hidden_dim=32, num_classes=1, num_heads=4, head_dim=8,
        num_steps=1, alpha=0.1
    )
    model.eval()
    batch = _tiny_batch()
    x0 = model.encoder(batch["x"]).detach().clone()
    
    # Manual one step
    block = model.blocks[0]
    g = block.norm(x0).detach().clone().requires_grad_(True)
    e = block.energy_from_g(g, batch["c_2"], batch["u_2"])
    grad_g, = torch.autograd.grad(e, g)
    expected_x1 = x0 - 0.1 * grad_g
    
    # Model one step
    # We need to intercept the encoder output
    with torch.no_grad():
        x_enc = model.encoder(batch["x"])
    
    # Run the block step directly
    x_next, _ = block.step(x_enc, batch["c_2"], batch["u_2"], alpha=0.1)
    
    assert torch.allclose(x_next, expected_x1, atol=1e-6)

    # Counter-check: grad_x should be different
    x_grad_test = x0.detach().clone().requires_grad_(True)
    e_x = block.energy(x_grad_test, batch["c_2"], batch["u_2"])
    grad_x, = torch.autograd.grad(e_x, x_grad_test)
    wrong_x1 = x0 - 0.1 * grad_x
    
    # They should be different because of the ELN Jacobian
    assert not torch.allclose(x_next, wrong_x1, atol=1e-6)


def test_et_head_mixing_effect():
    # Verify that Hw is not just an identity or zero and affects energy
    model = ETGraphClassifier(
        in_dim=8, hidden_dim=32, num_classes=1, num_heads=4, head_dim=8
    )
    batch = _tiny_batch()
    # Use larger values to overcome small init and ensure divergence
    g = torch.randn(batch["x"].size(0), 32) * 10.0
    
    with torch.no_grad():
        # Set Hw to identity
        model.blocks[0].attn.Hw.copy_(torch.eye(4))
        e_id = model.blocks[0].attn.energy(g, batch["c_2"], batch["u_2"])
        
        # Set Hw to something very different
        model.blocks[0].attn.Hw.fill_(5.0)
        e_fill = model.blocks[0].attn.energy(g, batch["c_2"], batch["u_2"])
        
        # Should be very different now
        diff = (e_id - e_fill).abs().item()
        assert diff > 1.0  # Should be huge now with g*10 and Hw=5


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
    )

    model = build_graph_tasks_model(args, task_type="binary", num_classes=1)
    assert isinstance(model, ETGraphClassifier)
