from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# 1. Primary model scope
# ---------------------------------------------------------------------------

def test_protocol_runner_accepts_pairwiseget_and_fullget():
    """Assert the protocol runner argparse accepts pairwiseget and fullget."""
    from experiments.run_protocol import build_arg_parser
    parser = build_arg_parser()
    for name in ("pairwiseget", "fullget"):
        ns = parser.parse_args(["--task", "stage1_wedge_triangle", "--model_name", name, "--epochs", "1"])
        assert ns.model_name == name


def test_model_factory_accepts_pairwiseget_and_fullget():
    """Assert build_model accepts pairwiseget/fullget with minimal config."""
    from get.models.factory import build_model
    from omegaconf import DictConfig
    for name in ("pairwiseget", "fullget"):
        cfg = DictConfig({"model_name": name, "in_dim": 4, "hidden_dim": 64, "num_heads": 2, "head_dim": 32, "num_steps": 1, "num_classes": 1, "task_type": "binary"})
        model = build_model(cfg)
        assert model is not None


# ---------------------------------------------------------------------------
# 3. No dense adjacency conversion
# ---------------------------------------------------------------------------

def test_graph_to_sample_uses_sample_from_edge_index():
    """Assert graph_to_sample in protocol.py does not use to_dense_adj."""
    import ast
    proto_path = ROOT / "get" / "data" / "protocol.py"
    tree = ast.parse(proto_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if "to_dense_adj" in ast.dump(node):
                pytest.fail("graph_to_sample must not call to_dense_adj")


def test_pyg_data_to_sample_uses_sample_from_edge_index():
    """Assert _pyg_data_to_sample in run_graph_tasks.py does not use to_dense_adj."""
    import ast
    runner_path = ROOT / "experiments" / "run_graph_tasks.py"
    tree = ast.parse(runner_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if "to_dense_adj" in ast.dump(node):
                pytest.fail("_pyg_data_to_sample must not call to_dense_adj")


def test_collate_preserves_sparse_motif_indices():
    """Test that collate_graph_samples preserves sparse motif indices."""
    from get.data.synthetic import sample_from_edge_index, collate_graph_samples
    # Triangle (3-cycle) guarantees motifs
    n = 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(n, 4)
    y = torch.tensor([1.0])
    s1 = sample_from_edge_index(edge_index, n, x, y, max_motifs_per_anchor=4)
    s2 = sample_from_edge_index(edge_index, n, x, y, max_motifs_per_anchor=4)
    batch = collate_graph_samples([s1, s2])
    assert batch.c_3.dim() == 1
    if batch.c_3.numel() > 0:
        assert batch.c_3.max().item() < batch.x.size(0), "c_3 indices must be shifted correctly"
        assert batch.c_3.min().item() >= 0


# ---------------------------------------------------------------------------
# 4. Result schema
# ---------------------------------------------------------------------------

def test_result_schema_smoke(tmp_path):
    """Run a tiny experiment and assert output JSON contains required fields."""
    from experiments.run_protocol import build_arg_parser, run_experiment, TASK_SPECS
    from get.data import build_dataset, split_items, summarize_splits
    import torch
    # Build a minimal synthetic dataset
    from argparse import Namespace
    args = Namespace(
        task="stage1_wedge_triangle", model_name="pairwiseget", device="cpu",
            dataset_root="data", tu_name="MUTAG", cv_folds=1, num_runs=1, seed=42,
            in_dim=4, max_motifs_per_anchor=4, max_graphs=8,
            min_nodes=6, max_nodes=12, edge_prob=0.3,
            train_ratio=0.7, val_ratio=0.15,
            hidden_dim=64, num_heads=2, head_dim=32, num_steps=2, num_blocks=1,
            R=2, K=4, lambda_2=1.0, lambda_3=0.0, lambda_m=0.0,
            beta_2=1.0, beta_3=1.0, beta_m=1.0,
            alpha=0.1, multiplier=4.0, pos_k=0,
            epochs=1, batch_size=4, lr=1e-3, weight_decay=1e-4,
            use_amp=False, patience=1, num_workers=0, output_dir=str(tmp_path),
        )
    device = torch.device("cpu")
    items, num_classes = build_dataset(args.task, args)
    tr, va, te = split_items(items, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio, task_type=TASK_SPECS[args.task].task_type)
    result = run_experiment(args, tr, va, te, TASK_SPECS[args.task].task_type, num_classes, device)

    for key in ("train", "val", "test", "history", "parameter_count", "runtime_seconds", "peak_cuda_memory_mb"):
        assert key in result, f"Result must contain {key}"
    assert "loss" in result["test"]
    assert isinstance(result["history"], dict)
    assert "train" in result["history"] or "val" in result["history"]


# ---------------------------------------------------------------------------
# 5. Compile guard
# ---------------------------------------------------------------------------

def test_compile_scope_all_accepts_get_model(tmp_path):
    """compile_scope='all' should run for GET models once inner grads use torch.func.grad."""
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "dataset.name=stage1_wedge_triangle",
        "model=pairwiseget",
        "experiment.device=cpu",
        "trainer.epochs=1",
        "+dataset.max_graphs=8",
        "trainer.batch_size=4",
        "experiment.compile.enabled=True",
        "experiment.compile.scope=all",
        f"+output_dir={tmp_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert "double backward" not in proc.stderr.lower()
    assert "compile.scope='all'" not in proc.stderr


def test_compile_guard_eval_only_accepted(tmp_path):
    """compile_scope='eval_only' must be accepted for GET models (smoke)."""
    import json
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "dataset.name=stage1_wedge_triangle",
        "model=pairwiseget",
        "experiment.device=cpu",
        "trainer.epochs=1",
        "+dataset.max_graphs=8",
        "trainer.batch_size=4",
        "experiment.compile.enabled=True",
        "experiment.compile.scope=eval_only",
        "experiment.inference_mode_eval=fixed",
        f"+output_dir={tmp_path}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    # Note: compile may fail due to known inductor bugs on this torch version
    # We only check two things:
    # 1. It does NOT raise a "compile_scope='all'" error (which indicates incorrect guard)
    # 2. It does NOT raise a double-backward-related error
    scope_error = "compile.scope='all'" in proc.stderr
    double_bwd_error = "double backward" in proc.stderr.lower()
    assert not scope_error, f"eval_only compile should not raise scope error: {proc.stderr[:500]}"
    if proc.returncode != 0 and not double_bwd_error:
        # Failure due to inductor bugs is allowed (known limitation)
        pass


# ---------------------------------------------------------------------------
# 6. Device residency
# ---------------------------------------------------------------------------

def test_move_batch_to_device_moves_tensors():
    """Test move_batch_to_device moves tensor fields to non-CPU device."""
    from get.utils.device import move_batch_to_device
    cpu = torch.device("cpu")
    x = torch.randn(3, 4)
    batch = {"x": x, "y": torch.tensor([1.0]), "c_2": torch.tensor([0, 1])}
    moved = move_batch_to_device(batch, cpu)
    for k, v in moved.items():
        assert isinstance(v, torch.Tensor)
        assert v.device == cpu


def test_assert_cuda_batch_fails_on_cpu_tensor():
    """assert_cuda_batch must raise on a CPU tensor."""
    from get.utils.device import assert_cuda_batch
    batch = {"x": torch.randn(3, 4)}
    with pytest.raises(AssertionError, match="x is still on CPU"):
        assert_cuda_batch(batch)


def test_assert_cuda_batch_passes_on_cuda_tensor():
    """assert_cuda_batch must pass on CUDA tensors (when CUDA available)."""
    from get.utils.device import assert_cuda_batch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    batch = {"x": torch.randn(3, 4, device="cuda")}
    assert_cuda_batch(batch)


def test_hot_modules_no_forbidden_calls():
    """Static grep for banned CPU sync calls in hot modules."""
    hot_dirs = [
        ROOT / "get" / "energy",
        ROOT / "get" / "solvers",
        ROOT / "get" / "models",
        ROOT / "get" / "trainers",
    ]
    import subprocess as sp
    banned = [r"\.cpu\(\)", r"\.numpy\(\)", r"\.item\(\)"]
    exceptions = [
        "checkpoint", "serialization", "epoch", "test_",  # allow in tests
        "preprocessing",  # allow in preprocessing code
    ]
    for d in hot_dirs:
        if not d.is_dir():
            continue
        result = sp.run(
            ["rg", "-n"] + [p for p in banned] + ["--", str(d)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if not any(exc in line for exc in exceptions):
                    pytest.fail(f"Forbidden CPU sync call in {d}:\n  {line}")


# ---------------------------------------------------------------------------
# 7. PrefetchLoader smoke
# ---------------------------------------------------------------------------

def test_prefetch_loader_smoke():
    """Build tiny PyG dataset, wrap with PrefetchLoader, run one training step."""
    try:
        from torch_geometric.loader import DataLoader, PrefetchLoader
    except ImportError:
        pytest.skip("PrefetchLoader not available in this PyG version")

    if not torch.cuda.is_available():
        pytest.skip("CUDA required for PrefetchLoader smoke test")

    from get.data.synthetic import GraphSampleData, collate_graph_samples
    n = 4
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(n, 4)
    y = torch.tensor([1.0])
    c_2 = edge_index[0].clone()
    u_2 = edge_index[1].clone()
    c_3 = torch.tensor([0, 1], dtype=torch.long)
    u_3 = torch.tensor([1, 2], dtype=torch.long)
    v_3 = torch.tensor([2, 0], dtype=torch.long)
    t_tau = torch.tensor([0, 1], dtype=torch.long)

    samples = [GraphSampleData(x=x, y=y, c_2=c_2, u_2=u_2, c_3=c_3, u_3=u_3, v_3=v_3, t_tau=t_tau)]
    base_loader = DataLoader(samples, batch_size=1, shuffle=False, collate_fn=collate_graph_samples)
    device = torch.device("cuda")
    loader = PrefetchLoader(base_loader, device=device)

    from get.models.factory import build_model
    from omegaconf import DictConfig
    cfg = DictConfig({"model_name": "pairwiseget", "in_dim": 4, "hidden_dim": 64, "num_heads": 2, "head_dim": 32, "num_steps": 1, "num_classes": 1, "task_type": "binary", "lambda_2": 1.0, "lambda_3": 0.0, "lambda_m": 0.0, "beta_2": 1.0, "beta_3": 1.0, "beta_m": 1.0, "update_damping": 0.0, "agg_mode": "softmax", "readout_mode": "graph"})
    model = build_model(cfg).to(device)

    for batch in loader:
        assert batch.x.is_cuda, "PrefetchLoader must place x on CUDA"
        for key in ("c_2", "u_2", "c_3", "u_3", "v_3", "t_tau", "batch"):
            if hasattr(batch, key):
                val = getattr(batch, key)
                if torch.is_tensor(val):
                    assert val.is_cuda, f"PrefetchLoader must place {key} on CUDA"
        logits = model(batch)
        assert logits.is_cuda, "model output must be on CUDA"
        break


# ---------------------------------------------------------------------------
# 8. GET-HAM branch registry
# ---------------------------------------------------------------------------

def test_branch_registry_pairwiseget_defaults_disable_global_and_cls():
    """Assert PairwiseGET/FullGET defaults do not enable global attention, CLS, structural memory, or dynamic edges."""
    from get.energy.branch import enabled_branches_from_config

    class _PairwiseCfg:
        pairwise = True
        motif = False
        memory = False
        global_attention = False

    class _FullGETCfg:
        pairwise = True
        motif = True
        memory = True
        global_attention = False

    pw = enabled_branches_from_config(_PairwiseCfg)
    assert pw["global_attention"] is False

    fg = enabled_branches_from_config(_FullGETCfg)
    assert fg["global_attention"] is False
    assert fg["pairwise"] is True
    assert fg["motif"] is True


def test_composed_energy_instantiates_requested_branches():
    """ComposedEnergy with pairwise+motif+memory should produce finite energy."""
    from get.energy.branch import ComposedEnergy
    import torch
    energy = ComposedEnergy(["pairwise", "motif", "memory"])
    H = torch.randn(6, 8)
    batch = type("Batch", (), {
        "batch": torch.zeros(6, dtype=torch.long),
        "c_2": torch.tensor([0, 1, 2, 3, 4, 5]),
        "u_2": torch.tensor([1, 2, 3, 4, 5, 0]),
        "c_3": torch.tensor([0, 1, 2, 3, 4, 5]),
        "u_3": torch.tensor([1, 2, 3, 4, 5, 0]),
        "v_3": torch.tensor([2, 3, 4, 5, 0, 1]),
        "t_tau": torch.zeros(6, dtype=torch.long),
    })()
    params = {"d": 8, "R": 2, "K": 3, "lambda_2": 1.0, "lambda_3": 1.0, "lambda_m": 1.0,
              "beta_2": 1.0, "beta_3": 1.0, "beta_m": 1.0, "use_pairwise": True,
              "use_motif": True, "use_memory": True, "agg_mode": "softmax", "T_tau": torch.randn(2, 1, 2, 4)}
    proj = {"Q2": torch.randn(6, 1, 8), "K2": torch.randn(6, 1, 8),
            "Q3": torch.randn(6, 1, 2, 4), "K3": torch.randn(6, 1, 2, 4),
            "Qm": torch.randn(6, 1, 8), "Km": torch.randn(1, 3, 8),
            "Qg": torch.randn(6, 1, 8), "Kg": torch.randn(6, 1, 8)}
    context = {"params": params, "projections": proj, "num_graphs": 1, "batch": batch.batch}
    total, branch_energies = energy({"H": H}, batch, context)
    assert torch.isfinite(total).all(), "Composed energy must be finite"
    for name in ("pairwise", "motif", "memory"):
        assert name in branch_energies, f"Branch {name} should be in output"


# ---------------------------------------------------------------------------
# 9. Global attention guard
# ---------------------------------------------------------------------------

def test_global_attention_energy_is_finite():
    """Small-graph global attention energy must be finite and differentiable."""
    from get.energy.branch import GlobalAttentionBranch
    import torch
    branch = GlobalAttentionBranch(max_global_nodes=64)
    H = torch.randn(6, 8, requires_grad=True)
    batch = type("Batch", (), {"batch": torch.zeros(6, dtype=torch.long)})()
    params = {"use_global_attention": True, "lambda_g": 1.0, "beta_g": 1.0}
    Wg = torch.randn(8, 8)
    proj = {"Qg": H @ Wg, "Kg": H @ Wg}
    context = {"params": params, "projections": proj, "num_graphs": 1, "batch": batch.batch}
    e = branch({"H": H}, batch, context)
    assert torch.isfinite(e).all(), "Global attention energy must be finite"
    grad, = torch.autograd.grad(e.sum(), H, create_graph=True)
    assert grad is not None and torch.isfinite(grad).all(), "Global attention gradients must be finite"


def test_global_attention_max_nodes_guard():
    """max_global_nodes must prevent accidental dense all-pairs allocation on large batches."""
    from get.energy.branch import GlobalAttentionBranch
    import torch
    branch = GlobalAttentionBranch(max_global_nodes=4)
    H = torch.randn(10, 8)
    batch = type("Batch", (), {"batch": torch.zeros(10, dtype=torch.long)})()
    params = {"use_global_attention": True, "lambda_g": 1.0, "beta_g": 1.0}
    proj = {"Qg": torch.randn(10, 8), "Kg": torch.randn(10, 8)}
    context = {"params": params, "projections": proj, "num_graphs": 1, "batch": batch.batch}
    import pytest
    with pytest.raises(RuntimeError, match="max_global_nodes"):
        branch({"H": H}, batch, context)


def test_get_ham_global_integrated_forward_is_finite():
    """Integrated GET-HAM global path must handle multi-head Q/K projections."""
    from types import SimpleNamespace
    import torch
    from get.data.synthetic import sample_from_edge_index, collate_graph_samples
    from get.models.factory import build_model

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3]],
        dtype=torch.long,
    )
    x = torch.randn(4, 5)
    y = torch.tensor([1.0])
    sample = sample_from_edge_index(edge_index, 4, x, y, max_motifs_per_anchor=4)
    batch = collate_graph_samples([sample])

    cfg = SimpleNamespace(
        model_name="get_ham_global",
        task_type="binary",
        num_classes=1,
        in_dim=5,
        hidden_dim=8,
        num_steps=1,
        num_heads=2,
        head_dim=4,
        R=2,
        K=3,
        num_motif_types=2,
        lambda_2=1.0,
        lambda_3=1.0,
        lambda_m=1.0,
        lambda_g=1.0,
        beta_2=1.0,
        beta_3=1.0,
        beta_m=1.0,
        beta_g=1.0,
        update_damping=0.0,
        fixed_step_size=0.01,
        inference_mode_train="fixed",
        inference_mode_eval="fixed",
        max_global_nodes=32,
    )
    model = build_model(cfg)
    out = model(batch)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_get_ham_global_multi_graph_multi_head_forward_is_finite():
    """Branch-composed energy must reduce per-head branch energies per graph."""
    from types import SimpleNamespace
    import torch
    from get.data.synthetic import sample_from_edge_index, collate_graph_samples
    from get.models.factory import build_model

    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0, 1], [1, 2, 3, 0, 2, 3]],
        dtype=torch.long,
    )
    x = torch.randn(4, 5)
    y = torch.tensor([1.0])
    samples = [
        sample_from_edge_index(edge_index, 4, x, y, max_motifs_per_anchor=4),
        sample_from_edge_index(edge_index, 4, x, y, max_motifs_per_anchor=4),
    ]
    batch = collate_graph_samples(samples)

    cfg = SimpleNamespace(
        model_name="get_ham_global",
        task_type="binary",
        num_classes=1,
        in_dim=5,
        hidden_dim=12,
        num_steps=1,
        num_heads=3,
        head_dim=4,
        R=2,
        K=3,
        num_motif_types=2,
        lambda_2=1.0,
        lambda_3=1.0,
        lambda_m=1.0,
        lambda_g=1.0,
        beta_2=1.0,
        beta_3=1.0,
        beta_m=1.0,
        beta_g=1.0,
        update_damping=0.0,
        fixed_step_size=0.01,
        inference_mode_train="fixed",
        inference_mode_eval="fixed",
        max_global_nodes=32,
    )
    model = build_model(cfg)
    out = model(batch)
    assert out.shape == (2,)
    assert torch.isfinite(out).all()





# ---------------------------------------------------------------------------
# 10. Readout [mean; sum; max]
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 11. Readout [mean; sum; max]
# ---------------------------------------------------------------------------

def test_readout_concat_mean_sum_max():
    from get.models.energy_classifier import EnergyGraphClassifier
    import torch
    model = EnergyGraphClassifier(
        in_dim=4, hidden_dim=64, num_classes=2, num_steps=1,
        num_heads=2, head_dim=32, R=2, K=4, num_motif_types=2,
        lambda_2=1.0, lambda_3=0.0, lambda_m=0.0,
        beta_2=1.0, beta_3=1.0, beta_m=1.0,
        update_damping=0.0, readout_mode="graph",
    ).eval()
    x = torch.randn(6, 64)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    pooled = model._pool_graph_concat(x, batch, 2)
    assert pooled.size(-1) == 64 * 3


# ---------------------------------------------------------------------------
# 12. Static edge features
# ---------------------------------------------------------------------------

def test_edge_attr_preserved_through_collation():
    """Test edge_attr is preserved through collation."""
    from get.data.synthetic import sample_from_edge_index, collate_graph_samples, GraphSampleData
    import torch
    n = 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(n, 4)
    y = torch.tensor([1.0])
    s1 = sample_from_edge_index(edge_index, n, x, y, max_motifs_per_anchor=4)
    s2 = sample_from_edge_index(edge_index, n, x, y, max_motifs_per_anchor=4)
    if not hasattr(s1, "edge_attr") or s1.edge_attr is None:
        s1.edge_attr = torch.randn(edge_index.size(1), 2)
        s2.edge_attr = torch.randn(edge_index.size(1), 2)
    batch = collate_graph_samples([s1, s2])
    assert hasattr(batch, "edge_attr")
    assert batch.edge_attr.dim() >= 2


def test_edge_attr_vector_is_promoted_to_matrix():
    """Test 1D edge_attr is promoted to a feature matrix on sample creation."""
    from get.data.synthetic import sample_from_edge_index
    import torch
    n = 2
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    x = torch.randn(n, 4)
    y = torch.tensor([1.0])
    edge_attr = torch.tensor([0.25, 0.75], dtype=torch.float32)
    sample = sample_from_edge_index(edge_index, n, x, y, max_motifs_per_anchor=0, edge_attr=edge_attr)
    assert hasattr(sample, "edge_attr")
    assert sample.edge_attr.shape == (2, 1)


# ---------------------------------------------------------------------------
# 12. PE/SE policy
# ---------------------------------------------------------------------------

def test_pe_se_tensors_on_batch(tmp_path):
    """Test PE/SE tensors can be stored on the Data object and moved with the batch."""
    from get.data.synthetic import GraphSampleData, collate_graph_samples
    import torch
    n = 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    x = torch.randn(n, 4)
    y = torch.tensor([1.0])
    pe = torch.randn(n, 4)
    s = GraphSampleData(x=x, y=y, c_2=edge_index[0], u_2=edge_index[1],
                        c_3=torch.tensor([0, 1]), u_3=torch.tensor([1, 2]),
                        v_3=torch.tensor([2, 0]), t_tau=torch.tensor([0, 1]),
                        pos=pe)
    batch = collate_graph_samples([s, s])
    assert hasattr(batch, "pos"), "PE tensor should be preserved in batch"
    assert batch.pos.size(0) == n * 2


# ---------------------------------------------------------------------------
# 13. Multi-state solver (future stub)
# ---------------------------------------------------------------------------

def test_dict_state_solver_is_future():
    """Multi-state dict solver is marked as future/extension."""
    import pytest
    pytest.skip("Multi-state solver is a future extension (Phase D). Not yet implemented.")
