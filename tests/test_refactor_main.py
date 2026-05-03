from __future__ import annotations

import pytest
from pathlib import Path

from omegaconf import OmegaConf

from main import run_from_cfg


def test_refactor_main_tiny_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create(
        {
            "seed": 1,
            "task": "binary_graph_classification",
            "experiment": OmegaConf.load(root / "configs" / "experiment" / "synthetic_graph.yaml"),
            "dataset": OmegaConf.load(root / "configs" / "dataset" / "synthetic_graph.yaml"),
            "model": OmegaConf.load(root / "configs" / "model" / "fullget.yaml"),
            "trainer": OmegaConf.load(root / "configs" / "trainer" / "default.yaml"),
        }
    )
    cfg = OmegaConf.merge(cfg, {"experiment": {"device": "cpu", "inference_mode_eval": "fixed"}})
    cfg = OmegaConf.merge(
        cfg,
        {
            "dataset": {
                "in_dim": 8,
                "num_train_graphs": 16,
                "num_val_graphs": 8,
                "num_test_graphs": 8,
                "min_nodes": 5,
                "max_nodes": 7,
                "edge_prob": 0.2,
                "max_motifs_per_anchor": 4,
            },
            "trainer": {
                "epochs": 2,
                "batch_size": 8,
                "eval_batch_size": 8,
                "patience": 2,
                "use_amp": False,
                "num_workers": 0,
            },
        },
    )

    metrics = run_from_cfg(cfg)
    assert "test" in metrics
    assert 0.0 <= metrics["test"]["acc"] <= 1.0
    assert metrics["epochs_ran"] >= 1


def test_refactor_main_quadratic_only_energy_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create(
        {
            "seed": 2,
            "task": "binary_graph_classification",
            "experiment": OmegaConf.load(root / "configs" / "experiment" / "synthetic_graph.yaml"),
            "dataset": OmegaConf.load(root / "configs" / "dataset" / "synthetic_graph.yaml"),
            "model": OmegaConf.load(root / "configs" / "model" / "fullget.yaml"),
            "trainer": OmegaConf.load(root / "configs" / "trainer" / "default.yaml"),
        }
    )
    cfg = OmegaConf.merge(
        cfg,
        {
            "experiment": {"device": "cpu", "inference_mode_eval": "fixed"},
            "model": {"energy_name": "quadratic_only"},
            "dataset": {"num_train_graphs": 12, "num_val_graphs": 6, "num_test_graphs": 6},
            "trainer": {"epochs": 1, "patience": 1, "use_amp": False, "num_workers": 0},
        },
    )
    metrics = run_from_cfg(cfg)
    assert "test" in metrics
    assert metrics["epochs_ran"] >= 1


def test_refactor_main_compile_eval_only_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create(
        {
            "seed": 3,
            "task": "binary_graph_classification",
            "experiment": OmegaConf.load(root / "configs" / "experiment" / "synthetic_graph.yaml"),
            "dataset": OmegaConf.load(root / "configs" / "dataset" / "synthetic_graph.yaml"),
            "model": OmegaConf.load(root / "configs" / "model" / "fullget.yaml"),
            "trainer": OmegaConf.load(root / "configs" / "trainer" / "default.yaml"),
        }
    )
    cfg = OmegaConf.merge(
        cfg,
        {
            "experiment": {
                "device": "cpu",
                "compile": {"enabled": True, "scope": "eval_only"},
                "inference_mode_train": "fixed",
                "inference_mode_eval": "fixed",
            },
            "dataset": {"num_train_graphs": 12, "num_val_graphs": 6, "num_test_graphs": 6},
            "trainer": {"epochs": 1, "patience": 1, "use_amp": False, "num_workers": 0},
        },
    )
    metrics = run_from_cfg(cfg)
    assert "test" in metrics
    assert metrics["epochs_ran"] >= 1


def test_refactor_main_compile_all_rejected_for_get_training(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create(
        {
            "seed": 4,
            "task": "binary_graph_classification",
            "experiment": OmegaConf.load(root / "configs" / "experiment" / "synthetic_graph.yaml"),
            "dataset": OmegaConf.load(root / "configs" / "dataset" / "synthetic_graph.yaml"),
            "model": OmegaConf.load(root / "configs" / "model" / "fullget.yaml"),
            "trainer": OmegaConf.load(root / "configs" / "trainer" / "default.yaml"),
        }
    )
    cfg = OmegaConf.merge(
        cfg,
        {
            "experiment": {
                "device": "cpu",
                "compile": {"enabled": True, "scope": "all"},
                "inference_mode_train": "fixed",
                "inference_mode_eval": "fixed",
            },
            "dataset": {"num_train_graphs": 12, "num_val_graphs": 6, "num_test_graphs": 6},
            "trainer": {"epochs": 1, "patience": 1, "use_amp": False, "num_workers": 0},
        },
    )
    with pytest.raises(ValueError, match="compile.scope='all' is unsupported"):
        run_from_cfg(cfg)
