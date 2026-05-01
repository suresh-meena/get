from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_cmd(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    return proc.stdout


def test_graph_tasks_runner_external_baseline_smoke(tmp_path: Path) -> None:
    out = tmp_path / "metrics.json"
    cmd = [
        sys.executable,
        "experiments/run_graph_tasks.py",
        "--model_name",
        "external_baseline",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--num_train_graphs",
        "16",
        "--num_val_graphs",
        "8",
        "--num_test_graphs",
        "8",
        "--batch_size",
        "8",
        "--eval_batch_size",
        "8",
        "--output",
        str(out),
    ]
    _run_cmd(cmd)
    assert out.exists()


def test_graph_tasks_runner_stage2_preset_smoke(tmp_path: Path) -> None:
    out = tmp_path / "metrics_stage2.json"
    cmd = [
        sys.executable,
        "experiments/run_graph_tasks.py",
        "--task_preset",
        "csl",
        "--model_name",
        "pairwiseget",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--num_train_graphs",
        "16",
        "--num_val_graphs",
        "8",
        "--num_test_graphs",
        "8",
        "--batch_size",
        "8",
        "--eval_batch_size",
        "8",
        "--output",
        str(out),
    ]
    _run_cmd(cmd)
    assert out.exists()


def test_graph_tasks_runner_stage4_preset_smoke(tmp_path: Path) -> None:
    out = tmp_path / "metrics_stage4.json"
    cmd = [
        sys.executable,
        "experiments/run_graph_tasks.py",
        "--task_preset",
        "graph_anomaly",
        "--model_name",
        "external_baseline",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--num_train_graphs",
        "16",
        "--num_val_graphs",
        "8",
        "--num_test_graphs",
        "8",
        "--batch_size",
        "8",
        "--eval_batch_size",
        "8",
        "--output",
        str(out),
    ]
    _run_cmd(cmd)
    assert out.exists()


def test_graph_tasks_runner_csl_cv_smoke(tmp_path: Path) -> None:
    out = tmp_path / "metrics_csl_cv.json"
    cmd = [
        sys.executable,
        "experiments/run_graph_tasks.py",
        "--dataset_name",
        "csl",
        "--task_preset",
        "csl",
        "--cv_folds",
        "3",
        "--max_graphs",
        "24",
        "--model_name",
        "pairwiseget",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--batch_size",
        "8",
        "--eval_batch_size",
        "8",
        "--output",
        str(out),
    ]
    _run_cmd(cmd)
    assert out.exists()
