from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_protocol_runner_stage1_smoke(tmp_path: Path) -> None:
    out = tmp_path / "stage1.json"
    cmd = [
        sys.executable,
        "experiments/run_protocol.py",
        "--task",
        "stage1_wedge_triangle",
        "--model_name",
        "pairwiseget",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--max_graphs",
        "16",
        "--batch_size",
        "8",
        "--eval_batch_size",
        "8",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    assert out.exists()


def test_protocol_runner_stage1_xorsat_smoke(tmp_path: Path) -> None:
    out = tmp_path / "stage1_xorsat.json"
    cmd = [
        sys.executable,
        "experiments/run_protocol.py",
        "--task",
        "stage1_xorsat",
        "--model_name",
        "pairwiseget",
        "--device",
        "cpu",
        "--epochs",
        "1",
        "--max_graphs",
        "16",
        "--batch_size",
        "8",
        "--eval_batch_size",
        "8",
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    assert out.exists()
