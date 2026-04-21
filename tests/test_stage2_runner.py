import subprocess
import sys
import os
from pathlib import Path


def _run(cmd, cwd):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _repo_root():
    # tests live at <repo>/tests
    return Path(__file__).resolve().parents[1]


def test_stage2_runner_help():
    repo = _repo_root()
    cmd = [sys.executable, "experiments/run_stage2.py", "--help"]
    rc, out = _run(cmd, cwd=repo)
    assert rc == 0, out
    assert "Unified Stage-2 runner" in out


def test_stage2_runner_smoke_graph_anomaly():
    repo = _repo_root()
    cmd = [
        sys.executable,
        "experiments/run_stage2.py",
        "--task",
        "graph_anomaly",
        "--num_graphs",
        "24",
        "--epochs",
        "1",
        "--batch_size",
        "8",
        "--seeds",
        "123",
        "--anomaly_label_rates",
        "0.4",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo)
    proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, env=env)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out
    assert "Saved outputs/stage2_graph_anomaly.json" in out
