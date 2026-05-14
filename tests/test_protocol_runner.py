from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_main_runner_stage1_smoke(tmp_path: Path) -> None:
    # Test using main.py with Hydra overrides
    cmd = [
        sys.executable,
        "main.py",
        "dataset.name=stage1_wedge_triangle",
        "model=pairwiseget",
        "experiment.device=cpu",
        "trainer.epochs=1",
        "+dataset.max_graphs=16",
        "trainer.batch_size=8",
        "trainer.eval_batch_size=8",
        f"+output_dir={tmp_path}",
    ]
    subprocess.run(cmd, check=True)
    assert (tmp_path / "metrics_stage1_wedge_triangle.json").exists()


def test_main_runner_stage1_xorsat_smoke(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "main.py",
        "dataset.name=stage1_xorsat",
        "model=pairwiseget",
        "experiment.device=cpu",
        "trainer.epochs=1",
        "+dataset.max_graphs=16",
        "trainer.batch_size=8",
        "trainer.eval_batch_size=8",
        f"+output_dir={tmp_path}",
    ]
    subprocess.run(cmd, check=True)
    assert (tmp_path / "metrics_stage1_xorsat.json").exists()

