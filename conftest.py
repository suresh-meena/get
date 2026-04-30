"""Pytest configuration for local package imports."""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _release_cuda_memory():
    yield
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
