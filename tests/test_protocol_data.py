from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import torch

from get.data import protocol as pdata


class _FakeData:
    def __init__(self, y: torch.Tensor, num_nodes: int = 3):
        self.num_nodes = num_nodes
        self.x = torch.arange(num_nodes * 2, dtype=torch.float32).view(num_nodes, 2)
        self.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        self.y = y
