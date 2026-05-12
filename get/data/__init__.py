"""Data utilities for the refactored stack."""

from .synthetic import SyntheticGraphDataset, collate_graph_samples
from .real_world import RealWorldGraphDataset
from .protocol import (
    build_dataset,
    ListGraphDataset,
    split_items,
    summarize_splits,
    TASK_SPECS,
    get_k_fold_splits,
)

__all__ = [
    "SyntheticGraphDataset",
    "collate_graph_samples",
    "RealWorldGraphDataset",
    "build_dataset",
    "ListGraphDataset",
    "split_items",
    "summarize_splits",
    "TASK_SPECS",
    "get_k_fold_splits",
]
