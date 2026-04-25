"""Shared utilities for the GET package."""

from .registry import (
    MODEL_REGISTRY,
    available_models,
    build_model,
    register_model,
)
from .graph import (
    build_undirected_adjacency,
    shortest_path_distances,
    degree_centrality,
    augment_with_virtual_node,
    _numba_csr_to_dense,
)
from .training import (
    build_adamw_optimizer,
    maybe_compile_model,
    random_flip_pe_signs,
    _is_get_model,
)
from .encoding import (
    laplacian_pe_from_adjacency,
    laplacian_pe_from_sparse_matrix,
    rwse_from_adjacency,
)

__all__ = [
    "MODEL_REGISTRY",
    "available_models",
    "build_model",
    "register_model",
    "build_undirected_adjacency",
    "shortest_path_distances",
    "degree_centrality",
    "augment_with_virtual_node",
    "_numba_csr_to_dense",
    "build_adamw_optimizer",
    "maybe_compile_model",
    "random_flip_pe_signs",
    "_is_get_model",
    "laplacian_pe_from_adjacency",
    "laplacian_pe_from_sparse_matrix",
    "rwse_from_adjacency",
]
