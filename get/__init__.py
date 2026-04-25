"""GET: Graph Energy Transformer.

Organized into subpackages:
  - get.energy/   : scalar energy functions and their gradients
  - get.models/   : GETModel, ETFaithful, baselines
  - get.data/     : batching, motif extraction, features, caching
  - get.utils/    : registry, graph utilities, training helpers
  - get.nn/       : shared neural network primitives (EnergyLayerNorm, MLP, etc.)

The original flat modules (energy.py, model.py, etc.) remain functional for
internal compatibility; new code should import from subpackages.
"""

# --- Data pipeline ---
from .data import (
    add_structural_node_features as add_structural_node_features,
    collate_get_batch as collate_get_batch,
    GETBatch as GETBatch,
    CachedGraphDataset as CachedGraphDataset,
    validate_get_batch as validate_get_batch,
)

# --- Core model ---
from .models.get_model import GETModel as GETModel

# --- ET core ---
from .models.et_core import (
    ETAttentionCore as ETAttentionCore,
    ETHopfieldCore as ETHopfieldCore,
    ETCoreBlock as ETCoreBlock,
)

# --- NN primitives ---
from .nn import (
    EnergyLayerNorm as EnergyLayerNorm,
    ETGraphMaskModulator as ETGraphMaskModulator,
)

# --- Baselines & factories ---
from .models.baselines import (
    PairwiseGET as PairwiseGET,
    FullGET as FullGET,
    GINBaseline as GINBaseline,
    GCNBaseline as GCNBaseline,
    GATBaseline as GATBaseline,
    ETFaithful as ETFaithful,
)

# --- Registry ---
from .utils.registry import (
    MODEL_REGISTRY as MODEL_REGISTRY,
    available_models as available_models,
    build_model as build_model,
    register_model as register_model,
)

# --- Graph utilities ---
from .utils.graph import (
    build_undirected_adjacency as build_undirected_adjacency,
    shortest_path_distances as shortest_path_distances,
    degree_centrality as degree_centrality,
    augment_with_virtual_node as augment_with_virtual_node,
)

# --- Training utilities ---
from .utils.training import (
    build_adamw_optimizer as build_adamw_optimizer,
    maybe_compile_model as maybe_compile_model,
    random_flip_pe_signs as random_flip_pe_signs,
)

# --- Encoding utilities ---
from .utils.encoding import (
    laplacian_pe_from_adjacency as laplacian_pe_from_adjacency,
    rwse_from_adjacency as rwse_from_adjacency,
)

__all__ = [
    # Data
    "add_structural_node_features",
    "collate_get_batch",
    "GETBatch",
    "CachedGraphDataset",
    "validate_get_batch",
    # Model
    "GETModel",
    # ET core
    "EnergyLayerNorm",
    "ETAttentionCore",
    "ETHopfieldCore",
    "ETCoreBlock",
    "ETGraphMaskModulator",
    # Baselines
    "PairwiseGET",
    "FullGET",
    "GINBaseline",
    "GCNBaseline",
    "GATBaseline",
    "ETFaithful",
    # Registry
    "MODEL_REGISTRY",
    "available_models",
    "build_model",
    "register_model",
    # Graph utilities
    "build_undirected_adjacency",
    "shortest_path_distances",
    "degree_centrality",
    "augment_with_virtual_node",
    # Training
    "build_adamw_optimizer",
    "maybe_compile_model",
    "laplacian_pe_from_adjacency",
    "random_flip_pe_signs",
    "rwse_from_adjacency",
]
