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
    add_structural_node_features,
    collate_get_batch,
    GETBatch,
    CachedGraphDataset,
    validate_get_batch,
)

# --- Core model ---
from .models.get_model import GETModel

# --- ET core ---
from .models.et_core import ETAttentionCore, ETHopfieldCore, ETCoreBlock

# --- NN primitives ---
from .nn import EnergyLayerNorm, ETGraphMaskModulator

# --- Baselines & factories ---
from .models.baselines import PairwiseGET, FullGET, GINBaseline, GCNBaseline, GATBaseline, ETFaithful

# --- Training utilities ---
from .utils.training import build_adamw_optimizer, maybe_compile_model, random_flip_pe_signs

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
    # Training
    "build_adamw_optimizer",
    "maybe_compile_model",
    "random_flip_pe_signs",
]
