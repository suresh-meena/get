# Graph Energy Transformer (GET)

A modular PyTorch implementation of the Graph Energy Transformer (GET), an energy-based graph neural network architecture.

## Repository Structure

The project is organized into subpackages for clarity and modularity:

- **`get/`**: Core library.
    - **`energy/`**: Implementation of scalar energy functions (Pairwise, Motif, Memory, Quadratic) and their analytical gradients.
    - **`models/`**: Architecture definitions including `GETModel`, `GETLayer`, and standard baselines (GCN, GAT, GIN).
    - **`data/`**: Data pipeline components, including CSR graph construction, motif extraction, and batching logic.
    - **`nn/`**: Shared neural network primitives like `EnergyLayerNorm`, `StableMLP`, and `ETGraphMaskModulator`.
    - **`utils/`**: Utilities for model registration, graph operations, and training helpers.
- **`experiments/`**: Testbed for empirical validation.
    - **`stage1/`**: Simple structural tests (Wedge discrimination, SAT factor graphs).
    - **`stage2/`**: Expressivity benchmarks (CSL dataset).
    - **`stage3/`**: Large-scale regression (ZINC).
- **`tests/`**: Comprehensive unit tests and robustness checks.

## Key Features

- **Analytical Gradients**: Manual implementation of energy gradients for high-performance deterministic unrolling.
- **Motif-Aware Energy**: Explicit higher-order structural branches for capturing local graph motifs (wedges, triangles).
- **Global Memory**: Integrated Hopfield-style associative memory for long-range dependency modeling.
- **Theoretical Guarantees**: Verified reductions to Modern Hopfield Networks and Energy Transformers.
- **Optimized Solver**: Support for both fixed-step and Armijo line-search gradient descent.

## Getting Started

### Installation

```bash
# Create and activate the conda environment
conda env create -f ENVIRONMENT.md
conda activate get
```

### Running Experiments

To run the CSL expressivity experiment:

```bash
export PYTHONPATH=.
python experiments/stage2/csl_expressivity.py --epochs 100
```

To run the ZINC regression experiment:

```bash
export PYTHONPATH=.
python experiments/stage3/zinc_regression.py --epochs 200
```

## Running Tests

We use `pytest` for unit testing and verification of mathematical properties:

```bash
python -m pytest tests/
```
