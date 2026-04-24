# Graph Energy Transformer (GET) Experiments

This directory contains the experimental code used to evaluate the GET architecture across four stages. To maintain a clean structure, legacy `expX_` wrapper scripts have been removed.

Please run the stage-specific scripts directly from the `code/` directory.

## Directory Structure

- `stage1/`: Mechanism diagnostics (identifying motif causality)
  - `wedge_discrimination.py`: Discrimination between density-matched graphs.
  - `triangle_counting_regression.py`: Counting triangles to test motif signals.
  - `sat_reasoning.py`: Ternary constraint reasoning (Max-3-SAT/3-XORSAT).
- `stage2/`: Expressivity
  - `csl_expressivity.py`: Circular Skip Link graphs benchmark.
  - `brec_expressivity.py`: BREC graph isomorphism benchmark.
- `stage3/`: Molecular transfer
  - `zinc_regression.py`: ZINC-12k constrained solubility.
  - `molhiv_classification.py`: OGBG-MolHIV classification.
  - `peptides_transfer.py`: LRGB Peptides functional/structural tasks.
- `stage4/`: ET-style transfer tasks
  - `runner.py`: Unified runner for TU graph classification and YelpChi/Amazon anomaly detection.
  - `plot.py`: Generates publication-ready plots for Stage 4 outputs.
- `shared/`: Shared utilities
  - `common.py`: Shared functions, data splitting, and unified `GETTrainer`.

## Usage Example

To run a specific experiment, ensure your `PYTHONPATH` includes the `code/` directory:

```bash
cd code
PYTHONPATH=. python experiments/stage3/zinc_regression.py --epochs 100 --model full --rwse_k 16
```
