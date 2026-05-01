# Graph Energy Transformer (GET)

A modular PyTorch refactor of Graph Energy Transformer (GET) energy components.

## Repository Structure

The codebase currently has two tracks:

- **`get/`**: Refactored core package.
  - **`energy/`**: Pure scalar energy branches (quadratic, pairwise, motif, memory) and composition in `GETEnergy`.
- **`configs/`**: Refactor-era configuration files (`config.yaml`, `model/`, `trainer/`).
- **`tests/`**: Lightweight Stage-0 style sanity checks for the refactored energy path.
- **`legacy/`**: Previous full training/experiment stack preserved as reference.

## Key Features

- **Autograd-first energy path**: No manual backward branch in `get/energy`; gradients come from PyTorch autograd.
- **Motif-aware energy**: Explicit higher-order motif branch with fused trilinear contraction.
- **Memory branch**: Hopfield-style global retrieval term.
- **Stage-0 sanity checks**: Gradcheck, permutation invariance, Armijo monotone descent, and branch-disable special-case tests.

## Getting Started

### Installation

```bash
# Create the environment (or use existing env `get`)
conda env create -f ENVIRONMENT.md
conda activate get
```

### Running Lightweight Tests

Use the refactored energy tests:

```bash
PYTHONPATH=. python -m pytest tests/test_energy.py tests/test_stage0_sanity.py -q
```

Run a tiny end-to-end refactor smoke:

```bash
PYTHONPATH=. python main.py trainer.epochs=2 dataset.num_train_graphs=16 dataset.num_val_graphs=8 dataset.num_test_graphs=8 experiment.device=cpu
```

Swap energy function from config:

```bash
PYTHONPATH=. python main.py model.energy_name=quadratic_only
```

Supported values: `get_full`, `pairwise_only`, `quadratic_only`.

Enable `torch.compile` (optional):

```bash
PYTHONPATH=. python main.py experiment.compile.enabled=true
```

Default scope is `experiment.compile.scope=eval_only`, which compiles eval path only and keeps training uncompiled (safer for solver training).
Use `experiment.compile.scope=all` to compile both train and eval paths.
For GET training, compile is guarded by default because solver training uses higher-order autodiff.
Set `experiment.compile.allow_double_backward=true` only if your backend supports it.

AMP is config-driven via `trainer.use_amp` and `trainer.amp_dtype` (`fp16` or `bf16`).

For older end-to-end training scripts, see `legacy/`.

## Running Tests

We use `pytest` for unit testing and verification of mathematical properties:

```bash
python -m pytest tests/
```
