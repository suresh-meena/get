# Architectural Review and Refactoring Guide: Graph Energy Transformer (GET)

## Executive Summary
The Graph Energy Transformer (GET) codebase suffers from significant technical debt, violating the KISS (Keep It Simple, Stupid) and DRY (Don't Repeat Yourself) principles. While the theoretical foundations outlined in the `main.tex` writeup are robust and mathematically sound, the implementation is heavily coupled, over-engineered in certain areas (e.g., manual gradient computations, monolithic training loops), and under-engineered in others (e.g., hardcoded hyperparameters, scattered `argparse` setups).

To ensure reproducible experiments, consistent baselines, high performance, and true modularity, the codebase requires a comprehensive refactoring. 

Below is a detailed breakdown of issues across the codebase and concrete, actionable suggestions for a full rewrite, with a strong emphasis on **Vectorization, Memory Efficiency, and Experiment Modularity**.

---

## 1. Experiment Configuration and Modularity
### Current Issues:
- **Rigid Stage Structure:** The codebase is split arbitrarily into `stage1`, `stage2`, `stage3`, and `stage4`. Adding a new experiment or removing an old one requires digging into these tightly coupled directories, copying boilerplate code, and creating new monolithic scripts.
- **Scattered `argparse` definitions:** Every script manually parses its own command-line arguments. There's no unified entry point.
- **Magic Numbers:** Hardcoded values like `pe_k=16`, `rwse_k=20`, `chunk_size = 131072`, and `hidden_dim=512` are buried within experimental scripts and model definitions.
- **Missing Protocol Gates:** The writeup dictates a strict sequence of experimental gates (Stage 0 sanity checks -> Stage 1 mechanism diagnostics -> Stages 2-4 transfer), but the code lacks a structured way to run these progressively or enforce the matching of baselines.

### Recommendations:
- **Adopt a Config Management Library (Hydra):** Every experiment should be defined purely by a YAML config. This eliminates magic numbers and centralizes hyperparameters.
- **Plug-and-Play Experiment Architecture:** Abandon the `stageX` directory structure. Instead, create a generic `main.py` entry point. An "experiment" should simply be a Hydra configuration file (e.g., `configs/experiment/zinc_regression.yaml`) that specifies the dataset, model, and trainer.
- **Task-Conditioned Wrappers:** As per the paper, the core GET mechanism must remain pure. We need modular "wrappers" that handle task-specific additions (like CLS tokens, LapPE, or bond-aware inputs) *only* when the experiment configuration demands it. 
- **Automated Sanity Checks (Stage 0):** Implement a dedicated `tests/` or `stage0/` pipeline that automatically verifies finite-difference gradients, permutation equivariance, and monotone decrease under accepted Armijo steps before any full training run.

---
## 2. Vectorization and Loop Elimination
### Current Issues:
- **Python-Level Loops in Critical Paths:** Several areas of the codebase rely on Python `for` loops where tensor vectorization is possible. For instance, evaluating multiple batches, computing metrics per graph, or iterating over motif chunks in `get/energy/motif.py` (`for start in range(0, motif_count, chunk_size):`).
- **Manual Metric Computation:** Metrics are computed manually inside `for` loops using list appends and `scikit-learn` aggregations (e.g., `accuracy_score`, `roc_auc_score`). This causes GPU-CPU sync bottlenecks and destroys performance.

### Recommendations:
- **Vectorize Everything:** Replace iterative chunking and Python loops with pure, vectorized PyTorch operations. If memory limits dictate chunking, leverage PyTorch's `vmap` or `unfold` semantics rather than manual indexing loops.
- **Use TorchMetrics:** Replace manual `scikit-learn` aggregations with `torchmetrics`. `torchmetrics` computes metrics directly on the GPU in a fully vectorized, batch-wise manner, eliminating the need to accumulate lists of predictions and transfer them to the CPU.

---

## 3. Memory Access and Computational Efficiency
### Current Issues:
- **Inefficient Hadamard Products and Memory Layouts:** The motif energy computation involves terms like `(K3[..., u_3, :, :] * K3[..., v_3, :, :] + T_tau_selected)`. Unoptimized, chained element-wise operations like this create massive intermediate tensors in memory. If the tensors are not contiguous, memory access becomes a severe bottleneck.
- **Redundant Graph Processing:** `_process_one_graph` recomputes CSR representations, incidence matrices, and structural features for the same graphs repeatedly if caching is disabled. The current caching loads entire processed datasets into RAM, which will OOM on large datasets.

### Recommendations:
- **Operator Fusion (via `torch.compile`):** Apply `torch.compile` to the energy functions. This will automatically fuse the pointwise operations (like the Hadamard product and addition in the motif branch) into a single GPU kernel, avoiding the allocation of intermediate memory buffers and drastically speeding up memory access.
- **Contiguous Memory:** Ensure that tensors accessed via advanced indexing (like `K3[..., u_3, :, :]`) are stored in a contiguous memory layout. Pre-allocate output buffers where applicable.
- **Efficient Feature Caching:** Move away from eagerly caching dicts in memory (`CachedGraphDataset`) toward PyTorch Geometric's robust `InMemoryDataset` or `Dataset` classes, processing and saving `.pt` files to disk once.

---

## 4. Readability and Model Purity
### Current Issues:
- **Manual Gradient Computations:** `get/energy/core.py` maintains two separate classes: `GETEnergy` and `GETEnergyWithGrad`. `GETEnergyWithGrad` manually computes the analytical pullback of the energy function. This code is dense, highly unreadable, duplicates the forward pass logic, and violates KISS.
- **Solver Logic in Model:** `GETModel` contains `_run_fixed_solver` and `_run_armijo_solver`. The inference dynamics (Armijo backtracking) are baked into the neural network `nn.Module`.
- **Bloated `__init__` Methods:** `GETLayer` and `GETModel` take over 30 arguments each.
- **Reinventing the Wheel:** The codebase sometimes implements custom operations or layers where standard PyTorch components would suffice and be better optimized.

### Recommendations:
- **Use Standard `torch.nn` Modules:** Wherever possible, replace custom layers or ad-hoc operations with standard `torch.nn` modules or `torch.nn.functional` equivalents. PyTorch's built-in modules are highly optimized (often backed by C++/CUDA) and inherently more readable. Avoid writing "random" custom code for standard operations.
- **Functional Purity (Inspired by JAX):** The model should act as a pure function that simply defines the energy $E(X)$. The dynamics updates ($X_{t+1} = X_t - \alpha \nabla_X E$) should be unrolled outside the core network structure using PyTorch's `autograd.grad` or `torch.func.grad`.
- **Rely on PyTorch Autograd:** Drop the manual backward pass (`GETEnergyWithGrad`). Trust PyTorch's autograd engine, combined with `torch.compile`, to generate optimized backward kernels automatically. This will cut the model code size in half and vastly improve readability.
- **Decouple Solvers:** Create a `get/solvers/` module. Implement `FixedStepSolver` and `ArmijoSolver` as separate entities that wrap the pure energy model.

---

## 5. Training and Evaluation Loops
### Current Issues:
- **Code Duplication:** There is a `GETTrainer` in `experiments/shared/common.py`. However, `experiments/stage4/shared.py` ignores it and implements its own `_train_graph_classification` and `_train_graph_binary_with_val` functions, leading to thousands of lines of duplicated logic.
- **Suboptimal AMP Usage:** Automatic Mixed Precision (AMP) is enabled/disabled via complex, manual `if` statements and `torch.autocast` blocks scattered throughout the loops.

### Recommendations:
- **Standardize on a Clean, Pure PyTorch Trainer:** Build a single, highly readable `Trainer` class in pure PyTorch. Pass the instantiated model, optimizer, loss function, and TorchMetrics object to it. 
- **Centralize AMP:** Handle `torch.autocast` and `GradScaler` cleanly within this single `Trainer` class. Remove all stage-specific duplicate loops.

---

## 6. Proposed Refactored Architecture
A strict adherence to modularity, readability, and performance would result in the following directory structure:

```text
code/
├── configs/                   # Hydra YAML configs (The single source of truth)
│   ├── experiment/            # e.g., stage1_sat.yaml, stage3_zinc.yaml
│   ├── model/                 # fullget.yaml, pairwiseget.yaml, gin.yaml
│   ├── dataset/               # dataset hyperparams (e.g., batch_size, transforms)
│   └── trainer/               # epochs, lr, max_grad_norm, etc.
├── get/
│   ├── data/
│   │   ├── datasets/          # Dataset loaders (ZINC, OGB, TU)
│   │   ├── transforms/        # PyG-style BaseTransforms (RWSE, LapPE, Motifs)
│   │   └── collator.py        # GETBatch collate_fn
│   ├── energy/                # Pure, readable energy scalar functions (E_quad, E_att, E_hn)
│   ├── models/                # Clean nn.Module definitions defining ONLY the architecture
│   ├── solvers/               # Unrolled inference dynamics (Armijo, FixedStep)
│   └── utils/                 # Logging, registry, graph utils
├── main.py                    # The single entry point utilizing Hydra
```

## 7. Concrete Action Plan for the Rewrite
1. **Unify Configuration:** Introduce Hydra. Move all hyperparameter dictionaries and `argparse` setups into `configs/`. 
2. **Refactor Data Pipeline:** Write a PyG `BaseTransform` for Motif Extraction and Positional Encodings. Apply this transform to datasets cleanly.
3. **Purify the Model & Decouple the Solvers:** Remove manual gradients (`GETEnergyWithGrad`). Extract `_run_armijo_solver` and `_run_fixed_solver` out of `GETModel`. Implement task-conditioned wrappers to keep the core GET model clean.
4. **Standardize the Trainer:** Build a single, vectorized `Trainer` class using `torchmetrics`. Remove all the `stageX` directories.
5. **Optimize with `torch.compile`:** Apply `torch.compile` to the unrolled solver loop to fuse operations (like the motif Hadamard product) for maximum memory efficiency.
6. **Implement Stage 0 Sanity Checks:** Build unit tests to strictly verify gradient correctness, permutation equivariance, and Armijo monotonicity to guarantee the mathematical claims in the paper.
7. **Ablation Support:** Ensure the config system makes it trivial to ablate features (e.g., toggling the memory bank, switching between open/closed motif support, or dropping the motif branch to run the `PairwiseGET` baseline with identically matched capacity).
8. **Build a Robust, Independent Test Suite (From Scratch):** Discard the current fragile tests. Build a comprehensive `pytest` suite from scratch that mathematically validates the energy functions. This must include:
    - **Finite Difference Checks:** Rigorously test the analytical gradients of all energy branches (pairwise, motif, memory, layernorm) against numerical approximations using `torch.autograd.gradcheck`.
    - **Equivariance Tests:** Assert that permuting the nodes of the input graph yields the exact same permuted output states and identical scalar energy.
    - **Monotonicity Guarantees:** Provide strict tests proving that accepted Armijo steps monotonically decrease the global energy function, confirming Theorem 1 of the paper.
    - **Special Case Verification:** Test that disabling the motif and memory branches (`lambda_3=0`, `lambda_m=0`) perfectly reduces the network to the expected ET/Hopfield limits as described in Table 1.