# ET Audit and Reproduction Plan

Scope:
- Compare the in-repo native ET implementation against the official JAX reference in `external/energy-transformer-graph`.
- Check which parts of the ET paper are actually reproduced here, and which parts are only ET-shaped.
- Identify the exact experiment settings needed for a trustworthy apples-to-apples comparison between ET and GET.
- Be skeptical: if a component is only “similar enough”, call it out.

Bottom line:
- The repo already has an ET-like PyTorch model in `get/models/et_classifier.py`.
- That model is not yet a faithful port of the official graph ET stack.
- The current graph runners can instantiate ET, but they do not yet reproduce the official ET preprocessing, tokenization, output path, or experiment protocol exactly.
- The safest path is to keep the official ET scaffold, then make the core energy block pluggable so GET can replace the ET core without losing ET’s outer experiment structure.

## 1. What the official ET graph code actually does

Main reference files:
- `external/energy-transformer-graph/src/model/core.py`
- `external/energy-transformer-graph/src/model/et.py`
- `external/energy-transformer-graph/src/graph_utils/tools.py`
- `external/energy-transformer-graph/src/graph_utils/tools_utils.py`
- `external/energy-transformer-graph/nbs/eval_tu.ipynb`
- `external/energy-transformer-graph/nbs/zinc.ipynb`
- `external/energy-transformer-graph/nbs/dd.ipynb`
- `external/energy-transformer-graph/nbs/mnist.ipynb`
- `external/energy-transformer-graph/nbs/cifar10.ipynb`

The official ET graph model is not just an energy block.
It is a full pipeline:

1. Input encoding
2. CLS token prep
3. Positional embedding prep
4. Graph correlation / adjacency preprocessing
5. Recurrent ET block iterations
6. Decoder
7. Task-specific head / loss

The key architectural pieces from the reference are:

- `EnergyLayerNorm` is a forward activation, not a trainable full `nn.LayerNorm`.
- The attention energy uses learned Q/K projections, per-head beta, and head mixing via `Hw`.
- The Hopfield channel is separate from attention and can be `relu`, `gelu`, or `lse`.
- The graph ET forward pass uses a CLS token and returns CLS output separately from node outputs.
- The official graph notebooks use positional embeddings derived from graph structure, not just raw node features.
- The official code can add stochasticity during recurrent updates (`noise_std`, `vary_noise`).

Important consequence:
- If a native port skips CLS token prep, positional embeddings, or the official output path, it is not the same benchmark even if the core energy equations look similar.

## 2. What the current native ET implementation does

Main file:
- `get/models/et_classifier.py`

Current status:
- There is a native PyTorch ET wrapper.
- It has an ET-style `EnergyLayerNorm`.
- It has a graph-masked attention energy and a Hopfield channel energy.
- It has stacked recurrent blocks.
- It is wired into the graph runners under `et` and `etfaithful`.

What it currently gets right:
- Uses a scalar/bias-style `EnergyLayerNorm` in the ET spirit.
- Uses per-head query/key projections and head mixing.
- Uses a Hopfield-style channel term with `relu`, `gelu`, or `lse`.
- Uses a recurrent update loop with gradient descent on an energy.

What it does not match yet:
- No CLS token path.
- No positional embedding preprocessing.
- No official graph correlation preprocessing.
- No decoder stage like the reference.
- No official noise injection path.
- No direct `get_graph`-style data utility layer.
- No faithful `return_stats`/evaluation structure matching the notebooks.
- It mean-pools graph embeddings instead of using the CLS token output path.

That last point matters a lot.
The official ET graph notebooks classify from the CLS token, not from mean pooled node states.

## 3. Fidelity gap table

| Area | Official ET reference | Native ET now | Risk level | What to do |
|---|---|---:|---:|---|
| Tokenization | CLS token + node tokens + positional embeddings | Only node features | High | Add ET-style token prep before the recurrent loop |
| Positional info | Laplacian/SVD positional embeddings from `get_pos` | None | High | Add a native PyTorch port of `get_pos` or equivalent |
| Graph correlation | `get_graph` builds dense adjacency / edge attrs and correlation input | Uses current batch edge incidence only | Medium | Add a dedicated ET graph-prep utility layer |
| Recurrent update | `x <- x - alpha * grad + noise` inside block/depth loops | Deterministic update only | Medium | Add optional noise / `vary_noise` / `noise_std` |
| Output head | Decoder returns CLS output and node outputs separately | Mean pool + readout | High | Switch to CLS readout for the ET benchmark path |
| Layer norm | Scalar gamma + optional bias | Mostly matches | Low | Keep, but verify exact init and bias semantics |
| Hopfield channel | ReLU / GELU / LSE variants | Present | Low | Good enough, but expose the same knobs explicitly |
| Attention energy | Q/K, per-head beta, head mixing `Hw` | Present | Medium | Verify exact masking and adjacency semantics on a tiny reference case |
| Bias controls | Separate flags for attn / chn / norm bias | Partial | Medium | Add the missing ET bias switches |
| Training loop | Notebook-driven JAX training / eval protocol | Unified PyTorch trainer | Medium | Match the official split and metric protocol per dataset |
| Evaluation protocol | 10-fold TU evaluation, anomaly detection with 1% and 40% train ratios, repeated runs | Current runners use broader GET-style protocol | High | Add ET-specific experiment scripts and config presets |

## 4. Critical mismatches in the current ET port

### 4.1 Mean pooling is not the ET graph classifier output path

Current code:
- `get/models/et_classifier.py:339-376`

Official reference:
- `external/energy-transformer-graph/src/model/et.py:142-170`

The reference ET graph pipeline adds a CLS token, runs recurrent updates, then decodes and returns the CLS output separately.
The current native ET model instead mean-pools node states and applies a readout.

This is not a harmless detail.
If you benchmark mean-pool ET against CLS-based ET, you are measuring a different model, not just a different implementation.

### 4.2 The reference uses ET graph preprocessing utilities that are not ported yet

Reference utilities:
- `external/energy-transformer-graph/src/graph_utils/tools.py:22-173`
- `external/energy-transformer-graph/src/graph_utils/tools_utils.py:1-121`

These utilities do several things the native path does not:

- convert graph batches to dense adjacency
- pad graphs to a fixed node cap
- add CLS connectivity to adjacency
- compute positional embeddings from the graph structure
- optionally include edge attributes
- split tensors across devices for the JAX notebook pipeline

The native PyTorch ET path currently consumes the GET batch format instead.
That is convenient, but it is not the same data contract.

### 4.3 The native ET path is missing several official ET knobs

Reference ET knobs:
- `use_biases_attn`
- `use_biases_chn`
- `use_biases_norm`
- `compute_corr`
- `vary_noise`
- `noise_std`
- `kernel_size`
- `kernel_dilation`
- `depth`
- `block`

Native ET currently exposes only a subset of these.

That means the model is not yet a complete port of the reference behavior.

### 4.4 Current tests only prove the model runs, not that it is faithful

Current ET test coverage:
- `tests/test_et_model.py:27-157`

Those tests verify:
- a forward pass works
- bias parameters exist
- one update rule matches the current code
- head mixing changes the energy
- the `etfaithful` alias wires into the runner

What they do not prove:
- CLS token behavior
- positional embedding behavior
- official ET graph preprocessing
- exact dataset protocol parity
- official anomaly / TU / ZINC notebook settings
- exact matching of reference gradients or energy traces

So the tests are useful smoke checks, but they are not a fidelity certificate.

## 5. Graph experiments from the ET reference

### 5.1 Paper-level graph experiments

From the paper abstract and appendix:
- graph anomaly detection
- graph classification

The paper’s anomaly detection setup uses:
- Yelp
- Amazon
- T-Finance
- T-Social
- 100 epochs
- Adam optimizer with learning rate 0.001
- best validation Macro-F1 for checkpoint selection
- training ratios of 1% and 40%
- 5 runs for reporting

The paper text also says large anomaly datasets use subgraph sampling per epoch to speed up training.

### 5.2 Official repo notebook suite

The official repo contains more than the paper abstract implies:

- `nbs/eval_tu.ipynb`
- `nbs/dd.ipynb`
- `nbs/zinc.ipynb`
- `nbs/mnist.ipynb`
- `nbs/cifar10.ipynb`
- plus eval variants

The TU notebook covers:
- MUTAG
- DD
- FRANKENSTEIN
- NCI1
- NCI109
- ENZYMES
- PROTEINS

The graph notebooks also show:
- TU evaluation with `use_node_attr=True`
- repeated n-fold evaluation
- ZINC with warmup cosine decay and AdamW
- image-to-graph tasks like MNIST and CIFAR10

Important skepticism:
- The repo notebook suite is broader than the paper narrative.
- Not every notebook is equally important for a GET-vs-ET comparison.
- If your goal is graph comparison only, TU + anomaly + ZINC are the core set.

## 6. What the current GET code path does instead

Relevant files:
- `get/models/energy_classifier.py`
- `get/energy/core.py`
- `get/solvers/gradient.py`
- `experiments/protocol/modeling.py`
- `experiments/run_graph_tasks.py`
- `experiments/protocol/data.py`

GET is structurally different:

- The energy is decomposed into quadratic, pairwise, motif, and memory branches.
- The solver is separated from the energy function.
- The current runners use GET-style graph batches with motif metadata.
- The default graph pipelines are tuned for GET, not for the ET reference tokenization.

That is fine for GET.
It is not fine if the goal is a faithful ET baseline.

## 7. Recommended design if ET should “mix well” with GET

The cleanest architecture is:

### 7.1 Keep the ET scaffold

Reuse the ET outer structure:
- encoder
- ET-style normalization
- graph token prep
- CLS token
- positional embeddings
- recurrent update loop
- decoder / readout
- ET experiment presets

### 7.2 Make the core pluggable

Use a core interface like:

- `energy(g, graph_context) -> scalar`
- `energy_and_grad(g, graph_context, create_graph=False) -> scalar, grad`

Then implement at least two cores:

- `OfficialETCore`
- `GETCore`

That gives you:
- a faithful ET baseline
- a hybrid “ET scaffold + GET core” comparison
- a path to compare architecture vs core contribution separately

### 7.3 Do not collapse ET into GET’s batch format

If you want apples-to-apples with the official ET reference, do not force ET to use the GET motif batch as its primary input contract.

Instead:
- port the ET utilities
- preserve the CLS token + positional embedding route
- then decide whether GET’s core can be dropped into that scaffold

That keeps the comparison honest.

## 8. Exact experiment parity checklist

### 8.1 Required for a trustworthy ET baseline

1. Port the ET data utilities
   - `get_graph`
   - `get_pos`
   - `batchify`
   - `to_device_split`
   - CLS-aware adjacency prep

2. Make the ET model output path match the reference
   - CLS token output
   - decoder head
   - optional node outputs if needed

3. Expose the full ET knobs
   - bias flags
   - noise flags
   - correlation mode
   - depth/block settings
   - kernel size / dilation

4. Add dataset-specific experiment scripts
   - TU 10-fold CV
   - anomaly 1% and 40% train ratios
   - ZINC regression
   - optional MNIST/CIFAR10 notebook parity if you want the full repo suite

5. Match the official selection metric
   - TU: classification accuracy / CV summary as used in the notebook
   - anomaly: validation Macro-F1
   - ZINC: MAE

6. Match the optimizer schedule where relevant
   - anomaly paper text says Adam lr 0.001
   - notebooks show AdamW + warmup cosine in some cases
   - do not silently mix those protocols

### 8.2 What should be compared between ET and GET

For each dataset / task:
- best validation metric
- final test metric
- number of parameters
- wall clock time
- solver iterations
- memory usage
- convergence stability

If you only compare accuracy, you miss the performance / stability story.
If you only compare speed, you miss whether the implementation is actually the same model.

## 9. Current runner issues to fix before calling ET “done”

These are the biggest practical issues in the repo today:

1. The native ET model still mean-pools instead of using CLS.
2. The ET preprocessing utilities are not fully ported.
3. The runner config files still have unresolved ETFaithful placeholders.
4. The existing ET tests do not check reference parity.
5. The current graph tasks use GET-centric batch semantics.
6. The protocol runner uses a different split regime from the official TU notebook.

Relevant files:
- `get/models/et_classifier.py:339-376`
- `external/energy-transformer-graph/src/model/et.py:142-170`
- `external/energy-transformer-graph/src/graph_utils/tools.py:117-173`
- `configs/models/catalog.yaml:35-50`
- `configs/models/stage4.yaml:35-50`
- `experiments/protocol/data.py:522-571`
- `tests/test_et_model.py:27-157`

## 10. Skeptical conclusion

The current ET work is a good start, but I would not yet call it a complete benchmark-quality ET port.

What is real:
- there is a native ET-like PyTorch model
- it can run
- it can be selected from the runners
- it uses an ET-style energy update

What is still missing:
- official graph ET preprocessing
- CLS-based output path
- exact notebook-level experiment reproduction
- complete ET knob coverage
- enough fidelity tests to trust cross-model benchmarking

Best next move:
- port the official ET utilities first
- make the scaffold pluggable
- then decide whether GET should replace the ET core or whether you want both cores under the same ET wrapper

That is the only way the final comparison will be honest.

## 12. Status Update After This Pass

Implemented in the native codebase:

- ET graph utilities now exist in `get/models/et_utils.py`.
- ETGraphClassifier now uses a CLS-token readout path by default.
- The ET block supports dynamic graph correlation, optional noise, and the missing bias toggles.
- The ET runner/config paths now accept the ET-specific parameters directly.
- Broken ET config placeholders were removed from the catalog files.
- ET tests now check:
  - EnergyLayerNorm formula parity
  - segment logsumexp correctness
  - CLS-token graph context construction
  - deterministic positional embeddings
  - ET config placeholder cleanup

Still intentionally not claimed as exact:

- full JAX notebook-by-notebook reproduction of every training curve
- exact ET graph correlation conv path from the reference image/video notebooks
- non-graph ET experiments outside the graph benchmark scope

So the current state is:
- faithful enough to compare ET-style architecture against GET in-repo
- much less misleading than the prior mean-pool / token-free version
- still not a byte-for-byte port of every external notebook experiment

## 13. Alignment Pass Completed

The remaining graph-side alignment gap was closed by switching the native ET path onto a dense adjacency / CLS-token / correlation-projection pipeline.

In practical terms:

- ET graph contexts now carry dense adjacency with CLS padding.
- ET blocks now consume dense attention masks instead of only sparse edge lists.
- Correlation is applied once from the initial token states, matching the reference notebook flow more closely.
- ET now exposes the same class of knobs that matter for the graph reference:
  - CLS token usage
  - positional embedding type
  - correlation toggle
  - noise toggle
  - attention/channel/norm bias toggles
  - block/depth count

Verification after this pass:
- `tests/test_et_model.py`
- `tests/test_energy_modularity.py`
- `tests/test_graph_tasks_runner.py`
- `tests/test_refactor_main.py`

All passed in the current environment.

## 11. Concrete implementation checklist

This is the order I would use if the goal is a real benchmark rather than a demo.

### Phase 1: Freeze the ET contract

1. Decide the ET-native input contract.
   - Preferred: mirror the official graph notebook contract, not the GET motif batch.
   - Required fields should include node features, adjacency/correlation input, CLS token path, and positional embeddings.

2. Port the official graph preprocessing utilities.
   - `get_graph`
   - `get_pos`
   - `batchify`
   - `to_device_split`
   - CLS-aware adjacency prep

3. Add a small adapter layer so ET can still consume the repo’s existing graph datasets.
   - Do not force GET’s motif metadata into ET if the reference does not use it.
   - Keep the adapter one-way and explicit.

Acceptance tests:
- A tiny graph batch reproduces the official token shapes.
- Positional embeddings are deterministic for a fixed seed.
- CLS token prep is present in the tensor path.

### Phase 2: Make the ET core faithful

4. Keep `EnergyLayerNorm` ET-style.
   - scalar gamma
   - optional bias
   - last-dimension normalization

5. Expose all ET knobs that matter for the graph code.
   - attention bias flag
   - channel bias flag
   - norm bias flag
   - `depth`
   - `block`
   - `noise_std`
   - `vary_noise`
   - `compute_corr`
   - `kernel_size`
   - `kernel_dilation`

6. Match the recurrent update semantics.
   - gradient step on the energy
   - optional noise injection
   - no accidental GET-style solver substitution unless you are intentionally running the hybrid benchmark

Acceptance tests:
- EnergyLayerNorm matches the closed-form formula exactly on a toy tensor.
- Attention head mixing changes the energy on a deterministic tiny graph.
- Segment logsumexp matches a naive grouped reference implementation.

### Phase 3: Build two benchmark modes

7. Native ET benchmark mode.
   - Uses the official ET scaffold.
   - Uses the ET core.
   - Uses ET experiment settings from the notebooks.

8. Hybrid ET/GET benchmark mode.
   - Uses the ET scaffold.
   - Replaces only the core with GET if you want to test whether GET’s energy design helps.
   - This should be explicitly named so nobody confuses it with the official ET baseline.

Acceptance tests:
- The benchmark runner can select both modes explicitly.
- The logs state whether the run is `et`, `hybrid_et_get`, or `get`.

### Phase 4: Reproduce the paper experiments

9. TU graph classification.
   - Use the TU datasets listed in the notebook.
   - Match the n-fold evaluation protocol.
   - Preserve the official node-attribute handling.

10. Graph anomaly detection.
   - Yelp
   - Amazon
   - T-Finance
   - T-Social
   - 1% and 40% training ratios
   - best validation Macro-F1 checkpointing
   - repeated runs

11. ZINC.
   - Use the notebook’s training settings.
   - Keep the regression metric separate from classification metrics.

Optional later:
- MNIST / CIFAR10 notebook parity if you want the full official suite, but do not block the graph benchmark on those.

### Phase 5: Benchmark discipline

12. Record both accuracy and runtime.
   - Accuracy / AUC / F1 / MAE depending on task
   - wall clock per epoch
   - solver iteration count
   - GPU memory if available

13. Keep the comparison symmetric.
   - same split
   - same optimizer
   - same batch size
   - same early-stopping policy
   - same number of parameter updates

14. Add regression tests for the experiment scripts themselves.
   - the CLI must instantiate the right model
   - the config must not silently ignore ET parameters
   - the runner must fail fast on unresolved placeholders

### Definition of done

I would call the ET port “done” only when all of these are true:

- The ET scaffold reproduces the official graph notebook tensor flow.
- The ET core matches the reference equations on toy cases.
- The benchmark runner can run ET, hybrid ET/GET, and GET under the same harness.
- The TU and anomaly experiments match the official protocol closely enough that the difference between models is meaningful.
- The tests prove the invariants above, not just that the code imports and runs.

## 13. Hotspot replacement checklist

This appendix is narrower than the ET reproduction plan. It focuses on CPU-side and preprocessing hotspots where a library API or a cache boundary is the best optimization, not a custom rewrite.

### Replace

1. ET positional embedding generation in the forward path.
   - Current location: `get/models/et_utils.py`
   - Replace with:
     - offline preprocessing using `torch_geometric.transforms.AddLaplacianEigenvectorPE`, or
     - a cached dataset field populated once at load time
   - Why:
     - dense `torch.linalg.eigh` per batch is expensive
     - this is preprocessing, not model logic
   - Acceptance:
     - the model forward no longer recomputes eigenvectors for every batch

2. Manual dense adjacency fills.
   - Current locations:
     - `experiments/protocol/data.py`
     - `get/data/real_world.py`
   - Replace with:
     - `torch_geometric.utils.to_dense_adj` when dense adjacency is needed
   - Why:
     - less handwritten indexing
     - easier to reason about batched graph conversion
   - Acceptance:
     - adjacency construction becomes a one-line library call in the loader path

3. Custom batch collation where PyG `Batch` can represent the same sample.
   - Current location: `get/data/synthetic.py`
   - Replace with:
     - `torch_geometric.data.Data` + `torch_geometric.loader.DataLoader`
   - Why:
     - PyG already optimizes node/edge index shifting
     - less Python work during batching
   - Caveat:
     - only do this if the motif tensors and optional `pos` can live cleanly on `Data`

4. Repeated ego-graph extraction for anomaly tasks.
   - Current location: `experiments/protocol/data.py`
   - Replace with:
     - cached ego-graph samples keyed by dataset version, `ego_hops`, and `max_graphs`
   - Why:
     - the expensive part is repeated subgraph extraction, not the API call itself
   - Acceptance:
     - repeated runs reuse cached ego-graph samples

### Keep

5. `numba.njit` motif extraction.
   - Current location: `get/data/synthetic.py`
   - Keep because:
     - the custom nested-neighborhood logic is exactly the kind of thing Numba helps with
     - it is already doing real work off the Python interpreter

6. PyG scatter-based segmented reductions.
   - Current location: `get/energy/ops.py`
   - Keep because:
     - the code already relies on efficient PyG scatter kernels
     - the remaining custom logic is mostly orchestration around those kernels

7. The vectorized brute-force SAT/XORSAT checks for the current benchmark sizes.
   - Current location: `experiments/protocol/data.py`
   - Keep for now because:
     - `n_vars` is still small enough that the cached assignment-table approach is acceptable
   - Revisit if:
     - benchmark sizes increase materially

### Cache

8. Real-world dataset conversion.
   - Current location: `get/data/real_world.py`
   - Cache:
     - converted `sample_from_adj(...)` items
     - remapped labels
   - Why:
     - TU datasets are deterministic for a fixed version, so there is no reason to rebuild samples every epoch

9. Protocol-stage dataset builds.
   - Current location: `experiments/protocol/data.py`
   - Cache:
     - stage1 synthetic datasets
     - stage2/3/4 processed samples
   - Why:
     - this is already partially done via `protocol_cache/*.pkl`
     - the remaining requirement is to keep the cache key stable and versioned

10. ET graph preprocessing outputs.
    - Current location: `get/models/et_utils.py`
    - Cache:
      - positional embeddings
      - CLS-augmented graph context
    - Why:
      - these are graph-structure features, not per-step model state

### Do not replace yet

11. `torch.linalg.eigh` inside a true offline preprocessing pass.
   - Keep only if:
     - it moves out of the forward path and into a cache or dataset transform
   - Otherwise:
     - it is too expensive to leave in the model path

12. The recurrent solver / energy update itself.
   - Do not replace with a library convenience function unless:
     - you are intentionally changing the model semantics
   - The solver is the model; it is not just plumbing

### Practical order

If the goal is performance without semantic drift, do these first:
1. cache ET positional embeddings
2. cache anomaly ego-graphs
3. convert real-world datasets once in `__init__`
4. consider a PyG `Batch` refactor only if the custom collator still shows up in profiling
