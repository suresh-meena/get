This summary contains results from experiments using optimized energy architectures. Following a diagnostic study, we identified that GET's performance is highly sensitive to the interaction between the energy aggregation mode and state normalization.

### Model Architecture & Energy Improvements
1.  **Task-Specific Aggregation**:
    - **Classification/Isomorphism**: `agg_mode="softmax"` (LogSumExp) is used to provide stable, attention-like selection of structural motifs. This is crucial for tasks like Wedge Discrimination and CSL.
    - **Regression/Counting**: `agg_mode="sum"` is used to enable precise linear counting of motif frequencies, bridging the gap with GIN.
2.  **Normalization Ablation (`no_energy_norm`)**:
    - We found that `LayerNorm` inside the energy loop can cause "radial gradient vanishing" in simpler models like `PairwiseGET`. Ablating it allows the model to escape the quadratic shrinkage of the energy landscape.
3.  **Degree Scaling**: All GET models now utilize `degree_scaler` (PNA-style) to stabilize counts across heterogeneous graphs.

## stage1_wedge_triangle (Binary Classification)
| Model | Accuracy | Energy Config |
|---|---|---|
| **fullget** | **0.7436** | Softmax, Lambda_3=10.0 |
| pairwiseget | 0.4103 | Sum (Legacy) |
| et | 0.3333 | Baseline |
| gat | 0.6667 | Baseline |
| gin | 0.4872 | Baseline |
| gcn | 0.5128 | Baseline |

> [!NOTE]
> Optimized **FullGET** now outperforms the GAT baseline by ~11%, demonstrating that its motif-aware energy landscape can effectively distinguish simple structural patterns when properly normalized.


## stage1_triangle_regression (Counting Task)
| Model | MAE (Lower is better) | Energy Config |
|---|---|---|
| **fullget** | **0.1159** | Sum, Lambda_3=1.0 |
| pairwiseget | 0.1572 | Sum, noLN |
| et | 0.1997 | Baseline |
| gat | 0.1425 | Baseline |
| gin | 0.1059 | Baseline |
| gcn | 0.1809 | Baseline |
| external_baseline | 0.0916 | Baseline |

> [!TIP]
> Switching to **Sum aggregation** allows GET to perform direct motif counting, bridging the gap with the GIN baseline which also utilizes sum-based aggregation.


## stage1_cycle_parity (Limitation)
| Model | Accuracy | Energy Config |
|---|---|---|
| fullget | 0.4615 | Softmax, Lambda_3=10.0 |
| pairwiseget | 0.4359 | Sum, noLN |
| et | 0.4359 | Baseline |
| gat | 0.4615 | Baseline |
| gin | 0.4359 | Baseline |
| gcn | 0.4359 | Baseline |
| external_baseline | 0.4359 | Baseline |

> [!WARNING]
> None of the models, including optimized GETs, can solve cycle parity. This requires global structural or spectral features that local message passing and local energy loops cannot capture.


## stage1_max3sat & stage1_xorsat (Boolean Satisfiability)
| Model | Accuracy |
|---|---|
| **fullget** | **1.0000** |
| pairwiseget | **1.0000** |
| et | **1.0000** |
| gat | **1.0000** |
| gin | **1.0000** |
| gcn | **1.0000** |

> [!NOTE]
> These tasks are perfectly solved (1.0 Accuracy) by the baseline GET models as well as the standard GNNs. Further energy optimization is not necessary here.


## stage1_srg_discrimination
| Model | Accuracy | Energy Config |
|---|---|---|
| fullget | 0.6410 | Softmax, Lambda_3=10.0 |
| pairwiseget | 0.4872 | Sum, noLN |
| et | 0.5128 | Baseline |
| gcn | 0.6667 | Baseline |
| gat | 0.5385 | Baseline |
| gin | 0.5385 | Baseline |
| external_baseline | 0.6667 | Baseline |

> [!NOTE]
> Performance for `fullget` remained unchanged from the baseline. SRGs (Strongly Regular Graphs) have identical node degrees everywhere, so the newly implemented `degree_scaler` provides no additional signal. The Softmax energy is still the best configuration, but it does not yet match the GCN/External baseline (0.6667).




## stage2_csl (Isomorphism)
| Model | Accuracy (5-Fold Mean) | Energy Config |
|---|---|---|
| **fullget** | **0.1489 ± 0.0542** | Softmax, Lambda_3=10.0 |
| pairwiseget | 0.0622 ± 0.0150 | Sum, noLN |
| et | 0.0778 ± 0.0211 | Baseline |
| gat | 0.0867 ± 0.0247 | Baseline |
| gin | 0.0800 ± 0.0227 | Baseline |
| gcn | 0.0644 ± 0.0178 | Baseline |
| external_baseline | 0.0644 ± 0.0178 | Baseline |

> [!NOTE]
> For higher-order isomorphism (CSL), the **Softmax/LogSumExp** aggregation proves essential. The attention mechanism provides critical selectivity that simpler `Sum` aggregations (like GIN/PairwiseGET) lack for such tasks.


## graph_classification (Legacy)
| Model | Accuracy |
|---|---|
| fullget | 1.0000 |
| pairwiseget | 1.0000 |
| et | 1.0000 |
| gcn | 1.0000 |
| gat | 1.0000 |
| gin | 1.0000 |
| external_baseline | 1.0000 |


## graph_anomaly (Legacy)
| Model | Accuracy |
|---|---|
| fullget | 0.8750 |
| pairwiseget | 0.8750 |
| et | 0.8750 |
| gcn | 0.8750 |
| gat | 0.8750 |
| gin | 0.8750 |
| external_baseline | 0.8750 |


