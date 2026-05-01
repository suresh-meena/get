# Synthetic Benchmark Results (Optimized Config)

This summary contains results from all synthetic experiments run with optimized settings (Armijo Solver, Lambda_3=10.0, 100 Epochs).

## stage1_wedge_triangle
| Model | Accuracy |
|---|---|
| fullget | 0.6410 |
| pairwiseget | 0.4103 |
| et | 0.3333 |
| gcn | 0.5128 |
| gat | 0.6667 |
| gin | 0.4872 |
| external_baseline | 0.5897 |


## stage1_triangle_regression
| Model | MAE (Lower is better) |
|---|---|
| fullget | 0.1354 |
| pairwiseget | 0.1697 |
| et | 0.1997 |
| gcn | 0.1809 |
| gat | 0.1425 |
| gin | 0.1059 |
| external_baseline | 0.0916 |


## stage1_cycle_parity
| Model | Accuracy |
|---|---|
| fullget | 0.4615 |
| pairwiseget | 0.5385 |
| et | 0.4359 |
| gcn | 0.4359 |
| gat | 0.4615 |
| gin | 0.4359 |
| external_baseline | 0.4359 |


## stage1_max3sat
| Model | Accuracy |
|---|---|
| fullget | 1.0000 |
| pairwiseget | 1.0000 |
| et | 1.0000 |
| gcn | 1.0000 |
| gat | 1.0000 |
| gin | 1.0000 |
| external_baseline | 1.0000 |


## stage1_xorsat
| Model | Accuracy |
|---|---|
| fullget | 1.0000 |
| pairwiseget | 1.0000 |
| et | 1.0000 |
| gcn | 1.0000 |
| gat | 1.0000 |
| gin | 1.0000 |
| external_baseline | 1.0000 |


## stage1_srg_discrimination
| Model | Accuracy |
|---|---|
| fullget | 0.6410 |
| pairwiseget | 0.5128 |
| et | 0.5128 |
| gcn | 0.6667 |
| gat | 0.5385 |
| gin | 0.5385 |
| external_baseline | 0.6667 |


## stage2_csl
| Model | Accuracy (5-Fold Mean) |
|---|---|
| fullget | 0.1644 ± 0.0257 |
| pairwiseget | 0.0778 ± 0.0211 |
| et | 0.0778 ± 0.0211 |
| gcn | 0.0644 ± 0.0178 |
| gat | 0.0867 ± 0.0247 |
| gin | 0.0800 ± 0.0227 |
| external_baseline | 0.0644 ± 0.0178 |


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


