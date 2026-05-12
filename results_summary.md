# 📊 Graph Energy Transformer Benchmark Results

| Task | bwgnn | et | fullget | fullget (1%) | fullget (40%) | gt | pairwiseget | pairwiseget (1%) | pairwiseget (40%) | unknown |
| :--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage1_max3sat | - | - | - | - | - | - | - | - | - | 0.5000 (AUC) |
| stage1_srg_discrimination | - | - | - | - | - | - | - | - | - | 0.5450 (AUC) |
| stage1_triangle_regression | - | - | - | - | - | - | - | - | - | 0.3136 (MAE) |
| stage1_wedge_triangle | - | - | - | - | - | - | - | - | - | 0.6694 (AUC) |
| stage1_xorsat | - | - | - | - | - | - | - | - | - | 0.5000 (AUC) |
| stage2_csl | 0.1000 (Acc) | 0.1000 (Acc) | 0.1000 (Acc) | - | - | 0.1000 (Acc) | 0.1000 (Acc) | - | - | N/A |
| stage3_molhiv | 0.6745 (AUC) | 0.7265 (AUC) | 0.6340 (AUC) | - | - | 0.6463 (AUC) | 0.7168 (AUC) | - | - | 0.6557 (AUC) |
| stage3_peptides | - | - | - | - | - | - | - | - | - | 0.6179 (MAE) |
| stage3_peptides_func | - | - | - | - | - | - | - | - | - | 0.5394 (AUC) |
| stage3_peptides_func_probe | 0.6457 (AUC) | 0.6758 (AUC) | 0.5809 (AUC) | - | - | 0.5860 (AUC) | 0.6402 (AUC) | - | - | 0.6487 (AUC) |
| stage3_peptides_struct_probe | 0.6910 (MAE) | 0.3613 (MAE) | 0.7012 (MAE) | - | - | 0.6767 (MAE) | 0.6877 (MAE) | - | - | 0.7058 (MAE) |
| stage3_zinc | 1.2129 (MAE) | 0.9717 (MAE) | 1.2399 (MAE) | - | - | 1.0511 (MAE) | 1.3036 (MAE) | - | - | 1.3493 (MAE) |
| stage4_amazon_anomaly | 0.7157 (AUC) | 0.6996 (AUC) | - | 0.7368 (AUC) | 0.7507 (AUC) | 0.4396 (AUC) | 0.7255 (AUC) | 0.7107 (AUC) | 0.7324 (AUC) | 0.7170 (AUC) |
| stage4_tfinance_anomaly | 0.8830 (AUC) | - | - | 0.8867 (AUC) | - | 0.9119 (AUC) | 0.8975 (AUC) | 0.8874 (AUC) | 0.8993 (AUC) | 0.8511 (AUC) |
| stage4_tsocial_anomaly | 0.5495 (AUC) | 0.5259 (AUC) | 0.5665 (AUC) | 0.5471 (AUC) | 0.5774 (AUC) | 0.5091 (AUC) | 0.5657 (AUC) | 0.4573 (AUC) | 0.5607 (AUC) | 0.5562 (AUC) |
| stage4_tu_classification | 0.6552 (Acc) | 0.6552 (Acc) | 0.6552 (Acc) | - | - | 0.7586 (Acc) | 0.6897 (Acc) | - | - | 0.8276 (Acc) |
| stage4_tu_enzymes | - | 0.2308 (Acc) | 0.2198 (Acc) | - | - | 0.2308 (Acc) | 0.2198 (Acc) | - | - | - |
| stage4_tu_mutagenicity | - | 0.6488 (Acc) | 0.6319 (Acc) | - | - | 0.6196 (Acc) | 0.5629 (Acc) | - | - | - |
| stage4_tu_nci1 | - | 0.6440 (Acc) | 0.4854 (Acc) | - | - | 0.4984 (Acc) | - | - | - | - |
| stage4_tu_nci109 | - | 0.6387 (Acc) | 0.5435 (Acc) | - | - | 0.5032 (Acc) | - | - | - | - |
| stage4_tu_proteins | - | 0.7202 (Acc) | 0.6131 (Acc) | - | - | 0.5952 (Acc) | 0.5893 (Acc) | - | - | - |
| stage4_yelpchi_anomaly | 0.8830 (AUC) | - | - | 0.8867 (AUC) | - | 0.9119 (AUC) | 0.8975 (AUC) | 0.8874 (AUC) | 0.8993 (AUC) | 0.5000 (AUC) |
