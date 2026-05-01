from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TaskSpec:
    task_type: str  # binary | multiclass | regression
    stage: str


TASK_SPECS: Dict[str, TaskSpec] = {
    "stage1_wedge_triangle": TaskSpec(task_type="binary", stage="1"),
    "stage1_triangle_regression": TaskSpec(task_type="regression", stage="1"),
    "stage1_cycle_parity": TaskSpec(task_type="binary", stage="1"),
    "stage1_max3sat": TaskSpec(task_type="binary", stage="1"),
    "stage1_xorsat": TaskSpec(task_type="binary", stage="1"),
    "stage1_srg_discrimination": TaskSpec(task_type="binary", stage="1"),
    "stage2_csl": TaskSpec(task_type="multiclass", stage="2"),
    "stage2_brec": TaskSpec(task_type="binary", stage="2"),
    "stage3_zinc": TaskSpec(task_type="regression", stage="3"),
    "stage3_molhiv": TaskSpec(task_type="binary", stage="3"),
    "stage3_peptides": TaskSpec(task_type="regression", stage="3"),
    "stage3_peptides_func": TaskSpec(task_type="binary", stage="3"),
    "stage4_tu_classification": TaskSpec(task_type="multiclass", stage="4"),
    "stage4_yelpchi_anomaly": TaskSpec(task_type="binary", stage="4"),
    "stage4_amazon_anomaly": TaskSpec(task_type="binary", stage="4"),
    "stage4_tfinance_anomaly": TaskSpec(task_type="binary", stage="4"),
    "stage4_tsocial_anomaly": TaskSpec(task_type="binary", stage="4"),
}

