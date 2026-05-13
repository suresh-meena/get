from __future__ import annotations

from argparse import Namespace

import torch

from experiments.run_protocol import _evaluate_brec_by_category
from get.data.synthetic import sample_from_adj


class _FakeTrainer:
    def _run_epoch(self, loader, train: bool = False):
        return {"loss": 0.1, "acc": 0.9, "auc": 0.8, "f1": 0.75}


def _make_sample(cat_id: int):
    adj = torch.tensor([[False, True], [True, False]])
    x = torch.zeros((2, 2), dtype=torch.float32)
    y = torch.tensor([1.0], dtype=torch.float32)
    s = sample_from_adj(adj, x, y, max_motifs_per_anchor=1)
    s.brec_category_id = torch.tensor([cat_id], dtype=torch.long)
    return s


def test_brec_category_reporting_groups_items() -> None:
    te_items = [_make_sample(0), _make_sample(0), _make_sample(1)]
    args = Namespace(
        batch_size=2,
        num_workers=0,
        _brec_category_names={0: "Basic", 1: "CFI"},
    )
    out = _evaluate_brec_by_category(_FakeTrainer(), args, te_items)
    assert "Basic" in out
    assert "CFI" in out
    assert out["Basic"]["acc"] == 0.9
    assert out["CFI"]["auc"] == 0.8
