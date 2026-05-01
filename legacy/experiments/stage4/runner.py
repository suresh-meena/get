from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shared.common import load_tu_dataset, save_results  # noqa: E402
from experiments.shared.plotting import load_and_plot  # noqa: E402
from experiments.shared.model_config import load_training_defaults  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Stage-2 runner for ET-style transfer tasks.")
    parser.add_argument("--task", choices=["graph_classification", "graph_anomaly"], required=True)
    parser.add_argument("--model_config", default="configs/models/catalog.yaml")
    parser.add_argument("--dataset", default="synth")
    parser.add_argument("--data_root", default="data/stage4")
    parser.add_argument("--num_graphs", type=int, default=120)
    parser.add_argument("--limit_graphs", type=int, default=0)
    parser.add_argument("--in_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--get_num_heads", type=int, default=4)
    parser.add_argument("--get_num_blocks", type=int, default=6)
    parser.add_argument("--cv_folds", type=int, default=1)
    parser.add_argument("--lambda_3", type=float, default=0.5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[123])
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--cache_processed", action="store_true", default=False)
    parser.add_argument("--cache_dir", default=".cache/get_data")
    parser.add_argument("--max_motifs", type=int, default=16)
    parser.add_argument("--anomaly_node_budget", type=int, default=2048)
    parser.add_argument("--anomaly_batch_cap", type=int, default=16)
    parser.add_argument("--anomaly_motif_budget", type=int, default=4096)
    parser.add_argument("--anomaly_motif_cap", type=int, default=8)
    parser.add_argument("--get_pe_k", type=int, default=0)
    parser.add_argument("--rwse_k", type=int, default=0)
    parser.add_argument("--ego_hops", type=int, default=1)
    parser.add_argument("--ego_limit", type=int, default=0)
    parser.add_argument("--ego_node_cap", type=int, default=512)
    parser.add_argument("--get_pairwise_et_mask", action="store_true", default=False)
    parser.add_argument("--get_norm_style", choices=["standard", "et"], default="et")
    parser.add_argument("--anomaly_label_rates", nargs="+", type=float, default=[0.01, 0.4])
    parser.add_argument("--weighted_bce", action="store_true", default=False)
    parser.add_argument("--et_num_heads", type=int, default=4)
    parser.add_argument("--et_head_dim", type=int, default=64)
    parser.add_argument("--et_num_blocks", type=int, default=6)
    parser.add_argument("--et_pe_k", type=int, default=8)
    parser.add_argument("--et_mask_mode", choices=["sparse", "official_dense"], default="sparse")
    parser.add_argument("--et_node_cap", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    training_defaults = load_training_defaults(args.model_config)
    args.use_amp = training_defaults.get("use_amp", None)
    args.amp_dtype = training_defaults.get("amp_dtype", None)
    if args.et_node_cap <= 0:
        args.et_node_cap = None
    if args.task == "graph_anomaly" and not args.weighted_bce:
        args.weighted_bce = True
    if args.dataset != "synth" and args.task == "graph_classification":
        try:
            _ = load_tu_dataset("MUTAG", limit=2)
        except Exception:
            pass

    random.seed(args.seeds[0])
    torch.manual_seed(args.seeds[0])

    if args.task == "graph_classification":
        from experiments.stage4.classification import run_graph_classification

        payload = run_graph_classification(args)
        out_name = "stage2_graph_classification"
    else:
        from experiments.stage4.anomaly import run_graph_anomaly

        payload = run_graph_anomaly(args)
        out_name = "stage2_graph_anomaly"

    metadata = {**vars(args), "stage": "stage4", "task": args.task, "dataset": args.dataset}
    json_path = save_results(out_name, payload, metadata=metadata)
    print(f"Saved outputs/{out_name}.json")
    load_and_plot(json_path, title=f"Stage 4 / {args.task.replace('_', ' ').title()} / {args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
