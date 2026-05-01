from __future__ import annotations

import argparse
from pathlib import Path

from experiments.shared.plotting import load_and_plot


def _resolve_json_paths(root: Path, pattern: str) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Stage 4 plots from JSON outputs.")
    parser.add_argument("--task", choices=["graph_classification", "graph_anomaly", "all"], default="all")
    parser.add_argument("--classification_json", default="outputs/stage4")
    parser.add_argument("--anomaly_json", default="outputs/stage4")
    parser.add_argument("--classification_out", default=None)
    parser.add_argument("--anomaly_out", default=None)
    args = parser.parse_args()

    if args.task in ("graph_classification", "all"):
        classification_root = Path(args.classification_json)
        classification_paths = _resolve_json_paths(classification_root, "*graph_classification*.json")
        for classification_json in classification_paths:
            output_path = Path(args.classification_out) if args.classification_out is not None else None
            out = load_and_plot(classification_json, output_path=output_path, title="Stage 4 / Graph Classification")
            print(f"Saved {out}")

    if args.task in ("graph_anomaly", "all"):
        anomaly_root = Path(args.anomaly_json)
        anomaly_paths = _resolve_json_paths(anomaly_root, "*graph_anomaly*.json")
        for anomaly_json in anomaly_paths:
            output_path = Path(args.anomaly_out) if args.anomaly_out is not None else None
            out = load_and_plot(anomaly_json, output_path=output_path, title="Stage 4 / Graph Anomaly")
            print(f"Saved {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())