from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_ORDER = ["pairwise", "fullget", "et_local", "et_complete", "gin"]
MODEL_LABEL = {
    "pairwise": "PairwiseGET",
    "fullget": "FullGET",
    "et_local": "ET-Local",
    "et_complete": "ET-Complete",
    "gin": "GIN",
}
MODEL_COLOR = {
    "pairwise": "#1f77b4",
    "fullget": "#2ca02c",
    "et_local": "#9467bd",
    "et_complete": "#8c564b",
    "gin": "#ff7f0e",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _history_mean_std(runs: list[dict], model_key: str, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    curves = []
    for run in runs:
        histories = run.get("histories", {})
        model_hist = histories.get(model_key)
        if not model_hist:
            continue
        metric = model_hist.get(metric_key)
        if not metric:
            continue
        curves.append(np.asarray(metric, dtype=float))

    if not curves:
        return np.asarray([]), np.asarray([])

    min_len = min(len(c) for c in curves)
    if min_len == 0:
        return np.asarray([]), np.asarray([])

    stacked = np.stack([c[:min_len] for c in curves], axis=0)
    return stacked.mean(axis=0), stacked.std(axis=0)


def plot_stage2_graph_classification(json_path: Path, output_path: Path | None = None) -> Path:
    payload = _load_json(json_path)
    runs = payload.get("runs", [])

    if output_path is None:
        output_path = json_path.with_name("stage2_graph_classification.png")

    plt.figure(figsize=(12, 5))

    for subplot_idx, metric_key, title in [
        (1, "train_loss", "Training Loss"),
        (2, "val_acc", "Validation Accuracy"),
    ]:
        ax = plt.subplot(1, 2, subplot_idx)
        for model_key in MODEL_ORDER:
            mean_curve, std_curve = _history_mean_std(runs, model_key, metric_key)
            if mean_curve.size == 0:
                continue
            x = np.arange(mean_curve.shape[0])
            color = MODEL_COLOR[model_key]
            ax.plot(x, mean_curve, color=color, linewidth=2.2, label=MODEL_LABEL[model_key])
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2, linestyle="--")
        if subplot_idx == 1:
            ax.set_ylabel("Loss")
        else:
            ax.set_ylabel("Accuracy")
        ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def plot_stage2_graph_anomaly(json_path: Path, output_path: Path | None = None) -> Path:
    payload = _load_json(json_path)
    summary = payload.get("summary", {})

    if output_path is None:
        output_path = json_path.with_name("stage2_graph_anomaly.png")

    label_rates = sorted(float(rate) for rate in summary.keys())
    if not label_rates:
        raise ValueError("No anomaly summary found in JSON payload.")

    xs = np.arange(len(label_rates))
    width = 0.18
    model_offsets = {
        "pairwise": -1.5 * width,
        "fullget": -0.5 * width,
        "et_local": 0.5 * width,
        "et_complete": 1.5 * width,
    }

    plt.figure(figsize=(10, 5))
    for model_key, offset in model_offsets.items():
        means = []
        stds = []
        for rate in label_rates:
            s = summary[str(rate)]
            means.append(float(s[f"{model_key}_mean"]))
            stds.append(float(s[f"{model_key}_std"]))
        plt.bar(
            xs + offset,
            means,
            width=width,
            yerr=stds,
            capsize=4,
            color=MODEL_COLOR[model_key],
            label=MODEL_LABEL[model_key],
        )

    plt.xticks(xs, [f"{rate:.2f}" for rate in label_rates])
    plt.xlabel("Labeled anomaly rate")
    plt.ylabel("AUC")
    plt.ylim(0.0, 1.0)
    plt.title("Stage-2 Graph Anomaly")
    plt.grid(axis="y", alpha=0.2, linestyle="--")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Stage-2 plots from JSON outputs.")
    parser.add_argument("--task", choices=["graph_classification", "graph_anomaly", "all"], default="all")
    parser.add_argument("--classification_json", default="outputs/stage2_graph_classification.json")
    parser.add_argument("--anomaly_json", default="outputs/stage2_graph_anomaly.json")
    parser.add_argument("--classification_out", default="outputs/stage2_graph_classification.png")
    parser.add_argument("--anomaly_out", default="outputs/stage2_graph_anomaly.png")
    args = parser.parse_args()

    if args.task in ("graph_classification", "all"):
        out = plot_stage2_graph_classification(Path(args.classification_json), Path(args.classification_out))
        print(f"Saved {out}")

    if args.task in ("graph_anomaly", "all"):
        out = plot_stage2_graph_anomaly(Path(args.anomaly_json), Path(args.anomaly_out))
        print(f"Saved {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
