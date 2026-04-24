from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Apply seaborn publication-ready style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
sns.set_palette("deep")

MODEL_ORDER = ["fullget", "et_faithful", "pairwise", "gin"]
MODEL_LABEL = {
    "fullget": "FullGET",
    "et_faithful": "ETFaithful",
    "pairwise": "PairwiseGET",
    "gin": "GIN",
}
# Publication-friendly distinct colors
MODEL_COLOR = {
    "fullget": "#D55E00",      # Vermillion
    "et_faithful": "#0072B2",  # Blue
    "pairwise": "#E69F00",     # Orange
    "gin": "#009E73",          # Green
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    axes = axes.flatten()

    for subplot_idx, (metric_key, title) in enumerate([
        ("train_loss", "Training Loss"),
        ("val_acc", "Validation Accuracy"),
    ]):
        ax = axes[subplot_idx]
        for model_key in MODEL_ORDER:
            mean_curve, std_curve = _history_mean_std(runs, model_key, metric_key)
            if mean_curve.size == 0:
                continue
            x = np.arange(mean_curve.shape[0])
            color = MODEL_COLOR.get(model_key, "#333333")
            ax.plot(x, mean_curve, color=color, linewidth=2.5, label=MODEL_LABEL.get(model_key, model_key))
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15, linewidth=0)

        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel("Epoch", fontweight='bold')
        ax.grid(alpha=0.3, linestyle="--")
        
        # Despine
        sns.despine(ax=ax)
        
        if subplot_idx == 0:
            ax.set_ylabel("Loss", fontweight='bold')
            ax.legend(frameon=False, loc="upper right")
        else:
            ax.set_ylabel("Accuracy", fontweight='bold')
            ax.legend(frameon=False, loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
    width = 0.35
    model_offsets = {"fullget": -0.5 * width, "et_faithful": 0.5 * width}

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for model_key, offset in model_offsets.items():
        means = []
        stds = []
        for rate in label_rates:
            s = summary[str(rate)]
            means.append(float(s[f"{model_key}_mean"]))
            stds.append(float(s[f"{model_key}_std"]))
        
        ax.bar(
            xs + offset,
            means,
            width=width * 0.9,
            yerr=stds,
            capsize=5,
            color=MODEL_COLOR.get(model_key, "#333333"),
            label=MODEL_LABEL.get(model_key, model_key),
            edgecolor='black',
            linewidth=1.2,
            error_kw=dict(lw=1.5, capthick=1.5)
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{rate*100:.0f}%" for rate in label_rates])
    ax.set_xlabel("Labeled Anomaly Rate", fontweight='bold')
    ax.set_ylabel("ROC-AUC", fontweight='bold')
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Graph Anomaly Detection", fontweight='bold', pad=15)
    
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax, bottom=True)
    ax.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
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
        if Path(args.classification_json).exists():
            out = plot_stage2_graph_classification(Path(args.classification_json), Path(args.classification_out))
            print(f"Saved {out}")

    if args.task in ("graph_anomaly", "all"):
        if Path(args.anomaly_json).exists():
            out = plot_stage2_graph_anomaly(Path(args.anomaly_json), Path(args.anomaly_out))
            print(f"Saved {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())