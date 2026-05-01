from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


HISTORY_PRIORITY = [
    "train_loss",
    "val_acc",
    "val_auc",
    "val_metric",
    "val_loss",
    "test_acc",
    "test_auc",
    "test_loss",
    "metric",
    "accuracy",
    "f1",
]

LABELS = {
    "train_loss": "Training Loss",
    "val_acc": "Validation Accuracy",
    "val_auc": "Validation AUC",
    "val_metric": "Validation Metric",
    "val_loss": "Validation Loss",
    "test_acc": "Evaluation Accuracy",
    "test_auc": "Evaluation AUC",
    "test_loss": "Evaluation Loss",
    "metric": "Metric",
    "accuracy": "Accuracy",
    "f1": "F1",
    "energy_trace": "Energy",
}


@dataclass
class PlotSeries:
    label: str
    history: dict[str, list[float]]
    energy_trace: list[float] | None = None


def _as_float(value) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return float(array.item())
    return float(array.mean())


def _normalize_energy_trace(energy_trace) -> list[float] | None:
    if energy_trace is None:
        return None
    if isinstance(energy_trace, (list, tuple)):
        return [_as_float(item) for item in energy_trace]
    if isinstance(energy_trace, np.ndarray):
        if energy_trace.ndim == 0:
            return [float(energy_trace.item())]
        if energy_trace.ndim == 1:
            return [float(value) for value in energy_trace.tolist()]
        return [float(step.mean()) for step in energy_trace]
    try:
        return [float(value) for value in list(energy_trace)]
    except Exception:
        return None


def _label_from_parts(parts: Iterable[str]) -> str:
    cleaned = [part for part in parts if part and part not in {"runs", "folds"}]
    return " / ".join(cleaned) if cleaned else "run"


def _looks_like_history_dict(node: dict) -> bool:
    return any(key in node for key in HISTORY_PRIORITY)


def _collect_series(node, label_parts: tuple[str, ...] = ()) -> list[PlotSeries]:
    series: list[PlotSeries] = []

    if isinstance(node, dict):
        if "history" in node and isinstance(node["history"], dict):
            energy = node.get("energy_trace")
            if energy is None and isinstance(node.get("extra"), dict):
                energy = node["extra"].get("energy_trace")
            series.append(
                PlotSeries(
                    label=_label_from_parts(label_parts),
                    history=node["history"],
                    energy_trace=_normalize_energy_trace(energy),
                )
            )
            return series

        if _looks_like_history_dict(node):
            energy = node.get("energy_trace")
            series.append(
                PlotSeries(
                    label=_label_from_parts(label_parts),
                    history=node,
                    energy_trace=_normalize_energy_trace(energy),
                )
            )
            return series

        for key, value in node.items():
            if key in {"summary", "metadata", "task", "dataset", "results"}:
                continue
            if key == "histories" and isinstance(value, dict):
                for child_key, child_value in value.items():
                    series.extend(_collect_series(child_value, label_parts + (str(child_key),)))
                continue
            if key == "energy_traces" and isinstance(value, dict):
                for child_key, child_value in value.items():
                    series.extend(_collect_series(child_value, label_parts + (str(child_key),)))
                continue
            if key == "energy_trace":
                series.append(
                    PlotSeries(
                        label=_label_from_parts(label_parts),
                        history={},
                        energy_trace=_normalize_energy_trace(value),
                    )
                )
                continue
            if isinstance(value, (dict, list, tuple)):
                next_parts = label_parts if key in {"runs", "folds"} else label_parts + (str(key),)
                series.extend(_collect_series(value, next_parts))
        return series

    if isinstance(node, list):
        if node and all(not isinstance(value, (dict, list, tuple)) for value in node):
            series.append(
                PlotSeries(
                    label=_label_from_parts(label_parts),
                    history={},
                    energy_trace=[_as_float(value) for value in node],
                )
            )
            return series
        for value in node:
            series.extend(_collect_series(value, label_parts))
        return series

    return series


def _aggregate_curves(curves_by_label: dict[str, list[list[float]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    aggregated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, curves in curves_by_label.items():
        usable = [np.asarray(curve, dtype=np.float64) for curve in curves if hasattr(curve, "__len__") and len(curve) > 0]
        if not usable:
            continue
        min_len = min(len(curve) for curve in usable)
        if min_len <= 0:
            continue
        stacked = np.stack([curve[:min_len] for curve in usable], axis=0)
        aggregated[label] = (stacked.mean(axis=0), stacked.std(axis=0))
    return aggregated


def _metric_key(histories: list[dict[str, list[float]]]) -> str | None:
    keys = set()
    for history in histories:
        keys.update(history.keys())
    for key in HISTORY_PRIORITY:
        if key in keys and key != "train_loss":
            return key
    return None


def _metric_title(metric_key: str | None, fallback: str) -> str:
    if metric_key is None:
        return fallback
    return LABELS.get(metric_key, metric_key.replace("_", " ").title())


def plot_payload(payload: dict, output_path: Path, title: str | None = None) -> Path:
    if isinstance(payload, dict) and "results" in payload and isinstance(payload["results"], dict):
        payload = payload["results"]

    series = _collect_series(payload)
    if not series:
        raise ValueError("No histories or energy traces found in payload.")

    history_groups: dict[str, list[dict]] = {}
    energy_groups: dict[str, list[list[float]]] = {}
    for item in series:
        if item.history:
            history_groups.setdefault(item.label, []).append(item.history)
        if item.energy_trace:
            energy_groups.setdefault(item.label, []).append(item.energy_trace)

    # Detect primary metric
    all_histories = [h for group in history_groups.values() for h in group]
    eval_key = _metric_key(all_histories)
    
    # 2x2 Grid setup
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    palette = sns.color_palette("deep", n_colors=max(len(history_groups), len(energy_groups), 3))
    model_colors = {label: palette[i % len(palette)] for i, label in enumerate(sorted(history_groups.keys()))}

    # --- Plot 1: Train Loss (0,0) ---
    ax = axes[0, 0]
    for label, group in sorted(history_groups.items()):
        curves = [h["train_loss"] for h in group if "train_loss" in h]
        if curves:
            agg_dict = _aggregate_curves({label: curves})
            if label in agg_dict:
                mean, std = agg_dict[label]
                x = np.arange(1, len(mean) + 1)
                ax.plot(x, mean, label=label, color=model_colors[label], linewidth=2)
                if len(mean) > 1:
                    ax.fill_between(x, mean - std, mean + std, color=model_colors[label], alpha=0.1)
    ax.set_title("Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False, loc="upper right")

    # --- Plot 2: Validation Metric (0,1) ---
    ax = axes[0, 1]
    if eval_key:
        for label, group in sorted(history_groups.items()):
            curves = [h[eval_key] for h in group if eval_key in h]
            if curves:
                agg_dict = _aggregate_curves({label: curves})
                if label in agg_dict:
                    mean, std = agg_dict[label]
                    x = np.arange(1, len(mean) + 1)
                    ax.plot(x, mean, label=label, color=model_colors[label], linewidth=2)
                    if len(mean) > 1:
                        ax.fill_between(x, mean - std, mean + std, color=model_colors[label], alpha=0.1)
        ax.set_title(_metric_title(eval_key, "Validation Performance"), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(_metric_title(eval_key, "Metric"))

    # --- Plot 3: Final Test Performance Bar (1,0) ---
    ax = axes[1, 0]
    bar_labels, bar_values = [], []
    for label, group in sorted(history_groups.items()):
        # Try to find final test result in payload for this label
        # If not explicit, take last val_metric
        val = None
        if eval_key and group:
            last_curves = [h[eval_key] for h in group if isinstance(h, dict) and eval_key in h]
            if last_curves:
                val = np.mean([c[-1] for c in last_curves if hasattr(c, "__len__")])
        
        if val is not None:
            bar_labels.append(label)
            bar_values.append(val)
    
    if bar_values:
        sns.barplot(x=bar_labels, y=bar_values, ax=ax, palette=palette, hue=bar_labels, legend=False)
        for i, v in enumerate(bar_values):
            ax.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold', fontsize=9)
    ax.set_title("Test Set Performance", fontweight="bold")
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=35, ha='right', fontsize=9)

    # --- Plot 4: Energy Descent (1,1) ---
    ax = axes[1, 1]
    if energy_groups:
        agg_energy = _aggregate_curves(energy_groups)
        for label, (mean, std) in sorted(agg_energy.items()):
            x = np.arange(len(mean))
            ax.plot(x, mean, label=label, color=model_colors.get(label, 'black'), linewidth=2)
            if len(mean) > 1:
                ax.fill_between(x, mean - std, mean + std, color=model_colors.get(label, 'black'), alpha=0.1)
        ax.set_title("Energy Descent (Inference)", fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Energy")
    else:
        ax.text(0.5, 0.5, "No Energy Traces", ha='center', va='center', alpha=0.5)

    if title:
        fig.suptitle(title, fontweight="bold", fontsize=18, y=0.98)

    sns.despine(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def load_and_plot(json_path: Path, output_path: Path | None = None, title: str | None = None) -> Path:
    import json

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if output_path is None:
        output_path = json_path.with_suffix(".png")
    return plot_payload(payload, output_path=output_path, title=title)
