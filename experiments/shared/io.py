from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path


def _to_jsonable(data):
    try:
        json.dumps(data)
        return data
    except TypeError:
        if isinstance(data, dict):
            return {str(k): _to_jsonable(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [_to_jsonable(v) for v in data]
        return str(data)


def _iter_histories(node, prefix=()):
    if isinstance(node, dict):
        if "history" in node and isinstance(node["history"], dict):
            yield prefix, node["history"]
            return
        if "histories" in node and isinstance(node["histories"], dict):
            for key, value in node["histories"].items():
                yield from _iter_histories(value, prefix + (str(key),))
        for key, value in node.items():
            if key in {"history", "histories", "metadata"}:
                continue
            if isinstance(value, (dict, list, tuple)):
                yield from _iter_histories(value, prefix + (str(key),))
        return
    if isinstance(node, (list, tuple)):
        for idx, value in enumerate(node):
            yield from _iter_histories(value, prefix + (str(idx),))


def _write_curves_csv(path: Path, payload: dict):
    rows = []
    for label_parts, history in _iter_histories(payload):
        if not isinstance(history, dict) or not history:
            continue
        keys = [k for k, v in history.items() if isinstance(v, (list, tuple)) and len(v) > 0]
        if not keys:
            continue
        max_len = max(len(history[k]) for k in keys)
        label = "/".join(label_parts) if label_parts else "run"
        for epoch_idx in range(max_len):
            row = {"series": label, "epoch": epoch_idx + 1}
            for key in keys:
                vals = history.get(key, [])
                row[key] = vals[epoch_idx] if epoch_idx < len(vals) else ""
            rows.append(row)

    if not rows:
        return
    columns = ["series", "epoch"]
    dynamic_cols = sorted({k for row in rows for k in row.keys() if k not in {"series", "epoch"}})
    columns.extend(dynamic_cols)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_results(name, payload, metadata=None):
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    suffix = os.environ.get("EXPERIMENT_OUTPUT_SUFFIX", "")
    base_path = outputs_dir / f"{name}{suffix}.json"

    result_data = payload if metadata is None else {"results": payload, "metadata": metadata}
    with base_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(result_data), f, indent=2)
    print(f"Saved {base_path}")

    if os.environ.get("EXPERIMENT_LEGACY_ONLY", "0") == "1":
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = outputs_dir / "runs" / f"{name}{suffix}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)

    if metadata is not None:
        config_path = run_dir / "config.yaml"
        try:
            import yaml  # type: ignore
            config_text = yaml.safe_dump(_to_jsonable(metadata), sort_keys=True)
        except Exception:
            config_text = json.dumps(_to_jsonable(metadata), indent=2)
        config_path.write_text(config_text, encoding="utf-8")

    _write_curves_csv(run_dir / "curves.csv", payload if isinstance(payload, dict) else {})
