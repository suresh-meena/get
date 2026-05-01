from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any


def _clean_component(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            cleaned.append(ch)
        else:
            cleaned.append("-")
    result = "".join(cleaned).strip("-_.")
    return result or None


def _seed_component(metadata: dict[str, Any]) -> str | None:
    seed = metadata.get("seed")
    if seed is not None:
        return _clean_component(f"seed_{seed}")

    seeds = metadata.get("seeds")
    if isinstance(seeds, (list, tuple)) and len(seeds) == 1:
        return _clean_component(f"seed_{seeds[0]}")
    if isinstance(seeds, (list, tuple)) and len(seeds) > 1:
        joined = "-".join(str(item) for item in seeds)
        return _clean_component(f"seeds_{joined}")
    return None


def artifact_path_parts(metadata: dict[str, Any] | None = None) -> list[str]:
    metadata = metadata or {}
    parts: list[str] = []

    for key in ("stage", "task", "dataset", "model", "variant"):
        cleaned = _clean_component(metadata.get(key))
        if cleaned is not None:
            parts.append(cleaned)

    seed_part = _seed_component(metadata)
    if seed_part is not None:
        parts.append(seed_part)

    return parts


def infer_artifact_metadata(name: str) -> dict[str, Any]:
    normalized = str(name).lower()
    inferred: dict[str, Any] = {}

    if normalized.startswith("exp1_"):
        inferred["stage"] = "stage1"
        inferred["task"] = normalized.removeprefix("exp1_")
    elif normalized.startswith("exp2_"):
        inferred["stage"] = "stage1"
        inferred["task"] = normalized.removeprefix("exp2_")
    elif normalized.startswith("exp3_") or normalized.startswith("brec_"):
        inferred["stage"] = "stage2"
        inferred["task"] = normalized.removeprefix("exp3_") if normalized.startswith("exp3_") else normalized
    elif normalized.startswith("exp8_"):
        inferred["stage"] = "stage3"
        inferred["task"] = normalized.removeprefix("exp8_")
    elif normalized.startswith("exp9_"):
        inferred["stage"] = "stage3"
        inferred["task"] = normalized.removeprefix("exp9_")
    elif normalized.startswith("exp10_"):
        inferred["stage"] = "stage3"
        inferred["task"] = normalized.removeprefix("exp10_")
    elif normalized.startswith("stage4_"):
        inferred["stage"] = "stage4"
        inferred["task"] = normalized.removeprefix("stage4_")

    task = inferred.get("task")
    if isinstance(task, str):
        if task.endswith("_results"):
            inferred["task"] = task.removesuffix("_results")
        elif task.endswith("_bench"):
            inferred["task"] = task.removesuffix("_bench")

    return inferred


def build_artifact_path(name: str, metadata: dict[str, Any] | None = None, root: str | Path = "outputs") -> Path:
    suffix = os.environ.get("EXPERIMENT_OUTPUT_SUFFIX", "")
    root_path = Path(root)
    resolved_metadata = dict(infer_artifact_metadata(name))
    if metadata:
        resolved_metadata.update(metadata)
    path_parts = artifact_path_parts(resolved_metadata)
    directory = root_path.joinpath(*path_parts) if path_parts else root_path
    return directory / f"{name}{suffix}.json"


def build_run_directory(name: str, metadata: dict[str, Any] | None = None, root: str | Path = "outputs") -> Path:
    suffix = os.environ.get("EXPERIMENT_OUTPUT_SUFFIX", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_path = Path(root)
    resolved_metadata = dict(infer_artifact_metadata(name))
    if metadata:
        resolved_metadata.update(metadata)
    path_parts = artifact_path_parts(resolved_metadata)
    directory = root_path.joinpath("runs", *path_parts) if path_parts else root_path / "runs"
    return directory / f"{name}{suffix}_{timestamp}"