from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
import torch

from get.utils.registry import build_model

_VAR_PATTERN = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")


def _resolve_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = _VAR_PATTERN.match(value.strip())
        if match is not None:
            key = match.group(1)
            if key not in context:
                raise KeyError(f"Unknown model-config variable '{key}'.")
            return context[key]
        return value
    if isinstance(value, dict):
        return {k: _resolve_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, context) for v in value]
    return value


def load_model_specs(config_path: str | Path) -> list[dict[str, Any]]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"Model config must contain a top-level 'models' list: {path}")
    models = data["models"]
    if not isinstance(models, list):
        raise ValueError(f"'models' must be a list in: {path}")
    return models


def load_model_catalog(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(f"Model catalog must contain a top-level 'models' mapping or list: {path}")
    return data


def _normalize_amp_dtype(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"fp16", "float16", "half"}:
            return torch.float16
        if key in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if key in {"none", "null", "off", "false"}:
            return None
    return value


def load_training_defaults(config_path: str | Path) -> dict[str, Any]:
    catalog = load_model_catalog(config_path)
    training = catalog.get("training", {})
    if training is None:
        return {}
    if not isinstance(training, dict):
        raise ValueError(f"'training' must be a mapping in: {config_path}")
    defaults = dict(training)
    if "amp_dtype" in defaults:
        defaults["amp_dtype"] = _normalize_amp_dtype(defaults.get("amp_dtype"))
    return defaults


def _normalize_catalog_specs(models: Any) -> list[dict[str, Any]]:
    if isinstance(models, list):
        normalized = []
        for spec in models:
            if not isinstance(spec, dict):
                raise ValueError("Each model spec must be a mapping.")
            normalized.append(dict(spec))
        return normalized

    if isinstance(models, dict):
        normalized = []
        for name, spec in models.items():
            if not isinstance(spec, dict):
                raise ValueError(f"Model spec for '{name}' must be a mapping.")
            merged = dict(spec)
            merged.setdefault("name", name)
            normalized.append(merged)
        return normalized

    raise ValueError("'models' must be a list or mapping in the model catalog.")


def _selected_model_names(
    catalog: dict[str, Any],
    *,
    stage: str | None = None,
    task: str | None = None,
    names: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if names is not None:
        return [str(name) for name in names]

    stages = catalog.get("stages", {})
    if isinstance(stages, dict) and stage is not None:
        stage_entry = stages.get(stage, {})
        if isinstance(stage_entry, dict):
            if task is not None and task in stage_entry:
                selected = stage_entry.get(task)
                if isinstance(selected, list):
                    return [str(name) for name in selected]
            default_selected = stage_entry.get("default")
            if isinstance(default_selected, list):
                return [str(name) for name in default_selected]

    models = catalog.get("models", {})
    if isinstance(models, dict):
        return [str(name) for name in models.keys()]
    if isinstance(models, list):
        selected = []
        for spec in models:
            if isinstance(spec, dict) and spec.get("name"):
                selected.append(str(spec["name"]))
        return selected
    return []


def instantiate_models_from_specs(
    specs: list[dict[str, Any]],
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    built: dict[str, Any] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each model spec must be a mapping.")
        if not bool(spec.get("enabled", True)):
            continue
        factory_name = spec.get("factory")
        if not factory_name:
            raise ValueError(f"Model spec missing 'factory': {spec}")
        label = spec.get("name", str(factory_name))
        params_raw = spec.get("params", {})
        if not isinstance(params_raw, dict):
            raise ValueError(f"'params' must be a mapping for model '{label}'.")
        params = _resolve_value(params_raw, context)
        built[str(label)] = build_model(str(factory_name), **params)
    return built


def instantiate_models_from_config(
    config_path: str | Path,
    *,
    context: dict[str, Any],
) -> dict[str, Any]:
    specs = load_model_specs(config_path)
    return instantiate_models_from_specs(specs, context=context)


def instantiate_models_from_catalog(
    config_path: str | Path,
    *,
    context: dict[str, Any],
    stage: str | None = None,
    task: str | None = None,
    names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    catalog = load_model_catalog(config_path)
    specs = _normalize_catalog_specs(catalog["models"])
    selected_names = set(_selected_model_names(catalog, stage=stage, task=task, names=names))
    built: dict[str, Any] = {}
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        if not bool(spec.get("enabled", True)):
            continue
        label = str(spec.get("name", spec.get("factory", "")))
        if selected_names and label not in selected_names:
            continue
        factory_name = spec.get("factory")
        if not factory_name:
            raise ValueError(f"Model spec missing 'factory': {spec}")
        params_raw = spec.get("params", {})
        if not isinstance(params_raw, dict):
            raise ValueError(f"'params' must be a mapping for model '{label}'.")
        params = _resolve_value(params_raw, context)
        built[label] = build_model(str(factory_name), **params)
    return built
