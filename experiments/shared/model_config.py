from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

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
