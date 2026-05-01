"""Model registry: name → factory mapping."""
from __future__ import annotations

from collections.abc import Callable


MODEL_REGISTRY: dict[str, Callable[..., object]] = {}


def _normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def register_model(name: str):
    key = _normalize_name(name)

    def decorator(factory: Callable[..., object]):
        MODEL_REGISTRY[key] = factory
        return factory

    return decorator


def build_model(name: str, *args, **kwargs):
    key = _normalize_name(name)
    if key not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[key](*args, **kwargs)


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)
