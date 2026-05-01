from __future__ import annotations

from typing import Any, Dict

import torch


def maybe_compile_model(model: torch.nn.Module, compile_cfg: Dict[str, Any] | None) -> torch.nn.Module:
    """
    Optionally wrap a model with torch.compile.

    Behavior:
    - disabled or missing config: return model unchanged
    - torch.compile missing: return model unchanged
    - compile failure: return model unchanged (safe fallback)
    """
    if not compile_cfg:
        return model

    enabled = bool(compile_cfg.get("enabled", False))
    if not enabled:
        return model

    allow_double_backward = bool(compile_cfg.get("allow_double_backward", False))
    if getattr(model, "requires_double_backward", False) and not allow_double_backward:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model

    backend = compile_cfg.get("backend", "inductor")
    dynamic = bool(compile_cfg.get("dynamic", True))
    mode = compile_cfg.get("mode", None)
    fullgraph = bool(compile_cfg.get("fullgraph", False))

    kwargs = {
        "backend": backend,
        "dynamic": dynamic,
        "fullgraph": fullgraph,
    }
    if mode is not None:
        kwargs["mode"] = mode

    try:
        return compile_fn(model, **kwargs)
    except Exception:
        return model
