from __future__ import annotations

from typing import Any, Dict

import torch


def maybe_compile_model(model: torch.nn.Module, compile_cfg: Dict[str, Any] | None) -> torch.nn.Module:
    if not compile_cfg:
        return model
    enabled = bool(compile_cfg.get("enabled", False))
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    if getattr(model, "requires_double_backward", False):
        return model
    if all(p.device.type == "cpu" for p in model.parameters()):
        return model
    try:
        return compile_fn(model, dynamic=True, fullgraph=False)
    except Exception:
        return model
