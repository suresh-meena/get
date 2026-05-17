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
    try:
        backend = str(compile_cfg.get("backend", "inductor"))
        dynamic = bool(compile_cfg.get("dynamic", True))
        fullgraph = bool(compile_cfg.get("fullgraph", False))
        mode = compile_cfg.get("mode", None)
        if mode is not None:
            mode = str(mode)
        return compile_fn(model, dynamic=dynamic, fullgraph=fullgraph, backend=backend, mode=mode)
    except Exception:
        return model
    enabled = bool(compile_cfg.get("enabled", False))
    if not enabled:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return model
    if getattr(model, "requires_double_backward", False):
        return model
    try:
        backend = str(compile_cfg.get("backend", "inductor"))
        dynamic = bool(compile_cfg.get("dynamic", True))
        fullgraph = bool(compile_cfg.get("fullgraph", False))
        mode = compile_cfg.get("mode", None)
        if mode is not None:
            mode = str(mode)
        # Disable donated buffers so compiled backward is compatible with
        # create_graph=True used during training with the fixed-step solver.
        return compile_fn(model, dynamic=dynamic, fullgraph=fullgraph, backend=backend, mode=mode)
    except Exception:
        return model
