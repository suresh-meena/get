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
    on_cpu = all(p.device.type == "cpu" for p in model.parameters())
    if on_cpu:
        return model
    try:
        return compile_fn(model, dynamic=True, fullgraph=False)
    except Exception:
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

    # CPU inductor compile can dominate short experiment runtime and make smoke
    # checks flaky. Keep default behavior fast unless explicitly forced.
    force_cpu_compile = bool(compile_cfg.get("force_cpu_compile", False))
    on_cpu = True
    for p in model.parameters():
        on_cpu = on_cpu and (p.device.type == "cpu")
        if not on_cpu:
            break
    if on_cpu and backend == "inductor" and not force_cpu_compile:
        return model

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
