import torch


def _is_get_model(model):
    try:
        from .model import GETModel
    except Exception:
        return False
    if isinstance(model, GETModel):
        return True
    # Wrapper baselines (e.g., ETLocalBaseline/ETCompleteBaseline) hold a GETModel in `model`.
    inner = getattr(model, "model", None)
    return isinstance(inner, GETModel)


def maybe_compile_model(model, enabled, model_name=None):
    if not enabled:
        return model

    name = model_name or model.__class__.__name__
    
    if _is_get_model(model):
        print(
            f"Warning: skipping torch.compile for {name}; GET training requires double-backward "
            "through the energy gradient and torch.compile/aot_autograd does not support this path reliably yet."
        )
        return model

    if not hasattr(torch, "compile"):
        print(f"Warning: torch.compile is unavailable; running {name} without compilation.")
        return model

    try:
        backend = None
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            # Triton kernels require newer GPUs; use aot_eager as a safe compiled fallback.
            if major < 7:
                backend = "aot_eager"

        if backend is not None:
            compiled_model = torch.compile(model, backend=backend)
            print(f"Enabled torch.compile for {name} (backend={backend}).")
        else:
            compiled_model = torch.compile(model)
            print(f"Enabled torch.compile for {name}.")
        return compiled_model
    except Exception as exc:
        print(f"Warning: torch.compile failed for {name}: {exc}")
        return model