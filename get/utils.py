import torch
import torch.optim as optim

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

def build_adamw_optimizer(model, lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8):
    """AdamW for supervised parameter learning, with no decay on scalars/norms/biases."""
    decay = []
    no_decay = []
    no_decay_markers = (
        "bias",
        "norm",
        "layernorm",
        "lambda_",
        "beta_",
        "eta_logit",
        "update_damping_logit",
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lowered = name.lower()
        if param.ndim <= 1 or any(marker in lowered for marker in no_decay_markers):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return optim.AdamW(groups, lr=lr, betas=betas, eps=eps)
