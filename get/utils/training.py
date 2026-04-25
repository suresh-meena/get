"""Training utilities: optimizer building, model compilation, PE sign flipping."""
import torch
import torch.optim as optim


def maybe_compile_model(model, enabled, model_name=None):
    if not enabled:
        return model

    name = model_name or model.__class__.__name__

    if not hasattr(torch, "compile"):
        print(f"Warning: torch.compile is unavailable; running {name} without compilation.")
        return model

    try:
        backend = "inductor"
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major < 7:
                backend = "aot_eager"

        compiled_model = torch.compile(model, backend=backend)
        print(f"Enabled torch.compile for {name} (backend={backend}).")
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


def random_flip_pe_signs(pe, training=False):
    """Random sign augmentation for Laplacian PE sign invariance."""
    if not training or pe is None or pe.numel() == 0:
        return pe
    signs = torch.randint(0, 2, (pe.size(1),), device=pe.device, dtype=torch.long)
    signs = (2 * signs - 1).to(dtype=pe.dtype)
    return pe * signs[None, :]


def _is_get_model(model):
    try:
        from get.models.get_model import GETModel
    except Exception:
        return False
    if isinstance(model, GETModel):
        return True
    inner = getattr(model, "model", None)
    return isinstance(inner, GETModel)
