import torch
import torch.optim as optim


def laplacian_pe_from_adjacency(adj, k, training=False):
    n = int(adj.size(0))
    if k <= 0:
        return adj.new_zeros((n, 0), dtype=torch.float32)
    if n <= 1:
        return adj.new_zeros((n, k), dtype=torch.float32)

    a = adj.to(dtype=torch.float32)
    deg = a.sum(dim=1)
    inv_sqrt_deg = torch.zeros_like(deg)
    valid = deg > 0
    inv_sqrt_deg[valid] = deg[valid].pow(-0.5)
    l = torch.eye(n, device=adj.device, dtype=torch.float32) - (inv_sqrt_deg[:, None] * a * inv_sqrt_deg[None, :])

    evals, evecs = torch.linalg.eigh(l)
    order = torch.argsort(evals)
    evecs = evecs[:, order]
    use = evecs[:, 1 : 1 + k]
    if use.size(1) < k:
        use = torch.cat(
            [use, torch.zeros((n, k - use.size(1)), device=adj.device, dtype=use.dtype)],
            dim=1,
        )

    if training and use.numel() > 0:
        use = random_flip_pe_signs(use, training=True)
    return use


def random_flip_pe_signs(pe, training=False):
    if not training or pe is None or pe.numel() == 0:
        return pe
    signs = torch.randint(0, 2, (pe.size(1),), device=pe.device, dtype=torch.long)
    signs = (2 * signs - 1).to(dtype=pe.dtype)
    return pe * signs[None, :]


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
