import torch.optim as optim


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
