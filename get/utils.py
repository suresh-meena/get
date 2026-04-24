import torch
import torch.optim as optim
import numpy as np
from numba import njit


def _adj_to_csr_utils(adj):
    """Convert dense adjacency tensor to CSR format."""
    # Move to CPU/numpy for Numba
    a = adj.detach().cpu().numpy()
    num_nodes = a.shape[0]
    # Find indices of non-zero entries
    rows, cols = np.where(a > 0)
    
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    # Count occurrences of each row index to build indptr
    row_counts = np.bincount(rows, minlength=num_nodes)
    indptr[1:] = np.cumsum(row_counts)
    
    indices = cols.astype(np.int64)
    return indptr, indices


@njit
def _numba_rwse(indptr, indices, k):
    """Compute RWSE using sparse random walk."""
    num_nodes = len(indptr) - 1
    rwse = np.zeros((num_nodes, k), dtype=np.float32)
    
    # deg_inv = 1 / deg
    deg_inv = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        d = indptr[i+1] - indptr[i]
        if d > 0:
            deg_inv[i] = 1.0 / d
            
    # p_curr[i, j] is probability of being at j after t steps, starting at i
    # We only need diagonals: p_curr[i, i]
    # For large k, full sparse matrix power is better, but for small k (16-24),
    # repeated sparse vector multiplication is efficient.
    
    for start_node in range(num_nodes):
        # prob vector for current start node
        p_vec = np.zeros(num_nodes, dtype=np.float32)
        p_vec[start_node] = 1.0
        
        for t in range(k):
            p_next = np.zeros(num_nodes, dtype=np.float32)
            for u in range(num_nodes):
                if p_vec[u] > 0:
                    prob_u = p_vec[u] * deg_inv[u]
                    for idx in range(indptr[u], indptr[u+1]):
                        v = indices[idx]
                        p_next[v] += prob_u
            p_vec = p_next
            rwse[start_node, t] = p_vec[start_node]
            
    return rwse


def rwse_from_adjacency(adj, k):
    n = int(adj.size(0))
    if k <= 0:
        return adj.new_zeros((n, 0), dtype=torch.float32)
    if n == 0:
        return adj.new_zeros((n, k), dtype=torch.float32)
    
    indptr, indices = _adj_to_csr_utils(adj)
    rwse_np = _numba_rwse(indptr, indices, k)
    
    return torch.from_numpy(rwse_np).to(device=adj.device, dtype=torch.float32)

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
    lap = torch.eye(n, device=adj.device, dtype=torch.float32) - (inv_sqrt_deg[:, None] * a * inv_sqrt_deg[None, :])

    evals, evecs = torch.linalg.eigh(lap)
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
