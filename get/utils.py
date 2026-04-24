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
def _numba_sp_vec_mul(indptr, indices, deg_inv, p_vec):
    """Sparse vector-matrix multiplication p_next = p_vec * (D^-1 A)."""
    num_nodes = len(indptr) - 1
    p_next = np.zeros(num_nodes, dtype=np.float32)
    for u in range(num_nodes):
        if p_vec[u] > 0:
            prob_u = p_vec[u] * deg_inv[u]
            for idx in range(indptr[u], indptr[u+1]):
                v = indices[idx]
                p_next[v] += prob_u
    return p_next


@njit
def _numba_rwse_sparse(indptr, indices, k, p_vec, p_next):
    """Compute RWSE using sparse random walk with pre-allocated workspace."""
    num_nodes = len(indptr) - 1
    rwse = np.zeros((num_nodes, k), dtype=np.float32)
    
    deg_inv = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        d = indptr[i+1] - indptr[i]
        if d > 0:
            deg_inv[i] = 1.0 / d
            
    for start_node in range(num_nodes):
        # Reset workspace
        p_vec.fill(0.0)
        p_vec[start_node] = 1.0
        
        for t in range(k):
            p_next.fill(0.0)
            for u in range(num_nodes):
                if p_vec[u] > 0:
                    prob_u = p_vec[u] * deg_inv[u]
                    for idx in range(indptr[u], indptr[u+1]):
                        v = indices[idx]
                        p_next[v] += prob_u
            # Swap p_vec and p_next to avoid copy
            # (In Numba we can't easily swap references to input buffers, 
            # so we copy back to p_vec)
            p_vec[:] = p_next[:]
            rwse[start_node, t] = p_vec[start_node]
            
    return rwse


def rwse_from_adjacency(num_nodes, indptr, indices, k):
    """Compute RWSE directly from CSR."""
    if k <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    
    # Pre-allocate workspace once per graph
    p_vec = np.zeros(num_nodes, dtype=np.float32)
    p_next = np.zeros(num_nodes, dtype=np.float32)
    
    rwse_np = _numba_rwse_sparse(indptr, indices, k, p_vec, p_next)
    return torch.from_numpy(rwse_np).to(dtype=torch.float32)


def laplacian_pe_from_sparse_matrix(num_nodes, indptr, indices, data, k, training=False):
    """Compute Laplacian PE directly from CSR matrix data."""
    if k <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    if num_nodes <= 1:
        return torch.zeros((num_nodes, k), dtype=torch.float32)

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        lap_sparse = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
        evals, evecs = eigsh(lap_sparse, k=k+1, which='SM', tol=1e-5)
        use = evecs[:, 1:k+1]
    except (ImportError, Exception):
        # Dense fallback
        lap = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        for i in range(num_nodes):
            for idx in range(indptr[i], indptr[i+1]):
                lap[i, indices[idx]] = float(data[idx])
        evals, evecs = torch.linalg.eigh(lap)
        use = evecs[:, 1 : 1 + k].cpu().numpy()

    if use.shape[1] < k:
        use = np.concatenate([use, np.zeros((num_nodes, k - use.shape[1]), dtype=np.float32)], axis=1)

    pe = torch.from_numpy(use).to(dtype=torch.float32)
    if training:
        pe = random_flip_pe_signs(pe, training=True)
    return pe


def laplacian_pe_from_adjacency(num_nodes, indptr, indices, k, training=False):
    """Compute Laplacian PE using sparse solver if possible."""
    from .data import _numba_build_sparse_laplacian
    l_indptr, l_indices, l_data = _numba_build_sparse_laplacian(num_nodes, indptr, indices)
    return laplacian_pe_from_sparse_matrix(num_nodes, l_indptr, l_indices, l_data, k, training)


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
