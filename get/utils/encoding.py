"""Positional and structural encoding utilities: LapPE, RWSE."""
import torch
import numpy as np
from numba import njit


def _adj_to_csr_utils(adj):
    """Convert dense adjacency tensor to CSR format."""
    a = adj.detach().cpu().numpy()
    num_nodes = a.shape[0]
    rows, cols = np.where(a > 0)
    rows = rows.astype(np.int32, copy=False)
    cols = cols.astype(np.int32, copy=False)

    indptr = np.zeros(num_nodes + 1, dtype=np.int32)
    row_counts = np.asarray(np.bincount(rows, minlength=num_nodes), dtype=np.int32)
    indptr[1:] = np.cumsum(row_counts, dtype=np.int32)

    indices = cols
    return indptr, indices


@njit
def _numba_rwse_sparse(indptr, indices, k, p_vec, p_next):
    """Compute RWSE using sparse random walk with pre-allocated workspace."""
    num_nodes = len(indptr) - 1
    rwse = np.zeros((num_nodes, k), dtype=np.float32)

    deg_inv = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        d = indptr[i + 1] - indptr[i]
        if d > 0:
            deg_inv[i] = 1.0 / d

    for start_node in range(num_nodes):
        p_vec.fill(0.0)
        p_vec[start_node] = 1.0

        for t in range(k):
            p_next.fill(0.0)
            for u in range(num_nodes):
                if p_vec[u] > 0:
                    prob_u = p_vec[u] * deg_inv[u]
                    for idx in range(indptr[u], indptr[u + 1]):
                        v = indices[idx]
                        p_next[v] += prob_u
            p_vec[:] = p_next[:]
            rwse[start_node, t] = p_vec[start_node]

    return rwse


def rwse_from_adjacency(num_nodes, indptr, indices, k):
    """Compute RWSE directly from CSR."""
    from get.data.positional import get_rwse

    return get_rwse(num_nodes, indptr, indices, k)


def laplacian_pe_from_sparse_matrix(num_nodes, indptr, indices, data, k, training=False):
    """Compute Laplacian PE directly from CSR matrix data."""
    from .graph import _numba_csr_to_dense
    from .training import random_flip_pe_signs

    if k <= 0:
        return torch.zeros((num_nodes, 0), dtype=torch.float32)
    if num_nodes <= 1:
        return torch.zeros((num_nodes, k), dtype=torch.float32)

    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh

        if k + 1 < num_nodes:
            lap_sparse = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))
            evals, evecs = eigsh(lap_sparse, k=k + 1, which='SM', tol=1e-5)
            use = evecs[:, 1:k + 1]
        else:
            raise ValueError("use dense fallback")
    except (ImportError, Exception):
        lap = _numba_csr_to_dense(num_nodes, indptr, indices, np.asarray(data, dtype=np.float32))
        evals, evecs = np.linalg.eigh(lap)
        use = evecs[:, 1:1 + k]

    if use.shape[1] < k:
        use = np.concatenate([use, np.zeros((num_nodes, k - use.shape[1]), dtype=np.float32)], axis=1)

    pe = torch.from_numpy(use).to(dtype=torch.float32)
    if training:
        pe = random_flip_pe_signs(pe, training=True)
    return pe


def laplacian_pe_from_adjacency(num_nodes, indptr, indices, k, training=False):
    """Compute Laplacian PE using sparse solver if possible."""
    from get.data.positional import _numba_build_sparse_laplacian
    l_indptr, l_indices, l_data = _numba_build_sparse_laplacian(num_nodes, indptr, indices)
    return laplacian_pe_from_sparse_matrix(num_nodes, l_indptr, l_indices, l_data, k, training)
