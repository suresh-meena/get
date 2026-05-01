"""Positional encoding helpers for GET data preparation."""

from __future__ import annotations

import numpy as np
import torch


def get_rwse(num_nodes, indptr, indices, k):
    if int(k) <= 0:
        return torch.zeros((int(num_nodes), 0), dtype=torch.float32)

    n = int(num_nodes)
    indptr_np = np.asarray(indptr, dtype=np.int32)
    indices_np = np.asarray(indices, dtype=np.int32)

    try:
        from scipy.sparse import csr_matrix

        data = np.empty(indices_np.size, dtype=np.float32)
        for row in range(n):
            start = int(indptr_np[row])
            end = int(indptr_np[row + 1])
            degree = end - start
            if degree > 0:
                data[start:end] = 1.0 / float(degree)

        transition_t = csr_matrix((data, indices_np, indptr_np), shape=(n, n)).transpose().tocsr()
        traces = np.zeros((n, int(k)), dtype=np.float32)
        state = np.zeros(n, dtype=np.float32)

        for start_node in range(n):
            state.fill(0.0)
            state[start_node] = 1.0
            for step in range(int(k)):
                state = transition_t.dot(state)
                traces[start_node, step] = state[start_node]

        return torch.from_numpy(traces).to(dtype=torch.float32)
    except Exception:
        adjacency = np.zeros((n, n), dtype=np.float32)
        for row in range(n):
            start = int(indptr_np[row])
            end = int(indptr_np[row + 1])
            adjacency[row, np.asarray(indices_np[start:end], dtype=np.int32)] = 1.0

        degree = adjacency.sum(axis=1)
        inv_degree = np.zeros_like(degree)
        nonzero = degree > 0
        inv_degree[nonzero] = 1.0 / degree[nonzero]

        transition = adjacency @ np.diag(inv_degree)
        current = np.eye(n, dtype=np.float32)
        traces = []
        for _ in range(int(k)):
            current = current @ transition
            traces.append(np.diag(current).astype(np.float32))

        return torch.from_numpy(np.stack(traces, axis=1)).to(dtype=torch.float32)


def _numba_build_sparse_laplacian(num_nodes, indptr, indices):
    n = int(num_nodes)
    indptr_np = np.asarray(indptr, dtype=np.int32)
    indices_np = np.asarray(indices, dtype=np.int32)
    row_lengths = np.diff(indptr_np)
    total_nnz = int(row_lengths.sum() + n)
    l_indptr = np.zeros(n + 1, dtype=np.int32)
    l_indices = np.empty(total_nnz, dtype=np.int32)
    l_data = np.empty(total_nnz, dtype=np.float32)

    l_indptr[1:] = np.cumsum(row_lengths + 1, dtype=np.int32)

    cursor = 0
    for row in range(n):
        neighbors = np.asarray(indices_np[int(indptr_np[row]) : int(indptr_np[row + 1])], dtype=np.int32)
        row_degree = int(neighbors.size)

        l_indices[cursor] = row
        l_data[cursor] = float(row_degree)
        cursor += 1
        if row_degree > 0:
            l_indices[cursor : cursor + row_degree] = neighbors
            l_data[cursor : cursor + row_degree] = -1.0
            cursor += row_degree

    return l_indptr, l_indices, l_data


__all__ = ["get_rwse", "_numba_build_sparse_laplacian"]