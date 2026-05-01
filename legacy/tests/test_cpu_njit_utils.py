import numpy as np

from get.utils import _numba_csr_to_dense


def test_numba_csr_to_dense_builds_expected_matrix():
    indptr = np.array([0, 2, 3], dtype=np.int64)
    indices = np.array([0, 1, 1], dtype=np.int64)
    data = np.array([1.0, -0.5, 2.0], dtype=np.float32)

    dense = _numba_csr_to_dense(2, indptr, indices, data)

    expected = np.array([[1.0, -0.5], [0.0, 2.0]], dtype=np.float32)
    assert dense.shape == (2, 2)
    assert np.allclose(dense, expected)