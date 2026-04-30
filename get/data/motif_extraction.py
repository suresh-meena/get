"""Compatibility shim for motif extraction helpers."""

from __future__ import annotations

from .batch import _numba_edges_to_csr, get_incidence_matrices

__all__ = ["_numba_edges_to_csr", "get_incidence_matrices"]