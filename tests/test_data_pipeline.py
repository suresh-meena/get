import numpy as np
from get.data.batch import _numba_edges_to_csr
import torch

from get.data.batch import _graph_dataset_cache_fingerprint, _process_one_graph, align_pairwise_edge_attr, collate_get_batch, get_incidence_matrices, validate_get_batch


def test_incidence_deterministic_under_edge_order_shuffle():
    num_nodes = 5
    edges_a = [(0, 1), (0, 2), (1, 2), (2, 3)]
    edges_b = [(2, 3), (1, 2), (0, 2), (0, 1)]

    edges_arr_a = np.ascontiguousarray(np.array(edges_a, dtype=np.int64).reshape(-1, 2))
    indptr_a, indices_a = _numba_edges_to_csr(num_nodes, edges_arr_a)
    out_a = get_incidence_matrices(num_nodes, indptr_a, indices_a, max_motifs_per_node=2)

    edges_arr_b = np.ascontiguousarray(np.array(edges_b, dtype=np.int64).reshape(-1, 2))
    indptr_b, indices_b = _numba_edges_to_csr(num_nodes, edges_arr_b)
    out_b = get_incidence_matrices(num_nodes, indptr_b, indices_b, max_motifs_per_node=2)
    for ta, tb in zip(out_a, out_b):
        assert torch.equal(ta, tb)


def test_motif_budget_keeps_boundary_ties():
    # Center node 0 has two closed wedges (1,2) and (3,4), plus open ones.
    num_nodes = 5
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]

    edges_arr = np.ascontiguousarray(np.array(edges, dtype=np.int64).reshape(-1, 2))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(
        num_nodes, indptr, indices, max_motifs_per_node=1
    )
    assert c_2.numel() > 0 and u_2.numel() > 0

    center_zero = c_3 == 0
    assert center_zero.any()
    # Because of tie-at-boundary rule with score 1, both closed motifs should remain.
    assert int(center_zero.sum().item()) == 2
    assert torch.all(t_tau[center_zero] == 1)


def test_zero_motif_budget_disables_motif_extraction():
    num_nodes = 6
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    edges_arr = np.ascontiguousarray(np.array(edges, dtype=np.int64).reshape(-1, 2))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    _c_2, _u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(
        num_nodes, indptr, indices, max_motifs_per_node=0
    )
    assert c_3.numel() == 0
    assert u_3.numel() == 0
    assert v_3.numel() == 0
    assert t_tau.numel() == 0


def test_align_pairwise_edge_attr_matches_directed_incidence_order():
    edges = [(0, 1), (1, 2)]
    edge_attr = torch.tensor([[10.0, 0.0], [20.0, 1.0]])
    c_2 = torch.tensor([0, 1, 1, 2], dtype=torch.long)
    u_2 = torch.tensor([1, 0, 2, 1], dtype=torch.long)

    aligned = align_pairwise_edge_attr(edges, edge_attr, c_2, u_2)
    expected = torch.tensor(
        [[10.0, 0.0], [10.0, 0.0], [20.0, 1.0], [20.0, 1.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(aligned, expected)


def test_align_pairwise_edge_attr_accepts_edge_index_tensor():
    edges = [(0, 1), (1, 2), (2, 0)]
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.tensor([[10.0, 0.0], [20.0, 1.0], [30.0, 2.0]])
    c_2 = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    u_2 = torch.tensor([1, 2, 0, 2, 0, 1], dtype=torch.long)

    aligned_from_list = align_pairwise_edge_attr(edges, edge_attr, c_2, u_2)
    aligned_from_tensor = align_pairwise_edge_attr(edge_index, edge_attr, c_2, u_2)

    assert torch.allclose(aligned_from_tensor, aligned_from_list)


def test_collate_offsets_and_ptr_are_consistent():
    g1 = {"x": torch.randn(3, 4), "edge_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long), "y": torch.tensor([1.0])}
    g2 = {"x": torch.randn(2, 4), "edge_index": torch.tensor([[0], [1]], dtype=torch.long), "y": torch.tensor([0.0])}

    batch = collate_get_batch([g1, g2])

    assert batch.x.shape[0] == 5
    assert torch.equal(batch.ptr, torch.tensor([0, 3, 5], dtype=torch.long))
    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 1, 1], dtype=torch.long))

    # Second graph node ids should be offset by 3, so directed incidence uses nodes >= 3.
    second_graph_mask = batch.c_2 >= 3
    if second_graph_mask.any():
        assert torch.all(batch.u_2[second_graph_mask] >= 3)


def test_collate_accepts_edge_index_without_edges_list():
    graph = {
        "x": torch.randn(3, 4),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        "edge_attr": torch.randn(3, 2),
        "y": torch.tensor([1.0]),
    }

    batch = collate_get_batch([graph])

    assert batch.c_2.numel() == 6
    assert batch.edge_attr.shape == (6, 2)


def test_process_one_graph_accepts_edge_index_tensor():
    graph = {
        "x": torch.randn(3, 4),
        "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        "edge_attr": torch.tensor([[1.0], [2.0], [3.0]]),
        "y": torch.tensor([1.0]),
    }

    item = _process_one_graph(graph, max_motifs=0, pe_k=0, rwse_k=0)

    assert item["c_2"].numel() == 6
    assert item["u_2"].numel() == 6
    assert item["aligned_edge_attr"].shape == (6, 1)


def test_collate_rejects_empty_graph_list():
    try:
        collate_get_batch([])
        assert False, "Expected ValueError for empty graph list"
    except ValueError as exc:
        assert "graph_list" in str(exc)


def test_dataset_cache_fingerprint_depends_on_full_content():
    shared_prefix = [
        {"x": torch.zeros(2, 1), "edges": [(0, 1)]},
        {"x": torch.ones(2, 1), "edges": [(0, 1)]},
        {"x": torch.full((2, 1), 2.0), "edges": [(0, 1)]},
        {"x": torch.full((2, 1), 3.0), "edges": [(0, 1)]},
        {"x": torch.full((2, 1), 4.0), "edges": [(0, 1)]},
    ]
    dataset_a = shared_prefix + [{"x": torch.full((2, 1), 5.0), "edges": [(0, 1)]}]
    dataset_b = shared_prefix + [{"x": torch.full((2, 1), 9.0), "edges": [(0, 1)]}]

    key_a = _graph_dataset_cache_fingerprint(dataset_a, "demo", 4, 2, 2)
    key_b = _graph_dataset_cache_fingerprint(dataset_b, "demo", 4, 2, 2)

    assert key_a != key_b


def test_validate_get_batch_accepts_consistent_batch():
    graph = {
        "x": torch.randn(3, 4),
        "edges": [(0, 1), (1, 2)],
        "edge_attr": torch.randn(2, 4),
        "pe": torch.randn(3, 2),
        "rwse": torch.randn(3, 2),
    }
    batch = collate_get_batch([graph])

    validate_get_batch(batch)


def test_validate_get_batch_rejects_shape_and_dtype_mismatches():
    graph = {
        "x": torch.randn(3, 4),
        "edges": [(0, 1), (1, 2)],
        "edge_attr": torch.randn(2, 4),
    }
    batch = collate_get_batch([graph])

    batch.edge_attr = torch.randn(batch.c_2.numel() + 1, 4)
    try:
        validate_get_batch(batch)
        assert False, "Expected a shape mismatch to be rejected"
    except ValueError as exc:
        assert "edge_attr" in str(exc)

    batch.edge_attr = torch.randn(batch.c_2.numel(), 4, dtype=torch.float64)
    try:
        validate_get_batch(batch)
        assert False, "Expected a dtype mismatch to be rejected"
    except ValueError as exc:
        assert "dtype" in str(exc)
