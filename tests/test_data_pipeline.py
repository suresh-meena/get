import torch

from get.data import align_pairwise_edge_attr, collate_get_batch, get_incidence_matrices


def test_incidence_deterministic_under_edge_order_shuffle():
    num_nodes = 5
    edges_a = [(0, 1), (0, 2), (1, 2), (2, 3)]
    edges_b = [(2, 3), (1, 2), (0, 2), (0, 1)]

    out_a = get_incidence_matrices(num_nodes, edges_a, max_motifs_per_node=2)
    out_b = get_incidence_matrices(num_nodes, edges_b, max_motifs_per_node=2)

    for ta, tb in zip(out_a, out_b):
        assert torch.equal(ta, tb)


def test_motif_budget_keeps_boundary_ties():
    # Center node 0 has two closed wedges (1,2) and (3,4), plus open ones.
    num_nodes = 5
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)]

    c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(
        num_nodes, edges, max_motifs_per_node=1
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
    _c_2, _u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(
        num_nodes, edges, max_motifs_per_node=0
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


def test_collate_offsets_and_ptr_are_consistent():
    g1 = {"x": torch.randn(3, 4), "edges": [(0, 1), (1, 2)], "y": torch.tensor([1.0])}
    g2 = {"x": torch.randn(2, 4), "edges": [(0, 1)], "y": torch.tensor([0.0])}

    batch = collate_get_batch([g1, g2])

    assert batch.x.shape[0] == 5
    assert torch.equal(batch.ptr, torch.tensor([0, 3, 5], dtype=torch.long))
    assert torch.equal(batch.batch, torch.tensor([0, 0, 0, 1, 1], dtype=torch.long))

    # Second graph node ids should be offset by 3, so directed incidence uses nodes >= 3.
    second_graph_mask = batch.c_2 >= 3
    if second_graph_mask.any():
        assert torch.all(batch.u_2[second_graph_mask] >= 3)


def test_collate_rejects_empty_graph_list():
    try:
        collate_get_batch([])
        assert False, "Expected ValueError for empty graph list"
    except ValueError as exc:
        assert "graph_list" in str(exc)
