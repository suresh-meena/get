import torch

from experiments.stage1.common import (
    match_pairwise_hidden_dim,
    prepare_stage1_graph,
    summarize_stage1_support,
)


def _support_graph():
    return {
        "x": torch.ones(5, 1),
        "edges": [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (3, 4)],
        "graph_id": 7,
    }


def test_prepare_stage1_graph_static_motif_appends_features_and_support_metadata():
    graph = prepare_stage1_graph(_support_graph(), feature_mode="static_motif", support_mode="exact", seed=13)

    assert graph["x"].shape == (5, 4)
    assert graph["candidate_motif_count"] == 6
    assert graph["retained_motif_count"] == 6
    assert graph["retained_closed"] == 2
    assert graph["retained_open"] == 4


def test_random_support_is_deterministic_for_a_fixed_seed():
    base = _support_graph()
    first = prepare_stage1_graph(base, feature_mode="core", support_mode="random", max_motifs=1, seed=99)
    second = prepare_stage1_graph(base, feature_mode="core", support_mode="random", max_motifs=1, seed=99)
    different = prepare_stage1_graph(base, feature_mode="core", support_mode="random", max_motifs=1, seed=100)

    assert torch.equal(first["c_3"], second["c_3"])
    assert torch.equal(first["u_3"], second["u_3"])
    assert torch.equal(first["v_3"], second["v_3"])
    assert torch.equal(first["t_tau"], second["t_tau"])
    assert first["retained_motif_count"] == second["retained_motif_count"]
    assert first["retained_motif_count"] == 2
    assert not (
        torch.equal(first["u_3"], different["u_3"]) and torch.equal(first["v_3"], different["v_3"])
    )


def test_rwse_features_are_appended_to_node_state():
    graph = prepare_stage1_graph(_support_graph(), feature_mode="rwse", rwse_k=2, support_mode="exact", seed=5)

    assert graph["x"].shape == (5, 3)
    assert graph["rwse"].shape == (5, 2)
    assert graph["pe"].shape == (5, 2)


def test_pairwise_hidden_dim_matching_is_close_to_full_model():
    matched_d, stats = match_pairwise_hidden_dim(
        in_dim=1,
        num_classes=1,
        full_hidden_dim=32,
        common_kwargs={"num_steps": 1, "num_heads": 1, "R": 2, "K": 4, "num_motif_types": 2},
    )

    assert matched_d >= 1
    assert stats["target_params"] > 0
    assert stats["matched_params"] > 0
    assert abs(stats["matched_params"] - stats["target_params"]) <= max(1, int(0.05 * stats["target_params"]))


def test_support_summary_aggregates_counts():
    dataset = [
        prepare_stage1_graph(_support_graph(), feature_mode="core", support_mode="exact", seed=1),
        prepare_stage1_graph(_support_graph(), feature_mode="core", support_mode="exact", seed=2),
    ]
    summary = summarize_stage1_support(dataset)

    assert summary["num_graphs"] == 2.0
    assert summary["candidate_motif_count"] == 12.0
    assert summary["retained_motif_count"] == 12.0
    assert summary["retained_fraction"] == 1.0