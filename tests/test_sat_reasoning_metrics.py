import torch

from experiments.stage1.sat_reasoning import build_sat_factor_graph, compute_metrics, custom_collate


def test_sat_reasoning_metrics_are_vectorized_and_correct():
    graph_1 = build_sat_factor_graph(
        3,
        [
            [(0, 1), (1, 1), (2, 1)],
            [(0, 1), (1, 1), (2, 1)],
        ],
        xor=False,
    )
    graph_2 = build_sat_factor_graph(
        3,
        [
            [(0, -1), (1, -1), (2, -1)],
            [(0, -1), (1, -1), (2, -1)],
        ],
        xor=False,
    )

    batch = custom_collate([graph_1, graph_2])
    logits = torch.full((batch.x.size(0), 1), 1.0)

    global_ratio, solved_ratio = compute_metrics(logits, batch, xor=False)

    assert global_ratio == 0.5
    assert solved_ratio == 0.5