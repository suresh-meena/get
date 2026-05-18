"""
Leakage control utilities for Stage 1 synthetic graph generation.

Ensures train/val/test splits and positive/negative pairs have matched
graph statistics to prevent statistical leakage.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from scipy.spatial.distance import euclidean, jensenshannon


def compute_degree_histogram(adj: torch.Tensor, bins: int = 10) -> np.ndarray:
    """Compute normalized degree histogram for a graph.
    
    Args:
        adj: adjacency matrix (n, n) boolean or float
        bins: number of histogram bins
        
    Returns:
        Normalized histogram of shape (bins,)
    """
    adj = adj.float()
    degrees = adj.sum(1).cpu().numpy()
    if degrees.size == 0:
        return np.zeros(bins)
    hist, _ = np.histogram(degrees, bins=bins, range=(0, degrees.max() + 1))
    hist = hist / hist.sum() if hist.sum() > 0 else hist
    return hist.astype(np.float32)


def compute_edge_count(adj: torch.Tensor) -> int:
    """Count edges in a graph (adjacency matrix)."""
    return int((adj.sum() / 2.0).item())


def compute_node_count(adj: torch.Tensor) -> int:
    """Count nodes in a graph (adjacency matrix)."""
    return adj.shape[0]


def compute_two_hop_count(adj: torch.Tensor) -> float:
    """Approximate count of 2-hop paths (diameter-2 neighborhoods).
    
    Computes the number of unique (u, w) pairs connected via any v,
    normalized by node count.
    """
    adj = adj.float()
    two_hop = (adj @ adj) > 0
    # Exclude 0-hop (self loops) and 1-hop (direct edges)
    two_hop = two_hop & ~torch.eye(adj.shape[0], dtype=torch.bool, device=adj.device)
    two_hop = two_hop & ~adj.bool()
    return float(two_hop.sum().item() / 2.0) / max(1.0, float(adj.shape[0]))


def graph_statistics_dict(adj: torch.Tensor) -> Dict[str, float]:
    """Compute all graph statistics as a dictionary."""
    return {
        "node_count": float(compute_node_count(adj)),
        "edge_count": float(compute_edge_count(adj)),
        "two_hop_count": compute_two_hop_count(adj),
        "avg_degree": float(adj.sum(1).float().mean().item()),
    }


def similarity_euclidean_degree(
    adj1: torch.Tensor,
    adj2: torch.Tensor,
    bins: int = 10,
) -> float:
    """Euclidean distance between normalized degree histograms.
    
    Returns: distance (0 = identical, higher = more different)
    """
    h1 = compute_degree_histogram(adj1, bins=bins)
    h2 = compute_degree_histogram(adj2, bins=bins)
    return euclidean(h1, h2)


def similarity_jensenshannon_degree(
    adj1: torch.Tensor,
    adj2: torch.Tensor,
    bins: int = 10,
) -> float:
    """Jensen-Shannon divergence between degree histograms.
    
    Returns: JS divergence (0 = identical, 1 = maximally different)
    """
    h1 = compute_degree_histogram(adj1, bins=bins)
    h2 = compute_degree_histogram(adj2, bins=bins)
    return jensenshannon(h1, h2)


def similarity_aggregate(
    adj1: torch.Tensor,
    adj2: torch.Tensor,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Aggregate similarity across multiple statistics.
    
    Args:
        adj1, adj2: adjacency matrices
        weights: dict with keys 'node_count', 'edge_count', 'two_hop_count'
                 default equal weighting
    
    Returns: aggregated similarity score (lower = more similar)
    """
    if weights is None:
        weights = {"node_count": 1.0, "edge_count": 1.0, "two_hop_count": 1.0}
    
    stats1 = graph_statistics_dict(adj1)
    stats2 = graph_statistics_dict(adj2)
    
    total_score = 0.0
    total_weight = sum(weights.values())
    
    for key in weights:
        if key not in stats1 or key not in stats2:
            continue
        val1 = stats1[key]
        val2 = stats2[key]
        # Normalize difference by maximum of the two values
        denom = max(abs(val1), abs(val2), 1e-6)
        dist = abs(val1 - val2) / denom
        total_score += weights[key] * dist
    
    return total_score / max(total_weight, 1e-6)


def match_graphs_greedy(
    graphs_pos: List[Tuple[int, torch.Tensor]],
    graphs_neg: List[Tuple[int, torch.Tensor]],
    similarity_fn=None,
) -> List[Tuple[int, int]]:
    """Greedily match positive and negative graphs with similar statistics.
    
    Args:
        graphs_pos: list of (idx, adjacency_matrix) for positive graphs
        graphs_neg: list of (idx, adjacency_matrix) for negative graphs
        similarity_fn: function(adj1, adj2) -> float (lower = more similar)
    
    Returns:
        List of (pos_idx, neg_idx) pairs in order of similarity
    """
    if similarity_fn is None:
        similarity_fn = similarity_aggregate
    
    # Compute all pairwise similarities
    similarities = []
    for pos_i, adj_pos in graphs_pos:
        for neg_j, adj_neg in graphs_neg:
            sim = similarity_fn(adj_pos, adj_neg)
            similarities.append((sim, pos_i, neg_j))
    
    # Sort by similarity (best matches first)
    similarities.sort(key=lambda x: x[0])
    
    # Greedily assign without replacement
    used_pos = set()
    used_neg = set()
    pairs = []
    
    for sim, pos_i, neg_j in similarities:
        if pos_i not in used_pos and neg_j not in used_neg:
            pairs.append((pos_i, neg_j))
            used_pos.add(pos_i)
            used_neg.add(neg_j)
            if len(pairs) == min(len(graphs_pos), len(graphs_neg)):
                break
    
    return pairs


def filter_by_statistic(
    graphs: List[Tuple[int, torch.Tensor]],
    target_value: float,
    tolerance: float,
    stat_key: str = "edge_count",
) -> List[Tuple[int, torch.Tensor]]:
    """Filter graphs by a specific statistic within tolerance.
    
    Args:
        graphs: list of (idx, adjacency_matrix)
        target_value: target value for the statistic
        tolerance: +/- tolerance window
        stat_key: 'node_count', 'edge_count', or 'two_hop_count'
    
    Returns:
        Filtered list of (idx, adj) tuples
    """
    filtered = []
    for idx, adj in graphs:
        stats = graph_statistics_dict(adj)
        value = stats[stat_key]
        if abs(value - target_value) <= tolerance:
            filtered.append((idx, adj))
    return filtered


def create_matched_pairs(
    graphs_with_labels: List[Tuple[int, torch.Tensor, float]],
    seed: int = 0,
    similarity_fn=None,
) -> List[Tuple[int, int, float, float]]:
    """Create matched positive/negative pairs from a dataset.
    
    Args:
        graphs_with_labels: list of (idx, adjacency_matrix, label) where label in {0, 1}
        seed: random seed for reproducibility
        similarity_fn: similarity function for matching
    
    Returns:
        List of (pos_idx, neg_idx, similarity, label_pair) tuples
    """
    if similarity_fn is None:
        similarity_fn = similarity_aggregate
    
    pos_graphs = [(idx, adj) for idx, adj, label in graphs_with_labels if label > 0.5]
    neg_graphs = [(idx, adj) for idx, adj, label in graphs_with_labels if label < 0.5]
    
    if not pos_graphs or not neg_graphs:
        return []
    
    # Match greedily
    pairs = match_graphs_greedy(pos_graphs, neg_graphs, similarity_fn)
    
    # Add similarity scores and labels
    result = []
    for pos_idx, neg_idx in pairs:
        # Find adjacency matrices
        adj_pos = next(adj for idx, adj in pos_graphs if idx == pos_idx)
        adj_neg = next(adj for idx, adj in neg_graphs if idx == neg_idx)
        sim = similarity_fn(adj_pos, adj_neg)
        result.append((pos_idx, neg_idx, sim, 1.0))  # label_pair always 1.0 (matched)
    
    return result


def apply_train_test_leakage_control(
    train_graphs: List[Tuple[int, torch.Tensor, float]],
    test_graphs: List[Tuple[int, torch.Tensor, float]],
    seed: int = 0,
    stat_keys: Optional[List[str]] = None,
) -> Tuple[List[int], List[int]]:
    """Identify and remove statistically similar graphs between train and test.
    
    Returns:
        (train_keep_indices, test_keep_indices) - indices to keep for uncontaminated split
    """
    if stat_keys is None:
        stat_keys = ["node_count", "edge_count", "two_hop_count"]
    
    rng = np.random.default_rng(seed)
    
    # Compute statistics for all graphs
    train_stats = [graph_statistics_dict(adj) for _, adj, _ in train_graphs]
    test_stats = [graph_statistics_dict(adj) for _, adj, _ in test_graphs]
    
    # Mark contaminated indices
    train_contam = set()
    test_contam = set()
    
    # For each test graph, check if similar train graph exists
    for test_idx, test_stat in enumerate(test_stats):
        for train_idx, train_stat in enumerate(train_stats):
            # Check if all statistics are very close
            all_close = True
            for key in stat_keys:
                val_train = train_stat[key]
                val_test = test_stat[key]
                # Use relative tolerance
                denom = max(abs(val_train), abs(val_test), 1e-6)
                rel_diff = abs(val_train - val_test) / denom
                if rel_diff > 0.1:  # 10% tolerance
                    all_close = False
                    break
            
            if all_close:
                # Mark both as contaminated
                train_contam.add(train_idx)
                test_contam.add(test_idx)
    
    train_keep = [i for i in range(len(train_graphs)) if i not in train_contam]
    test_keep = [i for i in range(len(test_graphs)) if i not in test_contam]
    
    return train_keep, test_keep


if __name__ == "__main__":
    # Quick test
    import torch
    
    # Create sample graphs
    adj1 = torch.zeros((5, 5), dtype=torch.bool)
    adj1[0, 1] = adj1[1, 0] = True
    adj1[1, 2] = adj1[2, 1] = True
    
    adj2 = torch.zeros((5, 5), dtype=torch.bool)
    adj2[0, 1] = adj2[1, 0] = True
    adj2[2, 3] = adj2[3, 2] = True
    
    print("Graph 1 stats:", graph_statistics_dict(adj1))
    print("Graph 2 stats:", graph_statistics_dict(adj2))
    print("Similarity (Euclidean):", similarity_euclidean_degree(adj1, adj2))
    print("Similarity (JS):", similarity_jensenshannon_degree(adj1, adj2))
    print("Similarity (Aggregate):", similarity_aggregate(adj1, adj2))
