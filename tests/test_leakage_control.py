"""
Tests for Stage 1 leakage-controlled dataset generation.
"""
import pytest
import torch
from argparse import Namespace
from get.data.protocol import build_dataset
from get.data.leakage_control import (
    compute_edge_count,
    compute_degree_histogram,
    compute_two_hop_count,
    similarity_aggregate,
)


class TestLeakageControl:
    """Test leakage-controlled Stage 1 generators."""
    
    def test_basic_statistic_computation(self):
        """Test that basic graph statistics can be computed."""
        # Create a simple graph
        adj = torch.zeros((5, 5), dtype=torch.bool)
        adj[0, 1] = adj[1, 0] = True
        adj[1, 2] = adj[2, 1] = True
        adj[2, 3] = adj[3, 2] = True
        
        edge_count = compute_edge_count(adj)
        degree_hist = compute_degree_histogram(adj, bins=5)
        twohop_count = compute_two_hop_count(adj)
        
        assert edge_count == 3
        assert degree_hist.shape == (5,)
        assert degree_hist.sum() > 0
        assert twohop_count >= 0
    
    @pytest.mark.parametrize(
        "task,num_graphs",
        [
            ("stage1_wedge_triangle", 50),
            ("stage1_wedge_triangle_matched", 50),
            ("stage1_wedge_triangle_degree_only", 50),
            ("stage1_wedge_triangle_edge_only", 50),
            ("stage1_cycle_parity", 50),
            ("stage1_cycle_parity_matched", 50),
            ("stage1_max3sat", 30),
            ("stage1_max3sat_matched", 30),
        ]
    )
    def test_generator_produces_graphs(self, task, num_graphs):
        """Test that each generator produces graphs."""
        args = Namespace(
            seed=42,
            max_graphs=num_graphs,
            min_nodes=5,
            max_nodes=10,
            edge_prob=0.2,
            in_dim=8,
            max_motifs_per_anchor=4,
        )
        
        graphs = build_dataset(task, args)[0]
        
        # Check that we get a list of samples
        assert isinstance(graphs, list)
        assert len(graphs) > 0
        
        # Check that each sample has required fields
        for sample in graphs:
            # Handle both dict and GraphSampleData objects
            if isinstance(sample, dict):
                assert "x" in sample  # node features
                assert "y" in sample  # label
                assert sample["y"].numel() > 0
                assert sample["y"][0].item() in {0.0, 1.0}
            else:
                # GraphSampleData object
                assert hasattr(sample, "x")  # node features
                assert hasattr(sample, "y")  # label
                assert sample.y.numel() > 0
                assert sample.y[0].item() in {0.0, 1.0}
    
    def test_matched_preserves_balance(self):
        """Test that matched generators produce balanced classes."""
        args = Namespace(
            seed=42,
            max_graphs=40,
            min_nodes=5,
            max_nodes=10,
            edge_prob=0.2,
            in_dim=8,
            max_motifs_per_anchor=4,
        )
        
        # Compare original vs matched
        graphs_orig = build_dataset("stage1_wedge_triangle", args)[0]
        graphs_matched = build_dataset("stage1_wedge_triangle_matched", args)[0]
        
        # Matched should have fewer or equal samples (due to keeping only matched pairs)
        assert len(graphs_matched) <= len(graphs_orig)
        
        # But should still have both classes
        labels_matched = []
        for g in graphs_matched:
            if isinstance(g, dict):
                labels_matched.append(float(g["y"][0].item()))
            else:
                labels_matched.append(float(g.y[0].item()))
        
        assert 0.0 in labels_matched
        assert 1.0 in labels_matched
    
    def test_baseline_generators_work(self):
        """Test that baseline (degree/edge/twohop only) generators work."""
        args = Namespace(
            seed=42,
            max_graphs=30,
            min_nodes=5,
            max_nodes=10,
            edge_prob=0.2,
            in_dim=8,
            max_motifs_per_anchor=4,
        )
        
        for task in ["stage1_wedge_triangle_degree_only", 
                     "stage1_wedge_triangle_edge_only",
                     "stage1_wedge_triangle_twohop_only"]:
            graphs = build_dataset(task, args)[0]
            assert len(graphs) > 0
            
            # Verify samples are valid
            for g in graphs:
                assert "x" in g
                assert "y" in g
                assert float(g["y"][0].item()) in {0.0, 1.0}
    
    def test_samples_have_edge_structure(self):
        """Test that generated samples have proper graph structure."""
        args = Namespace(
            seed=42,
            max_graphs=20,
            min_nodes=5,
            max_nodes=10,
            edge_prob=0.2,
            in_dim=8,
            max_motifs_per_anchor=4,
        )
        
        graphs = build_dataset("stage1_wedge_triangle_matched", args)[0]
        
        for sample in graphs:
            assert "x" in sample
            assert sample["x"].dim() == 2
            assert sample["x"].size(1) >= 1  # At least some features
            
            # Check for edge structure info
            assert "edge_index" in sample or "c_2" in sample
            
            # Verify label
            label = float(sample["y"][0].item())
            assert label in {0.0, 1.0}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
