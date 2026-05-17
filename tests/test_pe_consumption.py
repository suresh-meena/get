"""
Tests for positional encoding consumption in GET models.
"""
import pytest
import torch
import torch.nn as nn
from argparse import Namespace
from get.models.factory import build_model


class TestPEConsumption:
    """Verify that positional encodings are properly consumed by GET."""

    @pytest.mark.parametrize(
        "model_name,pos_k",
        [
            ("fullget_local", 8),
            ("nomotif_local", 8),
            ("fullget_global", 8),
            ("nomotif_global", 8),
            ("pairwise_only", 4),
            ("fullget_local", 0),  # pos_k=0 means no PE
        ]
    )
    def test_get_accepts_pe_in_batch(self, model_name, pos_k):
        """Test that GET models accept and process PE in batch dict."""
        cfg = Namespace(
            model_name=model_name,
            in_dim=8,
            hidden_dim=128,  # 4 heads * 32 dims per head
            num_heads=4,
            head_dim=32,
            num_classes=2,
            task_type="graph_binary",
            pos_k=pos_k,
        )
        model = build_model(cfg)
        model.eval()
        
        # Create minimal batch with PE
        batch = {
            "x": torch.randn(16, 8),  # 16 nodes, 8 features
            "batch": torch.arange(4).repeat_interleave(4),  # 4 graphs, 4 nodes each
            "y": torch.randint(0, 2, (4,)),  # 4 binary labels
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "c_2": torch.zeros(0, dtype=torch.long),
            "u_2": torch.zeros(0, dtype=torch.long),
            "c_3": torch.zeros(0, dtype=torch.long),
            "u_3": torch.zeros(0, dtype=torch.long),
            "v_3": torch.zeros(0, dtype=torch.long),
            "t_tau": torch.zeros(0, dtype=torch.long),
        }
        
        if pos_k > 0:
            batch["pos"] = torch.randn(16, pos_k)  # PE for each node
        
        # Forward pass should not raise
        with torch.no_grad():
            logits = model(batch)
        
        assert logits.shape == (4, 2)  # 4 samples, 2 classes
    
    def test_pe_concatenation_dimension(self):
        """Test that PE is correctly concatenated in the encoder."""
        in_dim = 8
        pos_k = 5
        hidden_dim = 128  # 4 heads * 32 dims per head
        
        cfg = Namespace(
            model_name="fullget_local",
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_heads=4,
            head_dim=32,
            num_classes=2,
            task_type="graph_binary",
            pos_k=pos_k,
        )
        model = build_model(cfg)
        
        # Verify encoder input dimension includes PE
        expected_encoder_in_dim = in_dim + pos_k
        first_layer = model.encoder[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.in_features == expected_encoder_in_dim, \
            f"Expected encoder input dim {expected_encoder_in_dim}, got {first_layer.in_features}"
    
    def test_pe_optional(self):
        """Test that PE is optional when pos_k=0 (batch without PE should work)."""
        cfg = Namespace(
            model_name="fullget_local",
            in_dim=8,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            num_classes=2,
            task_type="graph_binary",
            pos_k=0,  # No PE expected
        )
        model = build_model(cfg)
        model.eval()
        
        batch = {
            "x": torch.randn(16, 8),
            "batch": torch.arange(4).repeat_interleave(4),
            "y": torch.randint(0, 2, (4,)),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "c_2": torch.zeros(0, dtype=torch.long),
            "u_2": torch.zeros(0, dtype=torch.long),
            "c_3": torch.zeros(0, dtype=torch.long),
            "u_3": torch.zeros(0, dtype=torch.long),
            "v_3": torch.zeros(0, dtype=torch.long),
            "t_tau": torch.zeros(0, dtype=torch.long),
        }
        # No "pos" key in batch
        
        with torch.no_grad():
            logits = model(batch)
        
        # Should work with just x when pos_k=0
        assert logits.shape == (4, 2)
    
    def test_pe_shape_mismatch_handling(self):
        """Test graceful handling of PE with different shapes."""
        cfg = Namespace(
            model_name="fullget_local",
            in_dim=8,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            num_classes=2,
            task_type="graph_binary",
            pos_k=4,
        )
        model = build_model(cfg)
        model.eval()
        
        # PE with too many dimensions (will be trimmed)
        batch = {
            "x": torch.randn(16, 8),
            "batch": torch.arange(4).repeat_interleave(4),
            "y": torch.randint(0, 2, (4,)),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "c_2": torch.zeros(0, dtype=torch.long),
            "u_2": torch.zeros(0, dtype=torch.long),
            "c_3": torch.zeros(0, dtype=torch.long),
            "u_3": torch.zeros(0, dtype=torch.long),
            "v_3": torch.zeros(0, dtype=torch.long),
            "t_tau": torch.zeros(0, dtype=torch.long),
            "pos": torch.randn(16, 10),  # More than pos_k=4
        }
        
        with torch.no_grad():
            logits = model(batch)
        
        assert logits.shape == (4, 2)
        
        # PE with fewer dimensions (will be padded)
        batch["pos"] = torch.randn(16, 2)  # Less than pos_k=4
        
        with torch.no_grad():
            logits = model(batch)
        
        assert logits.shape == (4, 2)
    
    def test_pe_1d_to_2d_expansion(self):
        """Test that 1D PE is properly expanded to 2D."""
        cfg = Namespace(
            model_name="fullget_local",
            in_dim=8,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            num_classes=2,
            task_type="graph_binary",
            pos_k=4,
        )
        model = build_model(cfg)
        model.eval()
        
        batch = {
            "x": torch.randn(16, 8),
            "batch": torch.arange(4).repeat_interleave(4),
            "y": torch.randint(0, 2, (4,)),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "c_2": torch.zeros(0, dtype=torch.long),
            "u_2": torch.zeros(0, dtype=torch.long),
            "c_3": torch.zeros(0, dtype=torch.long),
            "u_3": torch.zeros(0, dtype=torch.long),
            "v_3": torch.zeros(0, dtype=torch.long),
            "t_tau": torch.zeros(0, dtype=torch.long),
            "pos": torch.randn(16),  # 1D PE
        }
        
        with torch.no_grad():
            logits = model(batch)
        
        assert logits.shape == (4, 2)
    
    def test_metadata_reports_pos_k(self):
        """Test that model metadata includes pos_k setting."""
        pos_k = 6
        cfg = Namespace(
            model_name="fullget_local",
            in_dim=8,
            hidden_dim=128,
            num_heads=4,
            head_dim=32,
            num_classes=2,
            task_type="graph_binary",
            pos_k=pos_k,
        )
        model = build_model(cfg)
        
        metadata = model.energy_metadata()
        assert "pos_k" in metadata
        assert metadata["pos_k"] == pos_k


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
