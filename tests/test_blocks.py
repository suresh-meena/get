import torch
from get.model import GETModel
from get.data import GETBatch
import pytest

def get_dummy_batch(num_nodes=10, num_graphs=2, in_dim=4):
    x = torch.randn(num_nodes, in_dim)
    batch = torch.randint(0, num_graphs, (num_nodes,))
    batch = torch.sort(batch)[0] # ensure contiguous graph ids
    ptr = torch.cat([torch.tensor([0]), torch.bincount(batch).cumsum(0)])
    
    # Just dummy incidence matrices
    c_2 = torch.randint(0, num_nodes, (15,))
    u_2 = torch.randint(0, num_nodes, (15,))
    c_3 = torch.randint(0, num_nodes, (5,))
    u_3 = torch.randint(0, num_nodes, (5,))
    v_3 = torch.randint(0, num_nodes, (5,))
    t_tau = torch.randint(0, 2, (5,), dtype=torch.int32)
    
    return GETBatch(x, c_2, u_2, c_3, u_3, v_3, t_tau, batch, ptr)

def test_model_blocks_parameter_sharing():
    # Model 1: 3 blocks, unshared
    model_unshared = GETModel(
        in_dim=4, d=16, num_blocks=3, share_block_weights=False,
        use_pairwise=True, use_motif=True, use_memory=True
    )
    # Model 2: 3 blocks, shared
    model_shared = GETModel(
        in_dim=4, d=16, num_blocks=3, share_block_weights=True,
        use_pairwise=True, use_motif=True, use_memory=True
    )
    
    batch = get_dummy_batch()
    model_unshared(batch, task_level='graph', inference_mode='fixed')
    model_shared(batch, task_level='graph', inference_mode='fixed')
    
    from torch.nn.parameter import UninitializedParameter
    unshared_params = sum(p.numel() for p in model_unshared.parameters() if p.requires_grad and not isinstance(p, UninitializedParameter))
    shared_params = sum(p.numel() for p in model_shared.parameters() if p.requires_grad and not isinstance(p, UninitializedParameter))
    
    # Unshared should have roughly 3x the parameters in the core layers
    assert unshared_params > shared_params

def test_model_blocks_forward():
    model = GETModel(
        in_dim=4, d=16, num_blocks=3, share_block_weights=False,
        use_pairwise=True, use_motif=True, use_memory=True, num_steps=2
    )
    batch = get_dummy_batch()
    
    out, energy_trace = model(batch, task_level='graph', inference_mode='fixed')
    assert out.shape == (2, 1) # num_graphs = 2, num_classes = 1
    assert len(energy_trace) == 6 # 3 blocks * 2 steps = 6
    
def test_model_blocks_armijo():
    model = GETModel(
        in_dim=4, d=16, num_blocks=2, share_block_weights=True,
        use_pairwise=True, use_motif=True, use_memory=True, num_steps=2
    )
    batch = get_dummy_batch()
    model.eval()
    
    out, energy_trace, stats = model(batch, task_level='node', inference_mode='armijo', return_solver_stats=True)
    assert out.shape == (10, 1) # num_nodes = 10, num_classes = 1
    assert len(energy_trace) == 4 # 2 blocks * 2 steps = 4
    assert stats['steps'] == 4
    
def test_model_blocks_gradients():
    torch.manual_seed(0)
    model = GETModel(
        in_dim=4, d=16, num_blocks=2, share_block_weights=False,
        use_pairwise=True, use_motif=True, use_memory=True, num_steps=1
    )
    batch = get_dummy_batch()
    
    out, _ = model(batch, task_level='graph', inference_mode='fixed')
    loss = out.sum()
    loss.backward()
    
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"


def test_shared_block_static_projections_are_reused(monkeypatch):
    model = GETModel(
        in_dim=4,
        d=16,
        num_blocks=3,
        share_block_weights=True,
        use_pairwise=True,
        use_motif=False,
        use_memory=False,
        num_heads=1,
    )
    batch = get_dummy_batch()
    batch.edge_attr = torch.randn(batch.c_2.numel(), 3)

    calls = {"edge_mlp": 0}
    original_forward = model.get_layers[0].edge_mlp.forward

    def counted_forward(*args, **kwargs):
        calls["edge_mlp"] += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.get_layers[0].edge_mlp, "forward", counted_forward)

    model._build_static_projections(batch)

    assert calls["edge_mlp"] == 1
            
if __name__ == "__main__":
    pytest.main([__file__])
