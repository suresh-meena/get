import torch
from types import SimpleNamespace
from get.model import GETLayer

def test_get_layer_grad():
    print("Verifying GETLayer analytical gradients...")
    d = 16
    R = 2
    K = 0
    num_nodes = 5
    
    # Use num_heads=1 for small d
    layer = GETLayer(d=d, R=R, K=K, num_heads=1, norm_style="et")
    X = torch.randn(num_nodes, d, requires_grad=True)
    
    # Mock batch data
    c_2 = torch.tensor([0, 1, 2])
    u_2 = torch.tensor([1, 2, 0])
    c_3 = torch.tensor([0])
    u_3 = torch.tensor([1])
    v_3 = torch.tensor([2])
    t_tau = torch.tensor([0])
    
    batch_data = SimpleNamespace(
        c_2=c_2, u_2=u_2, c_3=c_3, u_3=u_3, v_3=v_3, t_tau=t_tau,
        ptr=torch.tensor([0, num_nodes]),
        batch=torch.zeros(num_nodes, dtype=torch.long)
    )
    
    # 1. Autograd reference
    E_ref = layer.compute_energy(X, batch_data)
    grad_ref = torch.autograd.grad(E_ref, X)[0]
    
    # 2. Analytical
    E_ana, grad_ana = layer.energy_and_grad(X, batch_data)
    
    diff_E = (E_ref - E_ana).abs().max().item()
    diff_grad = (grad_ref - grad_ana).abs().max().item()
    
    print(f"  Energy diff: {diff_E:.2e}")
    print(f"  Grad diff:   {diff_grad:.2e}")
    assert diff_grad < 1e-4, "GETLayer grad verification failed!"

def test_et_faithful_grad():
    print("Verifying ETFaithfulCore analytical gradients...")
    from get.et_core import ETCoreBlock
    d = 16
    num_heads = 2
    head_dim = 8
    num_memories = 4
    num_nodes = 6
    
    core = ETCoreBlock(d, num_heads, head_dim, num_memories)
    g = torch.randn(num_nodes, d, requires_grad=True)
    
    c_aug = torch.tensor([0, 1, 2, 3])
    u_aug = torch.tensor([1, 2, 3, 0])
    
    # Sparse Mode
    E_ref = core.energy(g, c_aug, u_aug, None, mask_mode="sparse")
    grad_ref = torch.autograd.grad(E_ref, g)[0]
    
    E_ana, grad_ana = core.energy_and_grad(g, c_aug, u_aug, None, mask_mode="sparse")
    
    diff_grad_sparse = (grad_ref - grad_ana).abs().max().item()
    print(f"  Sparse Grad diff: {diff_grad_sparse:.2e}")
    
    # Dense Mode (Batched)
    B = 2
    N = 4
    g_batch = torch.randn(B, N, d, requires_grad=True)
    dense_mod = torch.randn(B, num_heads, N, N)
    
    E_ref_dense = core.energy(g_batch, None, None, None, mask_mode="official_dense", dense_modulation=dense_mod)
    grad_ref_dense = torch.autograd.grad(E_ref_dense.sum(), g_batch)[0]
    
    E_ana_dense, grad_ana_dense = core.energy_and_grad(g_batch, None, None, None, mask_mode="official_dense", dense_modulation=dense_mod)
    
    diff_grad_dense = (grad_ref_dense - grad_ana_dense).abs().max().item()
    print(f"  Dense Grad diff:  {diff_grad_dense:.2e}")
    
    assert diff_grad_sparse < 1e-4, "Sparse grad verification failed!"
    assert diff_grad_dense < 1e-4, "Dense grad verification failed!"

def test_compilation():
    print("Verifying maybe_compile_model support...")
    from get.utils import maybe_compile_model
    
    d = 16
    layer = GETLayer(d=d, K=0, num_heads=1, norm_style="et")
    maybe_compile_model(layer, enabled=True)
    
    X = torch.randn(5, d, requires_grad=True)
    c_2 = torch.tensor([0, 1])
    u_2 = torch.tensor([1, 0])
    batch_data = SimpleNamespace(
        c_2=c_2, u_2=u_2, c_3=torch.tensor([], dtype=torch.long),
        u_3=torch.tensor([], dtype=torch.long),
        v_3=torch.tensor([], dtype=torch.long),
        t_tau=torch.tensor([], dtype=torch.long),
        ptr=torch.tensor([0, 5]),
        batch=torch.zeros(5, dtype=torch.long)
    )
    
    # Warmup
    _ = layer(X, batch_data, step_size=0.1)
    # Execute
    X_next, E = layer(X, batch_data, step_size=0.1)
    print(f"  Execution successful. E: {E.item():.4f}")

if __name__ == "__main__":
    test_get_layer_grad()
    test_et_faithful_grad()
    test_compilation()
    print("All analytical gradients and compilation verified successfully!")
