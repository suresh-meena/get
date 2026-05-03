import torch
import pytest
from torch.autograd import gradcheck

from get.energy.quadratic import compute_quadratic_energy
from get.energy.pairwise import compute_pairwise_energy
from get.energy.motif import compute_motif_energy
from get.energy.memory import compute_memory_energy

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def dummy_data(device):
    num_nodes = 5
    num_graphs = 2
    d = 4
    
    # We use float64 for gradcheck
    X = torch.randn(num_nodes, d, dtype=torch.float64, device=device, requires_grad=True)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long, device=device)
    
    c_2 = torch.tensor([0, 0, 1, 1, 3, 4], dtype=torch.long, device=device)
    u_2 = torch.tensor([1, 2, 0, 2, 4, 3], dtype=torch.long, device=device)
    
    c_3 = torch.tensor([0, 0, 3], dtype=torch.long, device=device)
    u_3 = torch.tensor([1, 2, 4], dtype=torch.long, device=device)
    v_3 = torch.tensor([2, 1, 4], dtype=torch.long, device=device)
    t_tau = torch.tensor([0, 0, 1], dtype=torch.long, device=device)
    
    params = {
        'd': d,
        'beta_max': 5.0,
        'use_pairwise': True,
        'lambda_2': 1.0,
        'beta_2': 1.0,
        'pairwise_symmetric': True,
        'use_motif': True,
        'lambda_3': 0.5,
        'beta_3': 1.0,
        'R': 2,
        'T_tau': torch.randn(2, 2, 2, d//2, dtype=torch.float64, device=device), # num_motif_types, num_heads, R, head_dim
        'use_memory': True,
        'lambda_m': 1.0,
        'beta_m': 1.0,
        'K': 3
    }
    
    projections = {
        'Q2': torch.randn(num_nodes, 2, d//2, dtype=torch.float64, device=device), # num_nodes, num_heads, head_dim
        'K2': torch.randn(num_nodes, 2, d//2, dtype=torch.float64, device=device),
        'a_2': None,
        'Q3': torch.randn(num_nodes, 2, 2, d//2, dtype=torch.float64, device=device), # num_nodes, num_heads, R, head_dim
        'K3': torch.randn(num_nodes, 2, 2, d//2, dtype=torch.float64, device=device),
        'Qm': torch.randn(num_nodes, 2, d//2, dtype=torch.float64, device=device),
        'Km': torch.randn(2, 3, d//2, dtype=torch.float64, device=device) # num_heads, K, head_dim
    }
    
    return X, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, projections, num_nodes

def test_quadratic_energy_gradcheck(dummy_data):
    X, batch, num_graphs, _, _, _, _, _, _, _, _, _ = dummy_data
    
    def func(x):
        return compute_quadratic_energy(x, batch, num_graphs)
        
    assert gradcheck(func, (X,), eps=1e-6, atol=1e-4)


def test_quadratic_energy_is_graph_size_normalized():
    X = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    out = compute_quadratic_energy(X, batch, num_graphs=2)
    expected = torch.tensor([1.25, 4.5], dtype=torch.float32)
    assert torch.allclose(out, expected)

def test_pairwise_energy_gradcheck(dummy_data):
    X, batch, num_graphs, c_2, u_2, _, _, _, _, params, projections, num_nodes = dummy_data
    
    # We test gradient w.r.t Q2 and K2 to ensure the function is differentiable correctly
    Q2 = projections['Q2'].requires_grad_(True)
    K2 = projections['K2'].requires_grad_(True)
    
    def func(q, k):
        proj = projections.copy()
        proj['Q2'] = q
        proj['K2'] = k
        return compute_pairwise_energy(X, c_2, u_2, batch, num_graphs, params, proj, num_nodes)
        
    assert gradcheck(func, (Q2, K2), eps=1e-6, atol=1e-4)

def test_motif_energy_gradcheck(dummy_data):
    X, batch, num_graphs, _, _, c_3, u_3, v_3, t_tau, params, projections, num_nodes = dummy_data
    
    Q3 = projections['Q3'].requires_grad_(True)
    K3 = projections['K3'].requires_grad_(True)
    
    def func(q, k):
        proj = projections.copy()
        proj['Q3'] = q
        proj['K3'] = k
        return compute_motif_energy(X, c_3, u_3, v_3, t_tau, batch, num_graphs, params, proj, num_nodes)
        
    assert gradcheck(func, (Q3, K3), eps=1e-6, atol=1e-4)

def test_memory_energy_gradcheck(dummy_data):
    X, batch, num_graphs, _, _, _, _, _, _, params, projections, _ = dummy_data
    
    Qm = projections['Qm'].requires_grad_(True)
    Km = projections['Km'].requires_grad_(True)
    
    def func(q, k):
        proj = projections.copy()
        proj['Qm'] = q
        proj['Km'] = k
        return compute_memory_energy(X, batch, num_graphs, params, proj)
        
    assert gradcheck(func, (Qm, Km), eps=1e-6, atol=1e-4)
