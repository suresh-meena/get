import torch
import pytest
from get import GETModel
from get.data import collate_get_batch

def test_motif_symmetry():
    """Verify that s(i; j, k) == s(i; k, j) for motifs."""
    d = 16
    model = GETModel(in_dim=d, d=d, num_motif_types=2).cuda()
    model.eval()
    
    x = torch.randn(5, d, device="cuda")
    # Motif (0; 1, 2)
    c3 = torch.tensor([0, 0], device="cuda")
    u3 = torch.tensor([1, 2], device="cuda")
    v3 = torch.tensor([2, 1], device="cuda")
    t_tau = torch.tensor([0, 0], device="cuda")
    
    # We can't easily use FullGET for this specific check because it extracts motifs automatically.
    # We'll check the underlying energy function.
    from get.energy.motif import compute_motif_energy
    
    G = model.get_layers[0].layernorm(x)
    params = model.get_layers[0].get_params_dict()
    projections = model.get_layers[0]._build_projections(G)
    
    # compute_motif_energy(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes)
    batch = torch.zeros(5, dtype=torch.long, device="cuda")
    
    # Score for (0; 1, 2)
    E1 = compute_motif_energy(G, c3[:1], u3[:1], v3[:1], t_tau[:1], batch, 1, params, projections, 5)
    # Score for (0; 2, 1)
    E2 = compute_motif_energy(G, c3[1:], u3[1:], v3[1:], t_tau[1:], batch, 1, params, projections, 5)
    
    assert torch.allclose(E1, E2)
    print("Motif symmetry verified.")

def test_empty_graph_robustness():
    """Verify that empty graphs (no edges, no motifs) don't crash and return quadratic energy only."""
    d = 16
    model = GETModel(in_dim=d, d=d).cuda()
    model.eval()
    
    x = torch.randn(5, d, device="cuda")
    # No edges, no motifs
    batch = collate_get_batch([{"x": x, "edges": []}])
    batch = batch.to("cuda")
    
    with torch.no_grad():
        out, trace = model(batch)
    
    assert not torch.isnan(out).any()
    assert len(trace) > 0
    print("Empty graph robustness verified.")

def test_permutation_equivariance():
    """Verify that permuting nodes results in permuted gradients."""
    d = 16
    model = GETModel(in_dim=d, d=d).cuda()
    model.eval()
    
    x = torch.randn(5, d, device="cuda")
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    batch = collate_get_batch([{"x": x, "edges": edges}])
    batch = batch.to("cuda")
    
    # Original grad
    x_orig = x.clone().requires_grad_(True)
    batch.x = x_orig
    out, trace = model(batch)
    out.sum().backward()
    grad_orig = x_orig.grad.clone()
    
    # Permute
    perm = torch.randperm(5)
    inv_perm = torch.argsort(perm)
    x_perm = x[perm].clone().requires_grad_(True)
    
    new_edges = []
    # edge (u, v) becomes (perm_inv[u], perm_inv[v])? No, u, v are indices.
    # If node 0 is now at perm[0], then edge (0, 1) becomes (perm[0], perm[1]).
    # Wait, let's be careful.
    # x_perm[i] = x[perm[i]]
    # So if original node u is now at position i such that perm[i] = u.
    # So i = inv_perm[u].
    # Edge (u, v) becomes (inv_perm[u], inv_perm[v]).
    
    edges_perm = [(inv_perm[u].item(), inv_perm[v].item()) for u, v in edges]
    batch_perm = collate_get_batch([{"x": x_perm, "edges": edges_perm}])
    batch_perm = batch_perm.to("cuda")
    
    out_perm, trace_perm = model(batch_perm)
    out_perm.sum().backward()
    grad_perm = x_perm.grad.clone()
    
    # grad_perm[i] should be grad_orig[perm[i]]
    assert torch.allclose(grad_perm, grad_orig[perm], atol=1e-6)
    print("Permutation equivariance verified.")

if __name__ == "__main__":
    test_motif_symmetry()
    test_empty_graph_robustness()
    test_permutation_equivariance()
