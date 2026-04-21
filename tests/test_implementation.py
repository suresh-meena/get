import torch
import torch.nn as nn
from get import FullGET
from get.data import add_structural_node_features, collate_get_batch
from get.energy import compute_energy_GET
import networkx as nx
import random
import sys

def generate_random_graphs(num_graphs, n_min=5, n_max=12, d=4, p=0.4):
    graphs = []
    for _ in range(num_graphs):
        n = random.randint(n_min, n_max)
        G = nx.erdos_renyi_graph(n, p)
        edges = list(G.edges())
        x = torch.randn(n, d, dtype=torch.float64) # Float64 for gradcheck
        graphs.append({'x': x, 'edges': edges})
    return graphs

def test_finite_difference():
    print("Running Finite-Difference Checks...")
    graphs = generate_random_graphs(1, d=8)
    batch = collate_get_batch(graphs)
    
    # We need float64 for finite difference accuracy
    model = FullGET(in_dim=8, d=8, num_classes=1, num_steps=1).to(torch.float64)
    model.eval()
    
    # Function to check: energy computation given X
    def energy_func(X):
        # We need to manually apply layernorm here to treat X as the input variable
        G = model.get_layer.layernorm(X)
        params = model.get_layer.get_params_dict()
        E = compute_energy_GET(X, G, batch.c_2, batch.u_2, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, params)
        return E

    X = batch.x.clone().requires_grad_(True)
    
    try:
        # torch.autograd.gradcheck takes a function and inputs, checks analytical vs numerical
        test_passed = torch.autograd.gradcheck(energy_func, (X,), eps=1e-5, atol=1e-4, rtol=1e-4)
        print(f"Finite difference check passed: {test_passed}")
    except Exception as e:
        print(f"Finite difference check failed: {e}")
        sys.exit(1)

def test_equivariance():
    print("Running Equivariance Checks...")
    graphs = generate_random_graphs(1, n_min=8, n_max=8, d=4)
    g = graphs[0]
    
    # Original graph
    batch_orig = collate_get_batch([g])
    model = FullGET(in_dim=4, d=4, num_classes=1, num_steps=1).to(torch.float64)
    model.eval()
    
    X_orig = batch_orig.x.clone().requires_grad_(True)
    E_orig = compute_energy_GET(X_orig, model.get_layer.layernorm(X_orig), 
                                batch_orig.c_2, batch_orig.u_2, batch_orig.c_3, batch_orig.u_3, batch_orig.v_3, batch_orig.t_tau, model.get_layer.get_params_dict())
    grad_orig = torch.autograd.grad(E_orig, X_orig)[0]
    
    # Permute graph
    perm = torch.randperm(8)
    inv_perm = torch.argsort(perm)
    
    x_perm = g['x'][perm]
    edges_perm = [(inv_perm[u].item(), inv_perm[v].item()) for u, v in g['edges']]
    
    batch_perm = collate_get_batch([{'x': x_perm, 'edges': edges_perm}])
    X_perm = batch_perm.x.clone().requires_grad_(True)
    
    E_perm = compute_energy_GET(X_perm, model.get_layer.layernorm(X_perm), 
                                batch_perm.c_2, batch_perm.u_2, batch_perm.c_3, batch_perm.u_3, batch_perm.v_3, batch_perm.t_tau, model.get_layer.get_params_dict())
    grad_perm = torch.autograd.grad(E_perm, X_perm)[0]
    
    # Check energy invariance
    assert torch.allclose(E_orig, E_perm), f"Energy not invariant: {E_orig.item()} vs {E_perm.item()}"
    
    # Check gradient equivariance
    assert torch.allclose(grad_orig, grad_perm[inv_perm]), "Gradients are not equivariant"
    print("Equivariance check passed.")

def test_monotone_descent():
    print("Running Monotone Armijo Descent Checks...")
    graphs = generate_random_graphs(100, n_min=5, n_max=12, d=4)
    model = FullGET(in_dim=4, d=4, num_classes=1, num_steps=1).to(torch.float64)
    model.eval()
    
    failures = 0
    for g in graphs:
        batch = collate_get_batch([g])
        X = batch.x.clone().requires_grad_(True)
        
        G = model.get_layer.layernorm(X)
        E_0 = compute_energy_GET(X, G, batch.c_2, batch.u_2, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, model.get_layer.get_params_dict())
        
        grad_X = torch.autograd.grad(E_0, X)[0]
        
        # Backtracking line search
        eta = 1.0
        c = 0.5
        gamma = 0.5
        grad_norm_sq = (grad_X ** 2).sum()
        
        while True:
            X_1 = X - eta * grad_X
            G_1 = model.get_layer.layernorm(X_1)
            E_1 = compute_energy_GET(X_1, G_1, batch.c_2, batch.u_2, batch.c_3, batch.u_3, batch.v_3, batch.t_tau, model.get_layer.get_params_dict())
            
            if E_1 <= E_0 - c * eta * grad_norm_sq + 1e-8:
                # Armijo condition satisfied
                break
            
            eta *= gamma
            if eta < 1e-10:
                print(f"Failed to find step size. E0={E_0.item()}, E1={E_1.item()}, diff={E_1.item() - E_0.item()}")
                failures += 1
                break
                
        # The paper specifies: E_GET(X_t+1) <= E_GET(X_t) + 10^-8
        assert E_1 <= E_0 + 1e-8, f"Monotone check failed: E_1={E_1.item()} > E_0={E_0.item()}"
    
    if failures == 0:
        print("Monotone descent check passed on all 100 random instances.")
    else:
        print(f"Monotone descent failed on {failures} instances.")

def test_model_armijo_inference_mode():
    print("Running Model Armijo Inference Mode Check...")
    graphs = generate_random_graphs(1, n_min=10, n_max=10, d=4)
    batch = collate_get_batch(graphs)
    model = FullGET(in_dim=4, d=4, num_classes=1, num_steps=5).to(torch.float64)
    model.eval()

    out, energy_trace, solver_stats = model(
        batch,
        task_level='graph',
        inference_mode='armijo',
        return_solver_stats=True,
    )
    assert out.shape[0] == 1, "Graph output should have one row for one graph."
    assert len(energy_trace) == model.num_steps, "Armijo trace length mismatch."
    assert len(solver_stats['step_sizes']) == model.num_steps, "Step size trace length mismatch."
    assert len(solver_stats['backtracks']) == model.num_steps, "Backtrack trace length mismatch."
    assert len(solver_stats['accepted']) == model.num_steps, "Accepted trace length mismatch."

    # Armijo accepted steps are monotone by construction; failed searches keep X unchanged.
    for t in range(1, len(energy_trace)):
        assert energy_trace[t] <= energy_trace[t - 1] + 1e-8, (
            f"Armijo energy not monotone at step {t}: "
            f"{energy_trace[t]} > {energy_trace[t - 1]}"
        )
    print("Model Armijo inference mode check passed.")

def test_data_preparation_helpers():
    print("Running Data Preparation Helper Checks...")
    graph = {
        "x": torch.ones(3, 1),
        "edges": [(0, 1), (1, 2)],
        "edge_attr": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    }
    enriched = add_structural_node_features(graph, include_degree=True, include_motif_counts=True)
    assert enriched["x"].shape == (3, 4), "Expected original, degree, open-wedge, triangle features."

    batch = collate_get_batch([graph])
    assert batch.edge_attr.shape == (4, 2), "Undirected edge_attr should align to directed pairwise incidences."
    assert torch.allclose(batch.edge_attr[0], graph["edge_attr"][0])
    assert torch.allclose(batch.edge_attr[-1], graph["edge_attr"][1])
    print("Data preparation helper checks passed.")

if __name__ == "__main__":
    test_finite_difference()
    test_equivariance()
    test_monotone_descent()
    test_model_armijo_inference_mode()
    test_data_preparation_helpers()
    print("All Tier 0 implementation checks passed!")
