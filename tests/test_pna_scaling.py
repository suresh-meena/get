import torch
from get import FullGET
from get.data import collate_get_batch

def test_pna_scaling_changes_energy():
    d = 16
    in_dim = 4
    num_nodes = 10
    
    # Create two identical graphs
    x = torch.randn(num_nodes, in_dim)
    edges = [(i, (i+1)%num_nodes) for i in range(num_nodes)]
    graph = {"x": x, "edges": edges}
    batch = collate_get_batch([graph])
    
    # Model without PNA scaling
    model_no = FullGET(in_dim=in_dim, d=d, use_pna_scaling=False)
    model_no.eval()
    
    # Model with PNA scaling
    model_pna = FullGET(in_dim=in_dim, d=d, use_pna_scaling=True, avg_degree=1.0)
    model_pna.eval()
    
    # Copy weights
    model_pna.load_state_dict(model_no.state_dict())
    
    with torch.no_grad():
        out_no, trace_no, stats_no = model_no(batch, return_solver_stats=True)
        out_pna, trace_pna, stats_pna = model_pna(batch, return_solver_stats=True)
        
    # Energies should be different because of scaling
    assert not torch.allclose(torch.tensor(trace_no), torch.tensor(trace_pna))
    print("PNA scaling verified: energies differ as expected.")

if __name__ == "__main__":
    test_pna_scaling_changes_energy()
