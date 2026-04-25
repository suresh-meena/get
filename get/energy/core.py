"""Full GET energy composition: E_quad - E_att2 - E_att3 - E_mem."""
from .quadratic import compute_quadratic_energy
from .pairwise import compute_pairwise_energy
from .motif import compute_motif_energy
from .memory import compute_memory_energy


def compute_energy_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=None, degree_scaler=None):
    num_nodes = X.size(-2)
    E_quad = compute_quadratic_energy(X, batch, num_graphs)
    E_att2 = compute_pairwise_energy(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
    E_att3 = compute_motif_energy(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, degree_scaler=degree_scaler)
    E_mem = compute_memory_energy(G, batch, num_graphs, params, projections)

    if E_att2.dim() > E_quad.dim():
        E_att2 = E_att2.mean(dim=-2)
    if E_att3.dim() > E_quad.dim():
        E_att3 = E_att3.mean(dim=-2)
    if E_mem.dim() > E_quad.dim():
        E_mem = E_mem.mean(dim=-2)

    return E_quad - E_att2 - E_att3 - E_mem


def compute_energy_and_grad_GET(X, G, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections=None, degree_scaler=None):
    num_nodes = X.size(-2)
    E_quad = compute_quadratic_energy(X, batch, num_graphs)
    E_att2, (gQ2, gK2) = compute_pairwise_energy(G, c_2, u_2, batch, num_graphs, params, projections, num_nodes, return_grad=True, degree_scaler=degree_scaler)
    E_att3, (gQ3, gK3) = compute_motif_energy(G, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections, num_nodes, return_grad=True, degree_scaler=degree_scaler)
    E_mem, (gQm, _) = compute_memory_energy(G, batch, num_graphs, params, projections, return_grad=True)
    H = params.get('num_heads', 1)
    grads = {
        'grad_Q2': (gQ2 / H) if gQ2 is not None else None,
        'grad_K2': (gK2 / H) if gK2 is not None else None,
        'grad_Q3': (gQ3 / H) if gQ3 is not None else None,
        'grad_K3': (gK3 / H) if gK3 is not None else None,
        'grad_Qm': (gQm / H) if gQm is not None else None
    }
    E2 = E_att2.mean(dim=-2) if E_att2.dim() > E_quad.dim() else E_att2
    E3 = E_att3.mean(dim=-2) if E_att3.dim() > E_quad.dim() else E_att3
    Em = E_mem.mean(dim=-2) if E_mem.dim() > E_quad.dim() else E_mem

    return E_quad - E2 - E3 - Em, X, grads
