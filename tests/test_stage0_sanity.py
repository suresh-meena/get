import torch

from get.energy.core import GETEnergy
from get.energy.quadratic import compute_quadratic_energy


def _make_problem(device: torch.device, dtype: torch.dtype = torch.float64):
    torch.manual_seed(7)
    num_nodes = 6
    d = 4
    num_heads = 2
    head_dim = d // num_heads
    R = 2
    K = 3
    num_graphs = 1

    x = torch.randn(num_nodes, d, device=device, dtype=dtype)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    # Ring edges: each node has one outgoing support edge.
    c_2 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)
    u_2 = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.long, device=device)

    # Ring motifs: one motif per anchor (i, i+1, i+2).
    c_3 = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long, device=device)
    u_3 = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.long, device=device)
    v_3 = torch.tensor([2, 3, 4, 5, 0, 1], dtype=torch.long, device=device)
    t_tau = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long, device=device)

    params = {
        "d": d,
        "use_pairwise": True,
        "lambda_2": 1.0,
        "beta_2": 1.0,
        "pairwise_symmetric": False,
        "use_motif": True,
        "lambda_3": 0.6,
        "beta_3": 1.0,
        "R": R,
        "T_tau": torch.randn(2, num_heads, R, head_dim, device=device, dtype=dtype),
        "use_memory": True,
        "lambda_m": 0.7,
        "beta_m": 1.0,
        "K": K,
        "lambda_sum": 0.0,
    }
    km = torch.randn(num_heads, K, head_dim, device=device, dtype=dtype)

    return x, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R


def _projections_from_x(x: torch.Tensor, km: torch.Tensor, R: int):
    num_nodes, d = x.shape
    num_heads = km.shape[0]
    head_dim = d // num_heads

    xh = x.view(num_nodes, num_heads, head_dim)
    q2 = xh
    k2 = torch.roll(xh, shifts=1, dims=-1)
    q3 = xh.unsqueeze(2).expand(-1, -1, R, -1)
    k3 = torch.tanh(q3)
    qm = xh
    return {
        "Q2": q2,
        "K2": k2,
        "a_2": None,
        "Q3": q3,
        "K3": k3,
        "Qm": qm,
        "Km": km,
    }


def _energy_scalar(
    x: torch.Tensor,
    batch: torch.Tensor,
    num_graphs: int,
    c_2: torch.Tensor,
    u_2: torch.Tensor,
    c_3: torch.Tensor,
    u_3: torch.Tensor,
    v_3: torch.Tensor,
    t_tau: torch.Tensor,
    params: dict,
    km: torch.Tensor,
    R: int,
):
    projections = _projections_from_x(x, km=km, R=R)
    energy_vec = GETEnergy()(
        x,
        x,
        c_2,
        u_2,
        c_3,
        u_3,
        v_3,
        t_tau,
        batch,
        num_graphs,
        params,
        projections,
    )
    return energy_vec.sum()


def test_core_energy_returns_per_graph_values():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R = _make_problem(device)
    projections = _projections_from_x(x, km=km, R=R)
    energy = GETEnergy()(x, x, c_2, u_2, c_3, u_3, v_3, t_tau, batch, num_graphs, params, projections)
    assert energy.shape == (num_graphs,)


def test_energy_permutation_invariance():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R = _make_problem(device)
    e_ref = _energy_scalar(x, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R)

    perm = torch.randperm(x.size(0), device=device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=device)

    x_perm = x[perm]
    batch_perm = batch[perm]
    c_2_perm = inv_perm[c_2]
    u_2_perm = inv_perm[u_2]
    c_3_perm = inv_perm[c_3]
    u_3_perm = inv_perm[u_3]
    v_3_perm = inv_perm[v_3]

    e_perm = _energy_scalar(
        x_perm,
        batch_perm,
        num_graphs,
        c_2_perm,
        u_2_perm,
        c_3_perm,
        u_3_perm,
        v_3_perm,
        t_tau,
        params,
        km,
        R,
    )
    assert torch.allclose(e_ref, e_perm, atol=1e-7, rtol=1e-6)


def test_armijo_backtracking_monotone_decrease():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R = _make_problem(device)

    eta0 = 0.25
    gamma = 0.5
    c = 1e-4
    x_t = x.clone().detach()

    for _ in range(6):
        x_t = x_t.detach().requires_grad_(True)
        e_t = _energy_scalar(x_t, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R)
        grad, = torch.autograd.grad(e_t, x_t)
        grad_norm_sq = (grad * grad).sum()

        found = False
        eta = eta0
        for _ in range(20):
            x_candidate = (x_t - eta * grad).detach()
            e_candidate = _energy_scalar(
                x_candidate, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R
            )
            rhs = e_t.detach() - c * eta * grad_norm_sq.detach()
            if e_candidate <= rhs + 1e-12:
                found = True
                break
            eta *= gamma

        assert found
        assert e_candidate <= e_t.detach() + 1e-12
        x_t = x_candidate


def test_pairwise_only_special_case_matches_definition():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, batch, num_graphs, c_2, u_2, c_3, u_3, v_3, t_tau, params, km, R = _make_problem(device)
    projections = _projections_from_x(x, km=km, R=R)

    params_pairwise_only = dict(params)
    params_pairwise_only["use_motif"] = False
    params_pairwise_only["use_memory"] = False
    params_pairwise_only["lambda_3"] = 0.0
    params_pairwise_only["lambda_m"] = 0.0

    energy_total = GETEnergy()(
        x,
        x,
        c_2,
        u_2,
        c_3,
        u_3,
        v_3,
        t_tau,
        batch,
        num_graphs,
        params_pairwise_only,
        projections,
    )
    pairwise = GETEnergy().pairwise(
        x,
        c_2,
        u_2,
        batch,
        num_graphs,
        params_pairwise_only,
        projections,
        x.size(0),
    ).mean(dim=-1)
    quadratic = compute_quadratic_energy(x, batch, num_graphs)

    assert torch.allclose(energy_total, quadratic - pairwise, atol=1e-8, rtol=1e-6)
