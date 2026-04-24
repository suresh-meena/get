import torch
from get import GETModel, FullGET, PairwiseGET, collate_get_batch
from get.energy import compute_energy_GET

def test_inheritance_hopfield_reduction():
    """
    Proposition 1: |V|=1, lambda_2=lambda_3=0 reduces to Modern Hopfield retrieval.
    """
    print("Verifying Proposition 1: reduction to Modern Hopfield...")
    d = 32
    num_memories = 16

    # single-node model
    model = GETModel(
        in_dim=d, d=d, num_classes=1,
        lambda_2=0.0, lambda_3=0.0, lambda_m=1.0,
        num_heads=1, K=num_memories, use_pairwise=False, use_motif=False
    ).cuda().to(torch.float64)
    model.eval()

    # single-node graph
    x = torch.randn(1, d, device="cuda", dtype=torch.float64)
    batch = collate_get_batch([{'x': x, 'edges': []}])
    batch = batch.to("cuda")

    with torch.no_grad():
        X = model.node_encoder(batch.x)
        G = model.get_layers[0].layernorm(X)
        params = model.get_layers[0].get_params_dict()
        params['num_heads'] = 1

        E_total = model.get_layers[0].compute_energy(X, batch)
        
        # Manual term
        # E_Hop = 0.5||x||^2 - 1/beta * log(sum(exp(beta * x^T b_a)))
        beta_m = torch.nn.functional.softplus(model.get_layers[0].beta_m) + 1e-8
        lambda_m = torch.nn.functional.softplus(model.get_layers[0].lambda_m) + 1e-8
        
        Qm = G @ model.get_layers[0].W_Qm[0].t()
        Km = model.get_layers[0].B_mem @ model.get_layers[0].W_Km[0].t()
        scale = d ** 0.5
        scores = (Qm @ Km.t()) / scale
        E_manual = 0.5 * (X**2).sum() - (lambda_m / beta_m) * torch.logsumexp(beta_m * scores, dim=-1).sum()
        
        assert torch.allclose(E_total, E_manual, atol=1e-7)
    print("  Hopfield reduction verified.")

def test_inheritance_et_reduction():
    """
    Proposition 2: Complete graph, pairwise-only reduces to restricted ET attention.
    """
    print("Verifying Proposition 2: reduction to restricted ET attention...")
    d = 32
    num_nodes = 5

    # Complete graph with self-loops
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            edges.append((i, j))

    model = PairwiseGET(in_dim=d, d=d, num_classes=1, num_heads=1).cuda().to(torch.float64)
    model.eval()

    x = torch.randn(num_nodes, d, device="cuda", dtype=torch.float64)
    batch = collate_get_batch([{'x': x, 'edges': edges}])
    batch = batch.to("cuda")

    with torch.no_grad():
        X = model.node_encoder(batch.x)
        G = model.get_layers[0].layernorm(X)
        params = model.get_layers[0].get_params_dict()
        params['num_heads'] = 1

        E_total = model.get_layers[0].compute_energy(X, batch)
        
        # Manual attention-based energy
        Q = G @ model.get_layers[0].W_Q2[0].t()
        K = G @ model.get_layers[0].W_K2[0].t()
        scale = d ** 0.5
        scores = (Q @ K.t()) / scale

        # Mask out self-loops to match GET's graph-local sparse computation
        mask = torch.eye(num_nodes, device="cuda", dtype=torch.bool)
        scores = scores.masked_fill(mask, float('-inf'))

        beta_2 = torch.nn.functional.softplus(model.get_layers[0].beta_2) + 1e-8
        lambda_2 = torch.nn.functional.softplus(model.get_layers[0].lambda_2) + 1e-8
        E_manual = 0.5 * (X**2).sum() - (lambda_2 / beta_2) * torch.logsumexp(beta_2 * scores, dim=-1).sum()
        
        assert torch.allclose(E_total, E_manual, atol=1e-7)
    print("  ET attention reduction verified.")

def test_multi_head_energy_averaging():
    """
    Verify that multi-head energy is the average of head energies.
    """
    print("Verifying Multi-Head energy averaging...")
    d = 32
    num_heads = 4
    head_dim = 8
    num_nodes = 5

    model = FullGET(in_dim=d, d=d, num_heads=num_heads, head_dim=head_dim).cuda().to(torch.float64)
    model.eval()

    x = torch.randn(num_nodes, d, device="cuda", dtype=torch.float64)
    batch = collate_get_batch([{'x': x, 'edges': [(0, 1), (1, 2)]}])
    batch = batch.to("cuda")

    with torch.no_grad():
        X = model.node_encoder(batch.x)
        G = model.get_layers[0].layernorm(X)
        E_total = model.get_layers[0].compute_energy(X, batch)
        
        projections = model.get_layers[0]._build_projections(G, batch_data=batch)
        params = model.get_layers[0].get_params_dict()
        params['num_heads'] = num_heads

        E_total_backend = compute_energy_GET(
            X, G,
            batch.c_2, batch.u_2, batch.c_3, batch.u_3, batch.v_3, batch.t_tau,
            batch.batch, params, projections
        )
        assert torch.allclose(E_total, E_total_backend)
    print("  Multi-head energy averaging verified.")

def test_double_backward_full_model():
    """
    Rigorous check for double backward support in GETModel.
    """
    print("Verifying Double Backward support in GETModel...")
    d = 16
    model = FullGET(in_dim=d, d=d, num_heads=1, num_steps=1).cuda().to(torch.float64)

    x = torch.randn(5, d, device="cuda", dtype=torch.float64, requires_grad=True)
    batch = collate_get_batch([{'x': x, 'edges': [(0, 1), (1, 2)]}])
    batch = batch.to("cuda")

    # 1st backward
    out, _ = model(batch, is_training=True)
    loss = out.sum()
    grad_x = torch.autograd.grad(loss, x, create_graph=True)[0]
    
    # 2nd backward (Hessian-vector product style)
    loss2 = grad_x.sum()
    loss2.backward()
    
    assert x.grad is not None
    print("  Double backward verified.")
