import torch
import torch.nn as nn
from types import SimpleNamespace
from .energy import compute_energy_GET, compute_energy_and_grad_GET, _scatter_add_nd
from .et_core import ETGraphMaskModulator, EnergyLayerNorm


class StableMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, activation=nn.GELU, dropout=0.1, final_norm=True):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim) if final_norm else nn.Identity()

    def forward(self, x):
        return self.norm(self.net(x))


def _inv_softplus(x):
    return torch.log(torch.exp(torch.tensor(x)) - 1.0)


class GETLayer(nn.Module):
    def __init__(self, d=256, R=4, K=32, num_heads=1, head_dim=None, num_motif_types=2,
                 lambda_2=1.0, lambda_3=0.5, lambda_m=1.0, beta_2=1.0, beta_3=1.0, beta_m=1.0,
                 grad_clip_norm=1.0, beta_max=5.0, state_clip_norm=10.0, update_damping=1.0, learn_update_damping=False,
                 pairwise_symmetric=False, noise_std=0.0, use_pairwise=True, use_motif=True, use_memory=True,
                 norm_style="standard", pairwise_et_mask=False, pairwise_et_kernel_size=3):
        super().__init__()
        self.d = d
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or max(1, d // self.num_heads))
        self.R = R
        self.K = K
        self.num_motif_types = int(num_motif_types)
        self.noise_std = float(noise_std)
        self.grad_clip_norm = grad_clip_norm
        self.beta_max = beta_max
        self.state_clip_norm = state_clip_norm
        self.learn_update_damping = learn_update_damping
        self.pairwise_symmetric = bool(pairwise_symmetric)
        self.use_pairwise = bool(use_pairwise)
        self.use_motif = bool(use_motif)
        self.use_memory = bool(use_memory)
        self.norm_style = str(norm_style)
        self.pairwise_et_mask = bool(pairwise_et_mask)

        update_damping = min(max(float(update_damping), 1e-4), 1.0 - 1e-4)
        if learn_update_damping:
            self.update_damping_logit = nn.Parameter(torch.logit(torch.tensor(update_damping)))
        else:
            self.register_buffer("update_damping_value", torch.tensor(update_damping))

        self.lambda_2 = nn.Parameter(_inv_softplus(lambda_2))
        self.lambda_3 = nn.Parameter(_inv_softplus(lambda_3))
        self.lambda_m = nn.Parameter(_inv_softplus(lambda_m))
        self.beta_2 = nn.Parameter(_inv_softplus(beta_2))
        self.beta_3 = nn.Parameter(_inv_softplus(beta_3))
        self.beta_m = nn.Parameter(_inv_softplus(beta_m))

        # Always use EnergyLayerNorm to maintain theoretical guarantees and analytical pullback
        self.layernorm = EnergyLayerNorm(d, use_bias=True, eps=1e-5)

        self.W_Q2 = nn.Parameter(torch.empty(self.num_heads, self.head_dim, d))
        self.W_K2 = nn.Parameter(torch.empty(self.num_heads, self.head_dim, d))

        if self.use_motif:
            self.W_Q3 = nn.Parameter(torch.empty(self.num_heads, R * self.head_dim, d))
            self.W_K3 = nn.Parameter(torch.empty(self.num_heads, R * self.head_dim, d))
            self.T_tau = nn.Parameter(torch.empty(self.num_motif_types, self.num_heads, R, self.head_dim))
        else:
            self.W_Q3 = None
            self.W_K3 = None
            self.T_tau = None

        if self.use_memory and self.K > 0:
            self.W_Qm = nn.Parameter(torch.empty(self.num_heads, self.head_dim, d))
            self.W_Km = nn.Parameter(torch.empty(self.num_heads, self.head_dim, d))
            self.B_mem = nn.Parameter(torch.empty(K, d))
        else:
            self.W_Qm = None
            self.W_Km = None
            self.B_mem = None

        self.edge_mlp = nn.Sequential(
            nn.LazyLinear(d),
            nn.GELU(),
            nn.Linear(d, self.num_heads)
        )
        self.pairwise_mask_modulator = None
        if self.pairwise_et_mask:
            self.pairwise_mask_modulator = ETGraphMaskModulator(
                d=d,
                num_heads=self.num_heads,
                edge_feat_dim=None,
                kernel_size=pairwise_et_kernel_size
            )
        self.reset_parameters()

    @property
    def update_damping(self):
        if self.learn_update_damping:
            return torch.sigmoid(self.update_damping_logit)
        return self.update_damping_value

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q2, gain=1.0)
        nn.init.xavier_uniform_(self.W_K2, gain=1.0)
        if self.W_Q3 is not None:
            nn.init.xavier_uniform_(self.W_Q3, gain=0.5)
            nn.init.xavier_uniform_(self.W_K3, gain=0.5)
            nn.init.normal_(self.T_tau, std=0.02)
        if self.W_Qm is not None and self.B_mem is not None:
            nn.init.xavier_uniform_(self.W_Qm, gain=0.5)
            nn.init.xavier_uniform_(self.W_Km, gain=0.5)
            nn.init.normal_(self.B_mem, mean=0.0, std=1.0 / (self.d ** 0.5))

    def _compute_pairwise_et_bias(self, G, batch_data):
        if self.pairwise_mask_modulator is None:
            return None
        return self.pairwise_mask_modulator(G, batch_data)

    def _build_projections(self, G, static_projections=None, batch_data=None):
        static_projections = static_projections or {}
        projections = {}
        if self.use_pairwise:
            projections['Q2'] = torch.einsum("...nd, hzd -> ...hnz", G, self.W_Q2)
            projections['K2'] = torch.einsum("...nd, hzd -> ...hnz", G, self.W_K2)
            a_2 = static_projections.get('a_2')
            if a_2 is not None:
                if a_2.dim() == 2:
                    a_2 = a_2.permute(1, 0)
                elif a_2.dim() == 3:
                    a_2 = a_2.permute(0, 2, 1)
            et_bias = self._compute_pairwise_et_bias(G, batch_data)
            if et_bias is not None:
                a_2 = (a_2 + et_bias) if a_2 is not None else et_bias
            projections['a_2'] = a_2
        if self.use_motif and self.W_Q3 is not None:
            Q3 = torch.einsum("...nd, hzd -> ...hnz", G, self.W_Q3)
            K3 = torch.einsum("...nd, hzd -> ...hnz", G, self.W_K3)
            shape = list(Q3.shape[:-1]) + [self.R, self.head_dim]
            projections['Q3'] = Q3.view(*shape)
            projections['K3'] = K3.view(*shape)
        if self.use_memory and self.K > 0:
            projections['Qm'] = torch.einsum("...nd, hzd -> ...hnz", G, self.W_Qm)
            projections['Km'] = torch.einsum("kd, hzd -> hkz", self.B_mem, self.W_Km)
        return projections

    def get_params_dict(self):
        return {
            'd': self.head_dim, 'R': self.R, 'K': self.K,
            'lambda_2': self.lambda_2, 'lambda_3': self.lambda_3, 'lambda_m': self.lambda_m,
            'beta_2': self.beta_2, 'beta_3': self.beta_3, 'beta_m': self.beta_m,
            'beta_max': self.beta_max, 'pairwise_symmetric': self.pairwise_symmetric,
            'use_pairwise': self.use_pairwise, 'use_motif': self.use_motif, 'use_memory': self.use_memory,
            'T_tau': self.T_tau, 'W_Q2': self.W_Q2, 'W_K2': self.W_K2,
            'W_Q3': self.W_Q3, 'W_K3': self.W_K3, 'W_Qm': self.W_Qm, 'B_mem': self.B_mem,
        }

    def compute_energy(self, X, batch_data, static_projections=None):
        G = self.layernorm(X)
        projections = self._build_projections(G, static_projections, batch_data=batch_data)
        params = self.get_params_dict()
        params['num_heads'] = self.num_heads
        return compute_energy_GET(X, G, batch_data.c_2, batch_data.u_2, batch_data.c_3, batch_data.u_3, batch_data.v_3, batch_data.t_tau, batch_data.batch, params, projections)

    def energy_and_grad(self, X, batch_data, static_projections=None, create_graph=False):
        if not hasattr(self.layernorm, "backward"):
            if not X.requires_grad:
                X = X.requires_grad_(True)
            E = self.compute_energy(X, batch_data, static_projections)
            return E, torch.autograd.grad(E.sum(), X, create_graph=create_graph)[0]

        G = self.layernorm(X)
        projections = self._build_projections(G, static_projections, batch_data=batch_data)
        params = self.get_params_dict()
        params['num_heads'] = self.num_heads
        E, grad_X_quad, head_grads = compute_energy_and_grad_GET(X, G, batch_data.c_2, batch_data.u_2, batch_data.c_3, batch_data.u_3, batch_data.v_3, batch_data.t_tau, batch_data.batch, params, projections)

        grad_G_att = torch.zeros_like(G)
        W_Q2 = getattr(self, 'W_Q2', None)
        W_K2 = getattr(self, 'W_K2', None)
        W_Q3 = getattr(self, 'W_Q3', None)
        W_K3 = getattr(self, 'W_K3', None)
        W_Qm = getattr(self, 'W_Qm', None)

        if self.use_pairwise and W_Q2 is not None:
            gQ2, gK2 = head_grads.get('grad_Q2'), head_grads.get('grad_K2')
            if gQ2 is not None:
                term = (gQ2 @ W_Q2)
                grad_G_att += term.sum(dim=-3) if self.num_heads > 1 else term.squeeze(-3)
            if gK2 is not None:
                term = (gK2 @ W_K2)
                grad_G_att += term.sum(dim=-3) if self.num_heads > 1 else term.squeeze(-3)

        if self.use_motif and W_Q3 is not None:
            gQ3, gK3 = head_grads.get('grad_Q3'), head_grads.get('grad_K3')
            if gQ3 is not None:
                term = (gQ3.flatten(-2, -1) @ W_Q3)
                grad_G_att += term.sum(dim=-3) if self.num_heads > 1 else term.squeeze(-3)
            if gK3 is not None:
                term = (gK3.flatten(-2, -1) @ W_K3)
                grad_G_att += term.sum(dim=-3) if self.num_heads > 1 else term.squeeze(-3)

        if self.use_memory and W_Qm is not None:
            gQm = head_grads.get('grad_Qm')
            if gQm is not None:
                term = (gQm @ W_Qm)
                grad_G_att += term.sum(dim=-3) if self.num_heads > 1 else term.squeeze(-3)

        grad_X = grad_X_quad - self.layernorm.backward(X, grad_G_att)
        return E, grad_X

    def forward(self, X, batch_data, step_size, static_projections=None, is_training=True, apply_clipping=True, inference_mode='fixed'):
        if inference_mode == 'fixed':
            E, grad_X = self.energy_and_grad(X, batch_data, static_projections, create_graph=is_training)
            if apply_clipping:
                gnorm = torch.norm(grad_X, dim=-1, keepdim=True)
                grad_X = grad_X * (self.grad_clip_norm / gnorm.clamp(min=self.grad_clip_norm))
            X_next = X - (step_size * self.update_damping) * grad_X
            if apply_clipping:
                snorm = torch.norm(X_next, dim=-1, keepdim=True)
                X_next = X_next * (self.state_clip_norm / snorm.clamp(min=self.state_clip_norm))
            return X_next, E
        return self.energy_and_grad(X, batch_data, static_projections, create_graph=is_training)


class GETModel(nn.Module):
    def __init__(self, in_dim, d=256, num_classes=1, num_steps=8, tol=1e-4, compile=False, eta=0.05, eta_max=0.25, dropout=0.1, num_heads=1, head_dim=None, encoder_hidden_mult=2, readout_hidden_mult=2, pe_k=0, rwse_k=0, num_motif_types=2, use_cls_token=False, cls_self_loop=True, **layer_kwargs):
        super().__init__()
        self.d = d
        self.num_steps = num_steps
        self.tol = float(tol)
        self.eta_max = eta_max
        self.pe_k = pe_k
        self.rwse_k = rwse_k
        self.use_cls_token = bool(use_cls_token)
        self.cls_self_loop = bool(cls_self_loop)

        if self.pe_k > 0:
            self.pe_proj = nn.Linear(pe_k, d)
        if self.rwse_k > 0:
            self.rwse_proj = nn.Linear(rwse_k, d)

        self.node_encoder = StableMLP(in_dim, d, hidden_dim=max(d, encoder_hidden_mult * d), dropout=dropout, final_norm=True)
        self.get_layer = GETLayer(d, num_motif_types=num_motif_types, num_heads=num_heads, head_dim=head_dim, **layer_kwargs)

        eta = min(max(float(eta), 1e-4), eta_max - 1e-4)
        self.eta_logit = nn.Parameter(torch.logit(torch.tensor(eta / eta_max)))

        self.readout = nn.Sequential(
            nn.Linear(4 * d, readout_hidden_mult * d), nn.GELU(), nn.LayerNorm(readout_hidden_mult * d), nn.Dropout(dropout),
            nn.Linear(readout_hidden_mult * d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_classes)
        )
        self.node_readout = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_classes)
        )

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, d))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            self.cls_readout = nn.Sequential(
                nn.Linear(d, readout_hidden_mult * d), nn.GELU(), nn.LayerNorm(readout_hidden_mult * d), nn.Dropout(dropout),
                nn.Linear(readout_hidden_mult * d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_classes)
            )

        if compile and hasattr(torch, "compile"):
            try:
                self.get_layer.energy_and_grad = torch.compile(self.get_layer.energy_and_grad, dynamic=True)
                self._run_fixed_solver = torch.compile(self._run_fixed_solver, dynamic=True)
                print("INFO:    Successfully compiled GrET inference solvers.")
            except Exception:
                pass

    @property
    def eta(self):
        return self.eta_max * torch.sigmoid(self.eta_logit)

    def _readout(self, Z, batch_data, task_level, cls_positions=None):
        if task_level == 'node':
            num_nodes = batch_data.num_nodes - (len(cls_positions) if cls_positions is not None else 0)
            return self.node_readout(Z[:num_nodes])
        if task_level == 'graph':
            if cls_positions is not None:
                return self.cls_readout(Z[cls_positions])
            batch = batch_data.batch
            num_graphs = int(batch.max().item() + 1)
            counts = torch.bincount(batch, minlength=num_graphs).view(-1, 1).to(dtype=Z.dtype).clamp_min(1.0)
            
            z_sum = torch.zeros(num_graphs, self.d, dtype=Z.dtype, device=Z.device)
            z_sum.index_add_(0, batch, Z)
            z_mean = z_sum / counts
            
            z_max = torch.full((num_graphs, self.d), float('-inf'), dtype=Z.dtype, device=Z.device)
            if hasattr(torch, "index_reduce_"):
                z_max.index_reduce_(0, batch, Z, reduce="amax", include_self=False)
            else:
                z_max.scatter_reduce_(0, batch.view(-1, 1).expand_as(Z), Z, reduce="amax", include_self=False)
            z_max = torch.where(counts > 0, z_max, torch.zeros_like(z_max))
            
            z_sq_sum = torch.zeros(num_graphs, self.d, dtype=Z.dtype, device=Z.device)
            z_sq_sum.index_add_(0, batch, Z**2)
            z_var = (z_sq_sum / counts) - z_mean**2
            z_std = torch.sqrt(z_var.clamp_min(0.0) + 1e-6)
            
            return self.readout(torch.cat([z_mean, z_sum, z_max, z_std], dim=-1))
        raise ValueError(f"Unsupported task_level: {task_level}")

    def _build_static_projections(self, batch_data):
        static_projections = {}
        if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
            static_projections['a_2'] = self.get_layer.edge_mlp(batch_data.edge_attr)
        if self.get_layer.use_memory and self.get_layer.K > 0:
            static_projections['Km'] = torch.einsum("kd, hzd -> ...hkz", self.get_layer.B_mem, self.get_layer.W_Km)
        if self.get_layer.use_motif and self.get_layer.T_tau is not None and batch_data.t_tau.numel() > 0:
            t_tau = batch_data.t_tau
            T_params = self.get_layer.T_tau
            if t_tau.max() >= T_params.size(0):
                t_tau = torch.clamp(t_tau, max=T_params.size(0) - 1)
            static_projections['T_tau_selected'] = T_params[t_tau].transpose(0, 1)
        return static_projections

    def _augment_with_cls_token(self, X, batch_data):
        if not self.use_cls_token:
            return X, batch_data, None
        num_nodes, num_graphs = X.size(0), int(batch_data.batch.max().item() + 1)
        cls_tokens = self.cls_token.expand(num_graphs, -1)
        X = torch.cat([X, cls_tokens], dim=0)
        cls_positions = torch.arange(num_nodes, num_nodes + num_graphs, device=X.device)
        new_batch = torch.cat([batch_data.batch, torch.arange(num_graphs, device=X.device)], dim=0)
        if self.cls_self_loop:
            c_2 = torch.cat([batch_data.c_2, cls_positions, cls_positions], dim=0)
            u_2 = torch.cat([batch_data.u_2, cls_positions, cls_positions], dim=0)
        else:
            c_2, u_2 = batch_data.c_2, batch_data.u_2
        new_batch_data = SimpleNamespace(**{k: getattr(batch_data, k) for k in vars(batch_data) if k not in ['batch', 'c_2', 'u_2', 'num_nodes']})
        new_batch_data.batch = new_batch
        new_batch_data.c_2 = c_2
        new_batch_data.u_2 = u_2
        new_batch_data.num_nodes = num_nodes + num_graphs
        return X, new_batch_data, cls_positions

    def forward(self, batch_data, task_level='graph', inference_mode='fixed', return_solver_stats=False, **kwargs):
        X = self.node_encoder(batch_data.x)
        if self.pe_k > 0 and hasattr(batch_data, 'pe') and batch_data.pe is not None:
            X.add_(self.pe_proj(batch_data.pe))
        if self.rwse_k > 0 and hasattr(batch_data, 'rwse') and batch_data.rwse is not None:
            X.add_(self.rwse_proj(batch_data.rwse))

        X, solver_batch, cls_positions = self._augment_with_cls_token(X, batch_data)
        static_projections = self._build_static_projections(solver_batch)
        if inference_mode == 'fixed':
            X, energy_trace, stats = self._run_fixed_solver(X, solver_batch, static_projections)
        elif inference_mode == 'armijo':
            X, energy_trace, stats = self._run_armijo_solver(X, solver_batch, static_projections, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {inference_mode}")
        out = self._readout(X, solver_batch, task_level, cls_positions)
        return (out, energy_trace, stats) if return_solver_stats else (out, energy_trace)

    def _run_fixed_solver(self, X, solver_batch, static_projections, training_mode=True):
        energy_trace = []
        eta = self.eta
        X_current = X
        check_freq = 4
        
        for step_idx in range(self.num_steps):
            X_prev = X_current.detach().clone() if (self.tol > 0 and step_idx % check_freq == 0) else None
            
            # Use the layer's forward which now returns next state and energy
            X_next, E = self.get_layer(X_current, solver_batch, eta, static_projections, is_training=training_mode)
            energy_trace.append(E.detach())
            X_current = X_next
            
            if self.tol > 0 and X_prev is not None:
                # Tolerance check every 4 steps to reduce sync stalls
                with torch.no_grad():
                    diff = torch.norm(X_current - X_prev) / (torch.norm(X_prev) + 1e-6)
                    if diff < self.tol:
                        break
        return X_current, energy_trace, {}

    def _run_armijo_solver(self, X, batch_data, static_projections, armijo_c=0.1, armijo_gamma=0.5, armijo_eta0=0.2, armijo_max_backtracks=25, chunk_size=4):
        num_graphs = int(batch_data.batch.max().item() + 1)
        energy_trace = []
        stats = {'backtracks': [], 'steps': 0, 'step_sizes': [], 'accepted': []}
        eta0 = armijo_eta0

        # Hoist loop invariants
        etas_all = eta0 * (armijo_gamma ** torch.arange(armijo_max_backtracks, device=X.device, dtype=X.dtype))

        X_current = X.detach()
        for step_idx in range(self.num_steps):
            E_t, grad_X = self.get_layer.energy_and_grad(X_current, batch_data, static_projections=static_projections, create_graph=False)
            if self.tol > 0:
                gnorm = torch.norm(grad_X) / (torch.norm(X_current) + 1e-6)
                if gnorm < self.tol:
                    break

            gn_sq = (grad_X**2).sum(dim=-1)
            gn_sq_graph = _scatter_add_nd(gn_sq.new_zeros(num_graphs), batch_data.batch, gn_sq, dim=0)

            # Chunked evaluation to save VRAM and skip redundant work
            best_idx = torch.full((num_graphs,), armijo_max_backtracks - 1, device=X.device, dtype=torch.long)
            found_global = torch.zeros(num_graphs, device=X.device, dtype=torch.bool)
            
            for start_bt in range(0, armijo_max_backtracks, chunk_size):
                end_bt = min(start_bt + chunk_size, armijo_max_backtracks)
                etas = etas_all[start_bt:end_bt]
                
                # Evaluate current chunk
                X_tries = X_current.unsqueeze(0) - etas.view(-1, 1, 1) * grad_X.unsqueeze(0)
                E_tries = self.get_layer.compute_energy(X_tries, batch_data, static_projections=static_projections)
                E_target = E_t.unsqueeze(0) - armijo_c * etas.view(-1, 1) * gn_sq_graph.unsqueeze(0)
                
                valid = (E_tries <= E_target)
                found_chunk = valid.any(dim=0)
                
                # Update best indices for graphs that just found a valid step
                update_mask = found_chunk & (~found_global)
                if update_mask.any():
                    chunk_best = valid.to(torch.long).argmax(dim=0)
                    best_idx[update_mask] = start_bt + chunk_best[update_mask]
                    found_global[update_mask] = True
                
                # Early exit if all graphs in batch found a valid step size
                if found_global.all():
                    break

            # Final assignment
            # Use etas_all to get the final accepted states
            accepted_etas = etas_all[best_idx]
            # In-place update to reduce memory pressure
            X_current.sub_(accepted_etas[batch_data.batch].view(-1, 1) * grad_X)

            energy_trace.append(E_t.detach())
            stats['backtracks'].append(best_idx.float().mean().item())
            stats['step_sizes'].append(accepted_etas.mean().item())
            stats['accepted'].append(found_global.float().mean().item())
            stats['steps'] = step_idx + 1
        return X_current, energy_trace, stats


class PairwiseGET(GETModel):
    def __init__(self, in_dim, d=256, **kwargs):
        super().__init__(in_dim, d, use_pairwise=True, use_motif=False, use_memory=False, **kwargs)


class FullGET(GETModel):
    def __init__(self, in_dim, d=256, **kwargs):
        super().__init__(in_dim, d, **kwargs)
