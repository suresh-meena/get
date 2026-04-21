import torch
import torch.nn as nn
from .energy import compute_energy_GET


class StableMLP(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim=None,
        activation=nn.GELU,
        dropout=0.0,
        final_norm=True,
    ):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        layers = [
            nn.Linear(in_dim, hidden_dim),
            activation(),
        ]
        if hidden_dim > 1:
            layers.append(nn.LayerNorm(hidden_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_dim, out_dim),
                activation(),
            ]
        )
        if final_norm and out_dim > 1:
            layers.append(nn.LayerNorm(out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _inv_softplus(x):
    x = torch.as_tensor(float(x))
    if x <= 0:
        return torch.tensor(-20.0)
    return torch.log(torch.expm1(x))

class GETLayer(nn.Module):
    def __init__(self, d, R=1, K=32, 
                 lambda_2=1.0, lambda_3=1.0, lambda_m=1.0, 
                 beta_2=1.0, beta_3=1.0, beta_m=1.0,
                 grad_clip_norm=1.0, beta_max=5.0, state_clip_norm=10.0,
                 update_damping=1.0, learn_update_damping=False,
                 pairwise_symmetric=False, noise_std=0.0):
        super().__init__()
        self.d = d
        self.R = R
        self.K = K
        self.noise_std = float(noise_std)
        self.grad_clip_norm = grad_clip_norm
        self.beta_max = beta_max
        self.state_clip_norm = state_clip_norm
        self.learn_update_damping = learn_update_damping
        self.pairwise_symmetric = bool(pairwise_symmetric)
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
        
        self.layernorm = nn.LayerNorm(d, eps=1e-5)
        
        self.W_Q2 = nn.Parameter(torch.empty(d, d))
        self.W_K2 = nn.Parameter(torch.empty(d, d))
        
        self.W_Q3 = nn.Parameter(torch.empty(d, R * d))
        self.W_K3 = nn.Parameter(torch.empty(d, R * d))
        
        # Motif type embeddings (0=open, 1=closed)
        self.T_tau = nn.Parameter(torch.empty(2, R, d))
        
        if self.K > 0:
            self.W_Qm = nn.Parameter(torch.empty(d, d))
            self.W_Km = nn.Parameter(torch.empty(d, d))
            self.B_mem = nn.Parameter(torch.empty(K, d))
            
        self.edge_mlp = nn.Sequential(
            nn.Linear(5, d),
            nn.GELU(),
            nn.Linear(d, 1)
        )

        self.reset_parameters()

    @property
    def update_damping(self):
        if self.learn_update_damping:
            return torch.sigmoid(self.update_damping_logit)
        return self.update_damping_value

    def reset_parameters(self):
        for weight in (self.W_Q2, self.W_K2, self.W_Q3, self.W_K3):
            nn.init.xavier_uniform_(weight, gain=0.5)
        nn.init.normal_(self.T_tau, mean=0.0, std=0.02)
        if self.K > 0:
            nn.init.xavier_uniform_(self.W_Qm, gain=0.5)
            nn.init.xavier_uniform_(self.W_Km, gain=0.5)
            nn.init.normal_(self.B_mem, mean=0.0, std=1.0 / (self.d ** 0.5))
            
    def get_params_dict(self):
        params = {
            'd': self.d,
            'R': self.R,
            'K': self.K,
            'lambda_2': self.lambda_2,
            'lambda_3': self.lambda_3,
            'lambda_m': self.lambda_m,
            'beta_2': self.beta_2,
            'beta_3': self.beta_3,
            'beta_m': self.beta_m,
            'beta_max': self.beta_max,
            'pairwise_symmetric': self.pairwise_symmetric,
            
            'W_Q2': self.W_Q2,
            'W_K2': self.W_K2,
            
            'W_Q3': self.W_Q3,
            'W_K3': self.W_K3,
            'T_tau': self.T_tau,
        }
        if self.K > 0:
            params['W_Qm'] = self.W_Qm
            params['W_Km'] = self.W_Km
            params['B_mem'] = self.B_mem
        return params
            
    def _build_projections(self, G, static_projections=None):
        static_projections = static_projections or {}
        include_motif = torch.nn.functional.softplus(self.lambda_3).detach().item() > 1e-6

        weights = [self.W_Q2, self.W_K2]
        if include_motif:
            weights.extend([self.W_Q3, self.W_K3])
        if self.K > 0:
            weights.append(self.W_Qm)

        Z_all = G @ torch.cat(weights, dim=1)
        d, R = self.d, self.R

        offset = 0
        projections = {
            'Q2': Z_all[:, offset : offset + d],
            'a_2': static_projections.get('a_2'),
        }
        offset += d
        projections['K2'] = Z_all[:, offset : offset + d]
        offset += d

        if include_motif:
            projections['Q3'] = Z_all[:, offset : offset + R * d].view(-1, R, d)
            offset += R * d
            projections['K3'] = Z_all[:, offset : offset + R * d].view(-1, R, d)
            offset += R * d

        if self.K > 0:
            projections['Qm'] = Z_all[:, offset:]
            projections['Km'] = static_projections['Km']
        return projections

    def compute_energy(self, X, batch_data, static_projections=None):
        G = self.layernorm(X)
        params = self.get_params_dict()
        projections = self._build_projections(G, static_projections)
        return compute_energy_GET(
            X,
            G,
            batch_data.c_2,
            batch_data.u_2,
            batch_data.c_3,
            batch_data.u_3,
            batch_data.v_3,
            batch_data.t_tau,
            params,
            projections,
        )

    def energy_and_grad(self, X, batch_data, static_projections=None, create_graph=False):
        if not X.requires_grad:
            X = X.requires_grad_(True)
        E = self.compute_energy(X, batch_data, static_projections)
        grad_X = torch.autograd.grad(E, X, create_graph=create_graph)[0]
        return E, grad_X

    def forward(
        self,
        X,
        batch_data,
        step_size,
        static_projections=None,
        is_training=True,
        apply_clipping=True,
        apply_state_clipping=True,
    ):
        """
        Takes one step of gradient descent on the energy landscape.
        Uses projection fusion for speed.
        """
        with torch.enable_grad():
            if not is_training:
                X = X.detach()
            
            E, grad_X = self.energy_and_grad(
                X,
                batch_data,
                static_projections=static_projections,
                create_graph=is_training,
            )
            if apply_clipping and self.grad_clip_norm is not None:
                grad_norm = grad_X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                grad_scale = (self.grad_clip_norm / grad_norm).clamp(max=1.0)
                grad_X = grad_X * grad_scale
            if is_training and self.noise_std > 0.0:
                grad_X = grad_X + torch.randn_like(grad_X) * self.noise_std
            
        X_next = X - (step_size * self.update_damping) * grad_X
        if apply_state_clipping and self.state_clip_norm is not None:
            state_norm = X_next.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            state_scale = (self.state_clip_norm / state_norm).clamp(max=1.0)
            X_next = X_next * state_scale
        return X_next, E

class GETModel(nn.Module):
    def __init__(self, in_dim, d, num_classes, num_steps=8, compile=False,
                 eta=0.05, eta_max=0.25, dropout=0.1,
                 encoder_hidden_mult=2, readout_hidden_mult=2, pe_k=0, **layer_kwargs):
        super().__init__()
        self.d = d
        self.num_steps = num_steps
        self.eta_max = eta_max
        self.pe_k = pe_k
        
        if self.pe_k > 0:
            self.pe_proj = nn.Linear(pe_k, d)

        self.node_encoder = StableMLP(
            in_dim,
            d,
            hidden_dim=max(d, encoder_hidden_mult * d),
            dropout=dropout,
            final_norm=True,
        )
        
        self.get_layer = GETLayer(d, **layer_kwargs)
        eta = min(max(float(eta), 1e-4), eta_max - 1e-4)
        self.eta_logit = nn.Parameter(torch.logit(torch.tensor(eta / eta_max)))
        
        self.readout = nn.Sequential(
            nn.Linear(4 * d, readout_hidden_mult * d),
            nn.GELU(),
            nn.LayerNorm(readout_hidden_mult * d),
            nn.Dropout(dropout),
            nn.Linear(readout_hidden_mult * d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes)
        )
        
        self.node_readout = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes)
        )
        
    @property
    def eta(self):
        return self.eta_max * torch.sigmoid(self.eta_logit)

    def _build_static_projections(self, batch_data):
        static_projections = {}
        if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
            static_projections['a_2'] = self.get_layer.edge_mlp(batch_data.edge_attr).squeeze(-1)
        if self.get_layer.K > 0:
            static_projections['Km'] = self.get_layer.B_mem @ self.get_layer.W_Km
        return static_projections

    def _readout(self, X, batch_data, task_level):
        Z = self.get_layer.layernorm(X)
        if task_level == 'node':
            out = self.node_readout(Z)
            return out
        if task_level == 'graph':
            batch = batch_data.batch
            num_graphs = int(batch.max().item() + 1)

            batch_idx = batch.view(-1, 1).expand_as(Z)

            z_mean = torch.zeros(num_graphs, self.d, dtype=Z.dtype, device=Z.device)
            z_mean.scatter_reduce_(0, batch_idx, Z, reduce="mean", include_self=False)

            z_sum = torch.zeros(num_graphs, self.d, dtype=Z.dtype, device=Z.device)
            z_sum.scatter_add_(0, batch_idx, Z)

            z_max = torch.full((num_graphs, self.d), float('-inf'), dtype=Z.dtype, device=Z.device)
            z_max.scatter_reduce_(0, batch_idx, Z, reduce="amax", include_self=False)
            z_max = torch.where(z_max == float('-inf'), torch.zeros_like(z_max), z_max)

            z_sq_mean = torch.zeros(num_graphs, self.d, dtype=Z.dtype, device=Z.device)
            z_sq_mean.scatter_reduce_(0, batch_idx, Z**2, reduce="mean", include_self=False)
            z_std = torch.sqrt(torch.relu(z_sq_mean - z_mean**2) + 1e-6)

            z_pooled = torch.cat([z_mean, z_sum, z_max, z_std], dim=-1)
            out = self.readout(z_pooled)
            return out
        raise ValueError(f"Unsupported task_level: {task_level}")

    def _init_solver_stats(self, inference_mode):
        return {
            'mode': inference_mode,
            'step_sizes': [],
            'backtracks': [],
            'accepted': [],
            'grad_norms': [],
        }

    def _run_fixed_solver(self, X, batch_data, static_projections, disable_eval_clipping=False):
        energy_trace = []
        solver_stats = self._init_solver_stats('fixed')
        apply_clipping = not (disable_eval_clipping and not self.training)
        for _ in range(self.num_steps):
            X, E = self.get_layer(
                X,
                batch_data,
                self.eta,
                static_projections,
                is_training=self.training,
                apply_clipping=apply_clipping,
                apply_state_clipping=apply_clipping,
            )
            energy_trace.append(float(E.detach().item()))
        return X, energy_trace, solver_stats

    def _run_armijo_solver(
        self,
        X,
        batch_data,
        static_projections,
        armijo_c,
        armijo_gamma,
        armijo_eta0,
        armijo_max_backtracks,
    ):
        if self.training:
            raise ValueError("Armijo inference_mode is evaluation-only; call model.eval() first.")
        if not (0.0 < armijo_c < 1.0):
            raise ValueError(f"armijo_c must be in (0,1); got {armijo_c}.")
        if not (0.0 < armijo_gamma < 1.0):
            raise ValueError(f"armijo_gamma must be in (0,1); got {armijo_gamma}.")
        if armijo_max_backtracks < 1:
            raise ValueError("armijo_max_backtracks must be >= 1.")

        eta0 = float(self.eta.detach().item()) if armijo_eta0 is None else float(armijo_eta0)
        eta0 = max(eta0, 1e-8)

        energy_trace = []
        solver_stats = self._init_solver_stats('armijo')

        for _ in range(self.num_steps):
            X = X.detach().requires_grad_(True)
            E_t, grad_X = self.get_layer.energy_and_grad(
                X,
                batch_data,
                static_projections=static_projections,
                create_graph=False,
            )
            grad_norm_sq = (grad_X ** 2).sum()
            grad_norm = float(torch.sqrt(grad_norm_sq).detach().item())
            solver_stats['grad_norms'].append(grad_norm)

            if grad_norm_sq.detach().item() <= 1e-16:
                energy_trace.append(float(E_t.detach().item()))
                solver_stats['step_sizes'].append(0.0)
                solver_stats['backtracks'].append(0)
                solver_stats['accepted'].append(True)
                X = X.detach()
                continue

            eta_t = eta0
            accepted = False
            backtracks = 0
            X_next = X.detach()
            E_next = E_t.detach()

            for bt in range(armijo_max_backtracks):
                X_try = X - eta_t * grad_X
                E_try = self.get_layer.compute_energy(
                    X_try,
                    batch_data,
                    static_projections=static_projections,
                )
                armijo_rhs = E_t - armijo_c * eta_t * grad_norm_sq
                if E_try.detach().item() <= armijo_rhs.detach().item():
                    accepted = True
                    backtracks = bt
                    X_next = X_try.detach()
                    E_next = E_try.detach()
                    break
                eta_t *= armijo_gamma

            if not accepted:
                backtracks = armijo_max_backtracks
                eta_t = 0.0
                X_next = X.detach()
                E_next = E_t.detach()

            X = X_next
            energy_trace.append(float(E_next.item()))
            solver_stats['step_sizes'].append(float(eta_t))
            solver_stats['backtracks'].append(int(backtracks))
            solver_stats['accepted'].append(bool(accepted))

        return X, energy_trace, solver_stats
        
    def forward(
        self,
        batch_data,
        task_level='graph',
        inference_mode='fixed',
        armijo_c=1e-4,
        armijo_gamma=0.5,
        armijo_eta0=None,
        armijo_max_backtracks=25,
        return_solver_stats=False,
        disable_eval_clipping=False,
    ):
        x = batch_data.x
        if x.dim() == 1:
            x = x.view(-1, 1).float()
            
        X = self.node_encoder(x)
        
        if self.pe_k > 0 and hasattr(batch_data, 'pe') and batch_data.pe is not None:
            pe = batch_data.pe
            if self.training and pe.numel() > 0:
                signs = torch.randint(0, 2, (pe.size(1),), device=pe.device, dtype=torch.long)
                signs = (2 * signs - 1).to(dtype=pe.dtype)
                pe = pe * signs[None, :]
            X = X + self.pe_proj(pe)
            
        static_projections = self._build_static_projections(batch_data)

        if inference_mode == 'fixed':
            X, energy_trace, solver_stats = self._run_fixed_solver(
                X,
                batch_data,
                static_projections,
                disable_eval_clipping=disable_eval_clipping,
            )
        elif inference_mode == 'armijo':
            X, energy_trace, solver_stats = self._run_armijo_solver(
                X,
                batch_data,
                static_projections,
                armijo_c,
                armijo_gamma,
                armijo_eta0,
                armijo_max_backtracks,
            )
        else:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")

        out = self._readout(X, batch_data, task_level)
        if return_solver_stats:
            return out, energy_trace, solver_stats
        return out, energy_trace
