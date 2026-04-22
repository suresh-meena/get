import torch
import torch.nn as nn
from types import SimpleNamespace
from .energy import compute_energy_GET, compute_memory_entropy
from .et_core import ETGraphMaskModulator, EnergyLayerNorm
from .utils import random_flip_pe_signs


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
                 pairwise_symmetric=False, noise_std=0.0,
                 use_pairwise=True, use_motif=True, use_memory=True,
                 norm_style="standard",
                 pairwise_et_mask=False,
                 pairwise_et_kernel_size=3):
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
        
        if self.norm_style == "et":
            self.layernorm = EnergyLayerNorm(d, use_bias=True, eps=1e-5)
        elif self.norm_style == "standard":
            self.layernorm = nn.LayerNorm(d, eps=1e-5)
        else:
            raise ValueError(f"Unsupported norm_style: {self.norm_style}")
        
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
            nn.LazyLinear(d),
            nn.GELU(),
            nn.Linear(d, 1)
        )
        self.pairwise_mask_modulator = None
        if self.pairwise_et_mask:
            self.pairwise_mask_modulator = ETGraphMaskModulator(
                d=d,
                num_heads=1,
                edge_feat_dim=None,
                kernel_size=pairwise_et_kernel_size,
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
            'use_pairwise': self.use_pairwise,
            'use_motif': self.use_motif,
            'use_memory': self.use_memory,
            'norm_style': self.norm_style,
            'pairwise_et_mask': self.pairwise_et_mask,
            
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
            
    def _compute_pairwise_et_bias(self, G, batch_data):
        if self.pairwise_mask_modulator is None:
            return None
        if batch_data is None or not hasattr(batch_data, "batch"):
            return None
        c_2 = batch_data.c_2
        u_2 = batch_data.u_2
        edge_attr = getattr(batch_data, "edge_attr", None)
        if c_2.numel() == 0:
            return G.new_empty((0,))

        total_nodes = int(G.size(0))
        batch_vec = batch_data.batch
        num_graphs = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0
        out = G.new_zeros((c_2.numel(),))
        for g_idx in range(num_graphs):
            nodes = torch.nonzero(batch_vec == g_idx, as_tuple=False).flatten()
            n = int(nodes.numel())
            if n <= 0:
                continue
            node_flag = torch.zeros((total_nodes,), dtype=torch.bool, device=G.device)
            node_flag[nodes] = True
            edge_mask = node_flag[c_2] & node_flag[u_2]
            if not bool(edge_mask.any()):
                continue
            src_global = c_2[edge_mask]
            dst_global = u_2[edge_mask]
            inv = torch.full((total_nodes,), -1, dtype=torch.long, device=G.device)
            inv[nodes] = torch.arange(n, device=G.device, dtype=torch.long)
            src = inv[src_global]
            dst = inv[dst_global]
            adj = torch.zeros((n, n), dtype=torch.bool, device=G.device)
            adj[src, dst] = True
            if edge_attr is not None:
                edge_features = G.new_zeros((n, n, edge_attr.size(-1)))
                edge_features[src, dst] = edge_attr[edge_mask].to(dtype=G.dtype)
            else:
                edge_features = adj.to(dtype=G.dtype)
            mod = self.pairwise_mask_modulator.dense_modulation(
                x_local=G[nodes],
                edge_features=edge_features,
            ).squeeze(0)
            out[edge_mask] = mod[src, dst]
        return out

    def _build_projections(self, G, static_projections=None, batch_data=None):
        static_projections = static_projections or {}
        active_weights = []
        if self.use_pairwise:
            active_weights.extend([self.W_Q2, self.W_K2])
        if self.use_motif:
            active_weights.extend([self.W_Q3, self.W_K3])
        if self.use_memory and self.K > 0:
            active_weights.append(self.W_Qm)

        if not active_weights:
            return {}

        Z_all = G @ torch.cat(active_weights, dim=1)
        d, R = self.d, self.R

        offset = 0
        projections = {}
        if self.use_pairwise:
            projections['Q2'] = Z_all[:, offset : offset + d]
            a_2 = static_projections.get('a_2')
            et_bias = self._compute_pairwise_et_bias(G, batch_data)
            if et_bias is not None:
                a_2 = et_bias if a_2 is None else (a_2 + et_bias)
            projections['a_2'] = a_2
            offset += d
            projections['K2'] = Z_all[:, offset : offset + d]
            offset += d

        if self.use_motif:
            projections['Q3'] = Z_all[:, offset : offset + R * d].reshape(-1, R, d)
            offset += R * d
            projections['K3'] = Z_all[:, offset : offset + R * d].reshape(-1, R, d)
            offset += R * d

        if self.use_memory and self.K > 0:
            projections['Qm'] = Z_all[:, offset:]
            projections['Km'] = static_projections['Km']
        return projections

    def compute_energy(self, X, batch_data, static_projections=None):
        G = self.layernorm(X)
        params = self.get_params_dict()
        projections = self._build_projections(G, static_projections, batch_data=batch_data)
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
                 encoder_hidden_mult=2, readout_hidden_mult=2, pe_k=0,
                 use_cls_token=False, cls_self_loop=True, **layer_kwargs):
        super().__init__()
        self.d = d
        self.num_steps = num_steps
        self.eta_max = eta_max
        self.pe_k = pe_k
        self.use_cls_token = bool(use_cls_token)
        self.cls_self_loop = bool(cls_self_loop)
        
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

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, d))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            self.cls_readout = nn.Sequential(
                nn.Linear(d, readout_hidden_mult * d),
                nn.GELU(),
                nn.LayerNorm(readout_hidden_mult * d),
                nn.Dropout(dropout),
                nn.Linear(readout_hidden_mult * d, d),
                nn.GELU(),
                nn.LayerNorm(d),
                nn.Dropout(dropout),
                nn.Linear(d, num_classes),
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

    def _augment_with_cls_token(self, X, batch_data):
        if not self.use_cls_token:
            return X, batch_data, None, None

        num_nodes = int(X.size(0))
        num_graphs = int(batch_data.ptr.numel() - 1)
        cls_positions = torch.arange(
            num_nodes,
            num_nodes + num_graphs,
            dtype=torch.long,
            device=X.device,
        )
        node_positions = torch.arange(num_nodes, dtype=torch.long, device=X.device)
        cls_states = self.cls_token.to(dtype=X.dtype).expand(num_graphs, -1)
        X_aug = torch.cat([X, cls_states], dim=0)

        extra_c = []
        extra_u = []
        for g_idx in range(num_graphs):
            start = int(batch_data.ptr[g_idx].item())
            end = int(batch_data.ptr[g_idx + 1].item())
            if end <= start:
                continue
            nodes = torch.arange(start, end, dtype=torch.long, device=X.device)
            cls = cls_positions[g_idx]
            extra_c.extend([cls.expand_as(nodes), nodes])
            extra_u.extend([nodes, cls.expand_as(nodes)])
            if self.cls_self_loop:
                extra_c.append(cls.view(1))
                extra_u.append(cls.view(1))

        if extra_c:
            c_extra = torch.cat(extra_c, dim=0)
            u_extra = torch.cat(extra_u, dim=0)
            c_2 = torch.cat([batch_data.c_2, c_extra], dim=0)
            u_2 = torch.cat([batch_data.u_2, u_extra], dim=0)
        else:
            c_extra = batch_data.c_2.new_empty(0)
            c_2 = batch_data.c_2
            u_2 = batch_data.u_2

        edge_attr = batch_data.edge_attr
        if edge_attr is not None and c_extra.numel() > 0:
            cls_edge_attr = edge_attr.new_zeros((c_extra.numel(), *edge_attr.shape[1:]))
            edge_attr = torch.cat([edge_attr, cls_edge_attr], dim=0)

        cls_batch = torch.arange(num_graphs, dtype=batch_data.batch.dtype, device=batch_data.batch.device)
        batch_aug = torch.cat([batch_data.batch, cls_batch], dim=0)

        augmented_batch = SimpleNamespace(
            c_2=c_2,
            u_2=u_2,
            c_3=batch_data.c_3,
            u_3=batch_data.u_3,
            v_3=batch_data.v_3,
            t_tau=batch_data.t_tau,
            batch=batch_aug,
            ptr=batch_data.ptr,
            y=batch_data.y,
            edge_attr=edge_attr,
            pe=None,
            num_nodes=X_aug.size(0),
        )
        return X_aug, augmented_batch, cls_positions, node_positions

    def _readout(self, X, batch_data, task_level, cls_positions=None, node_positions=None):
        Z = self.get_layer.layernorm(X)
        if task_level == 'node':
            if node_positions is not None:
                Z = Z[node_positions]
            out = self.node_readout(Z)
            return out
        if task_level == 'graph':
            if cls_positions is not None:
                out = self.cls_readout(Z[cls_positions])
                return out
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
        G = self.get_layer.layernorm(X)
        solver_stats['memory_entropy'] = float(
            compute_memory_entropy(
                G,
                params=self.get_layer.get_params_dict(),
                projections=self.get_layer._build_projections(G, static_projections, batch_data=batch_data),
            ).detach().item()
        )
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

            accepted = False
            accepted_eta = 0.0
            accepted_backtracks = armijo_max_backtracks
            accepted_energy = float(E_t.detach().item())
            accepted_state = X.detach()

            # Sequential Armijo check; avoids batched energy assumptions.
            for bt in range(armijo_max_backtracks):
                eta_bt = eta0 * (armijo_gamma ** bt)
                X_try = (X - eta_bt * grad_X).detach()
                E_try = self.get_layer.compute_energy(
                    X_try,
                    batch_data,
                    static_projections=static_projections,
                )
                rhs = E_t - armijo_c * eta_bt * grad_norm_sq
                if bool(E_try <= rhs):
                    accepted = True
                    accepted_eta = float(eta_bt)
                    accepted_backtracks = bt
                    accepted_energy = float(E_try.detach().item())
                    accepted_state = X_try
                    break

            X = accepted_state
            energy_trace.append(accepted_energy)
            solver_stats['step_sizes'].append(accepted_eta)
            solver_stats['backtracks'].append(accepted_backtracks)
            solver_stats['accepted'].append(accepted)

        G = self.get_layer.layernorm(X)
        solver_stats['memory_entropy'] = float(
            compute_memory_entropy(
                G,
                params=self.get_layer.get_params_dict(),
                projections=self.get_layer._build_projections(G, static_projections, batch_data=batch_data),
            ).detach().item()
        )
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
            pe = random_flip_pe_signs(batch_data.pe, training=self.training)
            X = X + self.pe_proj(pe)

        X, solver_batch, cls_positions, node_positions = self._augment_with_cls_token(X, batch_data)
            
        static_projections = self._build_static_projections(solver_batch)

        if inference_mode == 'fixed':
            X, energy_trace, solver_stats = self._run_fixed_solver(
                X,
                solver_batch,
                static_projections,
                disable_eval_clipping=disable_eval_clipping,
            )
        elif inference_mode == 'armijo':
            X, energy_trace, solver_stats = self._run_armijo_solver(
                X,
                solver_batch,
                static_projections,
                armijo_c,
                armijo_gamma,
                armijo_eta0,
                armijo_max_backtracks,
            )
        else:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")

        out = self._readout(
            X,
            solver_batch,
            task_level,
            cls_positions=cls_positions,
            node_positions=node_positions,
        )
        if return_solver_stats:
            return out, energy_trace, solver_stats
        return out, energy_trace
