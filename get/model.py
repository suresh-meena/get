import torch
import torch.nn as nn
from types import SimpleNamespace
from .energy import compute_energy_GET, compute_memory_entropy, compute_energy_and_grad_GET
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
        ]
        if hidden_dim > 1:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
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
    def __init__(self, d=256, R=4, K=32, 
                 num_heads=1, head_dim=None,
                 num_motif_types=2,
                 lambda_2=1.0, lambda_3=0.5, lambda_m=1.0, 
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
        
        if self.norm_style == "et":
            self.layernorm = EnergyLayerNorm(d, use_bias=True, eps=1e-5)
        elif self.norm_style == "standard":
            self.layernorm = nn.LayerNorm(d, eps=1e-5)
        else:
            raise ValueError(f"Unsupported norm_style: {self.norm_style}")
        
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
            if weight is None:
                continue
            nn.init.xavier_uniform_(weight, gain=0.5)
        if self.T_tau is not None:
            nn.init.normal_(self.T_tau, mean=0.0, std=0.02)
        if self.W_Qm is not None and self.W_Km is not None and self.B_mem is not None:
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
        }
        params['W_Q2'] = self.W_Q2
        params['W_K2'] = self.W_K2
        if self.use_motif:
            params['W_Q3'] = self.W_Q3
            params['W_K3'] = self.W_K3
            params['T_tau'] = self.T_tau
        if self.use_memory and self.K > 0:
            params['W_Qm'] = self.W_Qm
            params['W_Km'] = self.W_Km
            params['B_mem'] = self.B_mem
        return params
            
    def _compute_pairwise_et_bias(self, G, batch_data):
        if self.pairwise_mask_modulator is None:
            return None
        if batch_data is None or not hasattr(batch_data, "batch"):
            return None
        c_2, u_2 = batch_data.c_2, batch_data.u_2
        if c_2.numel() == 0:
            return G.new_empty((0,))

        node_ptr = batch_data.ptr
        num_graphs = int(node_ptr.numel() - 1)
        max_n = int(torch.diff(node_ptr).max().item())
        
        edge_batch = batch_data.batch[c_2]
        graph_node_offsets = node_ptr[edge_batch]
        src_local = c_2 - graph_node_offsets
        dst_local = u_2 - graph_node_offsets
        
        bsz = num_graphs
        node_batch = batch_data.batch
        node_local = torch.arange(node_batch.size(0), device=G.device) - node_ptr[node_batch]
        
        if G.dim() == 3:
            num_trials = G.size(0)
            x_dense = G.new_zeros((num_trials, bsz, max_n, self.d))
            trial_idx = torch.arange(num_trials, device=G.device).view(-1, 1)
            x_dense[trial_idx, node_batch, node_local] = G
        else:
            x_dense = G.new_zeros((bsz, max_n, self.d))
            x_dense[node_batch, node_local] = G
        
        edge_attr = getattr(batch_data, "edge_attr", None)
        if edge_attr is not None:
            feat_dim = edge_attr.size(-1)
            e_dense = x_dense.new_zeros((bsz, max_n, max_n, feat_dim))
            e_dense[edge_batch, src_local, dst_local] = edge_attr.to(dtype=x_dense.dtype)
        else:
            e_dense = x_dense.new_zeros((bsz, max_n, max_n, 1))
            e_dense[edge_batch, src_local, dst_local] = 1.0
            
        if x_dense.dim() == 4:
            num_trials = x_dense.size(0)
            x_dense_flat = x_dense.view(-1, max_n, self.d)
            e_dense_flat = e_dense.repeat(num_trials, 1, 1, 1)
            mod_dense_flat = self.pairwise_mask_modulator.dense_modulation_batched(x_dense_flat, e_dense_flat) 
            mod_dense = mod_dense_flat.view(num_trials, bsz, self.num_heads, max_n, max_n)
            
            t_idx = torch.arange(num_trials, device=G.device).view(-1, 1, 1)
            l_idx = torch.arange(edge_batch.size(0), device=G.device).view(1, -1, 1)
            h_idx = torch.arange(self.num_heads, device=G.device).view(1, 1, -1)
            out = mod_dense[t_idx, edge_batch[l_idx], h_idx, src_local[l_idx], dst_local[l_idx]]
            out = out.permute(0, 2, 1) # [T, H, L]
        else:
            mod_dense = self.pairwise_mask_modulator.dense_modulation_batched(x_dense, e_dense) 
            l_idx = torch.arange(edge_batch.size(0), device=G.device).view(-1, 1)
            h_idx = torch.arange(self.num_heads, device=G.device).view(1, -1)
            out = mod_dense[edge_batch[l_idx], h_idx, src_local[l_idx], dst_local[l_idx]]
            out = out.permute(1, 0) # [H, L]
            
        return out

    def _build_projections(self, G, static_projections=None, batch_data=None):
        static_projections = static_projections or {}
        projections = {}
        
        if self.use_pairwise:
            Q2 = torch.einsum("...nd, hzd -> ...hnz", G, self.W_Q2)
            K2 = torch.einsum("...nd, hzd -> ...hnz", G, self.W_K2)
            projections['Q2'] = Q2
            projections['K2'] = K2
            
            a_2 = static_projections.get('a_2')
            if a_2 is not None:
                if a_2.dim() == 2:
                    a_2 = a_2.permute(1, 0) 
                elif a_2.dim() == 3:
                    a_2 = a_2.permute(0, 2, 1) 
            
            et_bias = self._compute_pairwise_et_bias(G, batch_data)
            if et_bias is not None:
                if a_2 is not None:
                    if a_2.dim() < et_bias.dim():
                        a_2 = a_2.unsqueeze(0)
                    a_2 = a_2 + et_bias
                else:
                    a_2 = et_bias
            projections['a_2'] = a_2

        if self.use_motif:
            Q3 = torch.einsum("...nd, hzd -> ...hnz", G, self.W_Q3)
            K3 = torch.einsum("...nd, hzd -> ...hnz", G, self.W_K3)
            shape = list(Q3.shape[:-1]) + [self.R, self.head_dim]
            projections['Q3'] = Q3.view(*shape)
            projections['K3'] = K3.view(*shape)

        if self.use_memory and self.K > 0:
            projections['Qm'] = torch.einsum("...nd, hzd -> ...hnz", G, self.W_Qm)
            projections['Km'] = torch.einsum("kd, hzd -> hkz", self.B_mem, self.W_Km)
            
        return projections

    def _get_flat_params_and_projections(self, G, projections):
        trial_batch = 1 if G.dim() == 2 else G.size(0)
        
        flat_projections = {}
        for k, v in projections.items():
            if v is None:
                flat_projections[k] = None
                continue
            if k in ['Q2', 'K2', 'Q3', 'K3', 'Qm']:
                if v.dim() == (4 if k in ['Q3', 'K3'] else 3):
                    flat_projections[k] = v
                else:
                    flat_projections[k] = v.reshape(-1, *v.shape[2:])
            elif k == 'Km':
                flat_projections[k] = v.repeat(trial_batch, 1, 1)
            elif k == 'a_2':
                if v.dim() == 2:
                    flat_projections[k] = v
                else:
                    flat_projections[k] = v.reshape(-1, v.size(-1))
                
        params = self.get_params_dict()
        params['d'] = self.head_dim
        if self.use_motif:
            T_tau = self.T_tau.permute(1, 0, 2, 3) 
            params['T_tau'] = T_tau.repeat(trial_batch, 1, 1, 1)
        
        params['W_Q2'] = self.W_Q2.repeat(trial_batch, 1, 1)
        params['W_K2'] = self.W_K2.repeat(trial_batch, 1, 1)
        if self.use_motif:
            params['W_Q3'] = self.W_Q3.repeat(trial_batch, 1, 1)
            params['W_K3'] = self.W_K3.repeat(trial_batch, 1, 1)
        if self.use_memory and self.K > 0:
            params['W_Qm'] = self.W_Qm.repeat(trial_batch, 1, 1)
            
        return params, flat_projections

    def compute_energy(self, X, batch_data, static_projections=None):
        G = self.layernorm(X)
        trial_batch = 1 if G.dim() == 2 else G.size(0)
        num_heads = self.num_heads
        
        projections = self._build_projections(G, static_projections, batch_data=batch_data)
        flat_params, flat_projections = self._get_flat_params_and_projections(G, projections)
        
        X_flat = X.repeat_interleave(num_heads, dim=0) if trial_batch > 1 else X.unsqueeze(0).repeat(num_heads, 1, 1)
        G_flat = G.repeat_interleave(num_heads, dim=0) if trial_batch > 1 else G.unsqueeze(0).repeat(num_heads, 1, 1)
        
        E_flat = compute_energy_GET(
            X_flat, G_flat,
            batch_data.c_2, batch_data.u_2, batch_data.c_3, batch_data.u_3, batch_data.v_3, batch_data.t_tau,
            flat_params, flat_projections,
        )
        E = E_flat.view(trial_batch, num_heads).mean(dim=-1)
        return E.squeeze(0) if trial_batch == 1 else E

    def energy_and_grad(self, X, batch_data, static_projections=None, create_graph=False):
        if not hasattr(self.layernorm, "backward"):
            if not X.requires_grad:
                X = X.requires_grad_(True)
            E = self.compute_energy(X, batch_data, static_projections)
            grad_X = torch.autograd.grad(E, X, create_graph=create_graph)[0]
            return E, grad_X

        G = self.layernorm(X)
        trial_batch = 1 if G.dim() == 2 else G.size(0)
        num_heads = self.num_heads
        
        projections = self._build_projections(G, static_projections, batch_data=batch_data)
        flat_params, flat_projections = self._get_flat_params_and_projections(G, projections)
        
        X_flat = X.repeat_interleave(num_heads, dim=0) if trial_batch > 1 else X.unsqueeze(0).repeat(num_heads, 1, 1)
        G_flat = G.repeat_interleave(num_heads, dim=0) if trial_batch > 1 else G.unsqueeze(0).repeat(num_heads, 1, 1)
        
        E_flat, grad_X_quad_flat, grad_G_att_flat = compute_energy_and_grad_GET(
            X_flat, G_flat,
            batch_data.c_2, batch_data.u_2, batch_data.c_3, batch_data.u_3, batch_data.v_3, batch_data.t_tau,
            flat_params, flat_projections,
        )
        
        E = E_flat.view(trial_batch, num_heads).mean(dim=-1)
        grad_X_quad = grad_X_quad_flat.view(trial_batch, num_heads, *X.shape).mean(dim=1)
        grad_G_att = grad_G_att_flat.view(trial_batch, num_heads, *G.shape).mean(dim=1)
        
        if trial_batch == 1:
            E = E.squeeze(0)
            grad_X_quad = grad_X_quad.squeeze(0)
            grad_G_att = grad_G_att.squeeze(0)
            
        grad_X_att = self.layernorm.backward(X, grad_G_att)
        grad_X = grad_X_quad - grad_X_att
        
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
        with torch.enable_grad():
            if not is_training:
                X = X.detach()
            E, grad_X = self.energy_and_grad(X, batch_data, static_projections, create_graph=is_training)
            if apply_clipping and self.grad_clip_norm is not None:
                gnorm = grad_X.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                grad_X = grad_X * (self.grad_clip_norm / gnorm).clamp(max=1.0)
            noise = None
            if is_training and self.noise_std > 0.0:
                noise = torch.randn_like(grad_X) * self.noise_std
            
        step = step_size * self.update_damping
        X_next = X - step * grad_X
        if noise is not None:
            X_next = X_next + torch.sqrt(step.clamp_min(1e-8)) * noise
        if apply_state_clipping and self.state_clip_norm is not None:
            snorm = X_next.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            X_next = X_next * (self.state_clip_norm / snorm).clamp(max=1.0)
        return X_next, E

class GETModel(nn.Module):
    def __init__(self, in_dim, d=256, num_classes=1, num_steps=8, compile=False,
                 eta=0.05, eta_max=0.25, dropout=0.1,
                 num_heads=1, head_dim=None,
                 encoder_hidden_mult=2, readout_hidden_mult=2, pe_k=0, rwse_k=0,
                 num_motif_types=2,
                 use_cls_token=False, cls_self_loop=True, **layer_kwargs):
        super().__init__()
        self.d = d
        self.num_steps = num_steps
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

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, d))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
            self.cls_readout = nn.Sequential(
                nn.Linear(d, readout_hidden_mult * d), nn.GELU(), nn.LayerNorm(readout_hidden_mult * d), nn.Dropout(dropout),
                nn.Linear(readout_hidden_mult * d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_classes)
            )

        # Optimization: Use torch.compile to fuse sequential unrolling kernels if requested
        if compile and hasattr(torch, "compile"):
            try:
                # Compile core energy and gradient logic
                self.get_layer.energy_and_grad = torch.compile(self.get_layer.energy_and_grad, dynamic=True)
                # Compile the fixed solver (the main training/inference path)
                self._run_fixed_solver = torch.compile(self._run_fixed_solver, dynamic=True)
                print("INFO:    Successfully compiled GrET inference solvers using torch.compile.")
            except Exception as e:
                print(f"WARNING: torch.compile failed: {e}. Falling back to eager execution.")
        
        self.node_readout = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, num_classes)
        )
        
    @property
    def eta(self):
        return self.eta_max * torch.sigmoid(self.eta_logit)

    def _build_static_projections(self, batch_data):
        static_projections = {}
        if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
            static_projections['a_2'] = self.get_layer.edge_mlp(batch_data.edge_attr)
        if self.get_layer.use_memory and self.get_layer.K > 0:
            static_projections['Km'] = torch.einsum("kd, hzd -> hkz", self.get_layer.B_mem, self.get_layer.W_Km)
        return static_projections

    def _augment_with_cls_token(self, X, batch_data):
        if not self.use_cls_token:
            return X, batch_data, None, None
        num_nodes, num_graphs = int(X.size(0)), int(batch_data.ptr.numel() - 1)
        cls_positions = torch.arange(num_nodes, num_nodes + num_graphs, dtype=torch.long, device=X.device)
        node_positions = torch.arange(num_nodes, dtype=torch.long, device=X.device)
        cls_states = self.cls_token.to(dtype=X.dtype).expand(num_graphs, -1)
        X_aug = torch.cat([X, cls_states], dim=0)
        node_indices, cls_per_node = torch.arange(num_nodes, device=X.device), cls_positions[batch_data.batch]
        c_extra_list, u_extra_list = [cls_per_node, node_indices], [node_indices, cls_per_node]
        if self.cls_self_loop:
            c_extra_list.append(cls_positions)
            u_extra_list.append(cls_positions)
        c_extra, u_extra = torch.cat(c_extra_list, dim=0), torch.cat(u_extra_list, dim=0)
        c_2, u_2 = torch.cat([batch_data.c_2, c_extra], dim=0), torch.cat([batch_data.u_2, u_extra], dim=0)
        edge_attr = batch_data.edge_attr
        if edge_attr is not None:
            edge_attr = torch.cat([edge_attr, edge_attr.new_zeros((c_extra.numel(), *edge_attr.shape[1:]))], dim=0)
        augmented_batch = SimpleNamespace(
            c_2=c_2, u_2=u_2, c_3=batch_data.c_3, u_3=batch_data.u_3, v_3=batch_data.v_3, t_tau=batch_data.t_tau,
            batch=torch.cat([batch_data.batch, torch.arange(num_graphs, dtype=batch_data.batch.dtype, device=batch_data.batch.device)], dim=0),
            ptr=batch_data.ptr, y=batch_data.y, edge_attr=edge_attr, pe=None, num_nodes=X_aug.size(0),
        )
        return X_aug, augmented_batch, cls_positions, node_positions

    def _readout(self, X, batch_data, task_level, cls_positions=None, node_positions=None):
        Z = self.get_layer.layernorm(X)
        if task_level == 'node':
            if node_positions is not None:
                Z = Z[node_positions]
            return self.node_readout(Z)
        if task_level == 'graph':
            if cls_positions is not None:
                return self.cls_readout(Z[cls_positions])
            batch = batch_data.batch
            num_graphs = int(batch.max().item() + 1)
            batch_idx = batch.view(-1, 1).expand_as(Z)
            counts = torch.bincount(batch, minlength=num_graphs).view(-1, 1).to(dtype=Z.dtype)
            z_sum = torch.zeros(num_graphs, self.d, dtype=Z.dtype, device=Z.device)
            z_sum.scatter_add_(0, batch_idx, Z)
            z_mean = z_sum / counts.clamp_min(1.0)
            z_max = torch.full((num_graphs, self.d), float('-inf'), dtype=Z.dtype, device=Z.device)
            z_max.scatter_reduce_(0, batch_idx, Z, reduce="amax", include_self=False)
            z_max = torch.where(counts > 0, z_max, torch.zeros_like(z_max))
            z_sq_sum = torch.zeros(num_graphs, self.d, dtype=Z.dtype, device=Z.device)
            z_sq_sum.scatter_add_(0, batch_idx, Z**2)
            z_std = torch.sqrt(torch.relu((z_sq_sum / counts.clamp_min(1.0)) - z_mean**2) + 1e-6)
            return self.readout(torch.cat([z_mean, z_sum, z_max, z_std], dim=-1))
        raise ValueError(f"Unsupported task_level: {task_level}")

    def _init_solver_stats(self, inference_mode):
        return {'mode': inference_mode, 'step_sizes': [], 'backtracks': [], 'accepted': [], 'grad_norms': []}

    def _run_fixed_solver(self, X, batch_data, static_projections, disable_eval_clipping=False, training_mode=None):
        energy_trace = []
        solver_stats = self._init_solver_stats('fixed')
        if training_mode is None:
            training_mode = self.training
        apply_clipping = not (disable_eval_clipping and not training_mode)
        for _ in range(self.num_steps):
            X, E = self.get_layer(
                X,
                batch_data,
                self.eta,
                static_projections,
                is_training=training_mode,
                apply_clipping=apply_clipping,
                apply_state_clipping=apply_clipping,
            )
            energy_trace.append(float(E.detach().item()))
        G = self.get_layer.layernorm(X)
        params = self.get_layer.get_params_dict()
        projs = self.get_layer._build_projections(G, static_projections, batch_data=batch_data)
        solver_stats['memory_entropy'] = float(
            compute_memory_entropy(G, params=params, projections=projs).detach().item()
        )
        return X, energy_trace, solver_stats

    def _run_armijo_solver(self, X, batch_data, static_projections, armijo_c, armijo_gamma, armijo_eta0, armijo_max_backtracks):
        if self.training:
            raise ValueError("Armijo inference_mode is evaluation-only; call model.eval() first.")
        eta0 = float(self.eta.detach().item()) if armijo_eta0 is None else float(armijo_eta0)
        energy_trace, solver_stats = [], self._init_solver_stats('armijo')
        with torch.enable_grad():
            for _ in range(self.num_steps):
                X = X.detach().requires_grad_(True)
                E_t, grad_X = self.get_layer.energy_and_grad(X, batch_data, static_projections=static_projections, create_graph=False)
                grad_norm_sq = (grad_X ** 2).sum()
                solver_stats['grad_norms'].append(float(torch.sqrt(grad_norm_sq).detach().item()))
                if grad_norm_sq.detach().item() <= 1e-16:
                    energy_trace.append(float(E_t.detach().item()))
                    solver_stats['step_sizes'].append(0.0)
                    solver_stats['backtracks'].append(0)
                    solver_stats['accepted'].append(True)
                    X = X.detach()
                    continue
                etas = eta0 * (armijo_gamma ** torch.arange(armijo_max_backtracks, device=X.device, dtype=X.dtype))
                X_tries = X.detach().unsqueeze(0) - etas.view(-1, 1, 1) * grad_X.detach().unsqueeze(0)
                E_tries = self.get_layer.compute_energy(X_tries, batch_data, static_projections=static_projections)
                success = (E_tries <= E_t - armijo_c * etas * grad_norm_sq)
                success_indices = torch.nonzero(success).view(-1)
                if success_indices.numel() > 0:
                    idx = int(success_indices[0].item())
                    accepted = True
                    accepted_eta = float(etas[idx].item())
                    accepted_backtracks = idx
                    accepted_energy = float(E_tries[idx].item())
                    accepted_state = X_tries[idx]
                else:
                    accepted = False
                    accepted_eta = 0.0
                    accepted_backtracks = armijo_max_backtracks
                    accepted_energy = float(E_t.detach().item())
                    accepted_state = X.detach()
                X = accepted_state
                energy_trace.append(accepted_energy)
                solver_stats['step_sizes'].append(accepted_eta)
                solver_stats['backtracks'].append(accepted_backtracks)
                solver_stats['accepted'].append(accepted)
        G = self.get_layer.layernorm(X)
        params = self.get_layer.get_params_dict()
        projs = self.get_layer._build_projections(G, static_projections, batch_data=batch_data)
        solver_stats['memory_entropy'] = float(
            compute_memory_entropy(G, params=params, projections=projs).detach().item()
        )
        return X, energy_trace, solver_stats
        
    def forward(self, batch_data, task_level='graph', inference_mode='fixed', armijo_c=1e-4, armijo_gamma=0.5, armijo_eta0=None, armijo_max_backtracks=25, return_solver_stats=False, disable_eval_clipping=False, is_training=None):
        training_mode = self.training if is_training is None else bool(is_training)
        x = batch_data.x
        if x.dim() == 1:
            x = x.view(-1, 1).float()
        X = self.node_encoder(x)
        if self.pe_k > 0 and hasattr(batch_data, 'pe') and batch_data.pe is not None:
            pe = random_flip_pe_signs(batch_data.pe, training=self.training)
            X = X + self.pe_proj(pe)
        if self.rwse_k > 0 and hasattr(batch_data, 'rwse') and batch_data.rwse is not None:
            X = X + self.rwse_proj(batch_data.rwse)
        X, solver_batch, cls_positions, node_positions = self._augment_with_cls_token(X, batch_data)
        static_projections = self._build_static_projections(solver_batch)
        if inference_mode == 'fixed':
            X, energy_trace, solver_stats = self._run_fixed_solver(X, solver_batch, static_projections, disable_eval_clipping=disable_eval_clipping, training_mode=training_mode)
        elif inference_mode == 'armijo':
            X, energy_trace, solver_stats = self._run_armijo_solver(X, solver_batch, static_projections, armijo_c, armijo_gamma, armijo_eta0, armijo_max_backtracks)
        else:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")
        out = self._readout(X, solver_batch, task_level, cls_positions=cls_positions, node_positions=node_positions)
        if return_solver_stats:
            return out, energy_trace, solver_stats
        return out, energy_trace
