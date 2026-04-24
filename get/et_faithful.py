import torch
import torch.nn as nn

from .et_core import ETCoreBlock, ETGraphMaskModulator, EnergyLayerNorm


class ETFaithfulGraphModel(nn.Module):
    """
    ET graph adapter built on modular ET core primitives:
    - ET core (EnergyLayerNorm + attention energy + CHN-ReLU memory)
    - graph adapter (CLS token + Laplacian PE + graph masking)
    """

    def __init__(
        self,
        in_dim,
        d,
        num_classes,
        num_steps=8,
        num_heads=1,
        head_dim=None,
        pe_k=16,
        rwse_k=0,
        eta=0.1,
        eta_max=0.25,
        K=32,
        allow_self=False,
        noise_std=0.0,
        grad_clip_norm=1.0,
        state_clip_norm=10.0,
        dropout=0.1,
        encoder_hidden_mult=2,
        readout_hidden_mult=2,
        et_official_mode=False,
        num_blocks=1,
        mask_mode="official_dense",
        node_cap=None,
        dense_kernel_size=3,  # compatibility placeholder for CLI
        share_block_weights=False,
    ):
        super().__init__()
        del dense_kernel_size
        self.d = int(d)
        self.num_steps = int(num_steps)
        self.num_blocks = int(max(1, num_blocks))
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or d)
        self.pe_k = int(pe_k)
        self.rwse_k = int(rwse_k)
        self.allow_self = bool(allow_self)
        self.noise_std = float(noise_std)
        self.grad_clip_norm = grad_clip_norm
        self.state_clip_norm = state_clip_norm
        self.node_cap = None if node_cap is None else int(node_cap)
        self.share_block_weights = bool(share_block_weights)

        self.mask_mode = str(mask_mode)
        if et_official_mode and self.mask_mode == "sparse":
            self.mask_mode = "official_dense"
        if self.mask_mode not in {"sparse", "official_dense"}:
            raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

        # Match the reference ET graph code: a linear projection in and a
        # linear classifier out, with the recurrent energy descent carrying the
        # capacity of the model.
        self.node_encoder = nn.Linear(in_dim, d)
        self.pe_proj = nn.Linear(self.pe_k, d) if self.pe_k > 0 else None
        self.rwse_proj = nn.Linear(self.rwse_k, d) if self.rwse_k > 0 else None
        self.cls_token = nn.Parameter(torch.zeros(1, d))

        self.norm_blocks = nn.ModuleList([EnergyLayerNorm(d, use_bias=True, eps=1e-5) for _ in range(self.num_blocks)])
        self.mask_modulator = None
        if self.mask_mode == "official_dense":
            self.mask_modulator = ETGraphMaskModulator(
                d=self.d,
                num_heads=self.num_heads,
                edge_feat_dim=None,
                kernel_size=3,
            )
        if self.share_block_weights:
            shared = ETCoreBlock(d=d, num_heads=self.num_heads, head_dim=self.head_dim, num_memories=K)
            self.core_blocks = nn.ModuleList([shared for _ in range(self.num_blocks)])
        else:
            self.core_blocks = nn.ModuleList(
                [ETCoreBlock(d=d, num_heads=self.num_heads, head_dim=self.head_dim, num_memories=K) for _ in range(self.num_blocks)]
            )

        eta = min(max(float(eta), 1e-4), float(eta_max) - 1e-4)
        self.eta_max = float(eta_max)
        self.eta_logit = nn.Parameter(torch.logit(torch.tensor(eta / self.eta_max)))

        self.readout = nn.Linear(d, num_classes)
        self.node_readout = nn.Linear(d, num_classes)

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @property
    def eta(self):
        return self.eta_max * torch.sigmoid(self.eta_logit)

    def _build_augmented_graph(self, batch_data, z_nodes):
        ptr = batch_data.ptr
        c2 = batch_data.c_2
        u2 = batch_data.u_2
        num_graphs = ptr.numel() - 1
        num_nodes = z_nodes.size(0)
        device = z_nodes.device
        dtype = z_nodes.dtype

        # 1. Integrate CLS tokens
        X_aug = torch.empty((num_nodes + num_graphs, self.d), dtype=dtype, device=device)
        X_aug[:num_nodes] = z_nodes
        X_aug[num_nodes:] = self.cls_token.expand(num_graphs, -1)
        
        cls_indices = torch.arange(num_nodes, num_nodes + num_graphs, device=device)
        node_batch = batch_data.batch
        node_cls_indices = cls_indices[node_batch]
        
        # 2. Vectorized augmented edges
        # c2/u2 + 2*N (cls-node bidirectional) + N or 2*N (self-loops)
        num_orig_edges = c2.numel()
        num_cls_edges = 2 * num_nodes
        num_self_loops = num_nodes + num_graphs if self.allow_self else num_graphs
        total_aug_edges = num_orig_edges + num_cls_edges + num_self_loops
        
        c_aug = torch.empty(total_aug_edges, dtype=torch.long, device=device)
        u_aug = torch.empty(total_aug_edges, dtype=torch.long, device=device)
        
        curr = 0
        # Original edges
        c_aug[curr : curr + num_orig_edges] = c2
        u_aug[curr : curr + num_orig_edges] = u2
        curr += num_orig_edges
        
        # CLS <-> Node edges
        node_indices = torch.arange(num_nodes, device=device)
        c_aug[curr : curr + num_nodes] = node_indices
        u_aug[curr : curr + num_nodes] = node_cls_indices
        curr += num_nodes
        c_aug[curr : curr + num_nodes] = node_cls_indices
        u_aug[curr : curr + num_nodes] = node_indices
        curr += num_nodes
        
        # Self-loops
        if self.allow_self:
            c_aug[curr : curr + num_nodes] = node_indices
            u_aug[curr : curr + num_nodes] = node_indices
            curr += num_nodes
            
        c_aug[curr : curr + num_graphs] = cls_indices
        u_aug[curr : curr + num_graphs] = cls_indices
        curr += num_graphs
        
        # 3. Handle PE
        if hasattr(batch_data, "pe") and batch_data.pe is not None:
            pe = batch_data.pe
            # Pre-allocated augmented PE
            pe_aug = torch.zeros((num_nodes + num_graphs, pe.size(-1)), dtype=dtype, device=device)
            pe_aug[:num_nodes] = pe.to(dtype=dtype)
            if self.pe_proj is not None:
                X_aug.add_(self.pe_proj(pe_aug))

        # RWSE
        if self.rwse_proj is not None and hasattr(batch_data, "rwse") and batch_data.rwse is not None:
            rwse = batch_data.rwse
            rwse_aug = torch.zeros((num_nodes + num_graphs, rwse.size(-1)), dtype=dtype, device=device)
            rwse_aug[:num_nodes] = rwse.to(dtype=dtype)
            X_aug.add_(self.rwse_proj(rwse_aug))

        return X_aug, c_aug, u_aug, cls_indices, node_indices, None

    def _prepare_dense_cache(self, x, cls_indices, node_indices, batch_data):
        if self.mask_mode != "official_dense" or self.mask_modulator is None:
            return None
        
        # x is [N_aug, D]. We need to reshape it into [B, max_n_aug, D]
        ptr = batch_data.ptr
        num_graphs = ptr.numel() - 1
        num_original_nodes = ptr[-1].item()
        
        # Max nodes in augmented graph: max(n_orig) + 1 (for CLS)
        max_n_orig = torch.diff(ptr).max().item()
        max_n_aug = int(max_n_orig + 1)
        
        # Prepare dense x_batch
        x_dense = x.new_zeros((num_graphs, max_n_aug, self.d))
        
        # Map original nodes
        node_batch = batch_data.batch
        node_local = torch.arange(num_original_nodes, device=x.device) - ptr[node_batch]
        # Augmented nodes have CLS at local 0, and original nodes at local 1..n
        x_dense[node_batch, node_local + 1] = x[node_indices]
        # Map CLS tokens to local 0
        x_dense[torch.arange(num_graphs, device=x.device), 0] = x[cls_indices]
        
        edge_attr = getattr(batch_data, "edge_attr", None)
        c2 = batch_data.c_2
        u2 = batch_data.u_2
        
        if edge_attr is not None:
            feat_dim = edge_attr.size(-1)
            e_dense = x.new_zeros((num_graphs, max_n_aug, max_n_aug, feat_dim))
            # Original edges
            edge_batch = node_batch[c2]
            src_local = c2 - ptr[edge_batch] + 1
            dst_local = u2 - ptr[edge_batch] + 1
            e_dense[edge_batch, src_local, dst_local] = edge_attr.to(dtype=x.dtype)
            
            # CLS <-> Node edges in e_dense (local 0 <-> 1..n)
            # We use zeros or a specific value for CLS edges? Default ET code uses 1.0 or specific features.
            # Here we follow the logic that CLS is fully connected.
            e_dense[:, 0, :, :] = 1.0
            e_dense[:, :, 0, :] = 1.0
        else:
            e_dense = x.new_zeros((num_graphs, max_n_aug, max_n_aug, 1))
            edge_batch = node_batch[c2]
            src_local = c2 - ptr[edge_batch] + 1
            dst_local = u2 - ptr[edge_batch] + 1
            e_dense[edge_batch, src_local, dst_local] = 1.0
            e_dense[:, 0, :, :] = 1.0
            e_dense[:, :, 0, :] = 1.0

        dense_modulation = self.mask_modulator.dense_modulation_batched(x_dense, e_dense)
        sizes = torch.diff(ptr) + 1
        return dense_modulation, sizes

    def _solve_dynamics(self, x_aug, c_aug, u_aug, cls_indices, node_indices, batch_data, static_projections=None):
        energy_trace = []
        step = self.eta
        x = x_aug
        
        # Initial dense cache if needed
        dense_cache = self._prepare_dense_cache(x, cls_indices, node_indices, batch_data)
        if dense_cache is None:
            dense_modulation = None
            dense_sizes = None
        else:
            dense_modulation, dense_sizes = dense_cache

        # Solver loop
        for block_idx in range(self.num_blocks):
            norm = self.norm_blocks[block_idx]
            core = self.core_blocks[block_idx]
            for step_idx in range(self.num_steps):
                g = norm(x)
                # Analytical gradient call
                e, grad_g = core.energy_and_grad(
                    g,
                    c_aug,
                    u_aug,
                    None, # graph_chunks is no longer used
                    mask_mode=self.mask_mode,
                    dense_modulation=dense_modulation,
                    dense_sizes=dense_sizes,
                    static_projections=static_projections,
                )
                
                # Pull back through norm
                grad_x = norm.backward(x, grad_g)

                noise = None
                if self.training and self.noise_std > 0:
                    noise = torch.randn_like(grad_x) * self.noise_std
                if self.grad_clip_norm is not None:
                    gnorm = grad_x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    grad_x = grad_x * (self.grad_clip_norm / gnorm).clamp(max=1.0)

                # In-place update for efficiency
                x.sub_(step * grad_x)
                if noise is not None:
                    x.add_(torch.sqrt(step.clamp_min(1e-8)) * noise)
                if self.state_clip_norm is not None:
                    snorm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    x.mul_((self.state_clip_norm / snorm).clamp(max=1.0))

                energy_trace.append(float(e.detach().item()))
        return x, energy_trace

    def _init_solver_stats(self, inference_mode):
        return {
            'mode': inference_mode,
            'step_sizes': [],
            'backtracks': [],
            'accepted': [],
            'grad_norms': [],
        }

    def _run_armijo_solver(
        self,
        x_aug,
        c_aug,
        u_aug,
        cls_indices,
        node_indices,
        batch_data,
        armijo_c=1e-4,
        armijo_gamma=0.5,
        armijo_eta0=None,
        armijo_max_backtracks=25,
        chunk_size=4,
        static_projections=None,
    ):
        if self.training:
            raise ValueError("Armijo inference_mode is evaluation-only; call model.eval() first.")
        
        eta0 = float(self.eta.detach().item()) if armijo_eta0 is None else float(armijo_eta0)
        eta0 = max(eta0, 1e-8)

        energy_trace = []
        solver_stats = self._init_solver_stats('armijo')
        x = x_aug

        # Pre-compute dense modulation
        dense_cache = self._prepare_dense_cache(x, cls_indices, node_indices, batch_data)
        if dense_cache is None:
            dense_modulation = None
            dense_sizes = None
        else:
            dense_modulation, dense_sizes = dense_cache

        etas_all = eta0 * (armijo_gamma ** torch.arange(armijo_max_backtracks, device=x.device, dtype=x.dtype))

        for block_idx in range(self.num_blocks):
            norm = self.norm_blocks[block_idx]
            core = self.core_blocks[block_idx]

            for _ in range(self.num_steps):
                g = norm(x)
                e_t, grad_g = core.energy_and_grad(
                    g,
                    c_aug,
                    u_aug,
                    None,
                    self.mask_mode,
                    dense_modulation=dense_modulation,
                    dense_sizes=dense_sizes,
                    static_projections=static_projections,
                )
                grad_x = norm.backward(x, grad_g)

                grad_norm_sq = (grad_x ** 2).sum()
                grad_norm = float(torch.sqrt(grad_norm_sq).detach().item())
                solver_stats['grad_norms'].append(grad_norm)

                if grad_norm_sq.detach().item() <= 1e-16:
                    energy_trace.append(float(e_t.detach().item()))
                    solver_stats['step_sizes'].append(0.0)
                    solver_stats['backtracks'].append(0)
                    solver_stats['accepted'].append(True)
                    continue

                # Chunked vectorized Armijo check
                accepted_eta = 0.0
                accepted_backtracks = armijo_max_backtracks
                accepted_energy = float(e_t.detach().item())
                accepted_state = x.detach()
                found = False
                
                for start_bt in range(0, armijo_max_backtracks, chunk_size):
                    end_bt = min(start_bt + chunk_size, armijo_max_backtracks)
                    etas = etas_all[start_bt:end_bt]
                    
                    x_tries = x.detach().unsqueeze(0) - etas.view(-1, 1, 1) * grad_x.detach().unsqueeze(0)
                    g_tries = norm(x_tries)
                    e_tries = core.energy(
                        g_tries,
                        c_aug,
                        u_aug,
                        None,
                        self.mask_mode,
                        dense_modulation=dense_modulation,
                        dense_sizes=dense_sizes,
                        static_projections=static_projections,
                    )
                    
                    rhs = e_t - armijo_c * etas * grad_norm_sq
                    success = (e_tries <= rhs)
                    
                    success_indices = torch.nonzero(success).view(-1)
                    if success_indices.numel() > 0:
                        idx = int(success_indices[0].item())
                        accepted_eta = float(etas[idx].item())
                        accepted_backtracks = start_bt + idx
                        accepted_energy = float(e_tries[idx].item())
                        accepted_state = x_tries[idx]
                        found = True
                        break
                
                x = accepted_state
                energy_trace.append(accepted_energy)
                solver_stats['step_sizes'].append(accepted_eta)
                solver_stats['backtracks'].append(accepted_backtracks)
                solver_stats['accepted'].append(found)
        
        return x, energy_trace, solver_stats

    def forward(self, batch_data, task_level="graph", inference_mode='fixed', return_solver_stats=False):
        x = batch_data.x
        if x.dim() == 1:
            x = x.view(-1, 1).float()
        x = x.to(dtype=self.cls_token.dtype)

        z_nodes = self.node_encoder(x)
        x_aug, c_aug, u_aug, cls_pos, node_pos, graph_chunks = self._build_augmented_graph(batch_data, z_nodes)
        
        # Motif bias hoisting
        static_projections = {}
        # In ETFaithful, motifs are not yet fully active in the sparse path, 
        # but if they were, we would hoist the T_tau here.
        
        if inference_mode == 'fixed':
            x_final, energy_trace = self._solve_dynamics(x_aug, c_aug, u_aug, cls_pos, node_pos, batch_data, static_projections=static_projections)
            solver_stats = {'mode': 'fixed', 'energy_trace': energy_trace}
        elif inference_mode == 'armijo':
            x_final, energy_trace, solver_stats = self._run_armijo_solver(x_aug, c_aug, u_aug, cls_pos, node_pos, batch_data, static_projections=static_projections)
        else:
            raise ValueError(f"Unsupported inference_mode: {inference_mode}")

        g_final = self.norm_blocks[-1](x_final)
        solver_stats["memory_entropy"] = self.core_blocks[-1].hn.entropy(g_final)

        if task_level == "graph":
            out = self.readout(x_final[cls_pos])
            if return_solver_stats:
                return out, energy_trace, solver_stats
            return out, energy_trace
        if task_level == "node":
            out = self.node_readout(x_final[node_pos])
            if return_solver_stats:
                return out, energy_trace, solver_stats
            return out, energy_trace
        raise ValueError(f"Unsupported task_level: {task_level}")
