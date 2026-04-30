"""ET core building blocks: attention energy, Hopfield energy, combined block."""
import math
import torch
import torch.nn as nn
from get.energy.ops import segment_logsumexp
from get.nn import EnergyLayerNorm, ETGraphMaskModulator

try:
    from torch_scatter import scatter as pyg_scatter
except (ImportError, OSError):
    pyg_scatter = None


class ETAttentionCore(nn.Module):
    """Official ET attention energy with optional sparse graph masking."""

    def __init__(self, d, num_heads=1, head_dim=None, beta_init=None):
        super().__init__()
        self.d = int(d)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or d)
        self.Wq = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d))
        self.Wk = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d))
        self.Hw = nn.Parameter(torch.empty(self.num_heads, self.num_heads))
        default_beta = 1.0 / math.sqrt(float(self.head_dim))
        beta = default_beta if beta_init is None else float(beta_init)
        self.betas = nn.Parameter(torch.full((self.num_heads,), beta))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.Wq, mean=0.0, std=0.002)
        nn.init.normal_(self.Wk, mean=0.0, std=0.002)
        nn.init.normal_(self.Hw, mean=0.0, std=0.002)

    def _format_dense_modulation(self, dense_modulation, batch_idx=None):
        if dense_modulation is None:
            return None
        if isinstance(dense_modulation, (list, tuple)):
            if batch_idx is None:
                if len(dense_modulation) != 1:
                    raise ValueError("batch_idx is required when dense_modulation is a list with more than one item.")
                dense_modulation = dense_modulation[0]
            else:
                dense_modulation = dense_modulation[batch_idx]
        if dense_modulation.dim() == 4:
            if dense_modulation.size(1) == self.num_heads:
                return dense_modulation.permute(0, 2, 3, 1).contiguous()
            if dense_modulation.size(-1) == self.num_heads:
                return dense_modulation.contiguous()
            raise ValueError("Expected head dim at axis 1 or -1 for dense modulation.")
        if dense_modulation.dim() == 3:
            if dense_modulation.size(0) == self.num_heads:
                return dense_modulation.permute(1, 2, 0).contiguous()
            if dense_modulation.size(-1) == self.num_heads:
                return dense_modulation.contiguous()
            return dense_modulation.unsqueeze(-1).contiguous()
        if dense_modulation.dim() == 2:
            return dense_modulation.unsqueeze(-1).contiguous()
        raise ValueError(f"Unsupported dense modulation rank: {dense_modulation.dim()}")

    def _project_qk(self, g):
        wq_t = self.Wq.transpose(1, 2)
        wk_t = self.Wk.transpose(1, 2)
        if g.dim() == 2:
            q = torch.matmul(g.unsqueeze(0), wq_t).permute(1, 0, 2).contiguous()
            k = torch.matmul(g.unsqueeze(0), wk_t).permute(1, 0, 2).contiguous()
        elif g.dim() == 3:
            q = torch.einsum("bnd,hdz->bnhz", g, wq_t).contiguous()
            k = torch.einsum("bnd,hdz->bnhz", g, wk_t).contiguous()
        else:
            q = torch.einsum("tbnd,hdz->tbnhz", g, wq_t).contiguous()
            k = torch.einsum("tbnd,hdz->tbnhz", g, wk_t).contiguous()
        return q, k

    def _dense_energy_batched(self, g, dense_modulation, dense_sizes=None, return_grad=False):
        q_all, k_all = self._project_qk(g)
        q_perm = q_all.permute(0, 2, 1, 3).contiguous()
        k_perm = k_all.permute(0, 2, 1, 3).contiguous()
        logits = torch.einsum("bhnz,bhmz->bhnm", q_perm, k_perm)
        logits = self.betas[None, :, None, None] * logits
        logits = logits.permute(0, 2, 3, 1).contiguous()
        logits = torch.matmul(logits, self.Hw)
        if dense_modulation is not None:
            logits = logits * self._format_dense_modulation(dense_modulation)
        if dense_sizes is None:
            max_n = int(g.size(1))
            dense_sizes = torch.full((g.size(0),), max_n, dtype=torch.long, device=g.device)
        else:
            max_n = int(g.size(1))
        node_ids = torch.arange(max_n, device=g.device)
        node_mask = node_ids[None, :] < dense_sizes[:, None]
        pair_mask = node_mask[:, :, None] & node_mask[:, None, :]
        finfo_min = torch.finfo(g.dtype).min
        logits = logits.masked_fill(~pair_mask[:, :, :, None], finfo_min)
        lse = torch.logsumexp(logits, dim=2)
        lse = torch.where(node_mask[:, :, None], lse, torch.zeros_like(lse))
        e = -(((lse.sum(dim=1) / self.betas[None, :]).sum()))
        if not return_grad:
            return e
        probs = torch.softmax(logits, dim=2)
        grad_logits = -(probs / self.betas[None, None, None, :])
        grad_qk_beta = grad_logits @ self.Hw.t()
        grad_qk = grad_qk_beta * self.betas[None, None, None, :]
        grad_q_all = torch.einsum("bnmh,bmhz->bnhz", grad_qk, k_all)
        grad_k_all = torch.einsum("bmnh,bmhz->bnhz", grad_qk, q_all)
        wq_t = self.Wq.transpose(1, 2)
        wk_t = self.Wk.transpose(1, 2)
        grad_g = torch.einsum("bnhz,hdz->bnd", grad_q_all, wq_t) + torch.einsum("bnhz,hdz->bnd", grad_k_all, wk_t)
        return e, grad_g

    def _dense_energy(self, g, graph_chunks, dense_modulation=None, dense_sizes=None, return_grad=False):
        del graph_chunks
        if g.dim() == 2:
            g_batch = g.unsqueeze(0)
            mod_batch = self._format_dense_modulation(dense_modulation) if dense_modulation is not None else None
            sizes = dense_sizes if dense_sizes is not None else torch.tensor([g.size(0)], device=g.device, dtype=torch.long)
            out = self._dense_energy_batched(g_batch, mod_batch, sizes, return_grad=return_grad)
            if return_grad:
                e, grad = out
                return e, grad.squeeze(0)
            return out

        if g.dim() == 3:
            mod_batch = self._format_dense_modulation(dense_modulation) if dense_modulation is not None else None
            return self._dense_energy_batched(g, mod_batch, dense_sizes, return_grad=return_grad)

        if g.dim() == 4:
            num_trials, bsz, max_n, d = g.shape
            g_flat = g.view(-1, max_n, d)
            mod_base = self._format_dense_modulation(dense_modulation) if dense_modulation is not None else None
            mod_flat = mod_base.repeat(num_trials, 1, 1, 1) if mod_base is not None else None
            sizes_flat = dense_sizes.repeat(num_trials) if dense_sizes is not None else None
            out = self._dense_energy_batched(g_flat, mod_flat, sizes_flat, return_grad=return_grad)
            if return_grad:
                e_flat, grad_flat = out
                return e_flat.view(num_trials), grad_flat.view(num_trials, bsz, max_n, d)
            return out.view(num_trials)

        raise ValueError(f"Unsupported dense input rank: {g.dim()}")

    def _sparse_energy(self, g, c_aug, u_aug, return_grad=False):
        q, k = self._project_qk(g)
        num_tokens = g.size(-2)
        num_heads = self.num_heads
        head_dim = self.head_dim
        is_batched = q.dim() == 4
        if is_batched:
            num_trials = q.size(0)
            q_sel, k_sel = q[:, c_aug], k[:, u_aug]
            logits = (q_sel * k_sel).sum(dim=-1).permute(0, 2, 1).reshape(-1, q_sel.size(1))
            betas = self.betas.repeat(num_trials)
            vals = betas[:, None] * logits
            vals_reshaped = vals.view(num_trials, num_heads, -1).transpose(1, 2)
            vals_mixed = torch.matmul(vals_reshaped, self.Hw).transpose(1, 2).reshape(-1, logits.size(1))
            lse = segment_logsumexp(vals_mixed, c_aug, num_tokens)
            e = -(lse.view(num_trials, num_heads, num_tokens).sum(dim=-1) / self.betas.unsqueeze(0)).sum(dim=-1)
            if not return_grad:
                return e
        else:
            q_sel, k_sel = q[c_aug], k[u_aug]
            logits = (q_sel * k_sel).sum(dim=-1).transpose(0, 1)
            vals = self.betas[:, None] * logits
            vals_mixed = torch.matmul(vals.transpose(0, 1), self.Hw).transpose(0, 1)
            lse = segment_logsumexp(vals_mixed, c_aug, num_tokens)
            e = (-(lse.sum(dim=-1) / self.betas).sum())
            if not return_grad:
                return e
        probs = torch.exp(vals_mixed - lse[:, c_aug])
        if is_batched:
            grad_vals_mixed = -(probs / betas[:, None])
            grad_vals = torch.matmul(grad_vals_mixed.view(num_trials, num_heads, -1).transpose(1, 2), self.Hw.t()).transpose(1, 2).reshape(-1, logits.size(1))
            grad_logits = (grad_vals * betas[:, None]).view(num_trials, num_heads, -1).transpose(1, 2)
            grad_q_sel = grad_logits.unsqueeze(-1) * k_sel
            grad_k_sel = grad_logits.unsqueeze(-1) * q_sel
            if pyg_scatter is not None:
                offsets = torch.arange(num_trials, device=g.device).view(-1, 1) * num_tokens
                c_aug_flat = (c_aug.unsqueeze(0) + offsets).view(-1)
                u_aug_flat = (u_aug.unsqueeze(0) + offsets).view(-1)
                grad_q = pyg_scatter(grad_q_sel.reshape(-1, num_heads, head_dim), c_aug_flat, dim=0, dim_size=num_trials * num_tokens, reduce="sum").view(num_trials, num_tokens, num_heads, head_dim)
                grad_k = pyg_scatter(grad_k_sel.reshape(-1, num_heads, head_dim), u_aug_flat, dim=0, dim_size=num_trials * num_tokens, reduce="sum").view(num_trials, num_tokens, num_heads, head_dim)
            else:
                idx_q = c_aug.view(1, -1, 1, 1).expand(num_trials, -1, num_heads, head_dim)
                grad_q = torch.zeros_like(q).scatter_add_(1, idx_q, grad_q_sel)
                idx_k = u_aug.view(1, -1, 1, 1).expand(num_trials, -1, num_heads, head_dim)
                grad_k = torch.zeros_like(k).scatter_add_(1, idx_k, grad_k_sel)
        else:
            grad_vals_mixed = -(probs / self.betas[:, None])
            grad_vals = torch.matmul(grad_vals_mixed.transpose(0, 1), self.Hw.t()).transpose(0, 1)
            grad_logits = grad_vals * self.betas[:, None]
            grad_q_sel = grad_logits.transpose(0, 1).unsqueeze(-1) * k_sel
            grad_k_sel = grad_logits.transpose(0, 1).unsqueeze(-1) * q_sel
            if pyg_scatter is not None:
                grad_q = pyg_scatter(grad_q_sel, c_aug, dim=0, dim_size=num_tokens, reduce="sum")
                grad_k = pyg_scatter(grad_k_sel, u_aug, dim=0, dim_size=num_tokens, reduce="sum")
            else:
                grad_q = torch.zeros_like(q).scatter_add_(0, c_aug.view(-1, 1, 1).expand(-1, num_heads, head_dim), grad_q_sel)
                grad_k = torch.zeros_like(k).scatter_add_(0, u_aug.view(-1, 1, 1).expand(-1, num_heads, head_dim), grad_k_sel)
        wq_t, wk_t = self.Wq.transpose(1, 2), self.Wk.transpose(1, 2)
        if is_batched:
            grad_g = torch.einsum("bnhz,hdz->bnd", grad_q, wq_t) + torch.einsum("bnhz,hdz->bnd", grad_k, wk_t)
        else:
            grad_g = torch.einsum("nhz,hdz->nd", grad_q, wq_t) + torch.einsum("nhz,hdz->nd", grad_k, wk_t)
        return e, grad_g

    def energy(self, g, c_aug, u_aug, graph_chunks, mask_mode="sparse", dense_modulation=None, dense_sizes=None, static_projections=None):
        del static_projections
        if mask_mode == "official_dense":
            return self._dense_energy(g, graph_chunks, dense_modulation=dense_modulation, dense_sizes=dense_sizes)
        return self._sparse_energy(g, c_aug, u_aug)

    def energy_and_grad(self, g, c_aug, u_aug, graph_chunks, mask_mode="sparse", dense_modulation=None, dense_sizes=None, static_projections=None):
        del static_projections
        if mask_mode == "official_dense":
            return self._dense_energy(g, graph_chunks, dense_modulation=dense_modulation, dense_sizes=dense_sizes, return_grad=True)
        return self._sparse_energy(g, c_aug, u_aug, return_grad=True)


class ETHopfieldCore(nn.Module):
    """Official ET CHN-ReLU energy term."""

    def __init__(self, d, num_memories):
        super().__init__()
        self.d = int(d)
        self.num_memories = int(max(0, num_memories))
        if self.num_memories > 0:
            self.Xi = nn.Parameter(torch.empty(self.d, self.num_memories))
            self.reset_parameters()
        else:
            self.register_parameter("Xi", None)

    def reset_parameters(self):
        if self.Xi is not None:
            nn.init.normal_(self.Xi, mean=0.0, std=0.02)

    def energy(self, g):
        if self.Xi is None:
            return g.new_zeros(()) if g.dim() == 2 else g.new_zeros(g.size(0))
        hid = g @ self.Xi
        if g.dim() == 3:
            return -0.5 * torch.relu(hid).pow(2).sum(dim=(1, 2))
        if g.dim() == 4:
            return -0.5 * torch.relu(hid).pow(2).sum(dim=(2, 3))
        return -0.5 * torch.relu(hid).pow(2).sum()

    def energy_and_grad(self, g):
        if self.Xi is None:
            return (g.new_zeros(()) if g.dim() == 2 else g.new_zeros(g.size(0))), torch.zeros_like(g)
        hid = g @ self.Xi
        relu_hid = torch.relu(hid)
        if g.dim() == 3:
            e = -0.5 * relu_hid.pow(2).sum(dim=(1, 2))
        elif g.dim() == 4:
            e = -0.5 * relu_hid.pow(2).sum(dim=(2, 3))
        else:
            e = -0.5 * relu_hid.pow(2).sum()
        grad_g = -(relu_hid @ self.Xi.t())
        return e, grad_g

    def entropy(self, g):
        if self.Xi is None:
            return 0.0
        probs = torch.softmax(g @ self.Xi, dim=-1)
        ent = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1).mean()
        return float(ent.detach().item())


class ETCoreBlock(nn.Module):
    """Reusable ET block: attention energy + CHN/Hopfield energy."""

    def __init__(self, d, num_heads, head_dim, num_memories, beta_init=None):
        super().__init__()
        self.attn = ETAttentionCore(d=d, num_heads=num_heads, head_dim=head_dim, beta_init=beta_init)
        self.hn = ETHopfieldCore(d=d, num_memories=num_memories)

    def energy(self, g, c_aug, u_aug, graph_chunks, mask_mode, dense_modulation=None, dense_sizes=None, static_projections=None):
        del static_projections
        return self.attn.energy(g, c_aug, u_aug, graph_chunks, mask_mode=mask_mode, dense_modulation=dense_modulation, dense_sizes=dense_sizes) + self.hn.energy(g)

    def energy_and_grad(self, g, c_aug, u_aug, graph_chunks, mask_mode, dense_modulation=None, dense_sizes=None, static_projections=None):
        del static_projections
        e_attn, grad_attn = self.attn.energy_and_grad(g, c_aug, u_aug, graph_chunks, mask_mode=mask_mode, dense_modulation=dense_modulation, dense_sizes=dense_sizes)
        e_hn, grad_hn = self.hn.energy_and_grad(g)
        return e_attn + e_hn, grad_attn + grad_hn


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
        self.share_block_weights = bool(share_block_weights)

        self.mask_mode = str(mask_mode)
        if et_official_mode and self.mask_mode == "sparse":
            self.mask_mode = "official_dense"
        if self.mask_mode not in {"sparse", "official_dense"}:
            raise ValueError(f"Unsupported mask_mode: {self.mask_mode}")

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
                kernel_size=int(dense_kernel_size),
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

        X_aug = torch.empty((num_nodes + num_graphs, self.d), dtype=dtype, device=device)
        X_aug[:num_nodes] = z_nodes
        X_aug[num_nodes:] = self.cls_token.expand(num_graphs, -1)

        cls_indices = torch.arange(num_nodes, num_nodes + num_graphs, device=device)
        node_batch = batch_data.batch
        node_cls_indices = cls_indices[node_batch]

        num_orig_edges = c2.numel()
        num_cls_edges = 2 * num_nodes
        num_self_loops = num_nodes + num_graphs if self.allow_self else num_graphs
        total_aug_edges = num_orig_edges + num_cls_edges + num_self_loops

        c_aug = torch.empty(total_aug_edges, dtype=torch.long, device=device)
        u_aug = torch.empty(total_aug_edges, dtype=torch.long, device=device)

        curr = 0
        c_aug[curr : curr + num_orig_edges] = c2
        u_aug[curr : curr + num_orig_edges] = u2
        curr += num_orig_edges

        node_indices = torch.arange(num_nodes, device=device)
        c_aug[curr : curr + num_nodes] = node_indices
        u_aug[curr : curr + num_nodes] = node_cls_indices
        curr += num_nodes
        c_aug[curr : curr + num_nodes] = node_cls_indices
        u_aug[curr : curr + num_nodes] = node_indices
        curr += num_nodes

        if self.allow_self:
            c_aug[curr : curr + num_nodes] = node_indices
            u_aug[curr : curr + num_nodes] = node_indices
            curr += num_nodes

        c_aug[curr : curr + num_graphs] = cls_indices
        u_aug[curr : curr + num_graphs] = cls_indices
        curr += num_graphs

        if hasattr(batch_data, "pe") and batch_data.pe is not None:
            pe = batch_data.pe
            pe_aug = torch.zeros((num_nodes + num_graphs, pe.size(-1)), dtype=dtype, device=device)
            pe_aug[:num_nodes] = pe.to(dtype=dtype)
            if self.pe_proj is not None:
                X_aug.add_(self.pe_proj(pe_aug))

        if self.rwse_proj is not None and hasattr(batch_data, "rwse") and batch_data.rwse is not None:
            rwse = batch_data.rwse
            rwse_aug = torch.zeros((num_nodes + num_graphs, rwse.size(-1)), dtype=dtype, device=device)
            rwse_aug[:num_nodes] = rwse.to(dtype=dtype)
            X_aug.add_(self.rwse_proj(rwse_aug))

        return X_aug, c_aug, u_aug, cls_indices, node_indices, None

    def _to_dense(self, x, cls_indices, node_indices, ptr, node_batch, num_graphs, max_n_aug):
        num_original_nodes = ptr[-1].item()
        x_dense = x.new_zeros((num_graphs, max_n_aug, self.d))
        node_local = torch.arange(num_original_nodes, device=x.device) - ptr[node_batch]
        x_dense[node_batch, node_local + 1] = x[node_indices]
        x_dense[torch.arange(num_graphs, device=x.device), 0] = x[cls_indices]
        return x_dense

    def _to_sparse(self, x_dense, cls_indices, node_indices, ptr, node_batch, num_graphs, x_like):
        num_original_nodes = ptr[-1].item()
        node_local = torch.arange(num_original_nodes, device=x_dense.device) - ptr[node_batch]
        x_sparse = torch.zeros_like(x_like)
        x_sparse[node_indices] = x_dense[node_batch, node_local + 1]
        x_sparse[cls_indices] = x_dense[torch.arange(num_graphs, device=x_dense.device), 0]
        return x_sparse

    def _prepare_dense_cache(self, x, cls_indices, node_indices, batch_data):
        if self.mask_mode != "official_dense" or self.mask_modulator is None:
            return None

        ptr = batch_data.ptr
        num_graphs = ptr.numel() - 1

        max_n_orig = torch.diff(ptr).max().item()
        max_n_aug = int(max_n_orig + 1)
        node_batch = batch_data.batch

        x_dense = self._to_dense(x, cls_indices, node_indices, ptr, node_batch, num_graphs, max_n_aug)

        edge_attr = getattr(batch_data, "edge_attr", None)
        c2 = batch_data.c_2
        u2 = batch_data.u_2

        if edge_attr is not None:
            feat_dim = edge_attr.size(-1)
            e_dense = x.new_zeros((num_graphs, max_n_aug, max_n_aug, feat_dim))
            edge_batch = node_batch[c2]
            src_local = c2 - ptr[edge_batch] + 1
            dst_local = u2 - ptr[edge_batch] + 1
            e_dense[edge_batch, src_local, dst_local] = edge_attr.to(dtype=x.dtype)
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
        return dense_modulation, sizes, ptr, node_batch, num_graphs, max_n_aug

    def _solve_dynamics(self, x_aug, c_aug, u_aug, cls_indices, node_indices, batch_data):
        energy_trace = []
        step = self.eta
        x = x_aug

        mask_mode = self.mask_mode
        dense_cache = self._prepare_dense_cache(x, cls_indices, node_indices, batch_data)
        if dense_cache is None:
            dense_modulation = None
            dense_sizes = None
        else:
            dense_modulation, dense_sizes, ptr, node_batch, num_graphs, max_n_aug = dense_cache

        for block_idx in range(self.num_blocks):
            norm = self.norm_blocks[block_idx]
            core = self.core_blocks[block_idx]
            for _ in range(self.num_steps):
                g = norm(x)
                if mask_mode == "official_dense":
                    g_dense = self._to_dense(g, cls_indices, node_indices, ptr, node_batch, num_graphs, max_n_aug)
                    e, grad_g_dense = core.energy_and_grad(
                        g_dense,
                        None,
                        None,
                        None,
                        mask_mode=mask_mode,
                        dense_modulation=dense_modulation,
                        dense_sizes=dense_sizes,
                    )
                    grad_g = self._to_sparse(grad_g_dense, cls_indices, node_indices, ptr, node_batch, num_graphs, g)
                else:
                    e, grad_g = core.energy_and_grad(
                        g,
                        c_aug,
                        u_aug,
                        None,
                        mask_mode=mask_mode,
                        dense_modulation=dense_modulation,
                        dense_sizes=dense_sizes,
                    )

                grad_x = norm.backward(x, grad_g)

                noise = None
                if self.training and self.noise_std > 0:
                    noise = torch.randn_like(grad_x) * self.noise_std
                if self.grad_clip_norm is not None:
                    gnorm = grad_x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    grad_x = grad_x * (self.grad_clip_norm / gnorm).clamp(max=1.0)

                x = x - step * grad_x
                if noise is not None:
                    x = x + torch.sqrt(step.clamp_min(1e-8)) * noise
                if self.state_clip_norm is not None:
                    snorm = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    x = x * (self.state_clip_norm / snorm).clamp(max=1.0)

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
    ):
        if self.training:
            raise ValueError("Armijo inference_mode is evaluation-only; call model.eval() first.")

        eta0 = float(self.eta.detach().item()) if armijo_eta0 is None else float(armijo_eta0)
        eta0 = max(eta0, 1e-8)

        energy_trace = []
        solver_stats = self._init_solver_stats('armijo')
        x = x_aug

        mask_mode = self.mask_mode
        dense_cache = self._prepare_dense_cache(x, cls_indices, node_indices, batch_data)
        if dense_cache is None:
            dense_modulation = None
            dense_sizes = None
        else:
            dense_modulation, dense_sizes, ptr, node_batch, num_graphs, max_n_aug = dense_cache

        etas_all = eta0 * (armijo_gamma ** torch.arange(armijo_max_backtracks, device=x.device, dtype=x.dtype))

        for block_idx in range(self.num_blocks):
            norm = self.norm_blocks[block_idx]
            core = self.core_blocks[block_idx]

            for _ in range(self.num_steps):
                g = norm(x)
                if mask_mode == "official_dense":
                    g_dense = self._to_dense(g, cls_indices, node_indices, ptr, node_batch, num_graphs, max_n_aug)
                    e_t, grad_g_dense = core.energy_and_grad(
                        g_dense,
                        None,
                        None,
                        None,
                        mask_mode,
                        dense_modulation=dense_modulation,
                        dense_sizes=dense_sizes,
                    )
                    grad_g = self._to_sparse(grad_g_dense, cls_indices, node_indices, ptr, node_batch, num_graphs, g)
                else:
                    e_t, grad_g = core.energy_and_grad(
                        g,
                        c_aug,
                        u_aug,
                        None,
                        mask_mode,
                        dense_modulation=dense_modulation,
                        dense_sizes=dense_sizes,
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

                accepted_eta = 0.0
                accepted_backtracks = armijo_max_backtracks
                accepted_energy = float(e_t.detach().item())
                accepted_state = x.detach()
                found = False

                for start_bt in range(0, armijo_max_backtracks, chunk_size):
                    end_bt = min(start_bt + chunk_size, armijo_max_backtracks)
                    etas = etas_all[start_bt:end_bt]

                    e_tries = []
                    x_tries_list = []
                    for eta_try in etas:
                        x_try = x.detach() - eta_try * grad_x.detach()
                        g_try = norm(x_try)
                        if mask_mode == "official_dense":
                            g_try_dense = self._to_dense(g_try, cls_indices, node_indices, ptr, node_batch, num_graphs, max_n_aug)
                            e_try = core.energy(
                                g_try_dense,
                                None,
                                None,
                                None,
                                mask_mode,
                                dense_modulation=dense_modulation,
                                dense_sizes=dense_sizes,
                            )
                        else:
                            e_try = core.energy(
                                g_try,
                                c_aug,
                                u_aug,
                                None,
                                mask_mode,
                                dense_modulation=dense_modulation,
                                dense_sizes=dense_sizes,
                            )
                        e_tries.append(e_try)
                        x_tries_list.append(x_try)

                    e_tries = torch.stack(e_tries, dim=0)
                    x_tries = torch.stack(x_tries_list, dim=0)

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
        x_aug, c_aug, u_aug, cls_pos, node_pos, _ = self._build_augmented_graph(batch_data, z_nodes)

        if inference_mode == 'fixed':
            x_final, energy_trace = self._solve_dynamics(x_aug, c_aug, u_aug, cls_pos, node_pos, batch_data)
            solver_stats = {'mode': 'fixed', 'energy_trace': energy_trace}
        elif inference_mode == 'armijo':
            x_final, energy_trace, solver_stats = self._run_armijo_solver(x_aug, c_aug, u_aug, cls_pos, node_pos, batch_data)
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
