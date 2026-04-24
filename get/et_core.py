import math

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from .energy import segment_logsumexp

try:
    from torch_scatter import scatter as pyg_scatter
except (ImportError, OSError):
    pyg_scatter = None


class EnergyLayerNorm(nn.Module):
    """ET-style LayerNorm with scalar gamma and optional vector bias."""

    def __init__(self, dim, use_bias=True, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self.use_bias = bool(use_bias)
        self.gamma = nn.Parameter(torch.ones(()))
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        xmeaned = x - x.mean(dim=-1, keepdim=True)
        v = self.gamma * xmeaned / torch.sqrt((xmeaned.pow(2)).mean(dim=-1, keepdim=True) + self.eps)
        if self.bias is not None:
            return v + self.bias
        return v

    def backward(self, x, grad_v):
        """Analytical pullback: dE/dx = (dE/dv) * (dv/dx)"""
        # x: [..., D], grad_v: [..., D]
        xmeaned = x - x.mean(dim=-1, keepdim=True)
        var = (xmeaned.pow(2)).mean(dim=-1, keepdim=True)
        std_inv = 1.0 / torch.sqrt(var + self.eps)
        std_inv_sq = std_inv**2
        
        term1 = grad_v
        term2 = grad_v.mean(dim=-1, keepdim=True)
        term3 = xmeaned * (grad_v * xmeaned).mean(dim=-1, keepdim=True) * std_inv_sq
        
        grad_x = self.gamma * std_inv * (term1 - term2 - term3)
        return grad_x


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
            dense_modulation = dense_modulation[batch_idx]
        if dense_modulation.dim() == 4:
            if dense_modulation.size(1) == self.num_heads:
                return dense_modulation.permute(0, 2, 3, 1).contiguous()
            if dense_modulation.size(-1) == self.num_heads:
                return dense_modulation.contiguous()
            raise ValueError("Expected head dimension at axis 1 or -1 for dense modulation.")
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
        # g: [N, D] or [B, N, D] or [TrialBatch, B, N, D]
        wq_t = self.Wq.transpose(1, 2)
        wk_t = self.Wk.transpose(1, 2)
        if g.dim() == 2:
            q = torch.matmul(g.unsqueeze(0), wq_t).permute(1, 0, 2).contiguous()  # [N, H, Z]
            k = torch.matmul(g.unsqueeze(0), wk_t).permute(1, 0, 2).contiguous()  # [N, H, Z]
        elif g.dim() == 3:
            q = torch.einsum("bnd,hdz->bnhz", g, wq_t).contiguous()  # [B, N, H, Z]
            k = torch.einsum("bnd,hdz->bnhz", g, wk_t).contiguous()  # [B, N, H, Z]
        else:
            q = torch.einsum("tbnd,hdz->tbnhz", g, wq_t).contiguous()
            k = torch.einsum("tbnd,hdz->tbnhz", g, wk_t).contiguous()
        return q, k

    def _dense_energy_batched(self, g, dense_modulation, dense_sizes=None, return_grad=False):
        q_all, k_all = self._project_qk(g)
        # q_all: [B, N, H, Z], k_all: [B, N, H, Z]
        
        q_perm = q_all.permute(0, 2, 1, 3).contiguous()  # [B, H, N, Z]
        k_perm = k_all.permute(0, 2, 1, 3).contiguous()  # [B, H, N, Z]
        logits = torch.einsum("bhnz,bhmz->bhnm", q_perm, k_perm)
        logits = self.betas[None, :, None, None] * logits
        logits = logits.permute(0, 2, 3, 1).contiguous()  # [B, N, N, H]
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
            
        probs = torch.softmax(logits, dim=2) # [B, N, N, H]
        grad_logits = - (probs / self.betas[None, None, None, :])
        grad_qk_beta = grad_logits @ self.Hw.t() # [B, N, N, H]
        grad_qk = grad_qk_beta * self.betas[None, None, None, :]
        
        grad_q_all = torch.einsum("bnmh,bmhz->bnhz", grad_qk, k_all)
        grad_k_all = torch.einsum("bmnh,bmhz->bnhz", grad_qk, q_all)
        
        wq_t = self.Wq.transpose(1, 2)
        wk_t = self.Wk.transpose(1, 2)
        grad_g = torch.einsum("bnhz,hdz->bnd", grad_q_all, wq_t) + torch.einsum("bnhz,hdz->bnd", grad_k_all, wk_t)
        return e, grad_g

    def _dense_energy(self, g, graph_chunks, dense_modulation=None, dense_sizes=None, return_grad=False):
        if (isinstance(dense_modulation, torch.Tensor) and dense_modulation.dim() == 4):
            is_trial_batched = g.dim() == 4
            if is_trial_batched:
                num_trials, bsz, max_n, d = g.shape
                g_flat = g.view(-1, max_n, d)
                mod_flat = dense_modulation.repeat(num_trials, 1, 1, 1)
                sizes_flat = dense_sizes.repeat(num_trials) if dense_sizes is not None else None
                res = self._dense_energy_batched(g_flat, mod_flat, sizes_flat, return_grad=return_grad)
                if return_grad:
                    e_flat, grad_g_flat = res
                    return e_flat.view(num_trials), grad_g_flat.view(num_trials, bsz, max_n, d)
                return res.view(num_trials)

            if g.dim() == 3:
                return self._dense_energy_batched(g, dense_modulation, dense_sizes, return_grad=return_grad)
            
            res = self._dense_energy_batched(g.unsqueeze(0), dense_modulation, dense_sizes, return_grad=return_grad)
            if return_grad:
                e, grad_g = res
                return e, grad_g.squeeze(0)
            return res
        
        if return_grad:
            e = self._dense_energy(g, graph_chunks, dense_modulation, dense_sizes, return_grad=False)
            grad_g = torch.autograd.grad(e, g)[0]
            return e, grad_g

        q_all, k_all = self._project_qk(g)
        e = g.new_zeros(())
        finfo_min = torch.finfo(g.dtype).min
        if graph_chunks is None:
            raise ValueError("graph_chunks must be provided if dense_modulation is not a batched 4D tensor.")

        for idx, chunk in enumerate(graph_chunks):
            start, end = chunk["start"], chunk["start"] + chunk["size"]
            mask = chunk["adj"].to(dtype=torch.bool)
            q, k = q_all[start:end], k_all[start:end]
            logits = torch.bmm(q.permute(1, 0, 2), k.permute(1, 2, 0))
            logits = self.betas[:, None, None] * logits
            logits = logits.permute(1, 2, 0).contiguous()  # [Q, K, H]
            logits = torch.matmul(logits, self.Hw)
            if dense_modulation is not None:
                mod = self._format_dense_modulation(dense_modulation, idx)
                logits = logits * mod[: chunk["size"], : chunk["size"]]
            logits = logits.masked_fill(~mask.unsqueeze(-1), finfo_min)
            lse = torch.logsumexp(logits, dim=1)  # [Q, H]
            e = e + (-(lse.sum(dim=0) / self.betas).sum())
        return e

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
            lse, exp_x, denom = segment_logsumexp(vals_mixed, c_aug, num_tokens, return_intermediates=True)
            e = -(lse.view(num_trials, num_heads, num_tokens).sum(dim=-1) / self.betas.unsqueeze(0)).sum(dim=-1)
            if not return_grad:
                return e
        else:
            q_sel, k_sel = q[c_aug], k[u_aug]
            logits = (q_sel * k_sel).sum(dim=-1).transpose(0, 1)
            vals = self.betas[:, None] * logits
            vals_mixed = torch.matmul(vals.transpose(0, 1), self.Hw).transpose(0, 1)
            lse, exp_x, denom = segment_logsumexp(vals_mixed, c_aug, num_tokens, return_intermediates=True)
            e = (-(lse.sum(dim=-1) / self.betas).sum())
            if not return_grad:
                return e
            
        probs = exp_x / denom[:, c_aug]
        if is_batched:
            grad_vals_mixed = - (probs / betas[:, None])
            grad_vals = torch.matmul(grad_vals_mixed.view(num_trials, num_heads, -1).transpose(1, 2), self.Hw.t()).transpose(1, 2).reshape(-1, logits.size(1))
            grad_logits = (grad_vals * betas[:, None]).view(num_trials, num_heads, -1).transpose(1, 2)
            grad_q_sel, grad_k_sel = grad_logits.unsqueeze(-1) * k_sel, grad_logits.unsqueeze(-1) * q_sel
            if pyg_scatter is not None:
                offsets = torch.arange(num_trials, device=g.device).view(-1, 1) * num_tokens
                c_aug_flat, u_aug_flat = (c_aug.unsqueeze(0) + offsets).view(-1), (u_aug.unsqueeze(0) + offsets).view(-1)
                grad_q = pyg_scatter(grad_q_sel.reshape(-1, num_heads, head_dim), c_aug_flat, dim=0, dim_size=num_trials * num_tokens, reduce="sum").view(num_trials, num_tokens, num_heads, head_dim)
                grad_k = pyg_scatter(grad_k_sel.reshape(-1, num_heads, head_dim), u_aug_flat, dim=0, dim_size=num_trials * num_tokens, reduce="sum").view(num_trials, num_tokens, num_heads, head_dim)
            else:
                idx_q = c_aug.view(1, -1, 1, 1).expand(num_trials, -1, num_heads, head_dim)
                grad_q = torch.zeros_like(q).scatter_add_(1, idx_q, grad_q_sel)
                idx_k = u_aug.view(1, -1, 1, 1).expand(num_trials, -1, num_heads, head_dim)
                grad_k = torch.zeros_like(k).scatter_add_(1, idx_k, grad_k_sel)
        else:
            grad_vals_mixed = - (probs / self.betas[:, None])
            grad_vals = torch.matmul(grad_vals_mixed.transpose(0, 1), self.Hw.t()).transpose(0, 1)
            grad_logits = grad_vals * self.betas[:, None]
            grad_q_sel, grad_k_sel = grad_logits.transpose(0, 1).unsqueeze(-1) * k_sel, grad_logits.transpose(0, 1).unsqueeze(-1) * q_sel
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

    def energy(self, g, c_aug, u_aug, graph_chunks, mask_mode="sparse", dense_modulation=None, dense_sizes=None):
        if mask_mode == "official_dense":
            return self._dense_energy(g, graph_chunks, dense_modulation=dense_modulation, dense_sizes=dense_sizes)
        return self._sparse_energy(g, c_aug, u_aug)

    def energy_and_grad(self, g, c_aug, u_aug, graph_chunks, mask_mode="sparse", dense_modulation=None, dense_sizes=None):
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
        grad_g = - (relu_hid @ self.Xi.t())
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

    def energy(self, g, c_aug, u_aug, graph_chunks, mask_mode, dense_modulation=None, dense_sizes=None):
        return self.attn.energy(g, c_aug, u_aug, graph_chunks, mask_mode=mask_mode, dense_modulation=dense_modulation, dense_sizes=dense_sizes) + self.hn.energy(g)

    def energy_and_grad(self, g, c_aug, u_aug, graph_chunks, mask_mode, dense_modulation=None, dense_sizes=None):
        e_attn, grad_attn = self.attn.energy_and_grad(g, c_aug, u_aug, graph_chunks, mask_mode=mask_mode, dense_modulation=dense_modulation, dense_sizes=dense_sizes)
        e_hn, grad_hn = self.hn.energy_and_grad(g)
        return e_attn + e_hn, grad_attn + grad_hn


class ETGraphMaskModulator(nn.Module):
    """
    ET graph appendix-style edge-conditioned mask:
    A_hat = Conv2D((X @ X^T)[..., None]) ⊙ A'
    """

    def __init__(self, d, num_heads, edge_feat_dim=None, kernel_size=3):
        super().__init__()
        self.d = int(d)
        self.num_heads = int(num_heads)
        self.edge_feat_dim = None if edge_feat_dim is None else int(edge_feat_dim)
        self.mask_conv = nn.Conv2d(in_channels=1, out_channels=self.num_heads, kernel_size=int(kernel_size), stride=1, padding=int(kernel_size)//2, bias=True)
        if self.edge_feat_dim is None:
            self.edge_proj = nn.LazyLinear(self.num_heads, bias=False)
        else:
            self.edge_proj = nn.Linear(self.edge_feat_dim, self.num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mask_conv.weight, a=5**0.5)
        if self.mask_conv.bias is not None:
            nn.init.zeros_(self.mask_conv.bias)
        if hasattr(self.edge_proj, "weight") and not isinstance(self.edge_proj.weight, UninitializedParameter):
            nn.init.xavier_uniform_(self.edge_proj.weight)

    def _prepare_edge_features(self, edge_features, x_dtype):
        if edge_features.dim() == 2:
            edge_features = edge_features.unsqueeze(-1)
        ef = edge_features.to(dtype=x_dtype)
        in_features = getattr(self.edge_proj, "in_features", None)
        if in_features is None or ef.size(-1) == in_features:
            return ef
        if in_features == 1:
            return ef.mean(dim=-1, keepdim=True)
        if ef.size(-1) == 1:
            return ef.expand(*ef.shape[:-1], in_features)
        if ef.size(-1) > in_features:
            return ef[..., :in_features]
        return torch.cat([ef, torch.zeros(*ef.shape[:-1], in_features - ef.size(-1), dtype=ef.dtype, device=ef.device)], dim=-1)

    def dense_modulation(self, x_local, edge_features):
        inner = (x_local @ x_local.transpose(0, 1)).unsqueeze(0).unsqueeze(0)
        conv_out = self.mask_conv(inner).squeeze(0)
        edge_heads = self.edge_proj(self._prepare_edge_features(edge_features, x_local.dtype)).permute(2, 0, 1)
        return conv_out * edge_heads

    def dense_modulation_batched(self, x_batch, edge_features_batch):
        inner = torch.matmul(x_batch, x_batch.transpose(-1, -2)).unsqueeze(1)
        conv_out = self.mask_conv(inner)
        edge_heads = self.edge_proj(self._prepare_edge_features(edge_features_batch, x_batch.dtype)).permute(0, 3, 1, 2)
        return conv_out * edge_heads
