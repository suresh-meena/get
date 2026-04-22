import math

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from .fused_ops import segment_reduce_1d


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


class ETAttentionCore(nn.Module):
    """Official ET attention energy with optional sparse graph masking."""

    def __init__(self, d, num_heads=1, head_dim=None, beta_init=None):
        super().__init__()
        self.d = int(d)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or d)
        self.Wq = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d))
        self.Wk = nn.Parameter(torch.empty(self.num_heads, self.head_dim, self.d))
        default_beta = 1.0 / math.sqrt(float(self.head_dim))
        beta = default_beta if beta_init is None else float(beta_init)
        self.betas = nn.Parameter(torch.full((self.num_heads,), beta))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.Wq, mean=0.0, std=0.002)
        nn.init.normal_(self.Wk, mean=0.0, std=0.002)

    def _project_qk(self, g):
        # g: [N, D]
        # W*: [H, Z, D] -> [H, D, Z] for batched matmul
        wq_t = self.Wq.transpose(1, 2)
        wk_t = self.Wk.transpose(1, 2)
        q = torch.matmul(g.unsqueeze(0), wq_t).permute(1, 0, 2).contiguous()  # [N, H, Z]
        k = torch.matmul(g.unsqueeze(0), wk_t).permute(1, 0, 2).contiguous()  # [N, H, Z]
        return q, k

    def _dense_energy(self, g, graph_chunks, dense_modulation=None):
        q_all, k_all = self._project_qk(g)
        e = g.new_zeros(())
        finfo_min = torch.finfo(g.dtype).min

        for idx, chunk in enumerate(graph_chunks):
            start = chunk["start"]
            end = start + chunk["size"]
            mask = chunk["adj"].to(dtype=torch.bool)
            q = q_all[start:end]
            k = k_all[start:end]
            # [Q,H,Z] -> [H,Q,Z], [K,H,Z] -> [H,Z,K]
            logits = torch.bmm(q.permute(1, 0, 2), k.permute(1, 2, 0))
            if dense_modulation is not None:
                logits = logits + dense_modulation[idx]
            logits = self.betas[:, None, None] * logits
            logits = logits.masked_fill(~mask.unsqueeze(0), finfo_min)
            lse = torch.logsumexp(logits, dim=-1)  # [H, Q]
            e = e + (-(lse.sum(dim=-1) / self.betas).sum())
        return e

    def _sparse_energy(self, g, c_aug, u_aug):
        q, k = self._project_qk(g)
        q_sel = q[c_aug]  # [E, H, Z]
        k_sel = k[u_aug]  # [E, H, Z]
        logits = (q_sel * k_sel).sum(dim=-1).transpose(0, 1)  # [H, E]
        num_tokens = int(g.size(0))
        e = g.new_zeros(())
        for h in range(self.num_heads):
            vals = self.betas[h] * logits[h]
            out, _ = segment_reduce_1d(vals, c_aug, num_tokens, reduce="max")
            max_per = out[c_aug]
            exp_shift = torch.exp(vals - max_per)
            sumexp, _ = segment_reduce_1d(exp_shift, c_aug, num_tokens, reduce="sum")
            lse = out + torch.log(sumexp.clamp_min(1e-12))
            e = e + (-(lse / self.betas[h]).sum())
        return e

    def energy(self, g, c_aug, u_aug, graph_chunks, mask_mode="sparse", dense_modulation=None):
        if mask_mode == "official_dense":
            return self._dense_energy(g, graph_chunks, dense_modulation=dense_modulation)
        return self._sparse_energy(g, c_aug, u_aug)


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
            return g.new_zeros(())
        hid = g @ self.Xi
        return -0.5 * torch.relu(hid).pow(2).sum()

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

    def energy(self, g, c_aug, u_aug, graph_chunks, mask_mode, dense_modulation=None):
        return self.attn.energy(
            g,
            c_aug,
            u_aug,
            graph_chunks,
            mask_mode=mask_mode,
            dense_modulation=dense_modulation,
        ) + self.hn.energy(g)


class ETGraphMaskModulator(nn.Module):
    """
    ET graph appendix-style edge-conditioned mask:
    A_hat = Conv2D(X ⊗ X) ⊙ A'
    """

    def __init__(self, d, num_heads, edge_feat_dim=None, kernel_size=3):
        super().__init__()
        self.d = int(d)
        self.num_heads = int(num_heads)
        self.edge_feat_dim = None if edge_feat_dim is None else int(edge_feat_dim)
        pad = int(kernel_size) // 2
        self.mask_conv = nn.Conv2d(
            in_channels=self.d,
            out_channels=self.num_heads,
            kernel_size=int(kernel_size),
            stride=1,
            padding=pad,
            bias=True,
        )
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
        if in_features is None:
            return ef
        if ef.size(-1) == in_features:
            return ef
        if in_features == 1 and ef.size(-1) > 1:
            return ef.mean(dim=-1, keepdim=True)
        if ef.size(-1) == 1 and in_features > 1:
            return ef.expand(*ef.shape[:-1], in_features)
        if ef.size(-1) > in_features:
            return ef[..., :in_features]
        pad = in_features - ef.size(-1)
        zeros = torch.zeros(*ef.shape[:-1], pad, dtype=ef.dtype, device=ef.device)
        return torch.cat([ef, zeros], dim=-1)

    def dense_modulation(self, x_local, edge_features):
        # x_local: [N, D], edge_features: [N, N] or [N, N, P]
        xt = x_local.transpose(0, 1)  # [D, N]
        outer = (xt.unsqueeze(-1) * xt.unsqueeze(-2)).unsqueeze(0)  # [1, D, N, N]
        conv_out = self.mask_conv(outer).squeeze(0)  # [H, N, N]
        ef = self._prepare_edge_features(edge_features, x_local.dtype)
        edge_heads = self.edge_proj(ef).permute(2, 0, 1)  # [H, N, N]
        return conv_out * edge_heads

    def dense_modulation_batched(self, x_batch, edge_features_batch):
        """
        Batched variant to avoid per-graph conv calls.
        x_batch: [B, N, D]
        edge_features_batch: [B, N, N] or [B, N, N, P]
        returns: [B, H, N, N]
        """
        xp = x_batch.permute(0, 2, 1).contiguous()  # [B, D, N]
        outer = xp.unsqueeze(-1) * xp.unsqueeze(-2)  # [B, D, N, N]
        conv_out = self.mask_conv(outer)  # [B, H, N, N]
        ef = self._prepare_edge_features(edge_features_batch, x_batch.dtype)
        edge_heads = self.edge_proj(ef).permute(0, 3, 1, 2)  # [B, H, N, N]
        return conv_out * edge_heads
