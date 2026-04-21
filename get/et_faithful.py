import torch
import torch.nn as nn

from .energy import positive_param, inverse_temperature
from .model import StableMLP


def _segment_logsumexp(x, segment_ids, num_segments):
    max_val = torch.full((num_segments,), float("-inf"), dtype=x.dtype, device=x.device)
    max_val.scatter_reduce_(0, segment_ids, x, reduce="amax", include_self=False)
    max_val = torch.where(max_val == float("-inf"), torch.zeros_like(max_val), max_val)
    x_centered = x - max_val[segment_ids]
    exp_x = torch.exp(x_centered)
    sum_exp = torch.zeros(num_segments, dtype=x.dtype, device=x.device)
    sum_exp.scatter_add_(0, segment_ids, exp_x)
    is_empty = (sum_exp == 0).to(dtype=x.dtype)
    return torch.log(sum_exp + is_empty) + max_val


def _laplacian_positional_encoding(adj, k, training=False):
    n = int(adj.size(0))
    if k <= 0:
        return adj.new_zeros((n, 0), dtype=torch.float32)
    if n <= 1:
        return adj.new_zeros((n, k), dtype=torch.float32)

    a = adj.to(dtype=torch.float32)
    deg = a.sum(dim=1)
    inv_sqrt_deg = torch.zeros_like(deg)
    valid = deg > 0
    inv_sqrt_deg[valid] = deg[valid].pow(-0.5)
    l = torch.eye(n, device=adj.device, dtype=torch.float32) - (inv_sqrt_deg[:, None] * a * inv_sqrt_deg[None, :])

    evals, evecs = torch.linalg.eigh(l)
    order = torch.argsort(evals)
    evecs = evecs[:, order]
    use = evecs[:, 1 : 1 + k]
    if use.size(1) < k:
        use = torch.cat([use, torch.zeros((n, k - use.size(1)), device=adj.device, dtype=use.dtype)], dim=1)

    if training and use.numel() > 0:
        signs = torch.randint(0, 2, (use.size(1),), device=adj.device, dtype=torch.long)
        signs = (2 * signs - 1).to(dtype=use.dtype)
        use = use * signs[None, :]
    return use


class ETFaithfulGraphModel(nn.Module):
    """Paper-inspired ET for graph tasks: CLS token + Laplacian PE + masked energy attention + HN memory."""

    def __init__(
        self,
        in_dim,
        d,
        num_classes,
        num_steps=8,
        num_heads=1,
        head_dim=None,
        pe_k=16,
        eta=0.05,
        eta_max=0.25,
        lambda_att=1.0,
        lambda_m=1.0,
        beta_att=1.0,
        beta_m=1.0,
        K=32,
        allow_self=False,
        noise_std=0.0,
        grad_clip_norm=1.0,
        state_clip_norm=10.0,
        beta_max=5.0,
        dropout=0.1,
        encoder_hidden_mult=2,
        readout_hidden_mult=2,
    ):
        super().__init__()
        self.d = d
        self.num_steps = int(num_steps)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or d)
        self.pe_k = int(pe_k)
        self.allow_self = bool(allow_self)
        self.noise_std = float(noise_std)
        self.grad_clip_norm = grad_clip_norm
        self.state_clip_norm = state_clip_norm
        self.beta_max = float(beta_max)

        self.node_encoder = StableMLP(
            in_dim,
            d,
            hidden_dim=max(d, encoder_hidden_mult * d),
            dropout=dropout,
            final_norm=True,
        )
        self.pe_proj = nn.Linear(self.pe_k, d)
        self.cls_token = nn.Parameter(torch.zeros(1, d))

        self.layernorm = nn.LayerNorm(d, eps=1e-5)
        self.W_Q = nn.Parameter(torch.empty(d, self.num_heads * self.head_dim))
        self.W_K = nn.Parameter(torch.empty(d, self.num_heads * self.head_dim))

        self.K = int(K)
        if self.K > 0:
            self.W_Qm = nn.Parameter(torch.empty(d, d))
            self.W_Km = nn.Parameter(torch.empty(d, d))
            self.B_mem = nn.Parameter(torch.empty(self.K, d))

        self.lambda_att = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(lambda_att)))))
        self.lambda_m = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(lambda_m)))))
        self.beta_att = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(beta_att)))))
        self.beta_m = nn.Parameter(torch.log(torch.expm1(torch.tensor(float(beta_m)))))

        eta = min(max(float(eta), 1e-4), float(eta_max) - 1e-4)
        self.eta_max = float(eta_max)
        self.eta_logit = nn.Parameter(torch.logit(torch.tensor(eta / self.eta_max)))

        self.readout = nn.Sequential(
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
            nn.Linear(d, num_classes),
        )

        self.reset_parameters()

    @property
    def eta(self):
        return self.eta_max * torch.sigmoid(self.eta_logit)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_Q, gain=0.5)
        nn.init.xavier_uniform_(self.W_K, gain=0.5)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        if self.K > 0:
            nn.init.xavier_uniform_(self.W_Qm, gain=0.5)
            nn.init.xavier_uniform_(self.W_Km, gain=0.5)
            nn.init.normal_(self.B_mem, mean=0.0, std=1.0 / (self.d ** 0.5))

    def _build_augmented_graph(self, batch_data, z_nodes):
        ptr = batch_data.ptr
        c2 = batch_data.c_2
        u2 = batch_data.u_2

        token_chunks = []
        center_ids = []
        nbr_ids = []
        graph_id_of_token = []
        cls_positions = []
        node_positions = []

        offset = 0
        for g_idx in range(ptr.numel() - 1):
            start = int(ptr[g_idx].item())
            end = int(ptr[g_idx + 1].item())
            n = end - start
            if n <= 0:
                continue

            z_g = z_nodes[start:end]
            adj = torch.zeros((n, n), dtype=torch.bool, device=z_nodes.device)
            mask = (c2 >= start) & (c2 < end)
            if bool(mask.any()):
                src = c2[mask] - start
                dst = u2[mask] - start
                adj[src, dst] = True

            adj_aug = torch.zeros((n + 1, n + 1), dtype=torch.bool, device=z_nodes.device)
            adj_aug[1:, 1:] = adj
            adj_aug[0, 1:] = True
            adj_aug[1:, 0] = True
            if self.allow_self:
                adj_aug.fill_diagonal_(True)

            pe = _laplacian_positional_encoding(adj_aug, self.pe_k, training=self.training)
            z_cls = self.cls_token.expand(1, -1)
            z_aug = torch.cat([z_cls, z_g], dim=0) + self.pe_proj(pe.to(dtype=z_g.dtype))
            token_chunks.append(z_aug)

            src_idx, dst_idx = torch.nonzero(adj_aug, as_tuple=True)
            center_ids.append(src_idx + offset)
            nbr_ids.append(dst_idx + offset)
            graph_id_of_token.append(torch.full((n + 1,), g_idx, dtype=torch.long, device=z_nodes.device))

            cls_positions.append(offset)
            node_positions.append(torch.arange(offset + 1, offset + n + 1, device=z_nodes.device, dtype=torch.long))
            offset += n + 1

        x_aug = torch.cat(token_chunks, dim=0)
        c_aug = torch.cat(center_ids, dim=0)
        u_aug = torch.cat(nbr_ids, dim=0)
        g_aug = torch.cat(graph_id_of_token, dim=0)
        cls_pos = torch.tensor(cls_positions, dtype=torch.long, device=z_nodes.device)
        node_pos = torch.cat(node_positions, dim=0)
        return x_aug, c_aug, u_aug, g_aug, cls_pos, node_pos

    def _compute_energy(self, x_aug, c_aug, u_aug):
        g = self.layernorm(x_aug)
        q = (g @ self.W_Q).view(-1, self.num_heads, self.head_dim)
        k = (g @ self.W_K).view(-1, self.num_heads, self.head_dim)

        scale = float(self.head_dim) ** 0.5
        ell = (q[c_aug] * k[u_aug]).sum(dim=-1) / scale
        ell = ell + (q[u_aug] * k[c_aug]).sum(dim=-1) / scale

        beta_att = inverse_temperature({"beta": self.beta_att}, "beta", beta_max=self.beta_max)
        lambda_att = positive_param({"lam": self.lambda_att}, "lam")

        num_tokens = int(x_aug.size(0))
        e_att = x_aug.new_zeros(())
        for h in range(self.num_heads):
            lse_h = _segment_logsumexp(beta_att * ell[:, h], c_aug, num_tokens)
            e_att = e_att + (lambda_att / beta_att) * lse_h.sum()

        e_mem = x_aug.new_zeros(())
        if self.K > 0 and positive_param({"lam": self.lambda_m}, "lam") > 1e-6:
            q_m = g @ self.W_Qm
            k_m = self.B_mem @ self.W_Km
            logits_m = (q_m @ k_m.t()) / (self.d ** 0.5)
            beta_m = inverse_temperature({"beta": self.beta_m}, "beta", beta_max=self.beta_max)
            lambda_m = positive_param({"lam": self.lambda_m}, "lam")
            e_mem = (lambda_m / beta_m) * torch.logsumexp(beta_m * logits_m, dim=1).sum()

        e_quad = 0.5 * (x_aug ** 2).sum()
        return e_quad - e_att - e_mem

    def _solve_dynamics(self, x_aug, c_aug, u_aug):
        energy_trace = []
        step = self.eta
        with torch.enable_grad():
            for _ in range(self.num_steps):
                if not self.training:
                    x_aug = x_aug.detach()
                if not x_aug.requires_grad:
                    x_aug = x_aug.requires_grad_(True)

                e = self._compute_energy(x_aug, c_aug, u_aug)
                grad = torch.autograd.grad(e, x_aug, create_graph=self.training)[0]

                if self.training and self.noise_std > 0:
                    grad = grad + torch.randn_like(grad) * self.noise_std

                if self.grad_clip_norm is not None:
                    gnorm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    grad = grad * (self.grad_clip_norm / gnorm).clamp(max=1.0)

                x_next = x_aug - step * grad
                if self.state_clip_norm is not None:
                    snorm = x_next.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    x_next = x_next * (self.state_clip_norm / snorm).clamp(max=1.0)

                energy_trace.append(float(e.detach().item()))
                x_aug = x_next
        return x_aug, energy_trace

    def forward(self, batch_data, task_level="graph"):
        x = batch_data.x
        if x.dim() == 1:
            x = x.view(-1, 1).float()

        z_nodes = self.node_encoder(x)
        x_aug, c_aug, u_aug, _g_aug, cls_pos, node_pos = self._build_augmented_graph(batch_data, z_nodes)
        x_final, energy_trace = self._solve_dynamics(x_aug, c_aug, u_aug)

        if task_level == "graph":
            out = self.readout(self.layernorm(x_final[cls_pos]))
            return out, energy_trace
        if task_level == "node":
            out = self.node_readout(self.layernorm(x_final[node_pos]))
            return out, energy_trace
        raise ValueError(f"Unsupported task_level: {task_level}")
