import torch
import torch.nn as nn

from .et_core import ETCoreBlock, ETGraphMaskModulator, EnergyLayerNorm
from .utils import laplacian_pe_from_adjacency
from .model import StableMLP


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
        eta=0.05,
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
        mask_mode="sparse",
        node_cap=None,
        dense_kernel_size=3,  # compatibility placeholder for CLI
        share_block_weights=True,
    ):
        super().__init__()
        del dense_kernel_size
        self.d = int(d)
        self.num_steps = int(num_steps)
        self.num_blocks = int(max(1, num_blocks))
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim or d)
        self.pe_k = int(pe_k)
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

        self.node_encoder = StableMLP(
            in_dim,
            d,
            hidden_dim=max(d, encoder_hidden_mult * d),
            dropout=dropout,
            final_norm=True,
        )
        self.pe_proj = nn.Linear(self.pe_k, d)
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

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    @property
    def eta(self):
        return self.eta_max * torch.sigmoid(self.eta_logit)

    def _build_augmented_graph(self, batch_data, z_nodes):
        ptr = batch_data.ptr
        c2 = batch_data.c_2
        u2 = batch_data.u_2
        pe_cls = getattr(batch_data, "pe_cls", None)
        pe_cls_ptr = getattr(batch_data, "pe_cls_ptr", None)

        token_chunks = []
        center_ids = []
        nbr_ids = []
        cls_positions = []
        node_positions = []
        graph_chunks = []

        offset = 0
        for g_idx in range(ptr.numel() - 1):
            start = int(ptr[g_idx].item())
            end = int(ptr[g_idx + 1].item())
            n = end - start
            if n <= 0:
                continue
            if self.node_cap is not None and n > self.node_cap:
                n = self.node_cap

            z_g = z_nodes[start : start + n]
            adj = torch.zeros((n, n), dtype=torch.bool, device=z_nodes.device)
            mask = (c2 >= start) & (c2 < end)
            if bool(mask.any()):
                src = c2[mask] - start
                dst = u2[mask] - start
                valid = (src < n) & (dst < n)
                if bool(valid.any()):
                    adj[src[valid], dst[valid]] = True

            adj_aug = torch.zeros((n + 1, n + 1), dtype=torch.bool, device=z_nodes.device)
            adj_aug[1:, 1:] = adj
            adj_aug[0, 1:] = True
            adj_aug[1:, 0] = True
            if self.allow_self:
                adj_aug.fill_diagonal_(True)

            if (
                pe_cls is not None
                and pe_cls_ptr is not None
                and not self.allow_self
                and int(pe_cls_ptr[g_idx + 1].item()) > int(pe_cls_ptr[g_idx].item())
            ):
                pe = pe_cls[int(pe_cls_ptr[g_idx].item()) : int(pe_cls_ptr[g_idx + 1].item())]
            else:
                pe = laplacian_pe_from_adjacency(adj_aug, k=self.pe_k, training=self.training)
            z_cls = self.cls_token.expand(1, -1)
            z_aug = torch.cat([z_cls, z_g], dim=0) + self.pe_proj(pe.to(dtype=z_g.dtype))
            token_chunks.append(z_aug)

            src_idx, dst_idx = torch.nonzero(adj_aug, as_tuple=True)
            center_ids.append(src_idx + offset)
            nbr_ids.append(dst_idx + offset)
            cls_positions.append(offset)
            node_positions.append(torch.arange(offset + 1, offset + n + 1, device=z_nodes.device, dtype=torch.long))
            graph_chunks.append({"start": offset, "size": n + 1, "adj": adj_aug, "orig_start": start})
            offset += n + 1

        x_aug = torch.cat(token_chunks, dim=0)
        c_aug = torch.cat(center_ids, dim=0)
        u_aug = torch.cat(nbr_ids, dim=0)
        cls_pos = torch.tensor(cls_positions, dtype=torch.long, device=z_nodes.device)
        node_pos = torch.cat(node_positions, dim=0)
        return x_aug, c_aug, u_aug, cls_pos, node_pos, graph_chunks

    def _solve_dynamics(self, x_aug, c_aug, u_aug, graph_chunks, batch_data):
        energy_trace = []
        step = self.eta
        x = x_aug
        dense_cache = None

        if self.mask_mode == "official_dense" and self.mask_modulator is not None:
            edge_attr = getattr(batch_data, "edge_attr", None)
            c2 = batch_data.c_2.to(device=x.device)
            u2 = batch_data.u_2.to(device=x.device)
            ptr = batch_data.ptr.to(device=x.device)
            bsz = len(graph_chunks)
            max_n = max(int(chunk["size"]) for chunk in graph_chunks)
            feat_dim = int(edge_attr.size(-1)) if edge_attr is not None else 1

            # Static plan for copying token states into padded dense tensors.
            sizes = torch.tensor([int(chunk["size"]) for chunk in graph_chunks], dtype=torch.long, device=x.device)

            if bsz > 0 and max_n > 0:
                batch_grid = torch.arange(bsz, dtype=torch.long, device=x.device)[:, None].expand(bsz, max_n)
                local_grid = torch.arange(max_n, dtype=torch.long, device=x.device)[None, :].expand(bsz, max_n)
                valid_grid = local_grid < sizes[:, None]
                x_batch_ids = batch_grid[valid_grid]
                x_local_ids = local_grid[valid_grid]
                x_source_ids = torch.arange(x.size(0), dtype=torch.long, device=x.device)
            else:
                x_batch_ids = torch.empty((0,), dtype=torch.long, device=x.device)
                x_local_ids = torch.empty((0,), dtype=torch.long, device=x.device)
                x_source_ids = torch.empty((0,), dtype=torch.long, device=x.device)

            if edge_attr is not None:
                if c2.numel() > 0:
                    graph_ids = torch.bucketize(c2, ptr[1:-1], right=True)
                    edge_starts = ptr[graph_ids]
                    edge_ends = ptr[graph_ids + 1]
                    edge_mask = (u2 >= edge_starts) & (u2 < edge_ends)
                else:
                    graph_ids = c2.new_empty((0,), dtype=torch.long)
                    edge_starts = c2.new_empty((0,), dtype=torch.long)
                    edge_mask = c2.new_empty((0,), dtype=torch.bool)

                if bool(edge_mask.any()):
                    edge_global_ids = torch.nonzero(edge_mask, as_tuple=False).reshape(-1).to(dtype=torch.long, device=x.device)
                    edge_batch_ids = graph_ids[edge_mask].to(dtype=torch.long, device=x.device)
                    edge_src_ids = (c2[edge_mask] - edge_starts[edge_mask] + 1).to(dtype=torch.long, device=x.device)
                    edge_dst_ids = (u2[edge_mask] - edge_starts[edge_mask] + 1).to(dtype=torch.long, device=x.device)
                else:
                    edge_batch_ids = torch.empty((0,), dtype=torch.long, device=x.device)
                    edge_src_ids = torch.empty((0,), dtype=torch.long, device=x.device)
                    edge_dst_ids = torch.empty((0,), dtype=torch.long, device=x.device)
                    edge_global_ids = torch.empty((0,), dtype=torch.long, device=x.device)
                edge_static_template = None
            else:
                edge_static_template = x.new_zeros((bsz, max_n, max_n, 1))
                for i, chunk in enumerate(graph_chunks):
                    size = int(chunk["size"])
                    adj_local = chunk["adj"].to(dtype=x.dtype)
                    edge_static_template[i, :size, :size, 0] = adj_local
                edge_batch_ids = edge_src_ids = edge_dst_ids = edge_global_ids = None

            if edge_attr is not None:
                edge_batch_static = None
            else:
                edge_batch_static = edge_static_template

            dense_cache = {
                "bsz": bsz,
                "max_n": max_n,
                "feat_dim": feat_dim,
                "sizes": sizes,
                "x_batch": x.new_zeros((bsz, max_n, x.size(-1))),
                "x_batch_ids": x_batch_ids,
                "x_local_ids": x_local_ids,
                "x_source_ids": x_source_ids,
                "edge_attr": edge_attr,
                "edge_batch": None if edge_attr is not None else edge_static_template,
                "edge_batch_ids": edge_batch_ids,
                "edge_src_ids": edge_src_ids,
                "edge_dst_ids": edge_dst_ids,
                "edge_global_ids": edge_global_ids,
                "edge_static_template": edge_static_template,
                "edge_batch_static": edge_batch_static,
            }

        with torch.enable_grad():
            for block_idx in range(self.num_blocks):
                norm = self.norm_blocks[block_idx]
                core = self.core_blocks[block_idx]
                for _ in range(self.num_steps):
                    if not self.training:
                        x = x.detach()
                    if not x.requires_grad:
                        x = x.requires_grad_(True)

                    # Official ET update uses grad wrt g, then updates x directly.
                    g = norm(x)
                    dense_modulation = None
                    dense_x_batch = None
                    if dense_cache is not None:
                        bsz = dense_cache["bsz"]
                        max_n = dense_cache["max_n"]
                        feat_dim = dense_cache["feat_dim"]
                        sizes = dense_cache["sizes"]
                        edge_attr = dense_cache["edge_attr"]
                        x_batch = dense_cache["x_batch"]
                        x_batch.zero_()
                        x_batch_ids = dense_cache["x_batch_ids"]
                        x_local_ids = dense_cache["x_local_ids"]
                        x_source_ids = dense_cache["x_source_ids"]
                        if x_batch_ids.numel() > 0:
                            x_batch[x_batch_ids, x_local_ids] = x[x_source_ids]
                        x_batch = x_batch.requires_grad_(True)

                        if edge_attr is not None:
                            edge_batch = x.new_zeros((bsz, max_n, max_n, feat_dim))
                            edge_batch_ids = dense_cache["edge_batch_ids"]
                            if edge_batch_ids.numel() > 0:
                                edge_src_ids = dense_cache["edge_src_ids"]
                                edge_dst_ids = dense_cache["edge_dst_ids"]
                                edge_global_ids = dense_cache["edge_global_ids"]
                                edge_batch[edge_batch_ids, edge_src_ids, edge_dst_ids, :] = edge_attr[edge_global_ids].to(
                                    dtype=x.dtype, device=x.device
                                )
                        else:
                            edge_batch = dense_cache["edge_batch"]

                        dense_modulation = self.mask_modulator.dense_modulation_batched(x_batch, edge_batch)  # [B, H, N, N]
                        g = norm(x_batch)
                        dense_x_batch = x_batch
                    e = core.energy(
                        g,
                        c_aug,
                        u_aug,
                        graph_chunks,
                        self.mask_mode,
                        dense_modulation=dense_modulation,
                    )
                    if dense_x_batch is not None:
                        grad_x_batch = torch.autograd.grad(e, dense_x_batch, create_graph=self.training)[0]
                        grad_g = x.new_zeros(x.shape)
                        grad_g[x_source_ids] = grad_x_batch[x_batch_ids, x_local_ids]
                    else:
                        grad_g = torch.autograd.grad(e, g, create_graph=self.training)[0]

                    if self.training and self.noise_std > 0:
                        grad_g = grad_g + torch.randn_like(grad_g) * self.noise_std
                    if self.grad_clip_norm is not None:
                        gnorm = grad_g.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        grad_g = grad_g * (self.grad_clip_norm / gnorm).clamp(max=1.0)

                    x_next = x - step * grad_g
                    if self.state_clip_norm is not None:
                        snorm = x_next.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        x_next = x_next * (self.state_clip_norm / snorm).clamp(max=1.0)

                    energy_trace.append(float(e.detach().item()))
                    x = x_next
        return x, energy_trace

    def forward(self, batch_data, task_level="graph", return_solver_stats=False):
        x = batch_data.x
        if x.dim() == 1:
            x = x.view(-1, 1).float()

        z_nodes = self.node_encoder(x)
        x_aug, c_aug, u_aug, cls_pos, node_pos, graph_chunks = self._build_augmented_graph(batch_data, z_nodes)
        x_final, energy_trace = self._solve_dynamics(x_aug, c_aug, u_aug, graph_chunks, batch_data)
        g_final = self.norm_blocks[-1](x_final)

        solver_stats = {
            "energy_trace": energy_trace,
            "memory_entropy": self.core_blocks[-1].hn.entropy(g_final),
        }

        if task_level == "graph":
            out = self.readout(g_final[cls_pos])
            if return_solver_stats:
                return out, energy_trace, solver_stats
            return out, energy_trace
        if task_level == "node":
            out = self.node_readout(g_final[node_pos])
            if return_solver_stats:
                return out, energy_trace, solver_stats
            return out, energy_trace
        raise ValueError(f"Unsupported task_level: {task_level}")
