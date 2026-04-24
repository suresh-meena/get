from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .registry import register_model
from .structural import shortest_path_distances


class _FeatureMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = max(int(out_dim), int(hidden_mult) * int(out_dim))
        layers = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _EdgeMessageBlock(nn.Module):
    def __init__(self, dim: int, edge_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.edge_dim = int(edge_dim)
        in_dim = 2 * dim + self.edge_dim
        self.msg = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, edge_attr: torch.Tensor | None = None):
        if src.numel() == 0:
            return self.norm(h)
        pieces = [h[src], h[dst]]
        if self.edge_dim > 0:
            if edge_attr is None:
                edge_attr = h.new_zeros((src.numel(), self.edge_dim))
            else:
                edge_attr = edge_attr.to(dtype=h.dtype, device=h.device)
                if edge_attr.dim() == 1:
                    edge_attr = edge_attr.unsqueeze(-1)
                if edge_attr.size(-1) < self.edge_dim:
                    pad = self.edge_dim - edge_attr.size(-1)
                    edge_attr = torch.cat([edge_attr, h.new_zeros((edge_attr.size(0), pad))], dim=-1)
                elif edge_attr.size(-1) > self.edge_dim:
                    edge_attr = edge_attr[..., : self.edge_dim]
            pieces.append(edge_attr)
        messages = self.msg(torch.cat(pieces, dim=-1))
        agg = h.new_zeros(h.shape)
        agg.index_add_(0, dst, messages)
        return self.norm(h + agg)


class _MaskedAttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(math.ceil(self.dim / self.num_heads))
        self.inner_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(dim, self.inner_dim)
        self.k_proj = nn.Linear(dim, self.inner_dim)
        self.v_proj = nn.Linear(dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, bias: torch.Tensor | None = None):
        n = int(h.size(0))
        if n == 0:
            return h
        q = self.q_proj(h).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(h).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(h).view(n, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.einsum("hnd,hmd->hnm", q, k) / math.sqrt(float(self.head_dim))
        if bias is not None:
            scores = scores + bias.to(dtype=scores.dtype, device=scores.device)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("hnm,hmd->hnd", attn, v).transpose(0, 1).contiguous().view(n, self.inner_dim)
        return self.norm(h + self.out_proj(out))


@dataclass
class _GraphChunk:
    start: int
    end: int
    node_ids: torch.Tensor
    src: torch.Tensor
    dst: torch.Tensor
    edge_attr: torch.Tensor | None


def _graph_chunks(batch_data) -> list[_GraphChunk]:
    ptr = batch_data.ptr
    c2 = batch_data.c_2
    u2 = batch_data.u_2
    edge_attr = getattr(batch_data, "edge_attr", None)
    chunks: list[_GraphChunk] = []
    for g_idx in range(int(ptr.numel() - 1)):
        start = int(ptr[g_idx].item())
        end = int(ptr[g_idx + 1].item())
        mask = (c2 >= start) & (c2 < end) & (u2 >= start) & (u2 < end)
        if bool(mask.any()):
            src = c2[mask] - start
            dst = u2[mask] - start
            ea = edge_attr[mask] if edge_attr is not None else None
        else:
            src = c2.new_empty((0,), dtype=torch.long)
            dst = c2.new_empty((0,), dtype=torch.long)
            ea = edge_attr.new_empty((0, *edge_attr.shape[1:])) if edge_attr is not None else None
        chunks.append(
            _GraphChunk(
                start=start,
                end=end,
                node_ids=torch.arange(start, end, device=ptr.device, dtype=torch.long),
                src=src,
                dst=dst,
                edge_attr=ea,
            )
        )
    return chunks


class _BaseGraphWrapper(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d: int,
        num_classes: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        pe_k: int = 0,
        rwse_k: int = 0,
        use_local: bool = True,
        use_attention: bool = True,
        use_spatial_bias: bool = False,
        use_virtual_node: bool = False,
        max_distance: int = 8,
        edge_dim: int = 0,
    ):
        super().__init__()
        self.d = int(d)
        self.num_classes = int(num_classes)
        self.num_layers = int(num_layers)
        self.use_local = bool(use_local)
        self.use_attention = bool(use_attention)
        self.use_spatial_bias = bool(use_spatial_bias)
        self.use_virtual_node = bool(use_virtual_node)
        self.max_distance = int(max_distance)
        self.edge_dim = int(edge_dim)

        self.node_encoder = _FeatureMLP(in_dim, d, dropout=dropout)
        self.pe_proj = nn.Linear(pe_k, d) if pe_k > 0 else None
        self.rwse_proj = nn.Linear(rwse_k, d) if rwse_k > 0 else None
        self.virtual_token = nn.Parameter(torch.zeros(1, d)) if self.use_virtual_node else None

        self.local_blocks = nn.ModuleList(
            [_EdgeMessageBlock(d, edge_dim=self.edge_dim, dropout=dropout) for _ in range(self.num_layers)]
        )
        self.attn_blocks = nn.ModuleList([_MaskedAttentionBlock(d, num_heads=num_heads, dropout=dropout) for _ in range(self.num_layers)])
        self.graph_head = nn.Sequential(
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.LayerNorm(2 * d),
            nn.Dropout(dropout),
            nn.Linear(2 * d, num_classes),
        )
        self.node_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )
        if self.use_spatial_bias:
            self.distance_bias = nn.Embedding(self.max_distance + 2, num_heads)
        else:
            self.distance_bias = None
        if self.virtual_token is not None:
            nn.init.normal_(self.virtual_token, mean=0.0, std=0.02)

    def _initial_node_states(self, batch_data):
        x = batch_data.x
        if x.dim() == 1:
            x = x.view(-1, 1).float()
        h = self.node_encoder(x.to(dtype=self.virtual_token.dtype if self.virtual_token is not None else x.dtype))
        if self.pe_proj is not None and hasattr(batch_data, "pe") and batch_data.pe is not None:
            h = h + self.pe_proj(batch_data.pe.to(dtype=h.dtype))
        if self.rwse_proj is not None and hasattr(batch_data, "rwse") and batch_data.rwse is not None:
            h = h + self.rwse_proj(batch_data.rwse.to(dtype=h.dtype))
        return h

    def _graph_bias(self, chunk: _GraphChunk, n: int, device, dtype):
        if not self.use_spatial_bias or self.distance_bias is None:
            return None
        edges = [(int(u), int(v)) for u, v in zip(chunk.src.tolist(), chunk.dst.tolist())]
        if self.use_virtual_node:
            v = n - 1
            edges.extend((i, v) for i in range(n - 1))
            edges.extend((v, i) for i in range(n - 1))
        dist = shortest_path_distances(n, edges, max_distance=self.max_distance).to(device=device)
        return self.distance_bias(dist).permute(2, 0, 1).to(dtype=dtype)

    def _graph_pool(self, h: torch.Tensor):
        if h.numel() == 0:
            return h.new_zeros((self.d,))
        mean = h.mean(dim=0)
        summ = h.sum(dim=0)
        mx = h.max(dim=0).values
        std = torch.sqrt(torch.relu((h.pow(2)).mean(dim=0) - mean.pow(2)) + 1e-6)
        return torch.cat([mean, summ, mx, std], dim=-1)

    def _run_graph(self, h: torch.Tensor, chunk: _GraphChunk):
        if self.use_virtual_node:
            h = torch.cat([h, self.virtual_token.to(dtype=h.dtype, device=h.device).expand(1, -1)], dim=0)
        n = int(h.size(0))
        edge_src = chunk.src
        edge_dst = chunk.dst
        edge_attr = chunk.edge_attr
        if self.use_virtual_node and n > 0:
            v = n - 1
            extra_src = torch.arange(n - 1, device=h.device, dtype=torch.long)
            extra_dst = torch.full((n - 1,), v, dtype=torch.long, device=h.device)
            edge_src = torch.cat([edge_src, extra_src, extra_dst], dim=0)
            edge_dst = torch.cat([edge_dst, extra_dst, extra_src], dim=0)
            if edge_attr is not None:
                zero = edge_attr.new_zeros((2 * (n - 1), *edge_attr.shape[1:]))
                edge_attr = torch.cat([edge_attr, zero], dim=0)
        bias = self._graph_bias(chunk, n, h.device, h.dtype)
        for local_block, attn_block in zip(self.local_blocks, self.attn_blocks):
            if self.use_local:
                h = local_block(h, edge_src, edge_dst, edge_attr=edge_attr)
            if self.use_attention:
                h = attn_block(h, bias=bias)
        return h

    def forward(self, batch_data, task_level="graph"):
        h0 = self._initial_node_states(batch_data)
        chunks = _graph_chunks(batch_data)
        node_outputs = []
        graph_outputs = []
        for chunk in chunks:
            h = h0[chunk.start : chunk.end]
            h = self._run_graph(h, chunk)
            if self.use_virtual_node:
                h_nodes = h[:-1]
            else:
                h_nodes = h
            graph_outputs.append(self._graph_pool(h))
            node_outputs.append(h_nodes)

        if task_level == "node":
            out = self.node_head(torch.cat(node_outputs, dim=0))
            return out, None
        graph_repr = torch.stack(graph_outputs, dim=0)
        out = self.graph_head(graph_repr)
        return out, None


class GraphGPSAdapter(_BaseGraphWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GraphormerAdapter(_BaseGraphWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_local", False)
        kwargs.setdefault("use_attention", True)
        kwargs.setdefault("use_spatial_bias", True)
        super().__init__(*args, **kwargs)


class ExphormerAdapter(_BaseGraphWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_local", True)
        kwargs.setdefault("use_attention", True)
        kwargs.setdefault("use_virtual_node", True)
        super().__init__(*args, **kwargs)


class GRITAdapter(_BaseGraphWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_local", True)
        kwargs.setdefault("use_attention", True)
        kwargs.setdefault("use_spatial_bias", True)
        super().__init__(*args, **kwargs)


class GPSEAdapter(_BaseGraphWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_local", True)
        kwargs.setdefault("use_attention", True)
        kwargs.setdefault("use_spatial_bias", True)
        super().__init__(*args, **kwargs)


class SignNetAdapter(_BaseGraphWrapper):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("use_local", False)
        kwargs.setdefault("use_attention", True)
        kwargs.setdefault("use_spatial_bias", True)
        super().__init__(*args, **kwargs)


class NotImplementedBaseline(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.name} is listed in the manuscript as a comparison baseline, "
            "but no faithful implementation is present in this repository yet."
        )


@register_model("graphgps")
def _build_graphgps(*args, **kwargs):
    return GraphGPSAdapter(*args, **kwargs)


@register_model("graphormer")
def _build_graphormer(*args, **kwargs):
    return GraphormerAdapter(*args, **kwargs)


@register_model("exphormer")
def _build_exphormer(*args, **kwargs):
    return ExphormerAdapter(*args, **kwargs)


@register_model("grit")
def _build_grit(*args, **kwargs):
    return GRITAdapter(*args, **kwargs)


@register_model("gpse")
def _build_gpse(*args, **kwargs):
    return GPSEAdapter(*args, **kwargs)


@register_model("signnet")
def _build_signnet(*args, **kwargs):
    return SignNetAdapter(*args, **kwargs)
