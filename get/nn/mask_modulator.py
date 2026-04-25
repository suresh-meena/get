"""ET graph-appendix edge-conditioned mask modulator."""
import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter


class ETGraphMaskModulator(nn.Module):
    """ET graph appendix-style edge-conditioned mask:
    A_hat = Conv2D((X @ X^T)[..., None]) ⊙ A'
    """

    def __init__(self, d, num_heads, edge_feat_dim=None, kernel_size=3):
        super().__init__()
        self.d = int(d)
        self.num_heads = int(num_heads)
        self.edge_feat_dim = None if edge_feat_dim is None else int(edge_feat_dim)
        self.mask_conv = nn.Conv2d(
            in_channels=1, out_channels=self.num_heads,
            kernel_size=int(kernel_size), stride=1,
            padding=int(kernel_size) // 2, bias=True
        )
        if self.edge_feat_dim is None:
            self.edge_proj = nn.LazyLinear(self.num_heads, bias=False)
        else:
            self.edge_proj = nn.Linear(self.edge_feat_dim, self.num_heads, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.mask_conv.weight, a=5 ** 0.5)
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

    def forward(self, x, batch_data=None):
        if batch_data is None:
            return self.dense_modulation(x, None) if x.dim() == 2 else self.dense_modulation_batched(x, None)

        num_nodes = x.size(0)
        c2 = batch_data.c_2
        u2 = batch_data.u_2
        edge_attr = getattr(batch_data, "aligned_edge_attr", getattr(batch_data, "edge_attr", None))

        ptr = batch_data.ptr
        num_graphs = ptr.numel() - 1
        max_n = torch.diff(ptr).max().item()

        x_dense = x.new_zeros((num_graphs, max_n, self.d))
        node_batch = batch_data.batch
        node_local = torch.arange(num_nodes, device=x.device) - ptr[node_batch]
        x_dense[node_batch, node_local] = x

        if edge_attr is not None:
            feat_dim = edge_attr.size(-1)
            e_dense = x.new_zeros((num_graphs, max_n, max_n, feat_dim))
            edge_batch = node_batch[c2]
            src_local = c2 - ptr[edge_batch]
            dst_local = u2 - ptr[edge_batch]
            e_dense[edge_batch, src_local, dst_local] = edge_attr.to(dtype=x.dtype)
        else:
            e_dense = x.new_zeros((num_graphs, max_n, max_n, 1))
            edge_batch = node_batch[c2]
            src_local = c2 - ptr[edge_batch]
            dst_local = u2 - ptr[edge_batch]
            e_dense[edge_batch, src_local, dst_local] = 1.0

        dense_mod = self.dense_modulation_batched(x_dense, e_dense)

        edge_batch = node_batch[c2]
        src_local = c2 - ptr[edge_batch]
        dst_local = u2 - ptr[edge_batch]
        sparse_mod = dense_mod[edge_batch, :, src_local, dst_local]

        return sparse_mod.transpose(0, 1)

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
