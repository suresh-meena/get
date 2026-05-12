import torch
import torch.nn.functional as F
from torch_geometric.utils import scatter, degree

ts_scatter = None
ts_scatter_add = None
ts_scatter_max = None
ts_scatter_logsumexp = None


def segment_reduce_1d(src: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, reduce="sum", dim: int = -1):
    """
    Optimized segmented reduction using PyTorch-Geometric's scatter.
    This uses high-performance C++ kernels specialized for sparse graph reductions.
    """
    if dim < 0:
        dim = src.dim() + dim
        
    out = scatter(src, segment_ids, dim=dim, dim_size=num_segments, reduce=reduce)
    
    # We still need counts for some scalers
    counts = bincount_1d(segment_ids, num_segments)
    return out, counts


def bincount_1d(segment_ids: torch.Tensor, num_segments: int):
    """Fast bincount wrapper."""
    return torch.bincount(segment_ids, minlength=num_segments)


def segment_logsumexp(x: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int = -1):
    """
    Computes logsumexp over segments using native PyTorch to support torch.compile.
    Using scatter for max reduction.
    """
    if dim < 0:
        dim = x.dim() + dim

    def _logsumexp_fwd(x_val, ids, num_seg, d):
        out_max = scatter(x_val, ids, dim=d, dim_size=num_seg, reduce="max")

        idx_shape = [1] * x_val.dim()
        idx_shape[d] = len(ids)
        idx = ids.view(*idx_shape).expand_as(x_val)
        
        max_expanded = torch.gather(out_max, d, idx)
        max_expanded = torch.where(max_expanded == float('-inf'), torch.zeros_like(max_expanded), max_expanded)
        x_shifted = x_val - max_expanded

        exp_x_shifted = torch.exp(x_shifted)
        out_sum_exp = scatter(exp_x_shifted, ids, dim=d, dim_size=num_seg, reduce="sum")

        neg_inf = torch.full_like(out_max, float("-inf"))
        out = torch.where(out_sum_exp == 0, neg_inf, torch.log(out_sum_exp.clamp_min(1e-12)) + out_max)
        return out

    from torch.utils.checkpoint import checkpoint
    if x.requires_grad:
        return checkpoint(_logsumexp_fwd, x, segment_ids, num_segments, dim, use_reentrant=False)
    else:
        return _logsumexp_fwd(x, segment_ids, num_segments, dim)


def scatter_add_nd(grad_buffer: torch.Tensor, indices: torch.Tensor, src: torch.Tensor, dim: int):
    """
    Memory-efficient N-dimensional scatter-add using scatter_add_.
    """
    if src.dtype != grad_buffer.dtype:
        src = src.to(dtype=grad_buffer.dtype)
    idx_shape = [1] * src.dim()
    idx_shape[dim] = indices.size(0)
    idx = indices.view(idx_shape).expand_as(src)
    return grad_buffer.scatter_add_(dim, idx, src)


def fused_motif_dot(
    Q3_c: torch.Tensor,
    K3_u: torch.Tensor,
    K3_v: torch.Tensor,
    T_tau: torch.Tensor,
) -> torch.Tensor:
    """
    Trilinear motif score contraction.
    
    Optimized for PyTorch Inductor fusion and memory efficiency.
    """
    return torch.einsum("...ij,...ij,...ij->...", Q3_c, K3_u, K3_v) + torch.einsum("...ij,...ij->...", Q3_c, T_tau)


def positive_param(params: dict, name: str):
    val = params[name]
    if isinstance(val, (float, int)):
        return val
    return F.softplus(val) + 1e-8


def inverse_temperature(params: dict, name: str, beta_max=None):
    beta = positive_param(params, name)
    if beta_max is not None:
        if torch.is_tensor(beta):
            beta = beta.clamp(max=beta_max)
        else:
            beta = min(beta, beta_max)
    return beta


def get_degree_from_incidence(c_2: torch.Tensor, num_nodes: int):
    """Use optimized PyG degree utility."""
    return degree(c_2, num_nodes=num_nodes, dtype=torch.float32)


def compute_degree_scaler(degrees: torch.Tensor, avg_degree: torch.Tensor | float, mode="pna"):
    if mode == "pna":
        if not torch.is_tensor(avg_degree):
            avg_degree = torch.tensor(float(avg_degree), device=degrees.device, dtype=degrees.dtype)
        else:
            avg_degree = avg_degree.to(device=degrees.device, dtype=degrees.dtype)
        avg_degree = avg_degree.clamp_min(1e-6)
        return torch.log(degrees + 1.0) / torch.log(avg_degree + 1.0)
    return torch.ones_like(degrees)


def positional_embeddings_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    k: int = 15,
    flip_sign: bool = False,
) -> torch.Tensor:
    """
    Compute Laplacian Eigenvector Positional Embeddings using PyG utilities.
    """
    from torch_geometric.utils import get_laplacian
    if k <= 0:
        return torch.empty((num_nodes, 0), device=edge_index.device)

    # L = I - D^-1/2 A D^-1/2
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index, normalization="sym", num_nodes=num_nodes
    )
    
    # For small graphs, dense eigh is fine. For large graphs, use sparse LOBPCG.
    if num_nodes < 2000:
        # Convert to dense for eigh
        L = torch.zeros((num_nodes, num_nodes), device=edge_index.device, dtype=torch.float32)
        L[edge_index_lap[0], edge_index_lap[1]] = edge_weight_lap
        evals, evecs = torch.linalg.eigh(L)
    else:
        # Use sparse LOBPCG
        indices = edge_index_lap
        values = edge_weight_lap
        L_sparse = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).to(device=edge_index.device)
        
        # Initial guess for eigenvectors
        X = torch.randn(num_nodes, k + 1, device=edge_index.device)
        
        # We want the smallest eigenvectors (smallest eigenvalues of L)
        # lobpcg finds the largest by default, but we can pass largest=False
        evals, evecs = torch.lobpcg(L_sparse, k=k+1, X=X, largest=False)
    
    # Skip the first eigenvector (constant)
    if evecs.size(1) < k + 1:
        evecs = torch.nn.functional.pad(evecs, (0, (k + 1) - evecs.size(1)))
    
    pos = evecs[:, 1 : k + 1]

    if flip_sign:
        flips = torch.randint(0, 2, pos.shape, device=pos.device, dtype=torch.float32)
        flips = flips * 2.0 - 1.0
        pos = pos * flips
        
    return pos
