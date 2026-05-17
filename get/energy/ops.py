import torch
import torch.nn.functional as F
from torch_geometric.utils import degree, scatter


def segment_logsumexp(x: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int = -1):
    if dim < 0:
        dim = x.dim() + dim

    index = segment_ids.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    size = [num_segments] + list(x.shape[1:])

    out_max = torch.scatter_reduce(x.new_full(size, float("-inf")), dim, index, x, "amax", include_self=False)

    idx_shape = [1] * x.dim()
    idx_shape[dim] = len(segment_ids)
    max_expanded = out_max.gather(dim, segment_ids.view(*idx_shape).expand_as(x))
    max_expanded = torch.where(torch.isneginf(max_expanded), torch.zeros_like(max_expanded), max_expanded)
    x_shifted = x - max_expanded

    out_sum = scatter(torch.exp(x_shifted), segment_ids, dim=dim, dim_size=num_segments, reduce="sum")
    return torch.log(out_sum.clamp_min(1e-12)) + out_max


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
        beta = beta.clamp(max=beta_max) if torch.is_tensor(beta) else min(beta, beta_max)
    return beta


def get_degree_from_incidence(c_2: torch.Tensor, num_nodes: int):
    """Use optimized PyG degree utility."""
    return degree(c_2, num_nodes=num_nodes, dtype=torch.float32)


def compute_degree_scaler(degrees: torch.Tensor, avg_degree: float | torch.Tensor, mode="pna"):
    if mode == "pna":
        avg = avg_degree if isinstance(avg_degree, torch.Tensor) else degrees.new_tensor(avg_degree)
        return torch.log(degrees + 1.0) / torch.log(avg.clamp_min(1e-6) + 1.0)
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
