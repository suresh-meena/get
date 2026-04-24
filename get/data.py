import torch
import os
import hashlib
import numpy as np
from numba import njit
from tqdm.auto import tqdm


@njit
def _numba_edges_to_csr(num_nodes, edges_arr):
    """Directly build CSR from edge array in Numba.
    Ensures simple undirected graph: symmetric, no self-loops, no duplicates, sorted.
    """
    # 1. Collect all valid directed edges
    num_input = edges_arr.shape[0]
    # At most 2 * num_input directed edges
    u_arr = np.empty(2 * num_input, dtype=np.int64)
    v_arr = np.empty(2 * num_input, dtype=np.int64)
    
    count = 0
    for i in range(num_input):
        u = edges_arr[i, 0]
        v = edges_arr[i, 1]
        if u != v and u < num_nodes and v < num_nodes:
            u_arr[count] = u
            v_arr[count] = v
            count += 1
            u_arr[count] = v
            v_arr[count] = u
            count += 1
            
    # 2. Sort by (u, v) to remove duplicates
    # Since Numba doesn't have an easy lexsort, we can sort a packed array if num_nodes is small enough,
    # but to be safe, we'll build a CSR with duplicates and then deduplicate inplace.
    
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for i in range(count):
        indptr[u_arr[i] + 1] += 1
        
    for i in range(num_nodes):
        indptr[i + 1] += indptr[i]
        
    indices_raw = np.empty(count, dtype=np.int64)
    curr_idx = indptr[:-1].copy()
    for i in range(count):
        u = u_arr[i]
        indices_raw[curr_idx[u]] = v_arr[i]
        curr_idx[u] += 1
        
    # Deduplicate and sort
    new_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    # worst case size is count
    indices = np.empty(count, dtype=np.int64)
    
    valid_count = 0
    for i in range(num_nodes):
        start = indptr[i]
        end = indptr[i+1]
        new_indptr[i] = valid_count
        if end > start:
            nbrs = indices_raw[start:end]
            nbrs.sort()
            # keep unique
            indices[valid_count] = nbrs[0]
            valid_count += 1
            for j in range(1, len(nbrs)):
                if nbrs[j] != nbrs[j-1]:
                    indices[valid_count] = nbrs[j]
                    valid_count += 1
    new_indptr[num_nodes] = valid_count
    
    return new_indptr, indices[:valid_count]


@njit
def _numba_motif_extraction(indptr, indices, max_motifs):
    """Numba-accelerated motif extraction with budget logic."""
    if max_motifs == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    num_nodes = len(indptr) - 1
    
    # Pass 1: Count total motifs to allocate arrays
    total_motifs = 0
    for i in range(num_nodes):
        start, end = indptr[i], indptr[i+1]
        deg = end - start
        if deg < 2:
            continue
        
        nbrs = indices[start:end]
        num_tri = 0
        num_wed = 0
        for p in range(deg):
            u = nbrs[p]
            u_start, u_end = indptr[u], indptr[u+1]
            u_nbrs = indices[u_start:u_end]
            for q in range(p + 1, deg):
                v = nbrs[q]
                # binary search for triangle check
                idx = np.searchsorted(u_nbrs, v)
                if idx < len(u_nbrs) and u_nbrs[idx] == v:
                    num_tri += 1
                else:
                    num_wed += 1
        
        if max_motifs > 0:
            if num_tri >= max_motifs:
                total_motifs += num_tri
            else:
                total_motifs += num_tri + num_wed
        else:
            total_motifs += num_tri + num_wed

    # Allocate output buffers
    c3 = np.empty(total_motifs, dtype=np.int64)
    u3 = np.empty(total_motifs, dtype=np.int64)
    v3 = np.empty(total_motifs, dtype=np.int64)
    tt = np.empty(total_motifs, dtype=np.int32)
    
    curr = 0
    for i in range(num_nodes):
        start, end = indptr[i], indptr[i+1]
        deg = end - start
        if deg < 2:
            continue
        
        nbrs = indices[start:end]
        num_tri = 0
        for p in range(deg):
            u = nbrs[p]
            u_start, u_end = indptr[u], indptr[u+1]
            u_nbrs = indices[u_start:u_end]
            for q in range(p + 1, deg):
                v = nbrs[q]
                idx = np.searchsorted(u_nbrs, v)
                if idx < len(u_nbrs) and u_nbrs[idx] == v:
                    num_tri += 1
        
        keep_wedges = True
        if max_motifs > 0 and num_tri >= max_motifs:
            keep_wedges = False
            
        # Two passes to ensure sorting: Triangles (1) then Wedges (0)
        for is_tri_pass in (1, 0):
            if is_tri_pass == 0 and not keep_wedges:
                continue
            for p in range(deg):
                u = nbrs[p]
                u_start, u_end = indptr[u], indptr[u+1]
                u_nbrs = indices[u_start:u_end]
                for q in range(p + 1, deg):
                    v = nbrs[q]
                    idx = np.searchsorted(u_nbrs, v)
                    is_tri = (idx < len(u_nbrs) and u_nbrs[idx] == v)
                    
                    if (is_tri and is_tri_pass == 1) or (not is_tri and is_tri_pass == 0):
                        c3[curr] = i
                        u3[curr] = u
                        v3[curr] = v
                        tt[curr] = 1 if is_tri else 0
                        curr += 1
                        
    return c3, u3, v3, tt


@njit
def _numba_count_motifs(indptr, indices):
    """Count triangles and open wedges per node using CSR."""
    num_nodes = len(indptr) - 1
    counts = np.zeros((num_nodes, 2), dtype=np.float32)
    
    for i in range(num_nodes):
        start, end = indptr[i], indptr[i+1]
        deg = end - start
        if deg < 2:
            continue
            
        nbrs = indices[start:end]
        tri_count = 0
        wed_count = 0
        for p in range(deg):
            u = nbrs[p]
            u_start, u_end = indptr[u], indptr[u+1]
            u_nbrs = indices[u_start:u_end]
            for q in range(p + 1, deg):
                v = nbrs[q]
                idx = np.searchsorted(u_nbrs, v)
                if idx < len(u_nbrs) and u_nbrs[idx] == v:
                    tri_count += 1
                else:
                    wed_count += 1
        counts[i, 0] = wed_count
        counts[i, 1] = tri_count
    return counts


def _to_long_tensor(values):
    return torch.tensor(values, dtype=torch.long)


def _is_integer_tensor(tensor):
    return tensor.dtype in {
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
    }


def _update_cache_hash(hasher, value):
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        hasher.update(f"tensor:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        hasher.update(tensor.numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(f"ndarray:{array.shape}:{array.dtype}".encode())
        hasher.update(array.tobytes())
        return
    if isinstance(value, dict):
        hasher.update(f"dict:{len(value)}".encode())
        for key in sorted(value.keys(), key=lambda item: str(item)):
            hasher.update(f"key:{key}".encode())
            _update_cache_hash(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(f"{type(value).__name__}:{len(value)}".encode())
        for item in value:
            _update_cache_hash(hasher, item)
        return
    if hasattr(value, "__dict__"):
        _update_cache_hash(hasher, vars(value))
        return
    if value is None:
        hasher.update(b"None")
        return
    hasher.update(repr(value).encode())


def _graph_dataset_cache_fingerprint(dataset, name, max_motifs, pe_k, rwse_k):
    hasher = hashlib.md5()
    for part in (name, max_motifs, pe_k, rwse_k, len(dataset)):
        hasher.update(repr(part).encode())
        hasher.update(b"|")
    for index in range(len(dataset)):
        hasher.update(f"graph:{index}".encode())
        _update_cache_hash(hasher, dataset[index])
    return hasher.hexdigest()[:8]


def validate_get_batch(batch):
    """Raise a clear ValueError when a GET batch has shape or dtype mismatches."""
    required_tensors = ["x", "c_2", "u_2", "c_3", "u_3", "v_3", "t_tau", "batch", "ptr"]
    for name in required_tensors:
        if not hasattr(batch, name):
            raise ValueError(f"GET batch is missing required field '{name}'.")

    x = batch.x
    if not torch.is_tensor(x):
        raise ValueError("GET batch x must be a tensor.")
    if x.dim() != 2:
        raise ValueError(f"GET batch x must be 2D [num_nodes, feat_dim], got shape {tuple(x.shape)}.")
    if not x.dtype.is_floating_point:
        raise ValueError(f"GET batch x must be floating point, got {x.dtype}.")

    num_nodes = int(x.size(0))

    for name in ["c_2", "u_2", "c_3", "u_3", "v_3", "batch", "ptr", "t_tau"]:
        tensor = getattr(batch, name)
        if not torch.is_tensor(tensor):
            raise ValueError(f"GET batch field '{name}' must be a tensor.")
        if not _is_integer_tensor(tensor):
            raise ValueError(f"GET batch field '{name}' must be an integer tensor, got {tensor.dtype}.")

    if batch.batch.numel() != num_nodes:
        raise ValueError(
            f"GET batch.batch has {batch.batch.numel()} entries but x has {num_nodes} nodes."
        )
    if batch.ptr.numel() < 2:
        raise ValueError("GET batch.ptr must contain at least [0, num_nodes].")
    if int(batch.ptr[0].item()) != 0:
        raise ValueError("GET batch.ptr must start at 0.")
    if int(batch.ptr[-1].item()) != num_nodes:
        raise ValueError(
            f"GET batch.ptr ends at {int(batch.ptr[-1].item())}, but x has {num_nodes} nodes."
        )
    if bool((batch.ptr[1:] < batch.ptr[:-1]).any()):
        raise ValueError("GET batch.ptr must be nondecreasing.")

    if batch.c_2.numel() != batch.u_2.numel():
        raise ValueError("GET batch pairwise incidence tensors c_2 and u_2 must have the same length.")
    if batch.c_3.numel() != batch.u_3.numel() or batch.c_3.numel() != batch.v_3.numel():
        raise ValueError("GET batch motif incidence tensors c_3, u_3, and v_3 must have the same length.")
    if batch.t_tau.numel() != batch.c_3.numel():
        raise ValueError("GET batch t_tau length must match the number of motifs.")

    for name, tensor in [("c_2", batch.c_2), ("u_2", batch.u_2), ("c_3", batch.c_3), ("u_3", batch.u_3), ("v_3", batch.v_3)]:
        if tensor.numel() > 0:
            min_index = int(tensor.min().item())
            max_index = int(tensor.max().item())
            if min_index < 0 or max_index >= num_nodes:
                raise ValueError(
                    f"GET batch field '{name}' contains node indices outside [0, {num_nodes - 1}]."
                )

    if getattr(batch, "edge_attr", None) is not None:
        edge_attr = batch.edge_attr
        if not torch.is_tensor(edge_attr):
            raise ValueError("GET batch edge_attr must be a tensor when present.")
        if not edge_attr.dtype.is_floating_point:
            raise ValueError(f"GET batch edge_attr must be floating point, got {edge_attr.dtype}.")
        if edge_attr.size(0) != batch.c_2.numel():
            raise ValueError(
                f"GET batch edge_attr has {edge_attr.size(0)} rows but c_2 has {batch.c_2.numel()} incidences."
            )
        if edge_attr.dtype != x.dtype:
            raise ValueError(
                f"GET batch edge_attr dtype {edge_attr.dtype} does not match x dtype {x.dtype}."
            )

    for name in ("pe", "rwse"):
        tensor = getattr(batch, name, None)
        if tensor is None:
            continue
        if not torch.is_tensor(tensor):
            raise ValueError(f"GET batch field '{name}' must be a tensor when present.")
        if tensor.size(0) != num_nodes:
            raise ValueError(
                f"GET batch field '{name}' has {tensor.size(0)} rows but x has {num_nodes} nodes."
            )
        if tensor.dtype != x.dtype:
            raise ValueError(f"GET batch field '{name}' dtype {tensor.dtype} does not match x dtype {x.dtype}.")

    pe_cls = getattr(batch, "pe_cls", None)
    pe_cls_ptr = getattr(batch, "pe_cls_ptr", None)
    if pe_cls is not None:
        if not torch.is_tensor(pe_cls):
            raise ValueError("GET batch pe_cls must be a tensor when present.")
        if pe_cls.dtype != x.dtype:
            raise ValueError(f"GET batch pe_cls dtype {pe_cls.dtype} does not match x dtype {x.dtype}.")
        if pe_cls_ptr is not None:
            if not torch.is_tensor(pe_cls_ptr):
                raise ValueError("GET batch pe_cls_ptr must be a tensor when present.")
            if int(pe_cls_ptr[-1].item()) != pe_cls.size(0):
                raise ValueError(
                    f"GET batch pe_cls_ptr ends at {int(pe_cls_ptr[-1].item())}, but pe_cls has {pe_cls.size(0)} rows."
                )

    return batch


@njit
def _numba_build_sparse_laplacian(num_nodes, indptr, indices):
    """Build sparse symmetric normalized Laplacian in CSR format."""
    # L = I - D^-1/2 A D^-1/2
    deg = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        deg[i] = indptr[i+1] - indptr[i]
        
    inv_sqrt_deg = np.zeros(num_nodes, dtype=np.float32)
    for i in range(num_nodes):
        if deg[i] > 0:
            inv_sqrt_deg[i] = 1.0 / np.sqrt(deg[i])
            
    # Count non-zeros for L: entries of A + diagonals
    nnz_a = indptr[num_nodes]
    nnz_l = nnz_a + num_nodes
    
    l_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    l_indices = np.empty(nnz_l, dtype=np.int64)
    l_data = np.empty(nnz_l, dtype=np.float32)
    
    curr = 0
    for i in range(num_nodes):
        l_indptr[i] = curr
        # Diagonal entry: 1.0 (if deg > 0)
        l_indices[curr] = i
        l_data[curr] = 1.0 if deg[i] > 0 else 0.0
        curr += 1
        
        # Off-diagonal entries: - inv_sqrt_deg[i] * inv_sqrt_deg[j]
        for idx in range(indptr[i], indptr[i+1]):
            j = indices[idx]
            if i == j:
                continue
            l_indices[curr] = j
            l_data[curr] = - inv_sqrt_deg[i] * inv_sqrt_deg[j]
            curr += 1
    l_indptr[num_nodes] = curr
    return l_indptr, l_indices, l_data


@njit
def _numba_augment_laplacian_cls(num_nodes, indptr, indices):
    """
    Augment a sparse normalized Laplacian with a CLS node.
    The CLS node is fully connected to all original nodes.
    """
    # New size: N + 1
    # CLS node adds: N new directed edges + 1 self-loop
    # Each existing node adds: 1 directed edge to CLS
    num_new_nodes = num_nodes + 1
    cls_idx = num_nodes
    
    # Original degree in A (without CLS)
    orig_deg_a = np.diff(indptr)
    # New degree in A (with CLS and self-loops)
    new_deg_a = orig_deg_a + 1
    cls_deg_a = num_nodes # CLS connected to everyone
    
    inv_sqrt_deg = np.zeros(num_new_nodes, dtype=np.float32)
    for i in range(num_nodes):
        if new_deg_a[i] > 0:
            inv_sqrt_deg[i] = 1.0 / np.sqrt(new_deg_a[i])
    if cls_deg_a > 0:
        inv_sqrt_deg[cls_idx] = 1.0 / np.sqrt(cls_deg_a)
        
    # Count NNZ: original nnz + 2*N (cls edges) + (N+1) (diagonals)
    nnz_orig = indptr[num_nodes]
    nnz_new = nnz_orig + 2 * num_nodes + num_new_nodes
    
    l_indptr = np.zeros(num_new_nodes + 1, dtype=np.int64)
    l_indices = np.empty(nnz_new, dtype=np.int64)
    l_data = np.empty(nnz_new, dtype=np.float32)
    
    curr = 0
    # Original nodes
    for i in range(num_nodes):
        l_indptr[i] = curr
        # Diagonal
        l_indices[curr] = i
        l_data[curr] = 1.0 if new_deg_a[i] > 0 else 0.0
        curr += 1
        # Original edges scaled
        for idx in range(indptr[i], indptr[i+1]):
            j = indices[idx]
            if i == j:
                continue
            l_indices[curr] = j
            l_data[curr] = - inv_sqrt_deg[i] * inv_sqrt_deg[j]
            curr += 1
        # Edge to CLS
        l_indices[curr] = cls_idx
        l_data[curr] = - inv_sqrt_deg[i] * inv_sqrt_deg[cls_idx]
        curr += 1
        
    # CLS node
    l_indptr[cls_idx] = curr
    # Diagonal
    l_indices[curr] = cls_idx
    l_data[curr] = 1.0 if cls_deg_a > 0 else 0.0
    curr += 1
    # Edges from CLS
    for j in range(num_nodes):
        l_indices[curr] = j
        l_data[curr] = - inv_sqrt_deg[cls_idx] * inv_sqrt_deg[j]
        curr += 1
        
    l_indptr[num_new_nodes] = curr
    return l_indptr, l_indices, l_data


def get_laplacian_pe(num_nodes, indptr, indices, k=16):
    # Directly use passed CSR
    from .utils import laplacian_pe_from_adjacency
    return laplacian_pe_from_adjacency(num_nodes, indptr, indices, k=k, training=False)


def get_cls_augmented_laplacian_pe(num_nodes, indptr, indices, k=16):
    # Directly build the augmented sparse Laplacian from passed CSR
    l_indptr, l_indices, l_data = _numba_augment_laplacian_cls(num_nodes, indptr, indices)
    
    from .utils import laplacian_pe_from_sparse_matrix
    return laplacian_pe_from_sparse_matrix(num_nodes + 1, l_indptr, l_indices, l_data, k=k, training=False)


def get_rwse(num_nodes, indptr, indices, k=16):
    from .utils import rwse_from_adjacency
    # Use optimized RWSE directly from CSR
    return rwse_from_adjacency(num_nodes, indptr, indices, k=k)


def get_incidence_matrices(num_nodes, indptr, indices, max_motifs_per_node=None):
    """
    Extracts the sparse support for pairwise and motif interactions from CSR components.
    """
    # Pairwise: vectorized repeat for center indices
    c_2 = np.repeat(np.arange(num_nodes), np.diff(indptr))
    u_2 = indices
    
    # Motifs: Numba-accelerated extraction
    m_limit = int(max_motifs_per_node) if max_motifs_per_node is not None else -1
    c_3, u_3, v_3, t_tau = _numba_motif_extraction(indptr, indices, m_limit)

    return (
        _to_long_tensor(c_2),
        _to_long_tensor(u_2),
        _to_long_tensor(c_3),
        _to_long_tensor(u_3),
        _to_long_tensor(v_3),
        torch.as_tensor(t_tau, dtype=torch.int32),
    )


def _edge_source_to_tensor(edge_source, device=None):
    if torch.is_tensor(edge_source):
        edges_tensor = edge_source.to(device=device, dtype=torch.long)
        if edges_tensor.dim() != 2:
            raise ValueError("edge_index must be a 2D tensor with shape [2, E] or [E, 2].")
        if edges_tensor.size(0) == 2 and edges_tensor.size(1) != 2:
            edges_tensor = edges_tensor.t()
        elif edges_tensor.size(-1) != 2:
            raise ValueError("edge_index must have shape [2, E] or [E, 2].")
        return edges_tensor.contiguous()

    edges_tensor = torch.as_tensor(edge_source, dtype=torch.long, device=device)
    if edges_tensor.numel() == 0:
        return edges_tensor.reshape(0, 2)
    if edges_tensor.dim() == 2 and edges_tensor.size(-1) == 2:
        return edges_tensor.contiguous()
    if edges_tensor.dim() == 2 and edges_tensor.size(0) == 2:
        return edges_tensor.t().contiguous()
    return edges_tensor.reshape(-1, 2).contiguous()


def _graph_edge_source(graph):
    return graph['edge_index'] if 'edge_index' in graph else graph['edges']


def align_pairwise_edge_attr(edge_source, edge_attr, c_2, u_2):
    if edge_attr is None:
        return None
    if edge_attr.size(0) == c_2.numel():
        return edge_attr
    edges_tensor = _edge_source_to_tensor(edge_source, device=c_2.device)
    if edge_attr.size(0) != edges_tensor.size(0):
        raise ValueError(
            "edge_attr must have one row per undirected input edge or one row per "
            f"directed pairwise incidence; got {edge_attr.size(0)} rows for "
            f"{edges_tensor.size(0)} edges and {c_2.numel()} incidences."
        )

    if c_2.numel() == 0:
        return edge_attr.new_empty((0, *edge_attr.shape[1:]))

    # Vectorized max node ID and key generation
    max_node_id = max(
        int(edges_tensor.max().item()) if edges_tensor.numel() > 0 else -1,
        int(c_2.max().item()) if c_2.numel() > 0 else -1,
        int(u_2.max().item()) if u_2.numel() > 0 else -1
    )
    stride = max_node_id + 1

    # Vectorized key construction
    undirected_key = edges_tensor[:, 0] * stride + edges_tensor[:, 1]
    reverse_key = (undirected_key % stride) * stride + (undirected_key // stride)
    
    directed_key = torch.cat([undirected_key, reverse_key], dim=0)
    directed_attr = torch.cat([edge_attr, edge_attr], dim=0)

    sort_idx = torch.argsort(directed_key)
    sorted_key = directed_key[sort_idx]
    sorted_attr = directed_attr[sort_idx]

    query_key = c_2.to(dtype=torch.long) * stride + u_2.to(dtype=torch.long)
    pos = torch.searchsorted(sorted_key, query_key)
    
    # Validation
    safe_pos = pos.clamp(max=max(sorted_key.numel() - 1, 0))
    invalid = (pos >= sorted_key.numel()) | (sorted_key[safe_pos] != query_key)
    if bool(invalid.any()):
        missing_idx = int(torch.nonzero(invalid, as_tuple=False)[0].item())
        src = int(c_2[missing_idx].item())
        dst = int(u_2[missing_idx].item())
        raise ValueError(f"Missing edge_attr for directed edge ({src}, {dst}).")
    return sorted_attr[pos]


def add_structural_node_features(
    graph,
    include_degree=True,
    include_motif_counts=False,
    normalize=True,
):
    """Return a graph copy with cheap task-preparation features appended to x."""
    if not include_degree and not include_motif_counts:
        return dict(graph)

    num_nodes = graph["x"].size(0)
    edges_arr = np.ascontiguousarray(np.array(graph["edges"], dtype=np.int64).reshape(-1, 2))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    
    features = []
    if include_degree:
        degree = torch.from_numpy(np.diff(indptr)).to(dtype=graph["x"].dtype, device=graph["x"].device)
        features.append(degree.view(-1, 1))

    if include_motif_counts:
        counts = _numba_count_motifs(indptr, indices)
        motif_counts = torch.from_numpy(counts).to(dtype=graph["x"].dtype, device=graph["x"].device)
        features.append(motif_counts)

    structural = torch.cat(features, dim=-1).to(device=graph["x"].device)
    if normalize and structural.numel() > 0:
        scale = structural.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
        structural = structural / scale

    enriched = dict(graph)
    enriched["x"] = torch.cat([graph["x"], structural], dim=-1)
    return enriched


class CachedGraphDataset:
    """Wrapper to compute and cache motif incidence matrices to disk."""
    def __init__(self, dataset, cache_dir=".cache/get_data", name="dataset", max_motifs=None, pe_k=0, rwse_k=0):
        self.dataset = dataset
        self.cache_dir = cache_dir
        self.name = name
        self.max_motifs = max_motifs
        self.pe_k = pe_k
        self.rwse_k = rwse_k
        os.makedirs(cache_dir, exist_ok=True)
        
        hash_val = _graph_dataset_cache_fingerprint(dataset, name, max_motifs, pe_k, rwse_k)
        self.cache_path = os.path.join(cache_dir, f"{name}_{hash_val}.pt")
        
        if os.path.exists(self.cache_path):
            print(
                f"Loading cached dataset from {self.cache_path} "
                f"(max_motifs={self.max_motifs}, pe_k={self.pe_k}, rwse_k={self.rwse_k})"
            )
            self.cached_data = torch.load(self.cache_path, weights_only=False)
        else:
            print(
                f"Processing and caching dataset to {self.cache_path} "
                f"(max_motifs={self.max_motifs}, pe_k={self.pe_k}, rwse_k={self.rwse_k})..."
            )
            self.cached_data = self._process_all()
            torch.save(self.cached_data, self.cache_path)
            
    def _process_all(self):
        # Determine number of workers: scale with CPU count but cap to avoid memory pressure
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        
        num_workers = min(multiprocessing.cpu_count() or 4, 8)
        processed = []
        
        # Parallel processing for incidence matrices and PEs
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _process_one_graph, 
                    g, self.max_motifs, self.pe_k, self.rwse_k
                ) for g in self.dataset
            ]
            for future in tqdm(futures, desc="Caching incidence matrices (parallel)"):
                processed.append(future.result())
        return processed


def _process_one_graph(g, max_motifs, pe_k, rwse_k):
    """Worker function for parallel processing."""
    num_nodes = g['x'].size(0)
    
    # BUILD CSR ONCE
    edge_source = _graph_edge_source(g)
    edges_tensor = _edge_source_to_tensor(edge_source)
    edges_arr = np.ascontiguousarray(edges_tensor.detach().cpu().numpy().reshape(-1, 2))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    
    c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(num_nodes, indptr, indices, max_motifs)
    
    item = dict(g)
    item['c_2'] = c_2
    item['u_2'] = u_2
    item['c_3'] = c_3
    item['u_3'] = u_3
    item['v_3'] = v_3
    item['t_tau'] = t_tau
    if pe_k > 0:
        item['pe'] = get_laplacian_pe(num_nodes, indptr, indices, pe_k)
        item['pe_cls'] = get_cls_augmented_laplacian_pe(num_nodes, indptr, indices, pe_k)
    
    if rwse_k > 0:
        item['rwse'] = get_rwse(num_nodes, indptr, indices, rwse_k)
    
    if 'edge_attr' in g:
        item['aligned_edge_attr'] = align_pairwise_edge_attr(edge_source, g['edge_attr'], c_2, u_2)
    
    return item


class GETBatch:
    """A data class representing a batch of graphs for the GET model."""
    def __init__(self, x, c_2, u_2, c_3, u_3, v_3, t_tau, batch, ptr, y=None, edge_attr=None, pe=None, pe_cls=None, pe_cls_ptr=None, rwse=None):
        self.x = x
        self.c_2 = c_2
        self.u_2 = u_2
        self.c_3 = c_3
        self.u_3 = u_3
        self.v_3 = v_3
        self.t_tau = t_tau
        self.batch = batch
        self.ptr = ptr
        self.y = y
        self.edge_attr = edge_attr
        self.pe = pe
        self.pe_cls = pe_cls
        self.pe_cls_ptr = pe_cls_ptr
        self.rwse = rwse
        self.num_nodes = x.size(0)

    def to(self, device, non_blocking=False):
        self.x = self.x.to(device, non_blocking=non_blocking)
        self.c_2 = self.c_2.to(device, non_blocking=non_blocking)
        self.u_2 = self.u_2.to(device, non_blocking=non_blocking)
        self.c_3 = self.c_3.to(device, non_blocking=non_blocking)
        self.u_3 = self.u_3.to(device, non_blocking=non_blocking)
        self.v_3 = self.v_3.to(device, non_blocking=non_blocking)
        self.t_tau = self.t_tau.to(device, non_blocking=non_blocking)
        self.batch = self.batch.to(device, non_blocking=non_blocking)
        self.ptr = self.ptr.to(device, non_blocking=non_blocking)
        if self.y is not None:
            self.y = self.y.to(device, non_blocking=non_blocking)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device, non_blocking=non_blocking)
        if self.pe is not None:
            self.pe = self.pe.to(device, non_blocking=non_blocking)
        if self.pe_cls is not None:
            self.pe_cls = self.pe_cls.to(device, non_blocking=non_blocking)
        if self.pe_cls_ptr is not None:
            self.pe_cls_ptr = self.pe_cls_ptr.to(device, non_blocking=non_blocking)
        if self.rwse is not None:
            self.rwse = self.rwse.to(device, non_blocking=non_blocking)
        return self


def collate_get_batch(graph_list, max_motifs=None):
    """
    Collate a list of graphs into a single large disconnected graph (PyG style).
    Uses pre-allocated tensors for efficiency.
    """
    if len(graph_list) == 0:
        raise ValueError("graph_list must contain at least one graph.")

    # 1. Pre-calculate total sizes
    total_nodes = 0
    total_edges_pair = 0
    total_motifs = 0
    total_pe_cls_nodes = 0
    has_y = 'y' in graph_list[0]
    has_edge_attr = 'aligned_edge_attr' in graph_list[0] or 'edge_attr' in graph_list[0]
    has_pe = 'pe' in graph_list[0]
    has_pe_cls = 'pe_cls' in graph_list[0]
    has_rwse = 'rwse' in graph_list[0]
    
    for g in graph_list:
        num_nodes = g['x'].size(0)
        total_nodes += num_nodes
        
        if 'c_2' in g:
            total_edges_pair += g['c_2'].numel()
            total_motifs += g['c_3'].numel()
        else:
            # Incidence matrices not cached, will compute on the fly
            # This is slow, but we need the counts for pre-allocation
            num_n = g['x'].size(0)
            edge_source = _graph_edge_source(g)
            e_arr = np.ascontiguousarray(_edge_source_to_tensor(edge_source).detach().cpu().numpy().reshape(-1, 2))
            iptr, idc = _numba_edges_to_csr(num_n, e_arr)
            c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(num_n, iptr, idc, max_motifs)
            g['c_2'], g['u_2'], g['c_3'], g['u_3'], g['v_3'], g['t_tau'] = c_2, u_2, c_3, u_3, v_3, t_tau
            total_edges_pair += c_2.numel()
            total_motifs += c_3.numel()
            
        if has_pe_cls:
            total_pe_cls_nodes += g['pe_cls'].size(0)

    # 2. Pre-allocate tensors
    device = graph_list[0]['x'].device
    x_dtype = graph_list[0]['x'].dtype
    
    x = torch.empty((total_nodes, graph_list[0]['x'].size(1)), dtype=x_dtype, device=device)
    c_2 = torch.empty(total_edges_pair, dtype=torch.long, device=device)
    u_2 = torch.empty(total_edges_pair, dtype=torch.long, device=device)
    c_3 = torch.empty(total_motifs, dtype=torch.long, device=device)
    u_3 = torch.empty(total_motifs, dtype=torch.long, device=device)
    v_3 = torch.empty(total_motifs, dtype=torch.long, device=device)
    t_tau = torch.empty(total_motifs, dtype=torch.int32, device=device)
    batch = torch.empty(total_nodes, dtype=torch.long, device=device)
    ptr = torch.empty(len(graph_list) + 1, dtype=torch.long, device=device)
    ptr[0] = 0
    
    y = torch.empty(len(graph_list), dtype=torch.float32, device=device) if has_y else None
    
    edge_attr = None
    if has_edge_attr:
        attr_sample = graph_list[0].get('aligned_edge_attr', graph_list[0].get('edge_attr'))
        edge_attr = torch.empty((total_edges_pair, attr_sample.size(1)), dtype=x_dtype, device=device)
        
    pe = torch.empty((total_nodes, graph_list[0]['pe'].size(1)), dtype=x_dtype, device=device) if has_pe else None
    
    pe_cls = None
    pe_cls_ptr = None
    if has_pe_cls:
        pe_cls = torch.empty((total_pe_cls_nodes, graph_list[0]['pe_cls'].size(1)), dtype=x_dtype, device=device)
        pe_cls_ptr = torch.empty(len(graph_list) + 1, dtype=torch.long, device=device)
        pe_cls_ptr[0] = 0
        
    rwse = torch.empty((total_nodes, graph_list[0]['rwse'].size(1)), dtype=x_dtype, device=device) if has_rwse else None

    # 3. Fill tensors
    n_curr, e_curr, m_curr, p_cls_curr = 0, 0, 0, 0
    
    for g_idx, g in enumerate(graph_list):
        num_nodes = g['x'].size(0)
        n_end = n_curr + num_nodes
        
        x[n_curr:n_end] = g['x']
        batch[n_curr:n_end] = g_idx
        ptr[g_idx + 1] = n_end
        
        c2, u2 = g['c_2'], g['u_2']
        e_cnt = c2.numel()
        e_end = e_curr + e_cnt
        c_2[e_curr:e_end] = c2 + n_curr
        u_2[e_curr:e_end] = u2 + n_curr
        
        c3, u3, v3, tt = g['c_3'], g['u_3'], g['v_3'], g['t_tau']
        m_cnt = c3.numel()
        m_end = m_curr + m_cnt
        c_3[m_curr:m_end] = c3 + n_curr
        u_3[m_curr:m_end] = u3 + n_curr
        v_3[m_curr:m_end] = v3 + n_curr
        t_tau[m_curr:m_end] = tt
        
        if has_y:
            y[g_idx] = torch.as_tensor(g['y']).view(-1)[0]
            
        if has_edge_attr:
            curr_attr = g.get('aligned_edge_attr')
            if curr_attr is None:
                curr_attr = align_pairwise_edge_attr(_graph_edge_source(g), g['edge_attr'], c2, u2)
            edge_attr[e_curr:e_end] = curr_attr
            
        if has_pe:
            pe[n_curr:n_end] = g['pe']
            
        if has_pe_cls:
            curr_pe_cls = g['pe_cls']
            p_cls_cnt = curr_pe_cls.size(0)
            p_cls_end = p_cls_curr + p_cls_cnt
            pe_cls[p_cls_curr:p_cls_end] = curr_pe_cls
            pe_cls_ptr[g_idx + 1] = p_cls_end
            p_cls_curr = p_cls_end
            
        if has_rwse:
            rwse[n_curr:n_end] = g['rwse']
            
        n_curr = n_end
        e_curr = e_end
        m_curr = m_end
        
    return GETBatch(x, c_2, u_2, c_3, u_3, v_3, t_tau, batch, ptr, y, edge_attr, pe=pe, pe_cls=pe_cls, pe_cls_ptr=pe_cls_ptr, rwse=rwse)
