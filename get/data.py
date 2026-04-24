import torch
import os
import hashlib
import numpy as np
from numba import njit
from tqdm.auto import tqdm


@njit
def _numba_edges_to_csr(num_nodes, edges_arr):
    """Directly build CSR from edge array in Numba."""
    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    for i in range(edges_arr.shape[0]):
        u, v = edges_arr[i, 0], edges_arr[i, 1]
        if u == v or u >= num_nodes or v >= num_nodes:
            continue
        indptr[u + 1] += 1
        indptr[v + 1] += 1
    
    for i in range(num_nodes):
        indptr[i + 1] += indptr[i]
        
    indices = np.empty(indptr[num_nodes], dtype=np.int64)
    curr_idx = indptr[:-1].copy()
    for i in range(edges_arr.shape[0]):
        u, v = edges_arr[i, 0], edges_arr[i, 1]
        if u == v or u >= num_nodes or v >= num_nodes:
            continue
        indices[curr_idx[u]] = v
        curr_idx[u] += 1
        indices[curr_idx[v]] = u
        curr_idx[v] += 1
        
    for i in range(num_nodes):
        indices[indptr[i]:indptr[i+1]].sort()
        
    return indptr, indices


@njit
def _numba_motif_extraction(indptr, indices, max_motifs):
    """Numba-accelerated motif extraction with budget logic."""
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


def get_laplacian_pe(num_nodes, edges_list, k=16):
    edges_arr = np.ascontiguousarray(np.array(edges_list, dtype=np.int64))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    
    from .utils import laplacian_pe_from_adjacency
    return laplacian_pe_from_adjacency(num_nodes, indptr, indices, k=k, training=False)


def get_cls_augmented_laplacian_pe(num_nodes, edges_list, k=16):
    edges_arr = np.ascontiguousarray(np.array(edges_list, dtype=np.int64))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    
    # Directly build the augmented sparse Laplacian
    l_indptr, l_indices, l_data = _numba_augment_laplacian_cls(num_nodes, indptr, indices)
    
    from .utils import laplacian_pe_from_sparse_matrix
    return laplacian_pe_from_sparse_matrix(num_nodes + 1, l_indptr, l_indices, l_data, k=k, training=False)


def get_rwse(num_nodes, edges_list, k=16):
    edges_arr = np.ascontiguousarray(np.array(edges_list, dtype=np.int64))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    
    from .utils import rwse_from_adjacency
    # Use optimized RWSE directly from CSR
    return rwse_from_adjacency(num_nodes, indptr, indices, k=k)


def get_incidence_matrices(num_nodes, edges_list, max_motifs_per_node=None):
    """
    Extracts the sparse support for pairwise and motif interactions from an edge list.
    
    Args:
        num_nodes: Total number of nodes in the graph
        edges_list: List of (u, v) pairs representing undirected edges
        max_motifs_per_node: Maximum number of anchored motifs B to retain per node.
        
    Returns:
        c_2, u_2: Pairwise center and neighbor indices (1D tensors)
        c_3, u_3, v_3: Motif center, first neighbor, and second neighbor indices (1D tensors)
        t_tau: Motif types (0 for open wedge, 1 for closed triangle) (1D tensor)
    """
    edges_arr = np.ascontiguousarray(np.array(edges_list, dtype=np.int64))
    indptr, indices = _numba_edges_to_csr(num_nodes, edges_arr)
    
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


def align_pairwise_edge_attr(edges_list, edge_attr, c_2, u_2):
    if edge_attr is None:
        return None
    if edge_attr.size(0) == c_2.numel():
        return edge_attr
    if edge_attr.size(0) != len(edges_list):
        raise ValueError(
            "edge_attr must have one row per undirected input edge or one row per "
            f"directed pairwise incidence; got {edge_attr.size(0)} rows for "
            f"{len(edges_list)} edges and {c_2.numel()} incidences."
        )

    if c_2.numel() == 0:
        return edge_attr.new_empty((0, *edge_attr.shape[1:]))

    # Vectorized max node ID and key generation
    edges_tensor = torch.as_tensor(edges_list, dtype=torch.long, device=c_2.device)
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
    edges_arr = np.ascontiguousarray(np.array(graph["edges"], dtype=np.int64))
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
        
        sample_str = str([g.get('edges', [])[:5] for g in dataset[:5]])
        hash_val = hashlib.md5((name + str(max_motifs) + str(pe_k) + str(rwse_k) + sample_str + str(len(dataset))).encode()).hexdigest()[:8]
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
        processed = []
        for g in tqdm(self.dataset, desc="Caching incidence matrices"):
            num_nodes = g['x'].size(0)
            c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(num_nodes, g['edges'], self.max_motifs)
            
            item = dict(g)
            item['c_2'] = c_2
            item['u_2'] = u_2
            item['c_3'] = c_3
            item['u_3'] = u_3
            item['v_3'] = v_3
            item['t_tau'] = t_tau
            if self.pe_k > 0:
                item['pe'] = get_laplacian_pe(num_nodes, g['edges'], self.pe_k)
                item['pe_cls'] = get_cls_augmented_laplacian_pe(num_nodes, g['edges'], self.pe_k)
            
            if self.rwse_k > 0:
                item['rwse'] = get_rwse(num_nodes, g['edges'], self.rwse_k)
            
            if 'edge_attr' in g:
                item['aligned_edge_attr'] = align_pairwise_edge_attr(g['edges'], g['edge_attr'], c_2, u_2)
            
            processed.append(item)
        return processed
        
    def __len__(self):
        return len(self.cached_data)
        
    def __getitem__(self, idx):
        return self.cached_data[idx]


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
    Each graph in graph_list is a dict: {'x': tensor, 'edges': list_of_tuples, 'y': tensor(optional)}
    """
    if len(graph_list) == 0:
        raise ValueError("graph_list must contain at least one graph.")

    x_list = []
    c_2_list, u_2_list = [], []
    c_3_list, u_3_list, v_3_list, t_tau_list = [], [], [], []
    batch_list = []
    ptr_list = [0]
    y_list = []
    edge_attr_list = []
    pe_list = []
    pe_cls_list = []
    pe_cls_ptr_list = [0]
    rwse_list = []
    
    node_offset = 0
    
    for g_idx, g in enumerate(graph_list):
        num_nodes = g['x'].size(0)
        x_list.append(g['x'])
        
        if 'c_2' in g:
            c_2, u_2 = g['c_2'], g['u_2']
            c_3, u_3, v_3, t_tau = g['c_3'], g['u_3'], g['v_3'], g['t_tau']
        else:
            c_2, u_2, c_3, u_3, v_3, t_tau = get_incidence_matrices(num_nodes, g['edges'], max_motifs)
        
        c_2_list.append(c_2 + node_offset)
        u_2_list.append(u_2 + node_offset)
        c_3_list.append(c_3 + node_offset)
        u_3_list.append(u_3 + node_offset)
        v_3_list.append(v_3 + node_offset)
        t_tau_list.append(t_tau)
        
        batch_list.append(torch.full((num_nodes,), g_idx, dtype=torch.long))
        ptr_list.append(ptr_list[-1] + num_nodes)
        
        if 'y' in g:
            # Canonicalize graph labels so mixed scalar vs [1]-shaped values
            # can be batched together without cat/stack shape errors.
            y = torch.as_tensor(g['y']).reshape(-1)
            if y.numel() != 1:
                raise ValueError(
                    "Expected one graph label per sample; "
                    f"got shape {tuple(torch.as_tensor(g['y']).shape)}."
                )
            y_list.append(y)
        if 'aligned_edge_attr' in g:
            edge_attr_list.append(g['aligned_edge_attr'])
        elif 'edge_attr' in g:
            edge_attr_list.append(align_pairwise_edge_attr(g['edges'], g['edge_attr'], c_2, u_2))
        
        if 'pe' in g:
            pe_list.append(g['pe'])
        if 'pe_cls' in g:
            pe_cls_list.append(g['pe_cls'])
            pe_cls_ptr_list.append(pe_cls_ptr_list[-1] + g['pe_cls'].size(0))
        
        if 'rwse' in g:
            rwse_list.append(g['rwse'])
            
        node_offset += num_nodes
        
    x = torch.cat(x_list, dim=0)
    c_2 = torch.cat(c_2_list, dim=0) if c_2_list else torch.empty(0, dtype=torch.long)
    u_2 = torch.cat(u_2_list, dim=0) if u_2_list else torch.empty(0, dtype=torch.long)
    c_3 = torch.cat(c_3_list, dim=0) if c_3_list else torch.empty(0, dtype=torch.long)
    u_3 = torch.cat(u_3_list, dim=0) if u_3_list else torch.empty(0, dtype=torch.long)
    v_3 = torch.cat(v_3_list, dim=0) if v_3_list else torch.empty(0, dtype=torch.long)
    t_tau = torch.cat(t_tau_list, dim=0) if t_tau_list else torch.empty(0, dtype=torch.long)
    batch = torch.cat(batch_list, dim=0)
    ptr = torch.tensor(ptr_list, dtype=torch.long)
    y = torch.cat(y_list, dim=0) if y_list else None
    edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
    pe = torch.cat(pe_list, dim=0) if pe_list else None
    pe_cls = torch.cat(pe_cls_list, dim=0) if pe_cls_list else None
    pe_cls_ptr = torch.tensor(pe_cls_ptr_list, dtype=torch.long) if pe_cls_list else None
    rwse = torch.cat(rwse_list, dim=0) if rwse_list else None
    
    return GETBatch(x, c_2, u_2, c_3, u_3, v_3, t_tau, batch, ptr, y, edge_attr, pe=pe, pe_cls=pe_cls, pe_cls_ptr=pe_cls_ptr, rwse=rwse)
