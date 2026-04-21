import torch


def _build_undirected_adjacency(num_nodes, edges_list):
    adj = [set() for _ in range(num_nodes)]
    for u, v in edges_list:
        u_i = int(u)
        v_i = int(v)
        adj[u_i].add(v_i)
        adj[v_i].add(u_i)
    return adj


def _extract_pairwise_indices(adj):
    c_2 = []
    u_2 = []
    for center, neighbors in enumerate(adj):
        for nbr in sorted(neighbors):
            c_2.append(center)
            u_2.append(nbr)
    return c_2, u_2


def _apply_motif_budget_with_ties(node_motifs, max_motifs_per_node):
    if max_motifs_per_node is None or len(node_motifs) <= max_motifs_per_node:
        return node_motifs
    if max_motifs_per_node <= 0:
        return []
    threshold_score = node_motifs[max_motifs_per_node - 1][3]
    return [m for m in node_motifs if m[3] >= threshold_score]


def _extract_motif_indices(adj, max_motifs_per_node=None):
    c_3 = []
    u_3 = []
    v_3 = []
    t_tau = []
    num_nodes = len(adj)
    adj_dense = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    for src, neighbors in enumerate(adj):
        if neighbors:
            nbr_idx = torch.tensor(sorted(neighbors), dtype=torch.long)
            adj_dense[src, nbr_idx] = True

    for center, neighbors_set in enumerate(adj):
        neighbors = sorted(neighbors_set)
        if len(neighbors) < 2:
            continue
        neighbors_t = torch.tensor(neighbors, dtype=torch.long)
        pair_idx = torch.triu_indices(neighbors_t.numel(), neighbors_t.numel(), offset=1)
        left = neighbors_t[pair_idx[0]]
        right = neighbors_t[pair_idx[1]]
        closed = adj_dense[left, right].to(dtype=torch.long)

        node_motifs = list(
            zip(
                [center] * left.numel(),
                left.tolist(),
                right.tolist(),
                closed.tolist(),
            )
        )

        node_motifs.sort(key=lambda x: (-x[3], x[1], x[2]))
        node_motifs = _apply_motif_budget_with_ties(node_motifs, max_motifs_per_node)

        for center_i, left_i, right_i, motif_type in node_motifs:
            c_3.append(center_i)
            u_3.append(left_i)
            v_3.append(right_i)
            t_tau.append(motif_type)

    return c_3, u_3, v_3, t_tau


def _to_long_tensor(values):
    return torch.tensor(values, dtype=torch.long)

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
    adj = _build_undirected_adjacency(num_nodes, edges_list)
    c_2, u_2 = _extract_pairwise_indices(adj)
    c_3, u_3, v_3, t_tau = _extract_motif_indices(adj, max_motifs_per_node=max_motifs_per_node)

    return (
        _to_long_tensor(c_2),
        _to_long_tensor(u_2),
        _to_long_tensor(c_3),
        _to_long_tensor(u_3),
        _to_long_tensor(v_3),
        _to_long_tensor(t_tau),
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

    max_node_id = -1
    for src, dst in edges_list:
        src_i = int(src)
        dst_i = int(dst)
        if src_i > max_node_id:
            max_node_id = src_i
        if dst_i > max_node_id:
            max_node_id = dst_i
    if c_2.numel() > 0:
        max_node_id = max(max_node_id, int(torch.max(c_2).item()), int(torch.max(u_2).item()))
    stride = max_node_id + 1

    undirected_key = torch.empty((len(edges_list),), dtype=torch.long)
    for idx, (src, dst) in enumerate(edges_list):
        src_i = int(src)
        dst_i = int(dst)
        undirected_key[idx] = src_i * stride + dst_i
    directed_key = torch.cat([undirected_key, (undirected_key % stride) * stride + (undirected_key // stride)], dim=0)
    directed_attr = torch.cat([edge_attr, edge_attr], dim=0)

    sort_idx = torch.argsort(directed_key)
    sorted_key = directed_key[sort_idx]
    sorted_attr = directed_attr[sort_idx]

    query_key = c_2.to(dtype=torch.long) * stride + u_2.to(dtype=torch.long)
    pos = torch.searchsorted(sorted_key, query_key)
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
    adj = _build_undirected_adjacency(num_nodes, graph["edges"])

    features = []
    if include_degree:
        degree = torch.tensor(
            [len(adj[i]) for i in range(num_nodes)],
            dtype=graph["x"].dtype,
            device=graph["x"].device,
        )
        features.append(degree.view(-1, 1))

    if include_motif_counts:
        open_wedges = []
        triangles = []
        for i in range(num_nodes):
            open_count = 0
            tri_count = 0
            neighbors = sorted(adj[i])
            for pos, j in enumerate(neighbors):
                for k in neighbors[pos + 1 :]:
                    if k in adj[j]:
                        tri_count += 1
                    else:
                        open_count += 1
            open_wedges.append(open_count)
            triangles.append(tri_count)
        motif_counts = torch.tensor(
            list(zip(open_wedges, triangles)),
            dtype=graph["x"].dtype,
            device=graph["x"].device,
        )
        features.append(motif_counts)

    structural = torch.cat(features, dim=-1).to(device=graph["x"].device)
    if normalize and structural.numel() > 0:
        scale = structural.abs().amax(dim=0, keepdim=True).clamp_min(1.0)
        structural = structural / scale

    enriched = dict(graph)
    enriched["x"] = torch.cat([graph["x"], structural], dim=-1)
    return enriched


class GETBatch:
    """A data class representing a batch of graphs for the GET model."""
    def __init__(self, x, c_2, u_2, c_3, u_3, v_3, t_tau, batch, ptr, y=None, edge_attr=None):
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
    
    node_offset = 0
    
    for g_idx, g in enumerate(graph_list):
        num_nodes = g['x'].size(0)
        x_list.append(g['x'])
        
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
            y_list.append(g['y'])
        if 'edge_attr' in g:
            edge_attr_list.append(align_pairwise_edge_attr(g['edges'], g['edge_attr'], c_2, u_2))
            
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
    
    return GETBatch(x, c_2, u_2, c_3, u_3, v_3, t_tau, batch, ptr, y, edge_attr)
