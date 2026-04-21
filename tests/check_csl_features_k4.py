import networkx as nx
import numpy as np
def generate_csl(n=41, k=2):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
        G.add_edge(i, (i + k) % n)
    return G

for k_class in [3, 4]:
    G = generate_csl(41, k_class)
    A = nx.to_numpy_array(G)
    A_k = [A]
    for _ in range(1, 5):
        A_k.append(A_k[-1] @ A)
    edges = list(G.edges(0))
    print(f"--- k={k_class} ---")
    for u, v in edges:
        feats = [A_k[k][u, v] for k in range(5)]
        print(f"Edge (0, {v}): {feats}")
