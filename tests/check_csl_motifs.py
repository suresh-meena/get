import networkx as nx
def generate_csl(n=41, k=2):
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        G.add_edge(i, (i + 1) % n)
        G.add_edge(i, (i + k) % n)
    return G

n = 41
ks = [2, 3, 4, 5, 6, 9, 11, 12, 13, 16]
for k in ks:
    G = generate_csl(n, k)
    triangles = sum(nx.triangles(G).values()) // 3
    # open wedges = total wedges - 3 * triangles
    total_wedges = sum(d * (d - 1) // 2 for d in dict(G.degree()).values())
    open_wedges = total_wedges - 3 * triangles
    print(f"k={k}: Triangles={triangles}, Open Wedges={open_wedges}")
