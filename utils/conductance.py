import networkx as nx

def conductance(G, S):
    """Calculate the conductance of a cut S in graph G for directed, weighted graphs."""
    S = set(S)
    T = set(G.nodes()) - S

    # Calculate cut size (total weight of edges crossing the cut)
    cut_S = sum(G[u][v]['weight'] for u, v in nx.edge_boundary(G, S, T))
    cut_T = sum(G[u][v]['weight'] for u, v in nx.edge_boundary(G, T, S))
    cut_size = min(cut_S, cut_T)

    # Calculate volume (total weight of edges going out of each part)
    volume_S = sum(G[u][v]['weight'] for u in S for v in G[u] if v in S) + cut_S + cut_T
    volume_T = sum(G[u][v]['weight'] for u in T for v in G[u] if v in T) + cut_S + cut_T

    volume = min(volume_S, volume_T)

    if cut_size == 0:
        return 0
    return cut_size / volume