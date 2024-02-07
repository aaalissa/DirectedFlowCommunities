import sys
sys.path.append("..")
from utils.dyetracing_classes import *

import networkx as nx
from cdlib import algorithms

def create_nx_graph(network_file, node_list):
    """
    Creates a networkx graph from a network file and a list of nodes.
    input: 
        network_file: path to network file in the format: start_node end_node weight
        node_list: list of Node objects
    output:
        G: undirected networkx graph (for visualization)
        G_di: directed networkx graph (for other community detection algorithms)
    """

    #create networkx for viz
    G = nx.Graph()

    for node in node_list:
        G.add_node(node.name)
    with open(network_file, 'r') as f:
        for row in f:
            # Assuming the edge pairs are space-separated
            parts = row.strip().split()
            if len(parts) < 2:
                continue  # Skip lines with insufficient data
            start, end = map(int, parts[:2])
            G.add_edge(start, end)

            weight = float(parts[2]) if len(parts) == 3 else None        

            #add edge weight
            if weight is not None:
                G[start][end]['weight'] = weight

    #directed graph for other community detection algos
    G_di = nx.DiGraph() 
    for node in node_list:
        G_di.add_node(node.name)
    with open(network_file, 'r') as f:
        for row in f:
            # Assuming the edge pairs are space-separated
            parts = row.strip().split()
            if len(parts) < 2:
                continue  # Skip lines with insufficient data
            start, end = map(int, parts[:2])
            G_di.add_edge(start, end)

            weight = float(parts[2]) if len(parts) == 3 else None        

            #add edge weight
            if weight is not None:
                G_di[start][end]['weight'] = weight
    
    return G, G_di

def run_benchmark_community_detection(G, G_di, true_memberships):
    """
    Runs community detection algorithms on a networkx graph and returns the results
    input:
        G: undirected networkx graph
        G_di: directed networkx graph
        true_memberships: list of true community memberships
    output:
        results: dictionary of results
    """
    results = {}
    #run baseline algorithms
    infomap_comms = algorithms.infomap(G_di)
    infomap_labels = np.array([community_label for community_label, community in enumerate((infomap_comms.communities)) for node in community])
    rb_pots_comms = algorithms.rb_pots(G_di, weights = 'weight')
    rb_pots_labels = np.array([community_label for community_label, community in enumerate((rb_pots_comms.communities)) for node in community])
    louvain_comms = algorithms.louvain(G)
    louvain_labels = np.array([community_label for community_label, community in enumerate((louvain_comms.communities)) for node in community])

    #evaluate results
    accuracy, ari, nmi = evaluate_clustering(true_memberships, infomap_labels)
    results['infomap'] = {'accuracy': accuracy, 'ari': ari, 'nmi': nmi}

    accuracy, ari, nmi = evaluate_clustering(true_memberships, rb_pots_labels)
    results['rb_pots'] = {'accuracy': accuracy, 'ari': ari, 'nmi': nmi}

    accuracy, ari, nmi = evaluate_clustering(true_memberships, louvain_labels)
    results['louvain'] = {'accuracy': accuracy, 'ari': ari, 'nmi': nmi}

    #need to add motif clustering results as well

    return results