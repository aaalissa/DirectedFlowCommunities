import sys
sys.path.append("..")

from utils.dyetracing_classes import *
import networkx as nx
import numpy as np
import pandas as pd
import csv


def setup_dyetracing_graph(file_name):
    """
    Creates a list of Nodes and Edges from a file in the format:
        start_node end_node weight
    """
    nodes_dict = {}
    edge_list = []

    with open(file_name, 'r') as f:
        for row in f:
            parts = row.strip().split()
            start, end = map(int, parts[:2])
 
            # Check if weight is provided
            weight = float(parts[2]) if len(parts) == 3 else None

            # Create Node instances if they don't exist
            if start not in nodes_dict:
                nodes_dict[start] = Node(name=start)
            if end not in nodes_dict:
                nodes_dict[end] = Node(name=end)

            # If self loop, skip it
            if start == end:
                continue

            #if weight is 0, skip it
            if weight == 0:
                continue

            # Create an Edge instance between start and end nodes
            edge = Edge(start_node=nodes_dict[start], end_node=nodes_dict[end]) #using default values for length and velocity
            if weight is not None:
                edge.u = float(weight)

            edge_list.append(edge)

    # Sort nodes by name
    sorted_dict = dict(sorted(nodes_dict.items(), key=lambda item: item[0]))
    node_list = (list(sorted_dict.values()))
    set_node_index(node_list)

    return node_list, edge_list

def setup_dyetracing_graph_from_csr(sparse_matrix):
    """
    Creates a list of Nodes and Edges from a file in the format:
        start_node end_node weight
    """
    nodes_dict = {}
    edge_list = []

    for (start, end), weight in np.ndenumerate(sparse_matrix.toarray()):
        # Create Node instances if they don't exist
        if start not in nodes_dict:
            nodes_dict[start] = Node(name=start)
        if end not in nodes_dict:
            nodes_dict[end] = Node(name=end)

        # If self loop, skip it
        if start == end:
            continue

        # If weight is 0, skip it
        if weight == 0:
            continue

        # Create an Edge instance between start and end nodes
        edge = Edge(start_node=nodes_dict[start], end_node=nodes_dict[end]) #using default values for length and velocity
        if weight is not None:
            edge.u = weight
        
        edge_list.append(edge)

    # Sort nodes by name
    sorted_dict = dict(sorted(nodes_dict.items(), key=lambda item: item[0]))
    node_list = (list(sorted_dict.values()))
    set_node_index(node_list)

    return node_list, edge_list

def set_node_index(node_list):
    """
    Sets the index of each node in the node_list to be its position in the list.
    """
    for i, node in enumerate(node_list):
        node.index = i

def read_communities(file_name):
    """
    Creates community membership in the form of a dictionary from a file in the format:
        node_id community_id
    """

    labels_dict = {}
    with open(file_name, 'r') as f:
        for row in f:
            parts = row.strip().split()
            node, membership = map(int, parts[:2])
            labels_dict[node] = membership

    return labels_dict

def load_nx_graph(node_list, network_file):
    # Initialize a directed graph
    G_dir = nx.DiGraph()

    # Add nodes to the graph
    for node in node_list:
        G_dir.add_node(node.name)

    # Read the network file and add edges
    with open(network_file, 'r') as f:
        for row in f:
            # Assuming the edge pairs are space-separated
            parts = row.strip().split()
            if len(parts) < 2:
                continue  # Skip lines with insufficient data
            start, end = map(int, parts[:2])
            G_dir.add_edge(start, end)

            # Check for edge weight
            weight = float(parts[2]) if len(parts) == 3 else None        

            # Add edge weight if available
            if weight is not None:
                G_dir[start][end]['weight'] = weight

    return G_dir

def faf_to_nx_digraph(edge_csv, coord_csv):
    G = nx.DiGraph()

    # Read coordinates and add nodes
    with open(coord_csv, newline='', encoding='utf-8-sig') as coord_file:
        skips = [20, 151, 159]  # skip Alaska, Honolulu, and Hawaii
        coords_csv = csv.reader(coord_file, delimiter=',')

        for coord in coords_csv:
            location_code = int(coord[3])  # each location has an integer code
            if location_code in skips:
                continue  # skip some locations
            location = coord[0]  # name of location
            lon = float(coord[1])  # longitude
            lat = float(coord[2])  # latitude

            # add node with location code and attributes
            G.add_node(location_code, location_name=location, position=(lat, lon))

    # Read edge data and add edges
    flows = np.genfromtxt(edge_csv, delimiter=',', filling_values=0)  # get numbers from CSV
    n_largest = 1200  # adjust this number as needed

    # Assuming region_mask and maxflow_i are defined elsewhere
    inter_region_flows = flows * region_mask(flows)  # flows between regions
    max_inter_region = maxflow_i(inter_region_flows, n=n_largest)
    sorted_inter_maxes = sorted(tuple(zip(*max_inter_region)), key=lambda x: -flows[x[0], x[1]])

    for i, (i_source, i_dest) in enumerate(sorted_inter_maxes):
        code_source = int(flows[i_source, 0])
        code_dest = int(flows[0, i_dest])
        G.add_edge(code_source, code_dest, weight=flows[i_source, i_dest])

    return G