import sys
sys.path.append("..")

import csv
import networkx as nx
import numpy as np
import pandas as pd

from utils.faf_utils import region_mask, maxflow_i
from utils.dyetracing_classes import * 
from utils.network_loader import set_node_index


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

    # Iterate over the matrix and add edges with non-zero weights, skipping self-loops
    num_nodes = flows.shape[0]
    for i in range(1, num_nodes):
        for j in range(1, num_nodes):
            if i == j:  # Skip self-loops
                continue
            weight = flows[i, j]
            if weight != 0:
                code_source = int(flows[i, 0])
                code_dest = int(flows[0, j])
                G.add_edge(code_source, code_dest, weight=weight)

    return G

def haversine_np(coord1, coord2):
    R = 6371  # Earth radius in kilometers

    lat1, lon1 = np.radians(coord1)
    lat2, lon2 = np.radians(coord2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def add_distances_to_edges(G):
    for u, v in G.edges():
        if 'position' in G.nodes[u] and 'position' in G.nodes[v]:
            coord1 = G.nodes[u]['position']
            coord2 = G.nodes[v]['position']
            distance = haversine_np(coord1, coord2)
            G[u][v]['distance'] = distance


def adjacency_matrix_csv_to_edge_list(csv_file_path, output_file_path, skip_nodes = [20, 151, 159]):
    if skip_nodes is None:
        skip_nodes = []

    with open(csv_file_path, 'r', newline='', encoding='utf-8-sig') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)[1:]  # Skip the first cell and read the headers (node names)

        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for i, row in enumerate(reader):
                start_node = row[0]  # The start node is in the first column

                # Skip this node if it's in the skip list
                if start_node in skip_nodes:
                    continue

                for j, weight in enumerate(row[1:]):  # Skip the first column
                    end_node = headers[j]  # The end node is determined by the header

                    if start_node == end_node:  # Skip self-loops
                        continue

                    # Skip this node if it's in the skip list
                    if end_node in skip_nodes or start_node == end_node:
                        continue

                    if weight == '':  # Replace empty string with 0
                        weight = '0'
                    if float(weight) != 0:  # Check if the weight is not zero
                        line = f"{start_node} {end_node} {weight}\n"
                        output_file.write(line)
