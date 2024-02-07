import sys
sys.path.append("..")
from utils.dyetracing_classes import *
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
import random

#TODO: make sure threshold is consistant across all functions.... would be helpful.

def hops_from_source(source_node):
    """
    Returns a dictionary with the number of hops from the source node to each node in the network using BFS.
    Parameters:
        - source_node (Node): The source node.
    Returns:
        - dict: A dictionary where keys are nodes and values are the number of hops from the source.
    """
    visited = set()
    queue = [(source_node, 0)]  # Each item in the queue is a tuple (node, distance_from_source)
    hops_dict = {}

    while queue:
        current_node, distance = queue.pop(0)
        
        if current_node in visited:
            continue

        visited.add(current_node)
        hops_dict[current_node] = distance

        for edge in current_node.connected_edges:
            if edge.start_node == current_node and edge.end_node not in visited:
                queue.append((edge.end_node, distance + 1))
            elif edge.end_node == current_node and edge.start_node not in visited:
                queue.append((edge.start_node, distance + 1))

    return hops_dict

def total_coverage(layers, threshold=0.001):
    """
    Calculate the total coverage based on the layers.
    """
    return sum(1 for node_values in layers if any(value >= threshold for value in node_values))

def percent_coverage(layers):
    """
    Calculate the percent coverage based on the layers.
    """
    return total_coverage(layers) / len(layers) * 100

def nodes_without_dye(zipped_layers, node_list, threshold=0.001):
    """
    Identifies nodes that do not have any dye.
    
    Parameters:
        - zipped_layers (numpy array or list): np array of shape (num nodes, num layers) or list of lists of length num nodes
        - node_list (list): List of Node objects
        
    Returns:
        - list: A list of nodes that do not have any dye.
    """
    no_dye_nodes = []

    for node, node_values in enumerate(zipped_layers):
        # Ensure node_values is iterable (list or numpy array)
        if not isinstance(node_values, (list, tuple, np.ndarray)):
            node_values = [node_values]
        # Check if any value for the node is below or equal to the threshold
        if any(value >= threshold for value in node_values):
            continue
        else:
            no_dye_nodes.append(node_list[node])

    return no_dye_nodes

def characterize_spread(source_node, node_list, simulation_results, verbose=True):
    """
    parameters:
        - source_node (int or Node): the source node of the simulation
        - node_list (list of all Nodes): the list of nodes in the network
        - simulation_results (absorbed values of all nodes): the results of the simulation
        - verbose (boolean): print or not print
    returns:
        - results (dict): a dictionary of the results of the simulation
    """
    results = {}

    # Check source node
    if isinstance(source_node, int):
        if source_node >= len(node_list) or source_node < 0:
            raise ValueError("Source node index is out of range")
        source_node = node_list[source_node]
    elif isinstance(source_node, Node):  # if source_node is a Node object, make sure Node Class is imported
        if source_node not in node_list:
            raise ValueError("Source node must be in the network")
    else:
        raise ValueError("Source node must be an integer or a Node object")

    total_nodes = len(node_list)
    hops_dict = hops_from_source(source_node)
    max_hops = max(hops_dict.values())

    #TODO: DETERMINE THRESHOLD VALUE! (0.00001 is a estimate)
    threshold = 0.00001
    dye_count_total = sum(1 for node in node_list if simulation_results[node.index] > threshold) 
    results['dye_count_total'] = (dye_count_total / total_nodes * 100, dye_count_total, total_nodes)
    if verbose:
        print(f'all nodes that contain dye: {results["dye_count_total"][0]}% - {results["dye_count_total"][1]} of {results["dye_count_total"][2]} total nodes')

    results['hops_data'] = {}
    for i in range(max_hops + 1):
        if i == 0:  # skip source node
            continue
        nodes_at_hop_i = [node for node, hops in hops_dict.items() if hops == i]
        dye_count = sum(1 for node in nodes_at_hop_i if simulation_results[node.index] > threshold)
        hop_percentage = dye_count / len(nodes_at_hop_i) * 100
        hop_i_concentration = [node.prior_concentration for node in nodes_at_hop_i]
        avg_unabsorbed = sum(hop_i_concentration) / len(hop_i_concentration)
        avg_absorbed = sum([simulation_results[node.index] for node in nodes_at_hop_i]) / len(hop_i_concentration)
        results['hops_data'][i] = {
            'hop_percentage': hop_percentage,
            'dye_count': dye_count,
            'avg_unabsorbed': avg_unabsorbed,
            'avg_absorbed': avg_absorbed
        }
        if verbose:
            print(f'\tnodes at hop {i} ({len(nodes_at_hop_i)}/{total_nodes}) that contain dye: {hop_percentage}% ({dye_count} nodes)')
            print(f'\tthe average unabsorbed amount at hop {i} is {avg_unabsorbed}')
            print(f'\tthe average absorbed amount at hop {i} is {avg_absorbed}')

    saturated_nodes = sum(1 for node in node_list if simulation_results[node.index] >= 0.99)
    results['saturated_nodes'] = saturated_nodes
    if saturated_nodes > 3 and verbose:
        print(f'\t\t{saturated_nodes} highly saturated nodes (>= 0.99), consider revising parameters')

    filtered_results = [result for result in simulation_results if result != 0 and result != 1]
    results['range'] = (min(filtered_results), max(filtered_results))
    if verbose:
        print(f"\tRange of absorbed concentrations: {results['range'][0]} to {results['range'][1]}")

    return results

def find_border_nodes(simulation_results, node_list, plot=False, verbose=False):
    """ 
    Find the border nodes of the simulation results.
    input: 
        simulation_results (list or dict): a list or dictionary of the simulation results
        node_list (list): a list of all the Node objects in the network
    output: 
        border_nodes (list): a list of the border nodes of Node objects
    """

    # Check if simulation_results is a list or a dictionary
    if isinstance(simulation_results, list):
        sorted_results = {node.index: value for node, value in zip(node_list, simulation_results)}
        sorted_results = {k: v for k, v in sorted(sorted_results.items(), key=lambda item: item[1], reverse=True)}
    elif isinstance(simulation_results, dict):
        sorted_results = {k: v for k, v in sorted(simulation_results.items(), key=lambda item: item[1], reverse=True)}
    else:
        raise ValueError("simulation_results must be either a list or a dictionary")

    # Sort dictionary by value
    #sorted_results = {k: v for k, v in sorted(simulation_results.items(), key=lambda item: item[1], reverse=True)}

    # Compute differences between consecutive elements
    test = list(sorted_results.values())
    differences = [test[i] - test[i+1] for i in range(len(test)-1)]

    # Determine a threshold for a significant drop
    threshold = 0.01

    # Identify indices where the drop exceeds the threshold
    significant_drops = [i for i, diff in enumerate(differences) if diff > threshold]

    # Determine the range of values for border nodes based on significant drops
    lower_bound = test[significant_drops[-1]]
    upper_bound = test[significant_drops[-3]]

    if verbose:
        print(upper_bound, lower_bound)

    # Find nodes whose values fall within this range
    # border_nodes = [node for node, value in sorted_results.items() if lower_bound <= value <= upper_bound]
    # border_nodes = [node_list[node] for node in border_nodes]

    border_nodes = [node_list[node] for node, value in islice(sorted_results.items(), significant_drops[-3], significant_drops[-1] + 1 ) if lower_bound <= value <= upper_bound]


    if plot:
        # Plot the simulation results
        plt.plot(test)
        plt.xlim(0, 50)
        #point upper and lower bounds as points
        plt.plot(significant_drops[-3], upper_bound, 'ro', label=f'upper bound ({upper_bound:.5f})')
        plt.plot(significant_drops[-1], lower_bound, 'bo', label=f'lower bound  ({lower_bound:.5f})')
        plt.legend()
        plt.title("Simulation Results")
        plt.show()

    return border_nodes


def get_lower_quartile_nodes(zipped_layers, node_list):
    """
    Calculate the mean values for each node and return the indices of nodes 
    with mean values in the lower quartile.
    
    Parameters:
    - zipped_layers: List of lists containing node values.
    
    Returns:
    - List of Node objects with mean values in the lower quartile.
    """
    mean_values = np.mean(zipped_layers, axis=1)
    quartile_value = np.percentile(mean_values, 25)

    # Get the indices of nodes with mean values below the 25th percentile
    lower_quartile_indices = np.where(mean_values <= quartile_value)[0].tolist()
    lower_quartile_nodes = [node_list[i] for i in lower_quartile_indices]
    
    return lower_quartile_nodes

def average_outgoing_edges(nodes):
    if not nodes:
        return None
    total_length = 0
    num_nodes = len(nodes)
    for node in nodes:
        total_length += len(node.outgoing_edges)
    return total_length / num_nodes

def find_incoming_edges_to_missed_nodes(missed_nodes):
    missed_node_incoming = set()
    for node in missed_nodes:
        for edge in node.incoming_edges:
            #add to set of missed_node_incoming
            missed_node_incoming.add(edge.start_node)
    return list(missed_node_incoming)

