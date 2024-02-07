import sys
sys.path.append("..")
from utils.dyetracing_classes import *
from utils.spread import *
import numpy as np
import networkx as nx

def find_low_betweenness_nodes(G, node_list, num_nodes):
    """
    find nodes with low betweenness centrality and out degree > 1
    inputs:
        G (networkx graph): graph of nodes and edges
        node_list (list): list of Node objects
        num_nodes (int): number of nodes to select

    returns:
        selected_node (list): list of Node objects
    """

    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)

    # Sort nodes by betweenness centrality
    sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get)

    # Select the specified number of nodes with the lowest betweenness centrality
    # and have out degree > 1
    selected_node_names = [node for node in sorted_nodes if G.out_degree(node) >= 1][:num_nodes]

    selected_nodes = []
    for node in node_list:
        if node.name in selected_node_names and len(node.outgoing_edges) > 1:
            #if node.name in node_list in source_nodes_names append to source_nodes
            selected_nodes.append(node)

    return selected_nodes

def find_high_betweenness_nodes(G, node_list, num_nodes):
    """
    find nodes with high betweenness centrality and out degree > 1
    inputs:
        G (networkx graph): graph of nodes and edges
        node_list (list): list of Node objects
        num_nodes (int): number of nodes to select
        
    returns:
        selected_node (list): list of Node objects
    """
    
    # Calculate betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(G)

    # Sort nodes by betweenness centrality
    sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)

    # Select the specified number of nodes with the highest betweenness centrality
    # and have out degree > 1
    selected_node_names = [node for node in sorted_nodes if G.out_degree(node) >= 1][:num_nodes]

    selected_nodes = []
    for node in node_list:
        if node.name in selected_node_names and len(node.outgoing_edges) > 1:
            #if node.name in node_list in source_nodes_names append to source_nodes
            selected_nodes.append(node)

    return selected_nodes

def find_random_nodes(node_list, num_nodes):
    """
    find random nodes with out degree > 1
    inputs:
        node_list (list): list of Node objects
        num_nodes (int): number of nodes to select

    returns:
        selected_node (list): list of Node objects
    """
    selected_nodes = []
    for _ in range(num_nodes):
        node = np.random.choice(node_list)
        while len(node.outgoing_edges) <= 1:
            node = np.random.choice(node_list)
        selected_nodes.append(node)

    return selected_nodes

def find_hybrid_source_nodes(G, node_list, num_nodes):
    """
    find 1/3 nodes with high betweenness centrality and out degree > 1
    find 1/3 nodes with low betweenness centrality and out degree > 1
    find 1/3 random nodes with out degree > 1
    make sure no duplicate nodes are selected

    inputs:
        G (networkx graph): graph of nodes and edges
        node_list (list): list of Node objects
        num_nodes (int): number of nodes to select
    
    returns:
        selected_node (list): list of Node objects
    """
    assert num_nodes <= len(node_list), "Number of nodes to select is greater than the number of nodes in the graph"

    betweenness_centrality = nx.betweenness_centrality(G)
    sorted_nodes = sorted(betweenness_centrality, key=betweenness_centrality.get)

    # Divide into thirds
    one_third = num_nodes // 3

    # Select 1/3 nodes with the highest betweenness centrality and out degree > 1
    high_centrality_nodes = [node for node in sorted_nodes if G.out_degree(node) >= 1][:one_third]
    low_centrality_nodes = [node for node in sorted_nodes if G.out_degree(node) >= 1][-one_third:]

    # Select random nodes with out-degree > 1, excluding already selected nodes
    potential_random_nodes = [node for node in G.nodes() if G.out_degree(node) > 1 and node not in high_centrality_nodes + low_centrality_nodes]
    random_nodes = random.sample(potential_random_nodes, min(len(potential_random_nodes), num_nodes - len(high_centrality_nodes + low_centrality_nodes)))

    # Combine all selected nodes, ensuring no duplicates
    selected_node_names = list(set(high_centrality_nodes + low_centrality_nodes + random_nodes))

    # If we don't have enough nodes due to overlaps or shortage, fill in with random choices
    if len(selected_node_names) < num_nodes:
        additional_nodes = random.sample([node for node in G.nodes() if node not in selected_node_names], num_nodes - len(selected_node_names))
        selected_node_names.extend(additional_nodes)

    selected_nodes = []
    for node in node_list:
        if node.name in selected_node_names and len(node.outgoing_edges) > 1:
            #if node.name in node_list in source_nodes_names append to source_nodes
            selected_nodes.append(node)

    return selected_nodes

def find_low_centrality_nodes(G, pr_weight, bc_weight, dc_weight, std_dev_threshold):
    # Calculate normalized centralities
    pagerank = nx.pagerank(G)
    max_pagerank = max(pagerank.values())
    pagerank_normalized = {node: value / max_pagerank for node, value in pagerank.items()}

    betweenness_centrality = nx.betweenness_centrality(G)
    betweenness_max = max(betweenness_centrality.values())
    betweenness_centrality_normalized = {node: value / betweenness_max for node, value in betweenness_centrality.items()}

    degree_centrality = nx.degree_centrality(G)
    max_degree = max(degree_centrality.values())
    degree_centrality_normalized = {node: value / max_degree for node, value in degree_centrality.items()}

    # Calculate combined centralities
    combined_centralities = {
        node: (
            pagerank_normalized[node] * pr_weight +
            betweenness_centrality_normalized[node] * bc_weight +
            degree_centrality_normalized[node] * dc_weight
        )
        for node in G.nodes()
    }

    # Calculate threshold for low centrality
    mean_combined = np.mean(list(combined_centralities.values()))
    std_combined = np.std(list(combined_centralities.values()))
    threshold = mean_combined - std_dev_threshold * std_combined

    # Find nodes with centrality less than the threshold and have out degree > 1
    low_centrality_nodes = [node for node, centrality in combined_centralities.items() if centrality < threshold and G.out_degree(node) >= 1]

    return low_centrality_nodes

def iter_node_selection(node_list, edge_list, params, n, min_percent_coverage= .90, min_percent_nodes = 0.125, verbose=False):
    """
    run iterative node selection algorithm
    inputs:
        node_list: list of Node objects
        edge_list: list of Edge objects
        params: dictionary of parameters
        n: number of random source nodes to select for each iteration
        proportional_min: minimum proportion of nodes that must be covered before stopping
    
    returns:
        zipped_layers: list of lists of length num node
    """

    # Extract parameters
    num_steps = params.get("num_steps")
    infinite_source = params.get("infinite_source")
    initial_dye_concentration = params.get("initial_dye_concentration")
    step_mode = params.get("step_mode", "proportional")

    total_nodes = len(node_list)
    avg_degree = np.mean([len(node.outgoing_edges) for node in node_list])

    # 1) Run simulation on n random source nodes with higher than average degree
    higher_degree_nodes = [node for node in node_list if len(node.outgoing_edges) > avg_degree]

    source_nodes_set = set()  # Use a set for faster look-up
    source_nodes = []

    for _ in range(n):
        if not higher_degree_nodes:  # If we've exhausted the higher degree nodes, break
            print("No more higher degree nodes to sample from, sampling from all nodes")
            higher_degree_nodes = node_list

        node = np.random.choice(higher_degree_nodes)
        while node in source_nodes_set:
            node = np.random.choice(higher_degree_nodes)

        higher_degree_nodes.remove(node)  # Remove the selected node from the list
        source_nodes.append(node)
        source_nodes_set.add(node)

    layers = []
    
    #all_border_nodes = []
    for source in source_nodes:
        reset_simulation(node_list, edge_list)
        l = run_simulation(node_list, edge_list, source.index, num_steps, infinite_source, init_dye_concentration=initial_dye_concentration, step_mode=step_mode)
        layers.append(l)

        # border_nodes = find_border_nodes(l, node_list)
        # all_border_nodes.extend(border_nodes)    
            
        if np.isnan(l).any():
            raise ValueError("Error intialization of simulation, NaN values detected.")

    if verbose:
        print(f"Simulation for source nodes {source_nodes} completed")

    zipped_layers = np.array(list(zip(*layers)))
    total_layers = len(zipped_layers[0])

    # 2) Calculate total coverage
    if verbose:
        print(f"Total coverage after initial simulations: {total_coverage(zipped_layers)}/{total_nodes} nodes")
    # if percent_coverage(zipped_layers) > min_percent_coverage * 100 and total_layers < total_nodes * min_percent_nodes:         # return if both conditions are met
    #     return zipped_layers

    # Repeat step 3 and step 4 until the the graph is covered and the number of layers is less than min of the total number of nodes (continue until both conditions are met)
    last_percent_coverage = None
    while percent_coverage(zipped_layers) < min_percent_coverage * 100 or total_layers < total_nodes * min_percent_nodes:
        if verbose:    
            print(f"Percent Coverage: {percent_coverage(zipped_layers)}")

        #3) Check for missed nodes
        missed_nodes = nodes_without_dye(zipped_layers, node_list)
        if len(missed_nodes) == 0:
            if verbose:
                print("No missed nodes, using lower quartile nodes instead.")
                lower_quartile_nodes = get_lower_quartile_nodes(zipped_layers, node_list)
                nodes_for_resimulation = lower_quartile_nodes
            break
        else:
            if average_outgoing_edges(missed_nodes) < 1:
                if verbose:
                    print("Missed nodes have no outgoing edges, selecting nodes with incoming edges instead.")
                missed_nodes = find_incoming_edges_to_missed_nodes(missed_nodes)
            nodes_for_resimulation = missed_nodes

        #break if no change in coverage or very few
        if last_percent_coverage == percent_coverage(zipped_layers) or len(missed_nodes) <= 3:
                print(f'\nFinal total coverage: {total_coverage(zipped_layers)} / {len(node_list)}')
                print(f'Total layers: {total_layers}')
                return zipped_layers


        if len(nodes_for_resimulation) == 0:
            nodes_for_resimulation = node_list
            if verbose:
                print("Error in Node selection, resimulating all nodes")
        if verbose:
            print(f'Number of nodes in resimulation set: {len(nodes_for_resimulation)}, missed nodes: {len(missed_nodes)})')
        
        #pick nodes with higher than average degree first
        higher_degree_nodes = [node for node in nodes_for_resimulation if len(node.outgoing_edges) > avg_degree]

        if len(higher_degree_nodes) < n:
            higher_degree_nodes = nodes_for_resimulation

        resimulation_nodes = []
        for _ in range(n):
            node = np.random.choice(higher_degree_nodes)

            while node in source_nodes_set or len(node.outgoing_edges) == 0:
                node = np.random.choice(nodes_for_resimulation)

            resimulation_nodes.append(node)
            source_nodes_set.add(node)

        if verbose:
            print(f"Selected Source Nodes: {resimulation_nodes}")

        #4) Run simulation using nodes identified in the subset from step 3
        for node in resimulation_nodes:
            l = run_simulation(node_list, edge_list, node.index, num_steps, infinite_source, init_dye_concentration=initial_dye_concentration, step_mode=step_mode)
            layers.append(l)
            if np.isnan(l).any():
                raise ValueError(f"NaN values detected, are there self loops or no outgoing edges? {node.outgoing_edges}")
        
        last_percent_coverage = percent_coverage(zipped_layers)

        zipped_layers = np.array(list(zip(*layers)))
        total_layers = len(zipped_layers[0])
        
    if verbose:
        print(f'\nFinal total coverage: {total_coverage(zipped_layers)} / {len(node_list)}')
        print(f'Total layers: {total_layers}')

    return zipped_layers


# def random_node_selection(node_list, edge_list, params, min_percent_nodes, selection_method = "random", verbose=False, return_source_nodes=False):
#     """
#     run random node selection algorithm
#     inputs:
#         node_list (list): list of Node objects
#         edge_list (list): list of Edge objects
#         params (dict): dictionary of parameters
#         min_percent_nodes (float): minimum percent of nodes to select, between 0 and 1
#         selection_method (str): method for selecting nodes, either "random" or "weighted_random"
    
#     returns:
#         zipped_layers: list of lists of length num node
#     """

#     #unpack params
#     num_steps = params.get("num_steps")
#     infinite_source = params.get("infinite_source")
#     initial_dye_concentration = params.get("initial_dye_concentration")
#     step_mode = params.get("step_mode", "proportional")

#     #node selection
#     total_nodes = len(node_list)
#     simuluation_nodes = [] #nodes that have atleast 1 outgoing edge
#     for node in node_list:
#         if len(node.outgoing_edges) > 0:
#             simuluation_nodes.append(node)

#     num_nodes = int(min_percent_nodes * total_nodes)
#     if len(simuluation_nodes) < num_nodes:
#         num_nodes = len(simuluation_nodes)
#         if verbose:
#             print(f"Not enough nodes to select from - defaulting to all possible simulation nodes ({num_nodes} nodes).")
    
#     if selection_method == "random": #select nodes at random that have atleast 1 outgoing edge
#         random_nodes = np.random.choice(simuluation_nodes, num_nodes, replace=False)

#     if selection_method == "weighted_random": #weight probability of selection by node degree
#         sim_nodes_degrees = [len(node.outgoing_edges) for node in simuluation_nodes]
#         node_probs = [degree/sum(sim_nodes_degrees) for degree in sim_nodes_degrees]
#         random_nodes = np.random.choice(simuluation_nodes, num_nodes, replace=False, p=node_probs)
    
#     #run simulation
#     layers = []
#     for node in random_nodes:
#         l = run_simulation(node_list, edge_list, node.index, num_steps, infinite_source, init_dye_concentration=initial_dye_concentration, step_mode=step_mode, troubleshooting_params=params)
#         if np.isnan(l).any():
#             raise ValueError(f"NaN values detected, are there self loops or no outgoing edges? {node.outgoing_edges}")
#         layers.append(l)
        
#     zipped_layers = np.array(list(zip(*layers)))
#     total_layers = len(zipped_layers[0])

#     if verbose:
#         print(f'Total coverage: {percent_coverage(zipped_layers)}')
#         print(f'Total layers: {total_layers}')

#     if return_source_nodes:
#         return zipped_layers, random_nodes
    
#     return zipped_layers
