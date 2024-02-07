import networkx as nx
from geopy.distance import geodesic
from itertools import combinations
from tqdm import tqdm

def calculate_distance(pos1, pos2):
    # Swap latitude and longitude due to weird formatting of the data
    swapped_pos1 = (pos1[1], pos1[0])
    swapped_pos2 = (pos2[1], pos2[0])
    return geodesic(swapped_pos1, swapped_pos2).nautical


def calculate_route_cost(distance, weight, cost_per_nmi_ton):
    return distance * weight * cost_per_nmi_ton

def find_cheapest_route(graph, origin, destination, weight, hubs):
    # Define cost per nmi-ton for different flight types
    cost_hub_hub = 0.496
    cost_spoke = 0.858
    cost_point_to_point = 5.154

    # Direct route (point-to-point)
    direct_distance = calculate_distance(graph.nodes[origin]['position'], graph.nodes[destination]['position'])
    direct_cost = calculate_route_cost(direct_distance, weight, cost_point_to_point)

    # Initialize cheapest cost as direct cost
    cheapest_cost = direct_cost

    # One stop at a hub
    for hub in hubs:
        if hub != origin and hub != destination:
            distance_to_hub = calculate_distance(graph.nodes[origin]['position'], graph.nodes[hub]['position'])
            distance_from_hub = calculate_distance(graph.nodes[hub]['position'], graph.nodes[destination]['position'])

            cost_to_hub = calculate_route_cost(distance_to_hub, weight, cost_spoke if origin not in hubs else cost_hub_hub)
            cost_from_hub = calculate_route_cost(distance_from_hub, weight, cost_spoke if destination not in hubs else cost_hub_hub)

            total_cost = cost_to_hub + cost_from_hub
            cheapest_cost = min(cheapest_cost, total_cost)

    # Two stops at hubs (only if there are at least 2 hubs)
    if len(hubs) >= 2:
        for hub1, hub2 in combinations(hubs, 2):
            if hub1 != origin and hub2 != destination and hub1 != hub2:
                distance_to_hub1 = calculate_distance(graph.nodes[origin]['position'], graph.nodes[hub1]['position'])
                distance_hub1_to_hub2 = calculate_distance(graph.nodes[hub1]['position'], graph.nodes[hub2]['position'])
                distance_from_hub2 = calculate_distance(graph.nodes[hub2]['position'], graph.nodes[destination]['position'])

                total_cost = calculate_route_cost(distance_to_hub1, weight, cost_spoke if origin not in hubs else cost_hub_hub) + \
                             calculate_route_cost(distance_hub1_to_hub2, weight, cost_hub_hub) + \
                             calculate_route_cost(distance_from_hub2, weight, cost_spoke if destination not in hubs else cost_hub_hub)

                cheapest_cost = min(cheapest_cost, total_cost)

    return cheapest_cost

def calculate_total_cost(graph, hubs):
    total_cost = 0

    # Get the total number of edges for the progress bar
    total_edges = graph.number_of_edges()

    # Iterate over all edges in the graph with a progress bar
    for (origin, destination, data) in tqdm(graph.edges(data=True), total=total_edges, desc="Calculating Costs"):
        weight = data['weight']
        cheapest_route_cost = find_cheapest_route(graph, origin, destination, weight, hubs)
        total_cost += cheapest_route_cost

    return total_cost