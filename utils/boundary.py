import numpy as np

def detect_boundaries(adj_matrix, dye_absorbed, threshold=0.5):
    """
    Detect potential boundary nodes based on dye absorption differences.

    Parameters:
    adj_matrix (2D list or np.array): Adjacency matrix of the network.
    dye_absorbed (list): Dye absorbed values for each node.
    threshold (float): Threshold for significant absorption difference.

    Returns:
    boundary_nodes (list): List of potential boundary nodes.
    """
    n = len(dye_absorbed)
    boundary_nodes = []

    for i in range(n):
        neighbors = np.where(adj_matrix[i] == 1)[0]
        for neighbor in neighbors:
            diff = abs(dye_absorbed[i] - dye_absorbed[neighbor])
            if diff / max(dye_absorbed[i], dye_absorbed[neighbor], 1) > threshold:
                boundary_nodes.append(i)
                break

    return boundary_nodes

def is_distinct_boundary(boundary_nodes, dye_absorbed, distinct_threshold=0.7):
    """
    Assess if the identified boundary nodes form a distinct boundary.

    Parameters:
    boundary_nodes (list): Identified potential boundary nodes.
    dye_absorbed (list): Dye absorbed values for each node.
    distinct_threshold (float): Threshold for distinctness.

    Returns:
    distinct_boundary (bool): True if a distinct boundary is formed.
    """
    if len(boundary_nodes) == 0:
        return False

    boundary_values = [dye_absorbed[node] for node in boundary_nodes]
    non_boundary_values = [dye_absorbed[i] for i in range(len(dye_absorbed)) if i not in boundary_nodes]

    # Calculate the average dye absorption for boundary and non-boundary nodes
    avg_boundary = np.mean(boundary_values)
    avg_non_boundary = np.mean(non_boundary_values)

    # Check if the difference between averages is significant
    return abs(avg_boundary - avg_non_boundary) / max(avg_boundary, avg_non_boundary, 1) > distinct_threshold

