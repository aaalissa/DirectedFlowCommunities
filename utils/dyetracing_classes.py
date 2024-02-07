import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from scipy.stats import mode

class Node:
    def __init__(self, name, dye_concentration=0.0, source_node=False, absorption_rate=0.5, infinite_source=False, absorption_limit = np.inf):
        self.name = name                    #best if it aligns with index in node list
        self.nickname = None                #nickname for node hehehehe
        self.index = None                   #index in node list

        self.connected_edges = []
        self.incoming_edges = []
        self.outgoing_edges = []

        self.source_node = source_node
        self.infinite_source = infinite_source

        self.incoming_dye = 0.0
        self.outgoing_dye = 0.0

        self.dye_concentration = dye_concentration
        self.prior_concentration = 0.0

        self.absorbed_amount = 0.0
        self.absorption_rate = absorption_rate
        self.absorption_limit = absorption_limit

    def add_edge(self, edge):
        self.connected_edges.append(edge)
        if edge.start_node == self:
            self.outgoing_edges.append(edge)
        if edge.end_node == self:
            self.incoming_edges.append(edge)

    
    def step(self, mode='equal'):
        """
        Distribute dye to outgoing edges based on current dye concentration and incoming edges.
        Parameters:
        - mode (str): 'equal' or 'proportional'. 'equal' divides the dye equally among edges, 
                    while 'proportional' divides it based on edge velocities.
        """
        # Calculate total incoming dye
        incoming_dye = sum(edge.C_end for edge in self.incoming_edges)
        self.incoming_dye = incoming_dye
        if self.incoming_dye < 0:
            print(f"\t\t{self} incoming dye values is negative!!! {incoming_dye:.5f}")
            #print(f"\t\t{[(edge, edge.C_end) for edge in self.incoming_edges]}")
        #print(f"\t\t{self} incoming dye: {incoming_dye:.5f}")

        # Caclulate absorbed dye
        potential_absorption = incoming_dye * self.absorption_rate

        
        # Check if the absorption limit is reached
        if self.absorbed_amount == self.absorption_limit:
            absorbed_dye = 0                                            # no more absorption
            self.absorbed_amount = self.absorption_limit                # update absorbed amount to limit
            self.outgoing_dye = self.incoming_dye                       # all dye goes to outgoing edges

        # Check if the absorption limit will be reached    
        elif self.absorbed_amount + potential_absorption >= self.absorption_limit:
            absorbed_dye = self.absorption_limit - self.absorbed_amount # absorb up to limit
            self.absorbed_amount = self.absorption_limit                # update absorbed amount to limit
            self.outgoing_dye = self.incoming_dye - absorbed_dye        # remaining dye goes to outgoing edges

        # Otherwise, absorb all potential absorption
        else:
            absorbed_dye = potential_absorption
            self.absorbed_amount += absorbed_dye
            self.outgoing_dye = self.incoming_dye - absorbed_dye

        incoming_dye -= absorbed_dye
        self.outgoing_dye = incoming_dye
        #print(f"\t\t{self} absorbed dye: {absorbed_dye:.5f}")
        
        # Calculate total dye
        total_dye = self.dye_concentration + incoming_dye
        #print(f"\t\t{self} total dye in node: {total_dye:.5f}, previous amount of dye: {self.dye_concentration:.5f}")

        #save prior concentration for references
        self.prior_concentration = total_dye

        # Distribute total dye to outgoing edges
        if not self.outgoing_edges:
            # If there are no outgoing edges, the dye stays in the node
            self.dye_concentration = total_dye
            return
        
        if mode == 'equal':
            equal_concentration = total_dye / len(self.outgoing_edges)
            for edge in self.outgoing_edges:
                edge.start_dye_concentration = equal_concentration
                
        elif mode == 'proportional':
            total_velocity = sum(edge.u for edge in self.outgoing_edges)
            for edge in self.outgoing_edges:
                edge.start_dye_concentration = total_dye * (edge.u / total_velocity)
                
        else:
            raise ValueError("Mode should be either 'equal' or 'proportional'")
        
        # Update node's dye concentration to account for distributed dye
        self.dye_concentration = total_dye - sum(edge.start_dye_concentration for edge in self.outgoing_edges)
        
        # This is to avoid negative dye concentrations from rounding errors
        if self.dye_concentration < 1e-25:  
            self.dye_concentration = 0
        
        assert self.dye_concentration >= 0, f"Node {self} has negative dye concentration {self.dye_concentration}"
    
    def create_dict(self, include_name = False):
        if include_name:
            return {'name': self.name, 'dye_concentration': self.dye_concentration, 'absorbed_amount': self.absorbed_amount, 'incoming_dye': self.incoming_dye, 'outgoing_dye': self.outgoing_dye}
        else:
            return {'dye_concentration': self.dye_concentration, 'absorbed_amount': self.absorbed_amount, 'incoming_dye': self.incoming_dye, 'outgoing_dye': self.outgoing_dye}

    def __repr__(self):
        return f"{self.name}"

class Edge:
    def __init__(self, start_node, end_node, length = 1, velocity = 1, decay_rate = 0.0, min_dt = 0.01):
        # Register this edge with the nodes
        self.start_node = start_node
        self.end_node = end_node
        start_node.add_edge(self)
        end_node.add_edge(self)

        self.L = length
        self.u = velocity
        self.lambda_decay = decay_rate

        self.dt = min_dt

        self.start_dye_concentration = 0.0
        self.C_center = 0.0  # Initial concentration at the center
        self.C_end = 0.0     # Initial concentration at the end

    def step(self): #calculate dye concentration at start, center and end of edge
        # Calculate new dye concentration at center of edge
        C_next_center = self.C_center - self.dt * self.u / (self.L/2) * (self.C_center - self.start_dye_concentration) - self.lambda_decay * self.dt * self.C_center
        
        # Calculate new dye concentration at end of edge
        C_next_end = self.C_end - self.dt * self.u / (self.L/2) * (self.C_end - self.C_center) - self.lambda_decay * self.dt * self.C_end
        
        # Update dye concentration at center and end of edge
        self.C_center = C_next_center
        self.C_end = C_next_end

        # Check for negative dye concentrations due to high decay rate, and set to 0
        if self.C_center < 0: 
            self.C_center = 0
            self.C_end = 0

        if self.C_end < 0:
            self.C_end = 0
    def __repr__(self):
        return f"edge(start={self.start_node.name}, end={self.end_node.name})"

def simulate(network_nodes, network_edges, T, print_final=False, history=True, step_mode='proportional'):
    """
    Simulate the dye flow through the network.

    Parameters:
    - network_nodes (list): A list of all nodes in the network.
    - network_edges (list): A list of all edges in the network.
    - T(int): The number of time steps to simulate.
    - print_final (bool): Whether to print the final dye concentrations at each node and edge.
    - history (bool): Whether to keep track of the history of dye concentrations at each node and edge.
    - step_mode (str): The mode of simulation to use. Either 'equal' or 'proportional'.
    """

    node_absorbed_history = []
    node_concentration_history = []
    n_temp_ab_history = []
    n_temp_con_history = []

    #run initalization step
    reset_source = False
    source_node_count = 0

    for node in network_nodes:
        #find source node
        if node.source_node == True:
            source_node_count = source_node_count + 1
            source_node = node
            assert node.dye_concentration > 0, f"Source node dye concentration is not set, please set initial dye concentration ({node.dye_concentration})"
            initial_dye_concentration = node.dye_concentration
            #node.absorption_rate = 0.0 #if source node, absorption rate is 0
            if node.infinite_source:
                reset_source = True
            node.step(mode=step_mode)
        
        n_temp_ab_history.append(node.absorbed_amount)
        n_temp_con_history.append(node.dye_concentration)
    
    assert source_node_count == 1, f"Source node count is {source_node_count}, should be 1, check for multiple source nodes"

    #should initialize as 0's for all nodes
    if history:
        node_absorbed_history.append(n_temp_ab_history)
        node_concentration_history.append(n_temp_con_history)
        
    #step through simulation
    for t in range(T):
        n_temp_ab_history = []
        n_temp_con_history = []

        if reset_source == True: #if infinite source, set source node dye concentration to initial concentration at every time step
            source_node.dye_concentration = initial_dye_concentration
    
        for edge in network_edges:
            edge.step()
      
        for node in network_nodes:
            node.step(mode=step_mode)
            if history:
                n_temp_ab_history.append(node.absorbed_amount)
                #TODO: add prior concentration to history? # FLAG !!!! is this what I want it to do? should this just be concentrations instead? 
                n_temp_con_history.append(node.prior_concentration)

        if history:
            node_absorbed_history.append(n_temp_ab_history.copy())
            node_concentration_history.append(n_temp_con_history.copy())

    if history:
        return node_absorbed_history, node_concentration_history

    return n_temp_ab_history, n_temp_con_history

def normalize(absorbed_history, source_node_index):
    """
    Normalize the absorbed history by dividing by the global maximum value.
    """
    if isinstance(absorbed_history, list):
        # turn into np array
        absorbed_history = np.array(absorbed_history)
        if len(absorbed_history.shape) == 1:
            # add dimension 
            absorbed_history = absorbed_history[np.newaxis, :]
    elif not isinstance(absorbed_history, np.ndarray):
        raise ValueError("Input should be a list or numpy array")

    # Find the global maximum value excluding the source node column
    global_max = np.max(np.delete(absorbed_history, source_node_index, axis=1))

    for i in range(absorbed_history.shape[0]):
        # set the value of the source node to 1
        absorbed_history[i][source_node_index] = 1

        # create a mask to exclude the source node column from normalization
        mask = np.ones(absorbed_history.shape[1], bool)
        mask[source_node_index] = 0
        absorbed_history[i][mask] /= global_max
        
    return absorbed_history

def run_simulation(node_list, edge_list, source_nodes, params):
    """
    Run the simulation for a list of source nodes and return the normalized absorbed history.
    inputs:
        node_list (list): list of Node objects
        edge_list (list): list of Edge objects
        source_nodes (list): list of source node
        params (dict): dictionary of parameters
    output:
        flow_profile (np.array): normalized absorbed history
    """
    absorbed = []

    step = 1
    for n in source_nodes:
        reset_simulation(node_list, edge_list)  # Resets the simulation environment
        load_params(node_list, edge_list, params)  # Loads the parameters into the simulation
        set_source_node(node_list, n.index, params)  # Designates a source node
        check_node_attributes(node_list, params, source_node=n.index)  # Checks if node attributes meet certain criteria

        # Runs the simulation and returns the history of test runs and concentrations
        run, _ = simulate(node_list, edge_list, params['num_steps'], history=True, step_mode="proportional")
        norm = normalize(run, n.index)  # Normalizes the test run data

        absorbed.append(norm[-1])  # Appends the last normalized value to absorbed list

        print(f"Layer {step}/{len(source_nodes)} completed", end='\r')
        step += 1

    print(f"Layer {len(source_nodes)}/{len(source_nodes)} completed")
    flow_profile = np.array(list(zip(*absorbed)))

    return flow_profile  # Returning the results for further use


def reset_simulation(network_nodes, network_edges):
    """
    Reset the attributes of each node and edge to their initial values.
    
    Parameters:
    - network_nodes (list): A list of all nodes in the network.
    - network_edges (list): A list of all edges in the network.
    """

    for node in network_nodes:
        node.prior_concentration = 0.0
        node.absorbed_amount = 0.0
        node.incoming_dye = 0.0
        node.outgoing_dye = 0.0
        node.dye_concentration = 0.0

    for edge in network_edges:
        edge.start_dye_concentration = 0.0
        edge.C_center = 0.0
        edge.C_end = 0.0

def load_params(node_list, edge_list, params):
    """
    Load parameters into nodes and edges.

    Parameters:
    - node_list (list): A list of all nodes.
    - edge_list (list): A list of all edges.
    - params (dict): Dictionary containing parameters to be loaded.
    """
    # Update all nodes
    for node in node_list:
        if "absorption_rate" in params:
            node.absorption_rate = params["absorption_rate"]
        if "absorption_limit" in params:
            node.absorption_limit = params["absorption_limit"]
        
    # Update all edges
    for edge in edge_list:
        if "decay_rate" in params:
            edge.lambda_decay = params["decay_rate"]
    
    # Update edge dt
    set_edge_dt(edge_list)

def set_source_node(node_list, source_node_index, params):
    """
    Set the source node and its attributes.

    Parameters:
    - node_list (list): A list of all nodes.
    - source_node (int): Index of the source node.
    - params (dict): Dictionary containing parameters for the source node.
    """
    
    assert source_node_index < len(node_list), f"Source node index {source_node_index} is out of range"

    # Reset old source nodes
    for node in node_list:
        node.source_node = False
        node.infinite_source = False

    # Update source node
    node_list[source_node_index].source_node = True
    node_list[source_node_index].infinite_source = True
    node_list[source_node_index].dye_concentration = 1

def set_edge_dt(edge_list):
    """
    set edge dt based on max u, dt = 1/max(u)
    inputs:
        edge_list (list): list of Edge objects
    """

    #find max u and set dt
    max_u = 0
    for edge in edge_list:
        if edge.u > max_u:
            max_u = edge.u
    
    dt = 1/max_u

    if dt > 0.01: #retain min dt of 0.01
        dt = 0.01

    for edge in edge_list:
        edge.dt = dt


def check_node_attributes(node_list, params, source_node = None):
    """
    Check that the node attributes match the given parameters.
    inputs:
        node_list (list): list of Node objects
        params (dict): dictionary of parameters
        source_node (int): index of source node, if None, will check all nodes
    """
    if source_node is not None:
        assert node_list[source_node].source_node == True, f"Source node {source_node} is not a source node"
    for node in node_list:
        if node.source_node == True:
            assert node.dye_concentration == params.get("initial_dye_concentration", node.dye_concentration), f"Source node ({node.name}) dye_concentration mismatch {node.dye_concentration} != {params.get('initial_dye_concentration', 'dye concentration not set')}"
            assert node.infinite_source == params.get("infinite_source", node.infinite_source), f"Node {node.name} infinite_source == {node.infinite_source} != {params.get('infinite_source', 'infinite_source not set')}"
            continue
        assert node.dye_concentration == 0, f"Node {node.name} dye_concentration mismatch, {node.dye_concentration} != 0"
        assert node.absorption_rate == params.get("absorption_rate", node.absorption_rate), f"Node {node.name} absorption_rate mismatch, {node.absorption_rate} != {params.get('absorption_rate', 'absorption_rate not set')}"
        assert node.absorption_limit == params.get("absorption_limit", node.absorption_limit), f"Node {node.name} absorption_limit mismatch, {node.absorption_limit} != {params.get('absorption_limit', 'absorption_limit not set')}"
        assert node.infinite_source == False, f"Node {node.name} is not a source node, but infinite_source == {node.infinite_source} != False"


def evaluate_clustering(ground_truth, labels):
    """ 
    Evaluate clustering results using accuracy, ARI, and NMI against ground truth 
    """

    #check inputs
    assert len(ground_truth) == len(labels), 'ground truth and labels must be the same size'
    assert isinstance(ground_truth, np.ndarray) and isinstance(labels, np.ndarray), 'ground truth and labels must be numpy arrays'

    labels_matched = np.zeros_like(labels)
    for cluster in np.unique(labels):
        mask = (labels == cluster)
        #labels_matched[mask] = mode(ground_truth[mask], keepdims=True)[0]
        #labels_matched[mask] = mode(ground_truth[mask])[0][0] #worked usually
        #labels_matched[mask] = mode(ground_truth[mask], keepdims=False)[0][0]
        mode_result = mode(ground_truth[mask], keepdims=False)[0]
        if np.isscalar(mode_result):
            labels_matched[mask] = mode_result
        else:
            labels_matched[mask] = mode_result[0]

    accuracy = accuracy_score(ground_truth, labels_matched)
    ari = adjusted_rand_score(ground_truth, labels_matched)
    nmi = normalized_mutual_info_score(ground_truth, labels_matched)
    
    return accuracy, ari, nmi

def cluster_and_evaluate(zipped_layers, cluster_range, true_membership=None, plot=True):
    """ 
    k-means clustering with automatic evaluation of the optimal number of clusters 
    """
    wcss = []
    silhouette_scores = []
    labels_list = []
    
    accuracy = []
    ari = []
    nmi = []

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(zipped_layers)
        labels = np.array(kmeans.labels_)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(zipped_layers, labels))
        labels_list.append(labels)
        
        if true_membership is not None:
            accuracy_temp, ari_temp, nmi_temp = evaluate_clustering(true_membership, labels)
            accuracy.append(accuracy_temp)
            ari.append(ari_temp)
            nmi.append(nmi_temp)

    # Plotting
    if plot:
        plt.figure(figsize=(12, 3))
        plt.subplot(1, 2, 1)
        plt.plot(cluster_range, wcss, 'ro-')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.title('Elbow Method For Optimal k')
        plt.subplot(1, 2, 2)
        plt.plot(cluster_range, silhouette_scores, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score For Optimal k')
        plt.tight_layout()
        plt.show()

    optimal_clusters_silhouette = cluster_range[np.argmax(silhouette_scores)]
    optimal_labels = labels_list[optimal_clusters_silhouette - cluster_range[0]]

    results = {
        "optimal_clusters": optimal_clusters_silhouette,
        "optimal_labels": optimal_labels,
        "labels_list": labels_list,
        "cluster_range": cluster_range,
    }



    if true_membership is not None:
        results["accuracy"] = accuracy[optimal_clusters_silhouette - cluster_range[0]]
        results["ari"] = ari[optimal_clusters_silhouette - cluster_range[0]]
        results["nmi"] = nmi[optimal_clusters_silhouette - cluster_range[0]]

    return results