import networkx as nx
import matplotlib.pyplot as plt


def generate_small_world_network(num_agents, k, p):
    G = nx.watts_strogatz_graph(n=num_agents, k=k, p=p)
    return G


def generate_scale_free_network(num_agents, m):
    G = nx.barabasi_albert_graph(n=num_agents, m=m)
    return G


def generate_scale_free_network_new(num_agents, m, leader_edges, other_edges_limits):
    G = nx.barabasi_albert_graph(n=num_agents, m=m)
    sorted_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)

    # Ensure the top two nodes (leaders) have exactly 15 connections
    leader1, leader2 = sorted_nodes[0][0], sorted_nodes[1][0]

    def adjust_node_degree(node, leader_edges):
        """ Adjust a node's degree to exactly leader_edges by adding/removing edges """
        current_degree = G.degree(node)

        if current_degree < leader_edges:
            # Add edges to random nodes that are not already connected
            for other_node in range(num_agents):
                if not G.has_edge(node, other_node) and node != other_node:
                    G.add_edge(node, other_node)
                    if G.degree(node) >= leader_edges:
                        break
        elif current_degree > leader_edges:
            # Remove edges randomly until we reach the target degree
            neighbors = list(G.neighbors(node))
            while G.degree(node) > leader_edges:
                G.remove_edge(node, neighbors.pop())

    # Adjust leader nodes to have exactly 15 edges
    adjust_node_degree(leader1, leader_edges)
    adjust_node_degree(leader2, leader_edges)

    for node, degree in sorted_nodes[2:]:
        if degree > other_edges_limits:
            adjust_node_degree(node, other_edges_limits)

    return G


def generate_random_network(num_agents, p):
    G = nx.erdos_renyi_graph(n=num_agents, p=p)
    return G


def analyze_network(G):
    clustering_coefficient = nx.average_clustering(G)
    try:
        average_path_length = nx.average_shortest_path_length(G)
    except nx.NetworkXError:
        average_path_length = float('inf')

    print(f"clustering_coefficient: {clustering_coefficient}")
    print(f"average_path_length: {average_path_length}")


def generate_network(network_type, num_agents, **kwargs):
    if network_type == "small_world":
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.1)
        return generate_small_world_network(num_agents, k, p)
    elif network_type == "scale_free":
        m = kwargs.get('m', 3)
        # return generate_scale_free_network(num_agents, m)
        return generate_scale_free_network_new(num_agents, m, 15, 7)

    elif network_type == "random":
        p = kwargs.get('p', 0.1)
        return generate_random_network(num_agents, p)
    else:
        raise ValueError("Unsupported network type")


def visualize_network(G):
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
    plt.title("Network Visualization")
    plt.show()
