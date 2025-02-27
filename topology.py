"""
Network topology reconfiguration module.
Implements dynamic topology adjustment based on traffic matrices using the Gale-Shapley algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def get_initial_connectivity(num_tors):
    """
    Initialize the connectivity matrix with all ToRs connected to each other.
    
    Args:
        num_tors: Number of Top-of-Rack switches
    
    Returns:
        connectivity: Matrix representing possible physical connections
    """
    connectivity = np.ones((num_tors, num_tors))
    np.fill_diagonal(connectivity, 0)
    return connectivity

def gale_shapley_matching(preferences_proposers, preferences_receivers, num_ports, traffic_matrix):
    """
    Implementation of the Gale-Shapley algorithm for stable matching.
    
    Args:
        preferences_proposers: List of preference lists for proposing ToRs
        preferences_receivers: List of preference lists for receiving ToRs
        num_ports: Number of ports per ToR
        traffic_matrix: Matrix of traffic demands
    
    Returns:
        matches: Dictionary of matching ToR pairs
    """
    num_tors = len(preferences_proposers)
    free_proposers = list(range(num_tors))
    matches = {}
    reverse_matches = {}
    remaining_ports = {i: num_ports for i in range(num_tors)}
    next_proposal = {i: 0 for i in range(num_tors)}

    while free_proposers:
        proposer = free_proposers[0]
        if next_proposal[proposer] >= num_tors or remaining_ports[proposer] <= 0:
            free_proposers.pop(0)
            continue

        receiver = preferences_proposers[proposer][next_proposal[proposer]]
        next_proposal[proposer] += 1

        if receiver == proposer or traffic_matrix[proposer, receiver] <= 0:
            continue

        if remaining_ports[receiver] > 0:
            matches[proposer] = receiver
            reverse_matches[receiver] = proposer
            remaining_ports[proposer] -= 1
            remaining_ports[receiver] -= 1
            free_proposers.pop(0)
        else:
            continue

    return matches

def reconfigure_topology(traffic_matrix, num_port, base_capacity=10):
    """
    Reconfigure network topology based on traffic demands.
    
    Args:
        traffic_matrix: Matrix of traffic demands between ToRs
        num_port: Number of ports per ToR
        base_capacity: Base capacity per link in Gbps
    
    Returns:
        topology_capacities: Matrix of link capacities
        connectivity: Matrix of possible physical connections
    """
    num_tors = traffic_matrix.shape[0]
    if traffic_matrix.shape != (num_tors, num_tors):
        raise ValueError(f"Traffic matrix must be {num_tors}x{num_tors}")
    
    if np.all(traffic_matrix == 0):
        print("Traffic matrix is all zeros. Returning empty topology.")
        return np.zeros((num_tors, num_tors)), get_initial_connectivity(num_tors)

    weight_matrix = (traffic_matrix + traffic_matrix.T).astype(float)
    
    weight_matrix_for_prefs = weight_matrix.copy()
    weight_matrix_for_prefs[weight_matrix_for_prefs == 0] = -np.inf

    preferences_proposers = [np.argsort(-weight_matrix_for_prefs[i]).tolist() for i in range(num_tors)]
    preferences_receivers = [np.argsort(-weight_matrix_for_prefs[:, i]).tolist() for i in range(num_tors)]

    matches = gale_shapley_matching(preferences_proposers, preferences_receivers, num_port, weight_matrix)

    topology = np.zeros((num_tors, num_tors))
    connectivity = get_initial_connectivity(num_tors)

    for proposer, receiver in matches.items():
        if topology[proposer, receiver] == 0:
            topology[proposer, receiver] = 1
            topology[receiver, proposer] = 1

    remaining_out_ports = np.full(num_tors, num_port) - np.sum(topology, axis=1)
    for i in range(num_tors):
        for j in range(i + 1, num_tors):
            if connectivity[i, j] > 0 and remaining_out_ports[i] > 0 and remaining_out_ports[j] > 0 and weight_matrix[i, j] > 0:
                ports_to_add = min(remaining_out_ports[i], remaining_out_ports[j])
                topology[i, j] += ports_to_add
                topology[j, i] += ports_to_add
                remaining_out_ports[i] -= ports_to_add
                remaining_out_ports[j] -= ports_to_add

    topology_capacities = topology * base_capacity

    active_links = int(np.sum(connectivity * (topology > 0)) / 2)
    used_ports = num_tors * num_port - int(sum(remaining_out_ports))
    avg_capacity = np.mean(topology_capacities[topology_capacities > 0]) if np.any(topology_capacities > 0) else 0
    print(f"Generated topology with {active_links} links, {used_ports}/{num_tors * num_port} ports used")
    print(f"Average capacity per link: {avg_capacity:.2f} Gbps")
    print(f"Topology capacities:\n{topology_capacities}")

    return topology_capacities, connectivity

def visualize_topology(topology, time_index=0, output_file="topology_clear.png"):
    """
    Visualize the network topology as a graph.
    
    Args:
        topology: Matrix of link capacities
        time_index: Time index for the visualization title
        output_file: Output file name for saving the visualization
    """
    G = nx.Graph()
    num_tors = topology.shape[0]
    for i in range(num_tors):
        G.add_node(i)
    for i in range(num_tors):
        for j in range(i + 1, num_tors):
            if topology[i, j] > 0:
                G.add_edge(i, j, weight=topology[i, j])

    plt.figure(figsize=(10, 8), dpi=200)
    pos = nx.spring_layout(G, seed=42, k=1.5)

    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold', font_color='black')

    edge_capacities = [topology[i, j] for i, j in G.edges()]
    if edge_capacities:
        max_capacity = max(edge_capacities)
        widths = [2 + 6 * (c / max_capacity) for c in edge_capacities]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.8, edge_color='gray')

    edge_labels = {(i, j): f"{int(topology[i, j])} Gbps" for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_weight='bold')

    plt.title(f"Reconfigured Network Topology (4 ToRs, Time {time_index})", fontsize=16, pad=20)
    plt.axis('off')

    active_links = len(G.edges())
    avg_capacity = np.mean(edge_capacities) if edge_capacities else 0
    info_text = f"Stats:\nToRs: {num_tors}\nLinks: {active_links}\nAvg Capacity: {avg_capacity:.1f} Gbps"
    plt.figtext(0.02, 0.02, info_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

    plt.savefig(output_file, bbox_inches='tight', dpi=200)
    print(f"Topology image saved as {output_file}")
    plt.show()
    plt.close()
