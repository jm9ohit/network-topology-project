"""
Examples showcasing the topology reconfiguration module.
This module demonstrates how to use the topology module for different scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
from topology import reconfigure_topology, visualize_topology

def test_with_4_tors():
    """
    Test the topology reconfiguration with 4 ToRs and different traffic patterns.
    """
    num_tors = 4  # Used for traffic matrix dimensions
    num_port = 2
    base_capacity = 10

    # Empty traffic matrix
    traffic_matrix = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    print(f"Sample Traffic Matrix:\n{traffic_matrix}")
    topology, _ = reconfigure_topology(traffic_matrix, num_port, base_capacity)
    visualize_topology(topology, output_file="topology_2ports.png")
    
    # Null traffic matrix test
    null_matrix = np.zeros((4, 4))
    print(f"\nNull Traffic Matrix:\n{null_matrix}")
    null_topology, _ = reconfigure_topology(null_matrix, num_port, base_capacity)
    visualize_topology(null_topology, output_file="topology_null.png")

if __name__ == "__main__":
    test_with_4_tors()
