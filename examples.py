import numpy as np
import matplotlib.pyplot as plt
from topology import reconfigure_topology, visualize_topology

def example_1():
    """Basic usage with 4 ToRs and varying traffic."""
    num_tors = 4
    num_port = 2
    
    # Create a sample traffic matrix with some traffic between ToRs
    traffic_matrix = np.array([
        [0, 5, 10, 2],
        [5, 0, 3, 8],
        [10, 3, 0, 7],
        [2, 8, 7, 0]
    ])
    
    print(f"Example 1 - Traffic Matrix:\n{traffic_matrix}")
    topology, connectivity = reconfigure_topology(traffic_matrix, num_port, base_capacity=10)
    visualize_topology(topology, output_file="example1_topology.png")

def example_2():
    """Larger network with 8 ToRs."""
    num_tors = 8
    num_port = 3
    
    # Create a random traffic matrix for 8 ToRs
    np.random.seed(42)
    traffic_matrix = np.random.randint(0, 15, size=(num_tors, num_tors))
    np.fill_diagonal(traffic_matrix, 0)  # No traffic to self
    
    print(f"Example 2 - Traffic Matrix:\n{traffic_matrix}")
    topology, connectivity = reconfigure_topology(traffic_matrix, num_port, base_capacity=10)
    visualize_topology(topology, output_file="example2_topology.png")

if __name__ == "__main__":
    example_1()
    example_2()
