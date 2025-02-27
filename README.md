# Network Topology Reconfiguration

A Python project for reconfiguring network topologies based on traffic patterns using the Gale-Shapley matching algorithm.

## Features
- Dynamic topology reconfiguration based on traffic matrices
- Gale-Shapley stable matching for optimal link allocation
- Network visualization with NetworkX and Matplotlib
- Configurable parameters for number of ToRs, ports, and link capacities

## Requirements
- Python 3.6+
- NumPy
- Matplotlib
- NetworkX

## Usage
```python
# Example usage
import numpy as np
from topology import reconfigure_topology, visualize_topology

# Create a traffic matrix
traffic_matrix = np.array([
    [0, 5, 10, 2],
    [5, 0, 3, 8],
    [10, 3, 0, 7],
    [2, 8, 7, 0]
])

# Reconfigure topology with 2 ports per ToR
topology, connectivity = reconfigure_topology(traffic_matrix, num_port=2, base_capacity=10)

# Visualize the resulting topology
visualize_topology(topology, output_file="my_topology.png")
