#!/bin/bash

# Create directories
mkdir -p data/vrp_instances

# Clone the repository
git clone https://github.com/PyVRP/Instances.git temp_instances

# Move CVRP instances to the target directory
cp temp_instances/CVRP/*.vrp data/vrp_instances/

# Clean up
rm -rf temp_instances

echo "VRP instances downloaded to data/vrp_instances/"
