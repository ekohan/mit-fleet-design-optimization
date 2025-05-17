"""
clustering module

This module provides functions and classes for clustering customers in the fleet optimization process.
"""

# Re-export public functions and classes from generator, heuristics, and common modules
from .common import (
    # Classes
    Cluster,
    ClusteringSettings,
)

from .generator import (
    # Main public function
    generate_clusters_for_configurations,
    
    # Functions used in tests
    _is_customer_feasible,
)

from .heuristics import (
    # Functions used in tests or by other modules
    compute_composite_distance,
    estimate_num_initial_clusters,
    get_cached_demand,
)

# Define what gets imported with "from fleetmix.clustering import *"
__all__ = [
    'generate_clusters_for_configurations',
    'Cluster',
    'ClusteringSettings',
    'compute_composite_distance',
    'estimate_num_initial_clusters',
    '_is_customer_feasible',
    'get_cached_demand',
] 