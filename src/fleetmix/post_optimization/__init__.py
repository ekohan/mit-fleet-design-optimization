"""
post_optimization module

This module provides functions for improving the solution quality after the initial optimization.
"""

# Re-export public functions from core
from .core import (
    # Main public function
    improve_solution,
    
    # Other functions that might be used externally
    generate_post_optimization_merges,
    validate_merged_cluster
)

# Define what gets imported with "from fleetmix.post_optimization import *"
__all__ = [
    'improve_solution',
    'generate_post_optimization_merges',
    'validate_merged_cluster'
] 