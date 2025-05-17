"""
merge_phase module

This module provides functions for improving the solution quality through cluster merging after the initial optimization.
"""

# Re-export public functions from core
from .core import (
    # Main public function
    improve_solution,
    
    # Other functions that might be used externally
    generate_merge_phase_clusters,
    validate_merged_cluster
)

# Define what gets imported with "from fleetmix.merge_phase import *"
__all__ = [
    'improve_solution',
    'generate_merge_phase_clusters',
    'validate_merged_cluster'
] 