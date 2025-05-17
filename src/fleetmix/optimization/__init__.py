"""
optimization module

This module provides functions for solving the Fleet Size-and-Mix optimization problem.
"""

# Re-export public functions from merge_phase
from .merge_phase import (
    # Main public function
    solve_fsm_problem,
    
    # Internal functions used in tests
    _create_model,
    _extract_solution,
    _validate_solution,
    _calculate_solution_statistics,
    _calculate_cluster_cost
)

# Define what gets imported with "from fleetmix.optimization import *"
__all__ = [
    'solve_fsm_problem',
    '_create_model',
    '_extract_solution',
    '_validate_solution',
    '_calculate_solution_statistics',
    '_calculate_cluster_cost'
] 