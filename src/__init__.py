"""
MIT Fleet Design Optimisation – Python Package

This package accompanies the research paper *Designing Multi-Compartment Vehicle Fleets for
Last-Mile Food Distribution Systems* and exposes a clear API for:

1. **Clustering** customers into capacity- and time-feasible groups (`src.clustering`).
2. **Optimising** the fleet size-and-mix via a mixed-integer linear model (`src.fsm_optimizer`).
3. Optional **merge phase** improvements (`src.merge_phase`).
4. **Utilities** for route-time estimation, logging, data conversion, and benchmarking.

Typical high-level workflow
--------------------------
>>> import pandas as pd, src
>>> clusters = src.clustering.generate_clusters_for_configurations(customers, configs, params)
>>> sol      = src.fsm_optimizer.solve_fsm_problem(clusters, configs, customers, params)
>>> sol      = src.merge_phase.improve_solution(sol, configs, customers, params)

All core functions follow Google-style docstrings and are directly linked to the sections and
equations of the paper so readers can cross-reference implementation with methodology.
"""

# Export the fleetmix package
from . import fleetmix
