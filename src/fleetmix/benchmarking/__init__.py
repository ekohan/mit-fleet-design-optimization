"""benchmarking package

Provides scripts and parsers that convert *standard VRP benchmark instances* (e.g., CVRP, flexible
MCVRP) into the data structures required by the fleet-design heuristic so researchers can reproduce
Tables 2–3 in Sections 5–6 of the paper.

Key entry points
----------------
• `run_benchmark.py`      – batch runner for synthetic CVRP → FSM tests.
• `run_all_mcvrp.py`      – runs Henke & Hübner (2015) instances through the pipeline.
• `run_all_cvrp.py`       – runs Uchoa et al. (2017) instances through the pipeline.
• `vrp_solver.py`         – thin wrapper around PyVRP to obtain lower/upper bounds.

All helper functions are re-exported via `__all__` for convenience.
"""

# Import model classes
from .models import MCVRPInstance, CVRPInstance, CVRPSolution

# Import parser functions and converter functions
from .parsers.mcvrp_parser import parse_mcvrp
from .parsers.cvrp_parser import CVRPParser
from .mcvrp_to_fsm import convert_mcvrp_to_fsm
from .cvrp_converter import convert_cvrp_to_fsm, CVRPBenchmarkType
from .vrp_interface import VRPType, convert_to_fsm

__all__ = [
    # Models
    "MCVRPInstance",
    "CVRPInstance", 
    "CVRPSolution",
    # Parsers
    "parse_mcvrp",
    "CVRPParser",
    # Converters
    "convert_mcvrp_to_fsm",
    "convert_cvrp_to_fsm",
    "CVRPBenchmarkType",
    "VRPType",
    "convert_to_fsm"
]
