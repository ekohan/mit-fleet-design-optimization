from src.core_types import BenchmarkType # noqa: F401

# BenchmarkType enum is now defined in src.core_types.py

class BenchmarkType(Enum):
    """Types of VRP benchmarks."""
    SINGLE_COMPARTMENT = "single_compartment"  # Upper bound - Separate VRPs per product
    MULTI_COMPARTMENT = "multi_compartment"    # Lower bound - Aggregate demand, post-process for compartments