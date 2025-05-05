"""Context managers for stubbing external dependencies in tests.

This module provides a collection of context manager functions that monkey-patch
heavy or external dependencies to make tests faster and more reliable. Instead of
monkeypatching in each test or using auto-use fixtures, these context managers
make the stubbing explicit and composable.

Usage example:

    def test_my_pipeline(monkeypatch, tmp_path):
        with stub_data_processing(monkeypatch), stub_clustering(monkeypatch), 
             stub_solver(monkeypatch), stub_save_results(monkeypatch, tmp_path):
            # Call code that would use these dependencies
            my_pipeline_function()
            
            # Now make assertions
            assert ...

The stubs are designed to be composable, so you can use only what you need.
"""
import contextlib
from pathlib import Path
import pandas as pd
import src.utils.data_processing
import src.utils.data_processing as dp_module
import src.clustering as clustering_module
import src.fsm_optimizer as fsm_module
import src.utils.save_results as save_module

@contextlib.contextmanager
def stub_data_processing(monkeypatch):
    """Stub demand file loading from data_processing."""
    # Create a COMPLETE replacement for load_customer_demand that avoids using the original function
    
    def full_stub_customer_demand(demand_file: str):
        print(f"Using STUB customer demand (ignoring {demand_file})")
        return pd.DataFrame([
            {'Customer_ID': 'C1', 'Latitude': 0.0, 'Longitude': 0.0,
             'Dry_Demand': 10, 'Chilled_Demand': 0, 'Frozen_Demand': 0},
            {'Customer_ID': 'C2', 'Latitude': 1.0, 'Longitude': 1.0,
             'Dry_Demand': 5, 'Chilled_Demand': 0, 'Frozen_Demand': 0}
        ])
    
    # Patch at the module level to completely replace the function
    monkeypatch.setattr("src.utils.data_processing.load_customer_demand", full_stub_customer_demand)
    
    yield

@contextlib.contextmanager
def stub_clustering(monkeypatch):
    """Stub clustering to return one trivial cluster."""
    monkeypatch.setattr(
        clustering_module,
        "generate_clusters_for_configurations",
        lambda *args, **kwargs: pd.DataFrame([{
            "Cluster_ID": "c1",
            "Customers": ["C1"],
            "Total_Demand": {"Dry":1, "Chilled":0, "Frozen":0},
            "Centroid_Latitude": 0.0,
            "Centroid_Longitude": 0.0,
            "Route_Time": 0.1,
            "Config_ID": 1
        }])
    )
    yield

@contextlib.contextmanager
def stub_solver(monkeypatch):
    """Stub FSM solver to return empty but valid solution."""
    monkeypatch.setattr(
        fsm_module,
        "solve_fsm_problem",
        lambda *args, **kwargs: {
            "selected_clusters": pd.DataFrame(),
            "missing_customers": set(),
            "vehicles_used": {},
            "total_fixed_cost": 0,
            "total_variable_cost": 0,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_penalties": 0,
            "solver_name": "stub",
            "solver_status": "Optimal"
        }
    )
    yield

@contextlib.contextmanager
def stub_save_results(monkeypatch, output_dir):
    """Stub save_optimization_results to write dummy output files."""
    def fake_save(*args, **kwargs):
        # Get the format from kwargs or default to excel
        format = kwargs.get('format', 'excel')
        print(f"Using stubbed save_results to save {format} output")
        
        # Create appropriate file extension
        ext = "xlsx" if format == "excel" else "json"
        (Path(output_dir) / f"output.{ext}").write_text("dummy")
    
    # Import and patch the module
    import src.utils.save_results
    monkeypatch.setattr(src.utils.save_results, "save_optimization_results", fake_save)
    yield

@contextlib.contextmanager
def stub_vrplib(monkeypatch):
    """Stub CVRPParser in src.benchmarking.cvrp_to_fsm."""
    import src.benchmarking.cvrp_to_fsm as mod
    # Dummy parser with minimal behavior
    class DummyParser:
        def __init__(self, path): pass
        def parse(self):
            class Inst:
                demands = {1: 1, 2: 1}
                capacity = 1
                num_vehicles = 1
                coordinates = {1: (0, 0), 2: (1, 1)}
                depot_id = 1
            return Inst()
        def parse_solution(self):
            return {'routes': [], 'cost': 0}
    monkeypatch.setattr(mod, 'CVRPParser', DummyParser)
    yield

@contextlib.contextmanager
def stub_vehicle_configurations(monkeypatch):
    """Stub generate_vehicle_configurations in src.benchmarking.cvrp_to_fsm."""
    import pandas as pd
    import src.benchmarking.cvrp_to_fsm as mod
    monkeypatch.setattr(
        mod,
        'generate_vehicle_configurations',
        lambda vehicles, goods: pd.DataFrame([
            {'Config_ID': 1, 'Vehicle_Type': 'A', 'Capacity': 1, 'Fixed_Cost': 0, 'Dry': 1, 'Chilled': 0, 'Frozen': 0}
        ])
    )
    yield

@contextlib.contextmanager
def stub_benchmark_clustering(monkeypatch):
    """Stub generate_clusters_for_configurations in src.benchmarking.cvrp_to_fsm."""
    import pandas as pd
    import src.benchmarking.cvrp_to_fsm as mod
    monkeypatch.setattr(
        mod,
        'generate_clusters_for_configurations',
        lambda *args, **kwargs: pd.DataFrame([
            {"Cluster_ID": "c1", "Customers": ["C1"],
             "Total_Demand": {"Dry": 1, "Chilled": 0, "Frozen": 0},
             "Centroid_Latitude": 0.0, "Centroid_Longitude": 0.0,
             "Route_Time": 0.1, "Config_ID": 1}
        ])
    )
    yield

@contextlib.contextmanager
def stub_demand(monkeypatch):
    """
    Stub customer demand loading for imports in any module.
    This is a more general solution than stub_data_processing.
    """
    def fake_demand(*args, **kwargs):
        print("Using explicit stubbed demand data")
        return pd.DataFrame([
            {'Customer_ID': 'C1', 'Latitude': 0.0, 'Longitude': 0.0,
             'Dry_Demand': 10, 'Chilled_Demand': 0, 'Frozen_Demand': 0},
            {'Customer_ID': 'C2', 'Latitude': 1.0, 'Longitude': 1.0,
             'Dry_Demand': 5, 'Chilled_Demand': 0, 'Frozen_Demand': 0}
        ])
    
    # Find all modules that might import this function
    import src.main
    import src.utils.data_processing
    import src.benchmarking.run_benchmark
    
    # Patch all known imports
    monkeypatch.setattr(src.utils.data_processing, "load_customer_demand", fake_demand)
    monkeypatch.setattr(src.main, "load_customer_demand", fake_demand)
    monkeypatch.setattr(src.benchmarking.run_benchmark, "load_customer_demand", fake_demand)
    
    yield 