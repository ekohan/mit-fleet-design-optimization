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
import fleetmix.clustering as clustering_module
import fleetmix.optimization as optimization_module

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
    monkeypatch.setattr("fleetmix.utils.data_processing.load_customer_demand", full_stub_customer_demand)
    
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
        optimization_module,
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
    # Import locally to avoid circular dependencies during test collection
    import fleetmix.utils.save_results as save_module_local

    def fake_save(*args, **kwargs):
        # Get the format from kwargs or default to excel
        format = kwargs.get('format', 'excel')
        print(f"Using stubbed save_results to save {format} output")
        
        # Create appropriate file extension
        ext = "xlsx" if format == "excel" else "json"
        (Path(output_dir) / f"output.{ext}").write_text("dummy")
    
    monkeypatch.setattr(save_module_local, "save_optimization_results", fake_save)
    yield

@contextlib.contextmanager
def stub_vrplib(monkeypatch):
    """Stub CVRPParser in src.benchmarking.cvrp_to_fsm and bypass missing-file guard for .vrp."""
    import fleetmix.benchmarking.cvrp_to_fsm as mod
    # Monkey-patch Path.exists to return True for .vrp under datasets/cvrp
    from pathlib import Path
    orig_exists = Path.exists
    def fake_exists(self):
        if self.suffix == '.vrp' and 'datasets/cvrp' in str(self):
            return True
        return orig_exists(self)
    monkeypatch.setattr(Path, 'exists', fake_exists)

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
def stub_mcvrp_parser(monkeypatch):
    """Stub MCVRP parser to return a dummy instance."""
    from fleetmix.benchmarking.models import MCVRPInstance
    from pathlib import Path
    
    def stub_parse_mcvrp(path):
        # Create a dummy MCVRP instance
        return MCVRPInstance(
            name="stub_instance",
            source_file=Path("stub_path"),
            dimension=3,
            capacity=100,
            vehicles=2,
            depot_id=1,
            coords={1: (0, 0), 2: (1, 1), 3: (2, 2)},
            demands={1: (0, 0, 0), 2: (10, 0, 0), 3: (0, 5, 5)}
        )
    
    monkeypatch.setattr("fleetmix.benchmarking.mcvrp_parser.parse_mcvrp", stub_parse_mcvrp)
    
    # Also patch Path.exists to return True for the MCVRP instance
    orig_exists = Path.exists
    def fake_exists(self):
        if self.name.endswith('.dat') and 'mcvrp' in str(self):
            return True
        return orig_exists(self)
    monkeypatch.setattr(Path, 'exists', fake_exists)
    
    yield

@contextlib.contextmanager
def stub_vehicle_configurations(monkeypatch):
    """Stub generate_vehicle_configurations in src.benchmarking.cvrp_to_fsm."""
    import pandas as pd
    import fleetmix.benchmarking.cvrp_to_fsm as mod
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
    import fleetmix.benchmarking.cvrp_to_fsm as mod
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
    
    # Patch all known imports in fleetmix modules
    import fleetmix.utils.data_processing as dp_mod
    import fleetmix.main as main_mod
    import fleetmix.benchmarking.run_benchmark as rb_mod
    monkeypatch.setattr(dp_mod, "load_customer_demand", fake_demand)
    monkeypatch.setattr(main_mod, "load_customer_demand", fake_demand)
    monkeypatch.setattr(rb_mod, "load_customer_demand", fake_demand)
    
    yield 