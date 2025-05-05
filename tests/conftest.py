import os
import sys
import pytest
from pathlib import Path
import subprocess

# Ensure project root is on sys.path so that `import src` works
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

@pytest.fixture(scope="session")
def small_vrp_path():
    """Path to a small CVRP instance file for component tests"""
    return Path(repo_root) / "src" / "benchmarking" / "cvrp_instances" / "X-n101-k25.vrp"

@pytest.fixture(scope="session")
def small_sol_path():
    """Path to the solution file for the small CVRP instance"""
    return Path(repo_root) / "src" / "benchmarking" / "cvrp_instances" / "X-n101-k25.sol"

@pytest.fixture(scope="session")
def mini_yaml():
    """Path to the minimal YAML config for integration tests"""
    return Path(repo_root) / "tests" / "_assets" / "smoke" / "mini.yaml"

@pytest.fixture(scope="session")
def mini_demand_csv():
    """Path to the minimal demand CSV for integration tests"""
    return Path(repo_root) / "tests" / "_assets" / "smoke" / "mini_demand.csv"

@pytest.fixture(autouse=True)
def tmp_results_dir(tmp_path, monkeypatch):
    """Redirect the project's results directory into a temp folder for integration tests"""
    # Create fake results dir
    fake_results = tmp_path / "results"
    fake_results.mkdir()
    # Run CLI commands from project root, but ensure results go to fake_results
    monkeypatch.setenv("PROJECT_RESULTS_DIR", str(fake_results))
    # Also redirect working directory to project root just in case
    monkeypatch.chdir(Path(__file__).parent.parent)
    return fake_results

@pytest.fixture(autouse=True)
def stub_heavy_steps(tmp_results_dir, monkeypatch):
    """Stub clustering, optimization, and saving to keep integration tests fast"""
    import pandas as pd
    from pandas import DataFrame
    from pathlib import Path
    import src.utils.data_processing as dp_module
    import src.main as main_mod

    # Stub clustering: return one trivial cluster
    monkeypatch.setattr(
        "src.clustering.generate_clusters_for_configurations",
        lambda *args, **kwargs: DataFrame([{
            "Cluster_ID": "c1",
            "Customers": ["1", "2"],
            "Total_Demand": {"Dry": 1, "Chilled": 0, "Frozen": 0},
            "Centroid_Latitude": 0.0,
            "Centroid_Longitude": 0.0,
            "Route_Time": 0.1,
            "Config_ID": 1
        }])
    )
    # Stub optimizer: return minimal solution dict
    monkeypatch.setattr(
        "src.fsm_optimizer.solve_fsm_problem",
        lambda *args, **kwargs: {
            "selected_clusters": args[0] if args else pd.DataFrame(),
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 100,
            "total_variable_cost": 0,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_penalties": 0,
            "solver_name": "stub",
            "solver_status": "Optimal"
        }
    )
    # Stub data processing: demand file loader and profiles dir
    asset_dir = Path(__file__).parent.parent / "tests" / "_assets"
    monkeypatch.setattr(dp_module, "get_demand_profiles_dir", lambda: asset_dir)
    monkeypatch.setattr(
        dp_module,
        "load_customer_demand",
        lambda demand_file: pd.DataFrame([{
            'Customer_ID': 'C1', 'Latitude': 0.0, 'Longitude': 0.0,
            'Dry_Demand': 1, 'Chilled_Demand': 0, 'Frozen_Demand': 0
        }])
    )
    # Stub main module's loader to use our dp_module stub
    monkeypatch.setattr(main_mod, "load_customer_demand", dp_module.load_customer_demand)
    # Stub clustering import in main to return one trivial cluster
    monkeypatch.setattr(
        main_mod,
        "generate_clusters_for_configurations",
        lambda *args, **kwargs: pd.DataFrame([{
            "Cluster_ID": "c1",
            "Customers": ["1", "2"],
            "Total_Demand": {"Dry": 1, "Chilled": 0, "Frozen": 0},
            "Centroid_Latitude": 0.0,
            "Centroid_Longitude": 0.0,
            "Route_Time": 0.1,
            "Config_ID": 1
        }])
    )
    # Stub optimize import in main to return minimal solution
    monkeypatch.setattr(
        main_mod,
        "solve_fsm_problem",
        lambda *args, **kwargs: {
            "selected_clusters": args[0] if args else pd.DataFrame(),
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 100,
            "total_variable_cost": 0,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_penalties": 0,
            "solver_name": "stub",
            "solver_status": "Optimal"
        }
    )
    # Stub save_optimization_results in main module to our fake_save
    def fake_save(*args, **kwargs):
        fmt = kwargs.get('format', 'excel')
        ext = 'xlsx' if fmt == 'excel' else 'json'
        f = Path(tmp_results_dir) / f"output.{ext}"
        f.write_text('dummy')
    monkeypatch.setattr(main_mod, "save_optimization_results", fake_save)

    # Stub save_optimization_results to write a dummy file into tmp_results_dir
    def fake_save(*args, **kwargs):
        fmt = kwargs.get('format', 'excel')
        ext = 'xlsx' if fmt == 'excel' else 'json'
        f = Path(tmp_results_dir) / f"output.{ext}"
        f.write_text('dummy')
    monkeypatch.setattr(
        "src.utils.save_results.save_optimization_results",
        fake_save
    )

# Toy data fixtures for FSM tests
@pytest.fixture
def toy_fsm_core_data():
    import pandas as pd
    from src.config.parameters import Parameters
    clusters_df = pd.DataFrame({
        'Cluster_ID': [1],
        'Customers': [['C1']],
        'Total_Demand': [{'Dry': 1, 'Chilled': 0, 'Frozen': 0}],
        'Config_ID': [1],
        'Centroid_Latitude': [0.0],
        'Centroid_Longitude': [0.0],
        'Route_Time': [1.0],
        'Method': ['test']
    })
    config_df = pd.DataFrame([{
        'Config_ID': 1,
        'Vehicle_Type': 'X',
        'Capacity': 10,
        'Fixed_Cost': 5,
        'Dry': 1,
        'Chilled': 0,
        'Frozen': 0
    }])
    customers_df = pd.DataFrame([{'Customer_ID': 'C1', 'Dry_Demand': 0, 'Chilled_Demand': 0, 'Frozen_Demand': 0}])
    params = Parameters.from_yaml()
    return clusters_df, config_df, customers_df, params

@pytest.fixture
def toy_fsm_edge_data():
    import pandas as pd
    from src.config.parameters import Parameters
    clusters_df = pd.DataFrame([{
        'Cluster_ID': 1,
        'Customers': ['C1', 'C2'],
        'Total_Demand': {'Dry': 2, 'Chilled': 0, 'Frozen': 0},
        'Route_Time': 1.0
    }])
    config_df = pd.DataFrame([{
        'Config_ID': 1,
        'Vehicle_Type': 'A',
        'Capacity': 5,
        'Fixed_Cost': 10,
        'Dry': 1,
        'Chilled': 0,
        'Frozen': 0
    }])
    params = Parameters.from_yaml()
    return clusters_df, config_df, params

# Toy data fixture for FSM model build tests
@pytest.fixture
def toy_fsm_model_build_data():
    import pandas as pd
    from src.config.parameters import Parameters
    clusters_df = pd.DataFrame([{
        'Cluster_ID': 'k1',
        'Customers': [1, 2],
        'Total_Demand': {'Dry': 5, 'Chilled': 0, 'Frozen': 0},
        'Route_Time': 1.0
    }])
    config_df = pd.DataFrame([{
        'Config_ID': 'v1',
        'Capacity': 10,
        'Fixed_Cost': 100,
        'Dry': 1,
        'Chilled': 0,
        'Frozen': 0
    }])
    params = Parameters.from_yaml('src/config/default_config.yaml')
    return clusters_df, config_df, params

# Smoke dataset fixtures
@pytest.fixture(scope="session")
def smoke_cluster_data():
    """Load the canonical smoke cluster configuration from YAML."""
    import yaml
    from pathlib import Path
    path = Path(__file__).parent / "_assets" / "smoke" / "mini.yaml"
    with path.open() as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="session")
def smoke_demand_data():
    """Load the canonical smoke demand data from CSV."""
    import csv
    from pathlib import Path
    path = Path(__file__).parent / "_assets" / "smoke" / "mini_demand.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        return list(reader) 