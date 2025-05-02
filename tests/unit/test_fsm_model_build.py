import pandas as pd
import pulp
import pytest

from src.fsm_optimizer import _create_model
from src.config.parameters import Parameters


def test_create_model_basic():
    # Create a single cluster with two customers
    clusters_df = pd.DataFrame([{
        'Cluster_ID': 'k1',
        'Customers': [1, 2],
        'Total_Demand': {'Dry': 5, 'Chilled': 0, 'Frozen': 0},
        'Route_Time': 1.0  # dummy route time
    }])
    # Single vehicle configuration that can serve Dry
    configurations_df = pd.DataFrame([{
        'Config_ID': 'v1',
        'Capacity': 10,
        'Fixed_Cost': 100,
        'Dry': 1,
        'Chilled': 0,
        'Frozen': 0
    }])
    # Load default parameters from YAML
    params = Parameters.from_yaml('src/config/default_config.yaml')

    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations_df, params)

    # Model should be a pulp problem
    assert isinstance(model, pulp.LpProblem)
    # y_vars contains our cluster
    assert 'k1' in y_vars
    # x_vars contains decision for (v1,k1)
    assert ('v1', 'k1') in x_vars
    # c_vk has a cost entry
    assert ('v1', 'k1') in c_vk

    # Check that customer coverage constraints exist for both customers
    cons_names = list(model.constraints.keys())
    assert any('Customer_Coverage_1' in name for name in cons_names)
    assert any('Customer_Coverage_2' in name for name in cons_names)

    # Objective should include x_v1_k1
    obj_str = str(model.objective)
    assert 'x_v1_k1' in obj_str 