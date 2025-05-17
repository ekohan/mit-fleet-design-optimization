import logging
import pandas as pd

import src.fsm_optimizer as fsm
from src.config.parameters import Parameters

def test_create_model_counts(toy_fsm_core_data):
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    model, y_vars, x_vars, c_vk = fsm._create_model(clusters_df, config_df, params)
    # Exactly one cluster variable and one assignment x-var
    assert len(y_vars) == 1, "Should create one y var"
    assert len(x_vars) == 1, "Should create one x var per vehicle-cluster"
    # Constraint names include customer coverage and vehicle assignment
    cons_names = list(model.constraints.keys())
    assert any('Customer_Coverage_' in name for name in cons_names)
    assert any('Vehicle_Assignment_' in name for name in cons_names)


def test_extract_solution():
    import pulp
    # Build clusters DataFrame
    clusters_df = pd.DataFrame({
        'Cluster_ID': [1, 2],
        'Customers': [['C1'], ['C2']]
    })
    # Create y-vars: only cluster 1 selected
    y1 = pulp.LpVariable('y_1', cat='Binary')
    y2 = pulp.LpVariable('y_2', cat='Binary')
    y1.varValue = 1
    y2.varValue = 0
    y_vars = {1: y1, 2: y2}
    # Create x-vars: assign vehicle 10 to cluster 1, vehicle 20 to cluster 2
    xA1 = pulp.LpVariable('x_10_1', cat='Binary')
    xB2 = pulp.LpVariable('x_20_2', cat='Binary')
    xA1.varValue = 1
    xB2.varValue = 1
    x_vars = {(10, 1): xA1, (20, 2): xB2}
    selected = fsm._extract_solution(clusters_df, y_vars, x_vars)
    # Only cluster 1 should be selected, with Config_ID mapped to 10
    assert list(selected['Cluster_ID']) == [1]
    assert list(selected['Config_ID']) == [10]


def test_capacity_violation_model_warning(toy_fsm_core_data, caplog):
    clusters_df, config_df, customers_df, params = toy_fsm_core_data
    # Build base data and violate capacity so no config is feasible
    clusters_df.at[0, 'Total_Demand'] = {'Dry': 100, 'Chilled': 0, 'Frozen': 0}
    # Capture warnings from model construction
    caplog.set_level(logging.WARNING, logger='src.fsm_optimizer')
    # Create model
    model, y_vars, x_vars, c_vk = fsm._create_model(clusters_df, config_df, params)
    # Assert that 'NoVehicle' variable was injected for unserviceable cluster
    assert any(v == 'NoVehicle' for v, k in x_vars.keys()), "Should inject NoVehicle for infeasible cluster"
    # Check warning about unserviceable cluster
    assert any(
        rec[0] == 'src.fsm_optimizer' and 
        rec[1] == logging.WARNING and
        'serve' in rec[2].lower()
        for rec in caplog.record_tuples
    ), "Expected warning about unserviceable cluster"
    