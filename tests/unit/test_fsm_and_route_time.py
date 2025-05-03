import pandas as pd
import pulp
import logging
import pytest

from src.fsm_optimizer import _create_model, _extract_solution, _validate_solution
from src.config.parameters import Parameters
from src.utils.route_time import _legacy_estimation, _bhh_estimation, estimate_route_time

# 1. FSM Optimizer Tests

def make_toy_clusters_and_configs():
    # Two clusters: c1 with A,B; c2 with C
    clusters_df = pd.DataFrame([
        {
            'Cluster_ID': 'c1',
            'Customers': ['A', 'B'],
            'Total_Demand': {'Dry': 1, 'Chilled': 0, 'Frozen': 0},
            'Route_Time': 1.0
        },
        {
            'Cluster_ID': 'c2',
            'Customers': ['C'],
            'Total_Demand': {'Dry': 1, 'Chilled': 0, 'Frozen': 0},
            'Route_Time': 1.0
        }
    ])
    # Two vehicle configs, both can serve Dry
    configurations_df = pd.DataFrame([
        {'Config_ID': 'v1', 'Capacity': 10, 'Fixed_Cost': 5, 'Dry': 1, 'Chilled': 0, 'Frozen': 0},
        {'Config_ID': 'v2', 'Capacity': 10, 'Fixed_Cost': 6, 'Dry': 1, 'Chilled': 0, 'Frozen': 0},
    ])
    return clusters_df, configurations_df


def test_create_model_basic_constraints_and_vars():
    clusters_df, configurations_df = make_toy_clusters_and_configs()
    params = Parameters.from_yaml()
    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations_df, params)

    # There should be exactly 3 customer coverage constraints (A, B, C)
    coverage = [n for n in model.constraints if n.startswith('Customer_Coverage_')]
    assert len(coverage) == 3

    # There should be 2 vehicle assignment constraints (one per cluster)
    assignment = [n for n in model.constraints if n.startswith('Vehicle_Assignment_')]
    assert len(assignment) == 2

    # x_vars count: 2 clusters × 2 configs =4
    assert len([k for k in x_vars if k != ('NoVehicle', )]) == 4


def test_light_load_threshold_monotonicity():
    clusters_df, configurations_df = make_toy_clusters_and_configs()
    thresholds = [0.0, 0.5, 1.0]
    costs = []
    for t in thresholds:
        params = Parameters.from_yaml()
        params.light_load_penalty = 100
        params.light_load_threshold = t
        model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations_df, params)
        # sum of all cost coefficients
        total_cost = sum(c_vk[v, k] for (v, k) in c_vk)
        costs.append(total_cost)
    # Costs should be non-decreasing
    assert costs[0] <= costs[1] <= costs[2]


def test_capacity_infeasibility_injects_no_vehicle(caplog):
    # Single cluster demand > capacity
    clusters_df = pd.DataFrame([
        {
            'Cluster_ID': 'c1',
            'Customers': ['X'],
            'Total_Demand': {'Dry': 100, 'Chilled': 0, 'Frozen': 0},
            'Route_Time': 1.0
        }
    ])
    configurations_df = pd.DataFrame([
        {'Config_ID': 'v1', 'Capacity': 10, 'Fixed_Cost': 5, 'Dry': 1, 'Chilled': 0, 'Frozen': 0},
    ])
    params = Parameters.from_yaml()
    caplog.set_level(logging.WARNING, logger='src.fsm_optimizer')
    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations_df, params)

    # 'NoVehicle' var should be in x_vars
    assert any(v == 'NoVehicle' for (v, k) in x_vars)
    # There should be a warning about unserviceable cluster
    assert any('cannot be served' in rec.message.lower() for rec in caplog.records)


def test_extract_and_validate_solution():
    # Prepare clusters and fake solver output
    clusters_df = pd.DataFrame([
        {'Cluster_ID': 'k1', 'Customers': ['C1']},
        {'Cluster_ID': 'k2', 'Customers': ['C2']}
    ])
    # Build y_vars
    y1 = pulp.LpVariable('y_k1', cat='Binary'); y1.varValue = 1
    y2 = pulp.LpVariable('y_k2', cat='Binary'); y2.varValue = 0
    y_vars = {'k1': y1, 'k2': y2}
    # Build x_vars: assign 'config1' to k1, 'config2' to k2
    x1 = pulp.LpVariable('x_conf1_k1', cat='Binary'); x1.varValue = 1
    x2 = pulp.LpVariable('x_conf2_k2', cat='Binary'); x2.varValue = 1
    x_vars = {('conf1', 'k1'): x1, ('conf2', 'k2'): x2}

    selected = _extract_solution(clusters_df, y_vars, x_vars)
    # Only k1 should be selected
    assert list(selected['Cluster_ID']) == ['k1']
    assert list(selected['Config_ID']) == ['conf1']

    # Validate solution: with customers_df missing C2
    customers_df = pd.DataFrame([
        {'Customer_ID': 'C1', 'Dry_Demand': 1, 'Chilled_Demand': 0, 'Frozen_Demand': 0}
    ])
    missing = _validate_solution(selected, customers_df, configurations_df)
    assert missing == {'C2'}


# 2. Route-Time Tests

def test_bhh_bounds_against_service_and_legacy():
    n = 5
    svc = 30
    # Legacy estimation
    legacy = _legacy_estimation(n, svc)
    # Build cluster with same-location customers
    df = pd.DataFrame({'Latitude': [0.0]*n, 'Longitude': [0.0]*n})
    depot = {'latitude': 0.0, 'longitude': 0.0}
    bhh = _bhh_estimation(df, depot, svc, avg_speed=60)
    # BHH includes service time: ≥ n*svc/60
    assert bhh >= n * svc/60
    # And should be ≤ legacy (legacy includes +1h overhead)
    assert bhh <= legacy

@pytest.mark.parametrize("method", ['Legacy', 'BHH', 'TSP'])
def test_estimate_route_time_dispatch(method):
    df = pd.DataFrame({'Latitude': [], 'Longitude': []})
    depot = {'latitude':0.0, 'longitude':0.0}
    time, seq = estimate_route_time(df, depot, service_time=15, avg_speed=50, method=method)
    assert isinstance(time, float)
    assert isinstance(seq, list) 