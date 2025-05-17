import math
from pathlib import Path

import pytest

from fleetmix.benchmarking.mcvrp_to_fsm import convert_mcvrp_to_fsm
from fleetmix.benchmarking.mcvrp_parser import parse_mcvrp

def test_total_demand_preserved_and_expected_vehicles():
    # Path to a sample MCVRP instance
    dat_path = Path(__file__).parent.parent.parent / 'src' / 'fleetmix' / 'benchmarking' / 'datasets' / 'mcvrp' / '10_3_3_3_(01).dat'
    # Parse original instance
    instance = parse_mcvrp(dat_path)
    # Convert to FSM format
    df, params = convert_mcvrp_to_fsm(dat_path)

    # Sum demands for customers only
    total_orig = sum(sum(demand) for node, demand in instance.demands.items() if node != instance.depot_id)
    total_conv = df['Dry_Demand'].sum() + df['Chilled_Demand'].sum() + df['Frozen_Demand'].sum()
    assert pytest.approx(total_conv) == total_orig

    # Expected vehicles preserved and matches ceil(total / capacity)
    assert params.expected_vehicles == instance.vehicles
    assert params.expected_vehicles == math.ceil(total_orig / instance.capacity) 

def test_dataframe_schema_and_vehicle_config():
    # Path to a sample MCVRP instance
    dat_path = Path(__file__).parent.parent.parent / 'src' / 'fleetmix' / 'benchmarking' / 'datasets' / 'mcvrp' / '10_3_3_3_(01).dat'
    # Parse original instance
    instance = parse_mcvrp(dat_path)
    # Convert to FSM format
    df, params = convert_mcvrp_to_fsm(dat_path)

    # DataFrame should have exactly these columns in order
    expected_cols = ['Customer_ID', 'Latitude', 'Longitude', 'Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']
    assert list(df.columns) == expected_cols

    # Check types of each column
    assert df['Customer_ID'].dtype == object
    assert df['Latitude'].dtype == float
    assert df['Longitude'].dtype == float
    assert df['Dry_Demand'].dtype == int or df['Dry_Demand'].dtype == float
    assert df['Chilled_Demand'].dtype == int or df['Chilled_Demand'].dtype == float
    assert df['Frozen_Demand'].dtype == int or df['Frozen_Demand'].dtype == float

    # Check vehicle configuration in parameters
    assert 'MCVRP' in params.vehicles
    veh = params.vehicles['MCVRP']
    assert veh['capacity'] == instance.capacity
    assert veh['fixed_cost'] == 1000
    assert veh['compartments'] == {'Dry': True, 'Chilled': True, 'Frozen': True} 