import math
from pathlib import Path

import pytest

from src.benchmarking.mcvrp_to_fsm import convert_mcvrp_to_fsm
from src.benchmarking.mcvrp_parser import parse_mcvrp

def test_total_demand_preserved_and_expected_vehicles():
    # Path to a sample MCVRP instance
    dat_path = Path(__file__).parent.parent.parent / 'src' / 'benchmarking' / 'mcvrp_instances' / '10_3_3_3_(01).dat'
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