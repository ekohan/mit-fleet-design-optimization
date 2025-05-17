import logging
import sys
import pytest
import pandas as pd
from pathlib import Path

from src.benchmarking.cvrp_parser import CVRPParser
from src.benchmarking.cvrp_to_fsm import main as c2f_main, convert_cvrp_to_fsm, CVRPBenchmarkType
from src.benchmarking.cvrp_parser import CVRPInstance

# --- Parser tests -------------------------------------------------------------

@pytest.fixture(autouse=True)
def stub_vrplib(monkeypatch):
    # Fake vrplib.read_instance
    def fake_read_instance(path):
        return {
            'node_coord': [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            'demand': [0, 5, 5],
            'capacity': 10,
            'depot': [0]
            # no edge_weight_type key to test defaulting
        }
    # Fake vrplib.read_solution
    def fake_read_solution(path):
        return {'routes': [[0, 1], [2]], 'cost': 123.4}

    monkeypatch.setattr('src.benchmarking.cvrp_parser.vrplib.read_instance', fake_read_instance)
    monkeypatch.setattr('src.benchmarking.cvrp_parser.vrplib.read_solution', fake_read_solution)


def test_parse_fills_defaults(tmp_path):
    # Create dummy files
    vrp = tmp_path / 'X-n3-k2.vrp'
    sol = tmp_path / 'X-n3-k2.sol'
    vrp.write_text('')
    sol.write_text('')

    parser = CVRPParser(str(vrp))
    inst = parser.parse()
    # Default edge_weight_type
    assert inst.edge_weight_type == 'EUC_2D'
    # Capacity and demands
    assert inst.capacity == 10
    assert inst.demands == {1.0: 0.0, 2.0: 5.0, 3.0: 5.0} or isinstance(inst.demands, dict)
    # Coordinates mapping length
    assert len(inst.coordinates) == 3
    # Depot ID correct (1-based)
    assert inst.depot_id == 1
    # num_vehicles from name k2
    assert inst.num_vehicles == 2


def test_parse_solution_logs_warning(tmp_path, caplog):
    vrp = tmp_path / 'X-n4-k3.vrp'
    sol = tmp_path / 'X-n4-k3.sol'
    vrp.write_text('')
    sol.write_text('')

    parser = CVRPParser(str(vrp))
    caplog.set_level(logging.WARNING)
    solution = parser.parse_solution()
    # cost and vehicles
    assert solution.cost == 123.4
    assert solution.num_vehicles == len(solution.routes)
    assert solution.expected_vehicles == 3
    # Expect warning about mismatch k3 vs actual 2 routes
    assert any('differ' in rec.message.lower() for rec in caplog.records), "Expected vehicle count mismatch warning"

# --- Converter tests ----------------------------------------------------------

class DummyInst:
    def __init__(self, num_vehicles, demands, coords, depot_id=1):
        self.name = 'D'
        self.num_vehicles = num_vehicles
        self.demands = demands
        self.coordinates = coords
        self.capacity = 10
        self.depot_id = depot_id
        self.dimension = len(coords)

@pytest.fixture(autouse=True)
def stub_parser(monkeypatch):
    # ----------------------------------------------------------------------------
    # Create placeholder .vrp files so our pre-check in convert_cvrp_to_fsm will find them.
    # Contents are never parsed (we stub out CVRPParser).
    instances_dir = (
        Path(__file__).resolve().parents[2]  # project root
        / "src" / "benchmarking" / "cvrp_instances"
    )
    instances_dir.mkdir(parents=True, exist_ok=True)
    for name in ("dummy", "a", "b"):
        f = instances_dir / f"{name}.vrp"
        if not f.exists():
            f.write_text("")  # empty stub
    # ----------------------------------------------------------------------------

    class FakeParser:
        def __init__(self, path): pass
        def parse(self): return FakeParser.instance
    monkeypatch.setattr('src.benchmarking.cvrp_to_fsm.CVRPParser', FakeParser)
    return FakeParser

@pytest.mark.parametrize('btype,mult', [
    (CVRPBenchmarkType.NORMAL, 1),
    (CVRPBenchmarkType.SPLIT, 1),
    (CVRPBenchmarkType.SCALED, 2),
])
def test_convert_preserves_total_demand_and_expected_vehicles(stub_parser, btype, mult):
    # Build dummy instance: two customers
    demands = {1: 5.0, 2: 5.0}
    coords = {1: (0,0), 2: (1,1)}
    inst = DummyInst(num_vehicles=3, demands=demands, coords=coords)
    stub_parser.instance = inst

    df, params = convert_cvrp_to_fsm('dummy', btype, num_goods=2, split_ratios={'dry':0.6,'chilled':0.4})
    # Check total demand preserved (split sums to original)
    total = (df.get('Dry_Demand', 0).sum() + df.get('Chilled_Demand', 0).sum() + df.get('Frozen_Demand', 0).sum())
    # Only non-depot customers are included in conversion
    non_depot_total = sum(v for k,v in demands.items() if k != inst.depot_id)
    expected_total = non_depot_total * (mult if btype==CVRPBenchmarkType.SCALED else 1)
    assert pytest.approx(expected_total) == total
    # Expected vehicles
    assert params.expected_vehicles == inst.num_vehicles * (mult if btype==CVRPBenchmarkType.SCALED else 1)


def test_convert_combined_counts_and_good_columns(stub_parser):
    # Two instances combined
    demands = {1: 3.0, 2: 4.0}
    coords = {1:(0,0), 2:(1,1)}
    inst1 = DummyInst(2, demands, coords)
    inst2 = DummyInst(2, demands, coords)
    # stub parse to cycle through instances
    sequence = [inst1, inst2]
    def parse_cycle(self): return sequence.pop(0)
    stub_parser.parse = parse_cycle

    df, params = convert_cvrp_to_fsm(['a','b'], CVRPBenchmarkType.COMBINED)
    # Combined goods columns exist
    for col in ['Dry_Demand','Chilled_Demand','Frozen_Demand']:
        assert col in df.columns
    # Rows should be (dim-1)*2 = 2*2=4
    assert len(df) == (inst1.dimension - 1) * 2
    # Expected vehicles = sum num_vehicles
    assert params.expected_vehicles == inst1.num_vehicles + inst2.num_vehicles


def test_cvrp_to_fsm_info_flag(capsys):
    # Capture info output
    sys_argv = sys.argv
    sys.argv = ['cvrp_to_fsm', '--info']
    try:
        c2f_main()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert 'Benchmark Types' in out
    assert 'Usage Examples' in out
    sys.argv = sys_argv 