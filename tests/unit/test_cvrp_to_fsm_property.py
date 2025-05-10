import pandas as pd
import pytest
from src.benchmarking.cvrp_to_fsm import (
    convert_cvrp_to_fsm,
    CVRPBenchmarkType
)
from pathlib import Path

# Monkey-patch CVRPParser.parse to return a controlled instance
class DummyInst:
    def __init__(self):
        self.demands = {1: 10, 2: 20, 3: 30}
        self.capacity = 15
        self.num_vehicles = 2
        self.coordinates = {1:(0,0),2:(1,1),3:(2,2)}
        self.depot_id = 1

@pytest.fixture(autouse=True)
def patch_parser(monkeypatch):
    import src.benchmarking.cvrp_to_fsm as mod
    # Bypass convert_cvrp_to_fsm file-existence check for unit tests
    orig_exists = Path.exists
    def fake_exists(self):
        if self.suffix == '.vrp' and 'cvrp_instances' in str(self):
            return True
        return orig_exists(self)
    monkeypatch.setattr(Path, 'exists', fake_exists)
    class DummyParser:
        def __init__(self, path): pass
        def parse(self): return DummyInst()
        def parse_solution(self): pass
    monkeypatch.setattr(mod, 'CVRPParser', DummyParser)

@pytest.mark.parametrize("btype,mult", [
    (CVRPBenchmarkType.NORMAL, 1),
    (CVRPBenchmarkType.SPLIT, 1),
    (CVRPBenchmarkType.SCALED, 2),
])
def test_demand_preserved_and_expected(btype, mult):
    # Run conversion
    df, params = convert_cvrp_to_fsm('X', btype, num_goods=2, split_ratios={'dry':0.5,'chilled':0.5})
    total = df[['Dry_Demand','Chilled_Demand','Frozen_Demand']].sum().sum()
    # Original total for non-depot nodes: 20+30 = 50
    base_inst = DummyInst()
    orig_total = sum(v for k,v in base_inst.demands.items() if k != base_inst.depot_id)
    # NORMAL/SPLIT: total = orig_total; SCALED multiplies
    expected = orig_total * mult
    assert total == expected
    # expected_vehicles matches
    assert params.expected_vehicles == DummyInst().num_vehicles * mult


def test_combined_conversion_rows_and_vehicles():
    # For combined, number of rows = (dimension-1)*mult
    inst = DummyInst()
    df, params = convert_cvrp_to_fsm(['A','B'], CVRPBenchmarkType.COMBINED)
    # dimension-1 = 3
    assert len(df) == (len(inst.demands) - 1) * 2
    # vehicles = sum num_vehicles * count
    assert params.expected_vehicles == inst.num_vehicles * 2


def test_info_flag_captures_output(capsys):
    import src.benchmarking.cvrp_to_fsm as mod
    # Call main with info
    mod.main_impl = mod.main  # alias
    import sys
    sys.argv = ['prog','--info']
    mod.main()
    out, err = capsys.readouterr()
    assert 'Benchmark Types' in out
    assert 'Usage Examples' in out 