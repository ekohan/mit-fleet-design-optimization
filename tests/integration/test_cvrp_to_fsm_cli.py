import subprocess
import sys
import pytest
from tests.utils.stubs import (
    stub_vrplib, stub_vehicle_configurations, 
    stub_benchmark_clustering, stub_solver, stub_save_results
)


def test_cvrp_to_fsm_info_flag():
    cmd = [sys.executable, '-m', 'src.benchmarking.cvrp_to_fsm', '--info']
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0
    out = result.stdout.decode('utf-8')
    assert 'CVRP to FSM Conversion Tool' in out


def test_cvrp_to_fsm_normal_smoke(tmp_path, monkeypatch):
    # Use stubs to make this a genuinely fast smoke test
    with stub_vrplib(monkeypatch), \
         stub_vehicle_configurations(monkeypatch), \
         stub_benchmark_clustering(monkeypatch), \
         stub_solver(monkeypatch), \
         stub_save_results(monkeypatch, tmp_path):
        # Run through subprocess
        cmd = [
            sys.executable, '-m', 'src.benchmarking.cvrp_to_fsm',
            '--instance', 'X-n101-k25',
            '--benchmark-type', 'normal'
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        out = result.stdout.decode('utf-8')
        # Should mention converted demand and theoretical vehicles
        assert 'Total converted demand' in out
        assert 'Minimum theoretical vehicles' in out 