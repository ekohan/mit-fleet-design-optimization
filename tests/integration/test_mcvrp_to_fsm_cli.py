import subprocess
import sys
import pytest
from pathlib import Path
from tests.utils.stubs import (
    stub_vehicle_configurations,
    stub_benchmark_clustering,
    stub_solver,
    stub_save_results,
    stub_mcvrp_parser
)


def test_mcvrp_to_fsm_info_flag():
    cmd = [sys.executable, '-m', 'fleetmix.benchmarking.convert_mcvrp_to_fsm', '--info']
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0
    out = result.stdout.decode('utf-8')
    assert 'MCVRP-to-FSM Conversion Tool' in out


def test_mcvrp_to_fsm_smoke(tmp_path, monkeypatch):
    # Stub heavy dependencies for a fast smoke test
    with stub_vehicle_configurations(monkeypatch), \
         stub_benchmark_clustering(monkeypatch), \
         stub_solver(monkeypatch), \
         stub_mcvrp_parser(monkeypatch), \
         stub_save_results(monkeypatch, tmp_path):
        cmd = [
            sys.executable, '-m', 'fleetmix.benchmarking.convert_mcvrp_to_fsm',
            '--instance', '10_3_3_3_(01)',
            '--format', 'json'
        ]
        result = subprocess.run(cmd, capture_output=True)
        assert result.returncode == 0
        out = result.stdout.decode('utf-8')
        assert 'Optimization Results:' in out 