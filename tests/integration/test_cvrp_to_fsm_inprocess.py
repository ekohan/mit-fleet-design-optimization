import sys
import pytest
from fleetmix.benchmarking.cvrp_to_fsm import main, CVRPBenchmarkType
from tests.utils.stubs import stub_vrplib, stub_vehicle_configurations, stub_benchmark_clustering, stub_solver, stub_save_results


def test_info_flag_inprocess(capsys):
    sys.argv = ['prog', '--info']
    main()
    out, err = capsys.readouterr()
    assert 'CVRP to FSM Conversion Tool' in out
    assert 'Benchmark Types' in out


def test_normal_smoke_inprocess(tmp_path, monkeypatch, caplog):
    # Use explicit stub context managers for heavy dependencies
    with stub_vrplib(monkeypatch), \
         stub_vehicle_configurations(monkeypatch), \
         stub_benchmark_clustering(monkeypatch), \
         stub_solver(monkeypatch), \
         stub_save_results(monkeypatch, tmp_path):
        # Run CLI normal conversion
        sys.argv = ['prog', '--instance', 'X-k1', '--benchmark-type', 'normal']
        caplog.set_level('INFO')
        main()
        assert True 