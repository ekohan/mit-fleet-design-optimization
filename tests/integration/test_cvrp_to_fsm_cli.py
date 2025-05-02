import subprocess
import sys


def test_cvrp_to_fsm_info_flag():
    cmd = [sys.executable, '-m', 'src.benchmarking.cvrp_to_fsm', '--info']
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0
    out = result.stdout.decode('utf-8')
    assert 'CVRP to FSM Conversion Tool' in out


def test_cvrp_to_fsm_normal_smoke():
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