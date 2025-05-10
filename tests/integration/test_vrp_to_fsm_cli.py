import subprocess
import sys
import pytest
from pathlib import Path
from tests.utils.stubs import (
    stub_vrplib,
    stub_vehicle_configurations,
    stub_benchmark_clustering,
    stub_solver,
    stub_save_results
)


def test_vrp_to_fsm_info_flag():
    cmd = [sys.executable, '-m', 'src.benchmarking.vrp_to_fsm', '--info']
    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode == 0
    out = result.stdout.decode('utf-8')
    assert 'Unified VRP-to-FSM Conversion Tool' in out
    assert 'Supported VRP types:' in out


@ pytest.mark.parametrize('instance,extra_args,expected_out', [
    (
        ['10_3_3_3_(01)'],
        ['--vrp-type', 'mcvrp', '--format', 'json'],
        'Optimization Results:'
    ),
    (
        ['X-n101-k25'],
        ['--vrp-type', 'cvrp', '--benchmark-type', 'normal'],
        'Total converted demand:'
    ),
])
def test_vrp_to_fsm_smoke(tmp_path, monkeypatch, instance, extra_args, expected_out):
    # Use stubs to speed up external calls and I/O
    with stub_vrplib(monkeypatch), \
         stub_vehicle_configurations(monkeypatch), \
         stub_benchmark_clustering(monkeypatch), \
         stub_solver(monkeypatch), \
         stub_save_results(monkeypatch, tmp_path):
        cmd = [sys.executable, '-m', 'src.benchmarking.vrp_to_fsm'] + extra_args + ['--instance'] + instance
        result = subprocess.run(cmd, capture_output=True, check=True)
        out = result.stdout.decode('utf-8')
        assert expected_out in out 


# --- Helper to build expected full path for missing file messages ---
def _get_expected_path_fragment(filename: str, vrp_type: str) -> str:
    base_path = Path("src/benchmarking/") # Relative to project root
    if vrp_type == "cvrp":
        return str((base_path / "cvrp_instances" / f"{filename}.vrp").resolve())
    elif vrp_type == "mcvrp": # For MCVRP, the parser error is simpler
        # The mcvrp_parser.py raises FileNotFoundError with the path it was given
        # which is already <path_to_project>/src/benchmarking/mcvrp_instances/<filename>.dat
        return str((base_path / "mcvrp_instances" / f"{filename}.dat").resolve())
    return filename # Fallback, should not happen with current test cases

@pytest.mark.parametrize(
    "vrp_type, instance_names, benchmark_args, error_prefix, missing_file_stem",
    [
        (
            "cvrp",
            ["NonExistentCVRP"],
            ["--benchmark-type", "normal"],
            "CVRP instance file(s) not found:",
            "NonExistentCVRP"
        ),
        (
            "cvrp",
            ["X-n101-k25", "NonExistentCVRPCombined"],
            ["--benchmark-type", "combined"],
            "CVRP instance file(s) not found:", # cvrp_converter now lists all missing files.
                                                # For this test, only the second one is missing.
            "NonExistentCVRPCombined" # The test expects this specific missing file path fragment.
        ),
        (
            "mcvrp",
            ["NonExistentMCVRP"],
            [], # No benchmark type for MCVRP
            "MCVRP instance file not found:", # This is the prefix from mcvrp_parser.py
            "NonExistentMCVRP" # mcvrp_parser itself includes the .dat part in its error
        ),
    ]
)
def test_vrp_to_fsm_instance_not_found(
    tmp_path, monkeypatch, vrp_type, instance_names, benchmark_args, error_prefix, missing_file_stem
):
    instance_args = ["--instance"] + instance_names
    cmd = [
        sys.executable, 
        "src/benchmarking/vrp_to_fsm.py", 
        "--vrp-type", vrp_type
    ] + instance_args + benchmark_args

    result = subprocess.run(cmd, capture_output=True)
    assert result.returncode != 0, "CLI should exit with an error code for missing files"
    
    stderr_output = result.stderr.decode('utf-8')
    assert "usage:" in stderr_output, "Error output should include usage information"

    if vrp_type == "mcvrp":
        # The mcvrp_parser.py raises: FileNotFoundError(f"MCVRP instance file not found: {path}")
        # The vrp_to_fsm.py CLI calls parser.error(str(e)), which results in:
        # vrp_to_fsm.py: error: MCVRP instance file not found: /full/path/to/file.dat
        expected_path_str = _get_expected_path_fragment(missing_file_stem, vrp_type)
        expected_msg_fragment = f"{error_prefix} {expected_path_str}" # No single quotes around path for parser.error
    elif vrp_type == "cvrp" and len(instance_names) > 1 and missing_file_stem == "NonExistentCVRPCombined":
        # Specific case for combined CVRP where one file exists and one doesn't
        # The cvrp_converter lists only the *missing* resolved path.
        expected_path_str = _get_expected_path_fragment(missing_file_stem, vrp_type)
        expected_msg_fragment = f"{error_prefix} {expected_path_str}" # No comma if only one missing
    else: # Single CVRP missing file
        expected_path_str = _get_expected_path_fragment(missing_file_stem, vrp_type)
        expected_msg_fragment = f"{error_prefix} {expected_path_str}"

    assert expected_msg_fragment in stderr_output, \
        f"Error message fragment \n'{expected_msg_fragment}'\n not found in stderr:\n'{stderr_output}'" 