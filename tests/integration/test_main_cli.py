import sys
import pytest
from pathlib import Path

import src.main as main_mod


def test_main_generates_excel(tmp_results_dir, monkeypatch):
    # Ensure stub_heavy_steps has applied before reload
    monkeypatch.setenv('PROJECT_RESULTS_DIR', str(tmp_results_dir))
    # Set CLI args for Excel
    sys.argv = [
        'src.main',
        '--config', 'tests/_assets/mini.yaml',
        '--format', 'excel',
        '--demand-file', 'mini_demand.csv'
    ]
    # Run CLI entrypoint
    main_mod.main()
    # Check that stub wrote output.xlsx
    out_file = Path(tmp_results_dir) / 'output.xlsx'
    assert out_file.exists(), f"Expected output.xlsx in {tmp_results_dir}"
    assert out_file.read_text() == 'dummy'


def test_main_generates_json(tmp_results_dir, monkeypatch):
    monkeypatch.setenv('PROJECT_RESULTS_DIR', str(tmp_results_dir))
    sys.argv = [
        'src.main',
        '--config', 'tests/_assets/mini.yaml',
        '--format', 'json',
        '--demand-file', 'mini_demand.csv'
    ]
    main_mod.main()
    out_file = Path(tmp_results_dir) / 'output.json'
    assert out_file.exists(), f"Expected output.json in {tmp_results_dir}"
    assert out_file.read_text() == 'dummy' 