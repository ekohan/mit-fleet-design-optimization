import sys
import pytest
from fleetmix.cli.cvrp_to_fsm import main as cvrp_main
from fleetmix.cli.convert_mcvrp_to_fsm import main as mcvrp_main
from fleetmix.cli.vrp_to_fsm import main as unified_main


def test_cvrp_info_flag(capsys):
    sys.argv = ['prog', '--info']
    cvrp_main()
    out, err = capsys.readouterr()
    assert 'CVRP to FSM Conversion Tool' in out


def test_mcvrp_info_flag(capsys):
    sys.argv = ['prog', '--info']
    mcvrp_main()
    out, err = capsys.readouterr()
    assert 'MCVRP-to-FSM Conversion Tool' in out


def test_unified_info_flag(capsys):
    sys.argv = ['prog', '--info']
    unified_main()
    out, err = capsys.readouterr()
    assert 'Unified VRP-to-FSM Conversion Tool' in out 