import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.benchmarking.cvrp_parser import CVRPParser, CVRPInstance, CVRPSolution

# --- Mock Data ---
MOCK_INSTANCE_DATA = {
    'name': 'Mock-n5-k2',
    'dimension': 5,
    'capacity': 100,
    'node_coord': [(0, 0), (10, 10), (20, 5), (5, 20), (15, 15)], # Depot + 4 clients
    'demand': [0, 10, 20, 15, 25],
    'depot': [0], # Depot is node 0
    'edge_weight_type': 'EUC_2D'
}

MOCK_SOLUTION_DATA = {
    'routes': [[0, 1, 3, 0], [0, 2, 4, 0]], # Using 0-based indexing from vrplib
    'cost': 150.5
}

# --- Fixtures ---
@pytest.fixture
def mock_vrp_file(tmp_path):
    """Create a mock VRP file for testing."""
    mock_file = tmp_path / "X-n5-k2.vrp"
    mock_file.touch()
    return mock_file

@pytest.fixture
def mock_sol_file(tmp_path):
    """Creates a dummy .sol file for testing."""
    p = tmp_path / "X-n5-k2.sol"
    p.write_text("Route #1: 1 3\n...") # Content doesn't matter much due to mocking
    return p

# --- Test Cases ---

@patch('src.benchmarking.cvrp_parser.vrplib.read_instance')
def test_cvrp_parser_parse_success(mock_read_instance, mock_vrp_file):
    """Test successful parsing of a CVRP instance file."""
    mock_read_instance.return_value = MOCK_INSTANCE_DATA
    parser = CVRPParser(str(mock_vrp_file))
    instance = parser.parse()

    assert isinstance(instance, CVRPInstance)
    assert instance.name == 'X-n5-k2' # Should use stem
    assert instance.dimension == 5
    assert instance.capacity == 100
    assert instance.depot_id == 1 # Should be 1-based
    assert instance.num_vehicles == 2 # Parsed from filename stem if possible - need to adjust filename
    assert instance.edge_weight_type == 'EUC_2D'

    # Check coordinates (1-based keys)
    assert len(instance.coordinates) == 5
    assert 1 in instance.coordinates
    assert 5 in instance.coordinates
    assert instance.coordinates[1] == (0, 0)
    assert instance.coordinates[2] == (10, 10)

    # Check demands (1-based keys)
    assert len(instance.demands) == 5
    assert instance.demands[1] == 0.0 # Depot demand
    assert instance.demands[2] == 10.0
    assert instance.demands[5] == 25.0

    mock_read_instance.assert_called_once_with(str(mock_vrp_file))


@patch('src.benchmarking.cvrp_parser.vrplib.read_solution')
def test_cvrp_parser_parse_solution_success(mock_read_solution, mock_vrp_file, mock_sol_file):
    """Test successful parsing of a CVRP solution file."""
    # Need instance file to exist, even if mocked
    parser = CVRPParser(str(mock_vrp_file))
    # Mock the solution reading using the initial MOCK_SOLUTION_DATA
    # Note: MOCK_SOLUTION_DATA uses 0-based routes including depots
    mock_read_solution.return_value = MOCK_SOLUTION_DATA
    # Set instance name for vehicle check
    parser.instance_name = 'X-n5-k2'

    solution = parser.parse_solution()

    assert isinstance(solution, CVRPSolution)
    assert solution.cost == 150.5
    assert solution.num_vehicles == 2 # Number of routes found
    assert solution.expected_vehicles == 2 # Parsed from instance name

    # Check routes (1-based node IDs) based on MOCK_SOLUTION_DATA
    # vrplib routes [0, 1, 3, 0] -> should become [1, 2, 4, 1] in CVRPSolution (keeping depot?)
    # vrplib routes [0, 2, 4, 0] -> should become [1, 3, 5, 1]
    # The current parser code converts [[0, 1, 3, 0], [0, 2, 4, 0]] to [[1, 2, 4, 1], [1, 3, 5, 1]]
    # Let's verify this behavior based on the parser code:
    # routes = [[node + 1 for node in route] for route in solution_data['routes']]
    assert solution.routes == [[1, 2, 4, 1], [1, 3, 5, 1]]

    mock_read_solution.assert_called_once_with(str(mock_sol_file))


@patch('src.benchmarking.cvrp_parser.vrplib.read_solution')
def test_cvrp_parser_parse_solution_route_indexing(mock_read_solution, mock_vrp_file, mock_sol_file):
    """Test parsing solution with specific route format (0-based, no depots)."""
    parser = CVRPParser(str(mock_vrp_file))
    parser.instance_name = 'X-n5-k2' # Needed for expected_vehicles

    # Assume vrplib.read_solution returns routes WITHOUT depots and 0-based: [[1, 3], [2, 4]]
    # (Note: vrplib node indices start from 0, but clients are often numbered 1..N)
    # If vrplib returns [1, 3] meaning depot->node 1->node 3->depot,
    # these correspond to node IDs 2 and 4 in our 1-based system (depot=1).
    # If vrplib returns [2, 4] meaning depot->node 2->node 4->depot,
    # these correspond to node IDs 3 and 5 in our 1-based system.
    mock_read_solution.return_value = {'routes': [[1, 3], [2, 4]], 'cost': 160.0}
    solution = parser.parse_solution()

    # The parser adds 1 to each node ID:
    # [1, 3] -> [2, 4]
    # [2, 4] -> [3, 5]
    assert solution.routes == [[2, 4], [3, 5]]
    assert solution.cost == 160.0
    assert solution.num_vehicles == 2
    assert solution.expected_vehicles == 2

    mock_read_solution.assert_called_once_with(str(mock_sol_file))


def test_cvrp_parser_init_file_not_found(tmp_path):
    """Test parser initialization when the instance file does not exist."""
    non_existent_file = tmp_path / "non_existent.vrp"
    with pytest.raises(FileNotFoundError, match="CVRP instance file not found"):
        CVRPParser(str(non_existent_file))


@patch('src.benchmarking.cvrp_parser.vrplib.read_instance', side_effect=Exception("vrplib error"))
def test_cvrp_parser_parse_vrplib_error(mock_read_instance, mock_vrp_file):
    """Test handling of errors during vrplib.read_instance."""
    parser = CVRPParser(str(mock_vrp_file))
    with pytest.raises(Exception, match="vrplib error"):
        parser.parse()


@patch('src.benchmarking.cvrp_parser.vrplib.read_solution')
def test_cvrp_parser_parse_solution_file_not_found(mock_read_solution, mock_vrp_file):
    """Test parsing solution when the .sol file does not exist."""
    # Instance file exists, but .sol file does not
    parser = CVRPParser(str(mock_vrp_file))
    with pytest.raises(FileNotFoundError, match="Solution file not found"):
        parser.parse_solution()
    mock_read_solution.assert_not_called() # Should fail before calling vrplib


@patch('src.benchmarking.cvrp_parser.vrplib.read_solution')
def test_cvrp_parser_parse_solution_vehicle_mismatch(mock_read_solution, mock_vrp_file, mock_sol_file, caplog):
    """Test parsing solution when the number of routes differs from instance name 'k' value."""
    parser = CVRPParser(str(mock_vrp_file))
    # Mock the solution reading - return 3 routes
    mock_read_solution.return_value = {'routes': [[1], [2], [3, 4]], 'cost': 200}
    # Set instance name expecting 2 vehicles
    parser.instance_name = 'X-n5-k2'

    with caplog.at_level("WARNING"):
        solution = parser.parse_solution()

    assert solution.num_vehicles == 3
    assert solution.expected_vehicles == 2
    assert "Number of routes (3) differs from instance name k2" in caplog.text


@patch('src.benchmarking.cvrp_parser.vrplib.read_instance')
def test_cvrp_parser_parse_instance_name_no_k(mock_read_instance, mock_vrp_file):
    """Test parsing when instance name does not contain '-k'."""
    # Modify the mock file path stem
    mock_vrp_file_no_k = mock_vrp_file.with_name("InstanceWithoutK.vrp")
    mock_vrp_file_no_k.touch()

    mock_read_instance.return_value = MOCK_INSTANCE_DATA.copy() # Use base data
    parser = CVRPParser(str(mock_vrp_file_no_k))
    instance = parser.parse()

    assert instance.name == "InstanceWithoutK"
    assert instance.num_vehicles is None # Should be None if '-k' is missing

# Note: Testing the __main__ block execution requires more complex setup,
# often involving subprocess calls or refactoring the main block into a
# testable function. For library code, focusing on the class methods is key.
