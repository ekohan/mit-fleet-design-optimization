import pytest
import yaml
from pathlib import Path
from src.config.parameters import Parameters

# Create a dummy config file for testing
DUMMY_CONFIG_CONTENT = """
vehicles:
  TypeA:
    capacity: 1000
    fixed_cost: 100
variable_cost_per_hour: 15.0
avg_speed: 40
max_route_time: 8
service_time: 20
depot:
  latitude: 1.0
  longitude: 2.0
goods:
  - A
  - B
clustering:
  max_depth: 10
  method: kmedoids
  distance: euclidean
  geo_weight: 0.6
  demand_weight: 0.4
  route_time_estimation: 'Legacy'
demand_file: "dummy_demand.csv"
light_load_penalty: 50
light_load_threshold: 0.15
compartment_setup_cost: 25
format: 'json'
post_optimization: false
expected_vehicles: 10
"""

DUMMY_CONFIG_INVALID_WEIGHTS = """
vehicles: {}
variable_cost_per_hour: 10.0
avg_speed: 30
max_route_time: 10
service_time: 25
depot: {latitude: 4.7, longitude: -74.1}
goods: [Dry, Chilled, Frozen]
clustering:
  max_depth: 20
  method: combine
  distance: composite
  geo_weight: 0.8 # Invalid sum
  demand_weight: 0.3 # Invalid sum
  route_time_estimation: 'BHH'
demand_file: "test.csv"
light_load_penalty: 0
light_load_threshold: 0.20
compartment_setup_cost: 50
format: 'excel'
post_optimization: true
"""

@pytest.fixture
def dummy_config_file(tmp_path):
    """Fixture to create a temporary dummy YAML config file."""
    config_path = tmp_path / "dummy_config.yaml"
    config_path.write_text(DUMMY_CONFIG_CONTENT)
    return config_path

@pytest.fixture
def dummy_invalid_weights_file(tmp_path):
    """Fixture for a config file with invalid clustering weights."""
    config_path = tmp_path / "dummy_invalid_weights.yaml"
    config_path.write_text(DUMMY_CONFIG_INVALID_WEIGHTS)
    return config_path

def test_parameters_load_from_yaml(dummy_config_file):
    """Test loading parameters from a specified YAML file."""
    params = Parameters.from_yaml(dummy_config_file)

    assert isinstance(params, Parameters)
    # Check a few loaded values
    assert params.variable_cost_per_hour == 15.0
    assert params.avg_speed == 40
    assert params.depot == {'latitude': 1.0, 'longitude': 2.0}
    assert params.goods == ['A', 'B']
    assert params.clustering['method'] == 'kmedoids'
    assert params.clustering['geo_weight'] == 0.6
    assert params.clustering['demand_weight'] == 0.4
    assert params.light_load_penalty == 50
    assert params.compartment_setup_cost == 25
    assert params.format == 'json'
    assert params.post_optimization is False
    assert params.expected_vehicles == 10
    assert params.vehicles['TypeA']['capacity'] == 1000

def test_parameters_load_default_exists():
    """Test that the default config file exists and can be loaded."""
    default_config_path = Path(__file__).parent.parent.parent / 'src' / 'config' / 'default_config.yaml'
    assert default_config_path.exists(), "Default config file not found."

    try:
        params = Parameters.from_yaml() # Load default
        assert isinstance(params, Parameters)
        # Check a known default value
        assert isinstance(params.vehicles, dict)
        assert params.avg_speed > 0 # Should have a default positive speed
        assert params.clustering['geo_weight'] + params.clustering['demand_weight'] == pytest.approx(1.0)
        assert params.post_optimization is True # Check default boolean
        assert params.expected_vehicles == -1 # Check default integer
    except Exception as e:
        pytest.fail(f"Loading default config failed: {e}")

def test_parameters_post_init_validation_ok(dummy_config_file):
    """Test that validation passes for correct clustering weights."""
    try:
        Parameters.from_yaml(dummy_config_file) # Should not raise error
    except ValueError:
        pytest.fail("Validation incorrectly failed for valid weights.")

def test_parameters_post_init_validation_fail(dummy_invalid_weights_file):
    """Test that validation fails for incorrect clustering weights."""
    with pytest.raises(ValueError, match="Clustering weights must sum to 1.0"):
        Parameters.from_yaml(dummy_invalid_weights_file)

def test_parameter_dataclass_attributes():
    """Test direct instantiation and attribute access."""
    # Create params directly (skipping YAML load)
    params = Parameters(
        vehicles={'V1': {'capacity': 500, 'fixed_cost': 20}},
        variable_cost_per_hour=10.0,
        avg_speed=30,
        max_route_time=10,
        service_time=25,
        depot={'latitude': 4.7, 'longitude': -74.1},
        goods=['Dry', 'Chilled'],
        clustering={
            'max_depth': 15, 'method': 'minibatch_kmeans', 'distance': 'euclidean',
            'geo_weight': 0.7, 'demand_weight': 0.3, 'route_time_estimation': 'BHH'
        },
        demand_file="some_file.csv",
        light_load_penalty=100,
        light_load_threshold=0.2,
        compartment_setup_cost=50,
        format='excel',
        post_optimization=True,
        expected_vehicles=5
    )
    assert params.avg_speed == 30
    assert params.clustering['max_depth'] == 15
    assert params.goods == ['Dry', 'Chilled']
    assert params.expected_vehicles == 5
