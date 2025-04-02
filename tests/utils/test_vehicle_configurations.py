import pytest
import pandas as pd
from src.utils.vehicle_configurations import generate_vehicle_configurations

@pytest.fixture
def sample_vehicle_types():
    """Fixture for sample vehicle types."""
    return {
        'Small': {'capacity': 1000, 'fixed_cost': 50},
        'Large': {'capacity': 3000, 'fixed_cost': 150}
    }

@pytest.fixture
def sample_goods():
    """Fixture for sample goods."""
    return ['Dry', 'Chilled']

def test_generate_vehicle_configurations_two_goods(sample_vehicle_types, sample_goods):
    """Test generating configurations for two goods and two vehicle types."""
    configs_df = generate_vehicle_configurations(sample_vehicle_types, sample_goods)

    # Expected number of configs: num_vehicles * (2^num_goods - 1)
    # 2 vehicle types * (2^2 - 1) = 2 * 3 = 6
    assert len(configs_df) == 6
    assert isinstance(configs_df, pd.DataFrame)
    assert list(configs_df.columns) == ['Dry', 'Chilled', 'Vehicle_Type', 'Config_ID', 'Capacity', 'Fixed_Cost']

    # Check Config IDs are unique and sequential
    assert list(configs_df['Config_ID']) == [1, 2, 3, 4, 5, 6]

    # Check one specific configuration (e.g., Small, Dry only)
    small_dry_config = configs_df[(configs_df['Vehicle_Type'] == 'Small') & (configs_df['Dry'] == 1) & (configs_df['Chilled'] == 0)]
    assert len(small_dry_config) == 1
    assert small_dry_config.iloc[0]['Capacity'] == 1000
    assert small_dry_config.iloc[0]['Fixed_Cost'] == 50

    # Check another configuration (e.g., Large, both Dry and Chilled)
    large_both_config = configs_df[(configs_df['Vehicle_Type'] == 'Large') & (configs_df['Dry'] == 1) & (configs_df['Chilled'] == 1)]
    assert len(large_both_config) == 1
    assert large_both_config.iloc[0]['Capacity'] == 3000
    assert large_both_config.iloc[0]['Fixed_Cost'] == 150

    # Ensure the "all zero" configuration is skipped
    assert not ((configs_df['Dry'] == 0) & (configs_df['Chilled'] == 0)).any()

def test_generate_vehicle_configurations_three_goods(sample_vehicle_types):
    """Test generating configurations for three goods."""
    goods = ['Dry', 'Chilled', 'Frozen']
    configs_df = generate_vehicle_configurations(sample_vehicle_types, goods)

    # Expected number of configs: 2 vehicle types * (2^3 - 1) = 2 * 7 = 14
    assert len(configs_df) == 14
    assert list(configs_df.columns) == ['Dry', 'Chilled', 'Frozen', 'Vehicle_Type', 'Config_ID', 'Capacity', 'Fixed_Cost']

    # Ensure the "all zero" configuration is skipped
    assert not ((configs_df['Dry'] == 0) & (configs_df['Chilled'] == 0) & (configs_df['Frozen'] == 0)).any()

def test_generate_vehicle_configurations_single_good(sample_vehicle_types):
    """Test generating configurations for a single good."""
    goods = ['Ambient']
    configs_df = generate_vehicle_configurations(sample_vehicle_types, goods)

    # Expected number of configs: 2 vehicle types * (2^1 - 1) = 2 * 1 = 2
    assert len(configs_df) == 2
    assert list(configs_df.columns) == ['Ambient', 'Vehicle_Type', 'Config_ID', 'Capacity', 'Fixed_Cost']
    assert (configs_df['Ambient'] == 1).all() # Only the config with the good selected

    # Check configs generated
    small_config = configs_df[configs_df['Vehicle_Type'] == 'Small']
    large_config = configs_df[configs_df['Vehicle_Type'] == 'Large']
    assert len(small_config) == 1
    assert len(large_config) == 1
    assert small_config.iloc[0]['Capacity'] == 1000
    assert large_config.iloc[0]['Capacity'] == 3000


def test_generate_vehicle_configurations_no_goods():
    """Test generating configurations when the list of goods is empty."""
    vehicle_types = {'Standard': {'capacity': 2000, 'fixed_cost': 100}}
    goods = []
    configs_df = generate_vehicle_configurations(vehicle_types, goods)

    # Expected number of configs: 1 vehicle type * (2^0 - 1) = 1 * 0 = 0
    assert len(configs_df) == 0
    assert configs_df.empty

def test_generate_vehicle_configurations_no_vehicles():
    """Test generating configurations when the dictionary of vehicle types is empty."""
    vehicle_types = {}
    goods = ['Dry', 'Chilled']
    configs_df = generate_vehicle_configurations(vehicle_types, goods)

    assert len(configs_df) == 0
    assert configs_df.empty

# Note: The print_configurations function primarily prints output and doesn't return values,
# making it hard to unit test directly without capturing stdout. Testing the generation logic
# in generate_vehicle_configurations is usually sufficient.
