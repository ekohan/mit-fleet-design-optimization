import pytest
import numpy as np
from src.utils.coordinate_converter import CoordinateConverter, GeoBounds

# Define sample CVRP coordinates and geographic bounds for testing
SAMPLE_CVRP_COORDS = {
    1: (0, 0),    # Depot
    2: (100, 100),
    3: (200, 50),
    4: (150, -50) # Added a point with negative y
}

# Bogota Metro Area bounding box (approximate)
BOGOTA_BOUNDS = GeoBounds(
    min_lat=4.3333,
    max_lat=4.9167,
    min_lon=-74.3500,
    max_lon=-73.9167
)

# Simpler bounds for easier verification
SIMPLE_BOUNDS = GeoBounds(
    min_lat=0.0,
    max_lat=10.0,
    min_lon=0.0,
    max_lon=10.0
)

@pytest.fixture
def bogota_converter():
    """Fixture for a CoordinateConverter using Bogota bounds."""
    return CoordinateConverter(cvrp_coords=SAMPLE_CVRP_COORDS, geo_bounds=BOGOTA_BOUNDS)

@pytest.fixture
def simple_converter():
    """Fixture for a CoordinateConverter using simple 0-10 bounds."""
    # Simplified CVRP coords for simple bounds
    simple_cvrp_coords = {
        1: (0, 0),
        2: (10, 10),
        3: (5, 5)
    }
    return CoordinateConverter(cvrp_coords=simple_cvrp_coords, geo_bounds=SIMPLE_BOUNDS)

# --- Test Initialization ---

def test_coordinate_converter_init_default_bounds():
    """Test initialization with default bounds."""
    converter = CoordinateConverter(SAMPLE_CVRP_COORDS)
    # Default bounds are Bogota
    assert converter.geo_bounds.min_lat == pytest.approx(4.3333)
    assert converter.geo_bounds.max_lat == pytest.approx(4.9167)
    assert converter.geo_bounds.min_lon == pytest.approx(-74.3500)
    assert converter.geo_bounds.max_lon == pytest.approx(-73.9167)
    # Check if CVRP bounds are calculated correctly
    assert converter.min_x == 0
    assert converter.max_x == 200
    assert converter.min_y == -50
    assert converter.max_y == 100
    assert converter.scale > 0 # Scale should be positive

def test_coordinate_converter_init_custom_bounds(simple_converter):
    """Test initialization with custom bounds."""
    assert simple_converter.geo_bounds == SIMPLE_BOUNDS
    assert simple_converter.min_x == 0
    assert simple_converter.max_x == 10
    assert simple_converter.min_y == 0
    assert simple_converter.max_y == 10
    assert simple_converter.scale > 0

# --- Test GeoBounds ---

def test_geobounds_properties():
    """Test the properties of the GeoBounds dataclass."""
    bounds = GeoBounds(min_lat=10, max_lat=20, min_lon=30, max_lon=50)
    assert bounds.center == (15, 40)
    assert bounds.lat_span == 10
    assert bounds.lon_span == 20

# --- Test Conversion Methods ---

@pytest.mark.parametrize("x, y", [
    (0, 0),         # Depot
    (100, 100),     # Top-right
    (200, 50),      # Mid-right
    (150, -50),     # Bottom-right
    (100, 0)        # Mid-point x=avg, y=min
])
def test_to_geographic_within_bounds(bogota_converter, x, y):
    """Test if converted geographic coordinates fall within the specified bounds."""
    lat, lon = bogota_converter.to_geographic(x, y)
    assert bogota_converter.geo_bounds.min_lat <= lat <= bogota_converter.geo_bounds.max_lat, f"Latitude {lat} out of bounds for ({x},{y})"
    assert bogota_converter.geo_bounds.min_lon <= lon <= bogota_converter.geo_bounds.max_lon, f"Longitude {lon} out of bounds for ({x},{y})"

@pytest.mark.parametrize("x_cvrp, y_cvrp, expected_lat, expected_lon", [
    (5, 5, 5.0, 5.0),  # Center point
    (0, 0, 0.0, 0.0),  # Min point
    (10, 10, 10.0, 10.0), # Max point
    (0, 10, 10.0, 0.0), # Top-left
    (10, 0, 0.0, 10.0), # Bottom-right
])
def test_to_geographic_simple(simple_converter, x_cvrp, y_cvrp, expected_lat, expected_lon):
    """Test coordinate conversion with simple 1:1 scaling (ignoring cosine correction)."""
    # Note: This test assumes aspect ratio allows near 1:1 mapping for simplicity.
    # The actual conversion includes cosine correction. We check if it's roughly correct.
    lat, lon = simple_converter.to_geographic(x_cvrp, y_cvrp)
    # Need approx due to scaling and cosine correction nuances
    assert lat == pytest.approx(expected_lat, abs=1e-1)
    assert lon == pytest.approx(expected_lon, abs=1e-1)


def test_to_geographic_and_back(bogota_converter):
    """Test if converting to geographic and back yields the original coordinates."""
    original_x, original_y = 123.4, -45.6
    lat, lon = bogota_converter.to_geographic(original_x, original_y)
    converted_x, converted_y = bogota_converter.to_cvrp(lat, lon)

    assert converted_x == pytest.approx(original_x, abs=1e-4), "X coordinate mismatch after round trip"
    assert converted_y == pytest.approx(original_y, abs=1e-4), "Y coordinate mismatch after round trip"

def test_convert_all_coordinates_to_geo(bogota_converter):
    """Test converting a dictionary of CVRP coordinates to geographic."""
    geo_coords = bogota_converter.convert_all_coordinates(SAMPLE_CVRP_COORDS, to_geographic=True)
    assert len(geo_coords) == len(SAMPLE_CVRP_COORDS)
    for node_id, (lat, lon) in geo_coords.items():
        assert node_id in SAMPLE_CVRP_COORDS
        assert bogota_converter.geo_bounds.min_lat <= lat <= bogota_converter.geo_bounds.max_lat
        assert bogota_converter.geo_bounds.min_lon <= lon <= bogota_converter.geo_bounds.max_lon

def test_convert_all_coordinates_to_cvrp(bogota_converter):
    """Test converting a dictionary of geographic coordinates back to CVRP."""
    # First convert to geographic
    geo_coords = bogota_converter.convert_all_coordinates(SAMPLE_CVRP_COORDS, to_geographic=True)
    # Then convert back to CVRP
    cvrp_coords_back = bogota_converter.convert_all_coordinates(geo_coords, to_geographic=False)

    assert len(cvrp_coords_back) == len(SAMPLE_CVRP_COORDS)
    for node_id, (x, y) in cvrp_coords_back.items():
        original_x, original_y = SAMPLE_CVRP_COORDS[node_id]
        assert x == pytest.approx(original_x, abs=1e-4)
        assert y == pytest.approx(original_y, abs=1e-4)

# --- Test Edge Cases ---

def test_single_point_conversion():
    """Test conversion when only one CVRP point is provided."""
    coords = {1: (50, 50)}
    bounds = GeoBounds(min_lat=0, max_lat=10, min_lon=0, max_lon=10)
    # Expect a RuntimeWarning due to division by zero in scale calculation
    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
         CoordinateConverter(cvrp_coords=coords, geo_bounds=bounds)
    # If the implementation handles this (e.g., by setting scale to 1), adjust the test:
    # converter = CoordinateConverter(cvrp_coords=coords, geo_bounds=bounds)
    # assert converter.scale == 1 # Or whatever the defined behavior is
    # lat, lon = converter.to_geographic(50, 50)
    # assert lat == bounds.center[0] # Should map to center if scale is undefined or 1
    # assert lon == bounds.center[1]

def test_collinear_points_vertical():
    """Test conversion with points lying on a vertical line."""
    coords = {1: (50, 0), 2: (50, 100)}
    bounds = GeoBounds(min_lat=0, max_lat=10, min_lon=0, max_lon=10)
    # Expect a RuntimeWarning due to division by zero in x_scale calculation
    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        CoordinateConverter(cvrp_coords=coords, geo_bounds=bounds)

def test_collinear_points_horizontal():
    """Test conversion with points lying on a horizontal line."""
    coords = {1: (0, 50), 2: (100, 50)}
    bounds = GeoBounds(min_lat=0, max_lat=10, min_lon=0, max_lon=10)
    # Expect a RuntimeWarning due to division by zero in y_scale calculation
    with pytest.warns(RuntimeWarning, match="divide by zero encountered"):
        CoordinateConverter(cvrp_coords=coords, geo_bounds=bounds)

# Note: The validate_conversion function uses random sampling and prints output,
# making it less suitable for automated testing without modifications (e.g., mocking random, capturing stdout).
# It's more of a manual verification/demonstration tool.
