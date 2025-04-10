"""Route time estimation methods for vehicle routing."""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from haversine import haversine

from pyvrp import (
    Model, Client, Depot, VehicleType, ProblemData,
    GeneticAlgorithmParams, PopulationParams, SolveParams
)
# Import stopping criteria from the correct submodule
from pyvrp.stop import MaxIterations
import logging # Added for logging

logger = logging.getLogger(__name__) # Added logger

# Global cache for distance and duration matrices (populated if TSP method is used)
_matrix_cache = {
    'distance_matrix': None,
    'duration_matrix': None,
    'customer_id_to_idx': None,
    'depot_idx': 0  # Depot is always at index 0
}

def build_distance_duration_matrices(
    customers_df: pd.DataFrame,
    depot: Dict[str, float],
    avg_speed: float
) -> None:
    """
    Build global distance and duration matrices for all customers plus the depot
    and store them in the module-level cache `_matrix_cache`.

    Args:
        customers_df: DataFrame containing ALL customer data.
        depot: Depot location coordinates {'latitude': float, 'longitude': float}.
        avg_speed: Average vehicle speed (km/h).
    """
    if customers_df.empty:
        logger.warning("Cannot build matrices: Customer DataFrame is empty.")
        return

    logger.info(f"Building global distance/duration matrices for {len(customers_df)} customers...")
    # Create mapping from Customer_ID to matrix index (Depot is 0)
    customer_ids = customers_df['Customer_ID'].tolist()
    # Ensure unique IDs before creating map
    if len(set(customer_ids)) != len(customer_ids):
         logger.warning("Duplicate Customer IDs found. Matrix mapping might be incorrect.")
    customer_id_to_idx = {cid: idx + 1 for idx, cid in enumerate(customer_ids)}

    # Total locations = all customers + depot
    n_locations = len(customers_df) + 1

    # Prepare coordinates list: depot first, then all customers
    depot_coord = (depot['latitude'], depot['longitude'])
    # Ensure Latitude/Longitude columns exist
    if 'Latitude' not in customers_df.columns or 'Longitude' not in customers_df.columns:
        logger.error("Missing 'Latitude' or 'Longitude' columns in customer data.")
        raise ValueError("Missing coordinate columns in customer data.")
        
    customer_coords = list(zip(customers_df['Latitude'], customers_df['Longitude']))
    all_coords = [depot_coord] + customer_coords

    # Initialize matrices
    distance_matrix = np.zeros((n_locations, n_locations), dtype=int)
    duration_matrix = np.zeros((n_locations, n_locations), dtype=int)

    # Speed in km/s for duration calculation
    avg_speed_kps = avg_speed / 3600 if avg_speed > 0 else 0

    # Populate matrices
    for i in range(n_locations):
        for j in range(i + 1, n_locations):
            # Distance using Haversine (km)
            dist_km = haversine(all_coords[i], all_coords[j])

            # Store distance in meters (integer)
            distance_matrix[i, j] = distance_matrix[j, i] = int(dist_km * 1000)

            # Duration in seconds
            duration_seconds = (dist_km / avg_speed_kps) if avg_speed_kps > 0 else np.inf # Use infinity if speed is 0
            duration_matrix[i, j] = duration_matrix[j, i] = int(duration_seconds)

    # Update cache
    _matrix_cache['distance_matrix'] = distance_matrix
    _matrix_cache['duration_matrix'] = duration_matrix
    _matrix_cache['customer_id_to_idx'] = customer_id_to_idx
    logger.info(f"Successfully built and cached global matrices ({n_locations}x{n_locations}).")


def estimate_route_time(
    cluster_customers: pd.DataFrame,
    depot: Dict[str, float],
    service_time: float,
    avg_speed: float,
    method: str = 'Legacy',
    max_route_time: float = None
) -> float:
    """
    Estimate route time using different methods.
    
    Args:
        cluster_customers: DataFrame containing customer data
        depot: Depot location coordinates
        service_time: Service time per customer (minutes)
        avg_speed: Average vehicle speed (km/h)
        method: Route time estimation method
        max_route_time: Maximum route time in hours (optional)
        
    Returns:
        Estimated route time in hours
    """
    if method == 'Legacy':
        return _legacy_estimation(len(cluster_customers), service_time)

    elif method == 'BHH':
        return _bhh_estimation(
            cluster_customers, depot, service_time, avg_speed
        )
    
    elif method == 'TSP':
        return _pyvrp_tsp_estimation(
            cluster_customers, depot, service_time, avg_speed, max_route_time
        )
    
    else:
        raise ValueError(f"Unknown route time estimation method: {method}")

def _legacy_estimation(num_customers: int, service_time: float) -> float:
    """Original simple estimation method."""
    return 1 + num_customers * service_time / 60  # Convert minutes to hours

def _bhh_estimation(
    cluster_customers: pd.DataFrame,
    depot: Dict[str, float],
    service_time: float,
    avg_speed: float
) -> float:
    """
    Beardwood-Halton-Hammersley estimation method.
    L ≈ 0.765 * sqrt(n) * sqrt(A)
    """
    if len(cluster_customers) <= 1:
        return service_time / 60
        
    # Calculate service time component
    service_time_total = len(cluster_customers) * service_time / 60  # hours
    
    # Calculate depot travel component
    centroid_lat = cluster_customers['Latitude'].mean()
    centroid_lon = cluster_customers['Longitude'].mean()
    depot_to_centroid = haversine(
        (depot['latitude'], depot['longitude']),
        (centroid_lat, centroid_lon)
    )
    depot_travel_time = 2 * depot_to_centroid / avg_speed  # Round trip hours
    
    # Calculate intra-cluster travel component using BHH formula
    cluster_radius = max(
        haversine(
            (centroid_lat, centroid_lon),
            (lat, lon)
        )
        for lat, lon in zip(
            cluster_customers['Latitude'],
            cluster_customers['Longitude']
        )
    )
    cluster_area = np.pi * (cluster_radius ** 2)
    intra_cluster_distance = (
        0.765 * 
        np.sqrt(len(cluster_customers)) * 
        np.sqrt(cluster_area)
    )
    intra_cluster_time = intra_cluster_distance / avg_speed
    
    return service_time_total + depot_travel_time + intra_cluster_time

def _pyvrp_tsp_estimation(
    cluster_customers: pd.DataFrame,
    depot: Dict[str, float],
    service_time: float,  # minutes
    avg_speed: float,      # km/h
    max_route_time: float = None  # hours, optional parameter
) -> float:
    """
    Estimate route time by solving a TSP for the cluster using PyVRP.
    Assumes infinite capacity (single vehicle TSP).
    Uses precomputed global matrices from `_matrix_cache` if available.

    Args:
        cluster_customers: DataFrame containing customer data for the cluster.
        depot: Depot location coordinates {'latitude': float, 'longitude': float}.
        service_time: Service time per customer (minutes).
        avg_speed: Average vehicle speed (km/h).
        max_route_time: Maximum route time in hours (optional, defaults to 1 week if None).

    Returns:
        Estimated route time in hours.
    """
    num_customers = len(cluster_customers)
    
    # Handle edge cases: 0 or 1 customer
    if num_customers == 0:
        return 0.0
    if num_customers == 1:
        depot_coord = (depot['latitude'], depot['longitude'])
        cust_coord = (cluster_customers.iloc[0]['Latitude'], cluster_customers.iloc[0]['Longitude'])
        dist_to = haversine(depot_coord, cust_coord)
        dist_from = haversine(cust_coord, depot_coord)
        travel_time_hours = (dist_to + dist_from) / avg_speed
        service_time_hours = service_time / 60.0
        return travel_time_hours + service_time_hours

    # --- Prepare data for PyVRP TSP ---
    
    # Create PyVRP Depot object (scaling coordinates for precision)
    pyvrp_depot = Depot(
        x=int(depot['latitude'] * 10000), 
        y=int(depot['longitude'] * 10000)
    )

    # Create PyVRP Client objects
    pyvrp_clients = []
    for _, customer in cluster_customers.iterrows():
        pyvrp_clients.append(Client(
            x=int(customer['Latitude'] * 10000),
            y=int(customer['Longitude'] * 10000),
            delivery=[1],  # Dummy demand for TSP
            service_duration=int(service_time * 60) # Service time in seconds
        ))

    # Create a single VehicleType with effectively infinite capacity and duration
    # Capacity needs to be at least num_customers for dummy demands
    # Use max_route_time if provided, otherwise use a week as effectively infinite
    max_duration_seconds = int((max_route_time or 24 * 7) * 3600)  # Convert hours to seconds
    vehicle_type = VehicleType(
        num_available=1, 
        capacity=[num_customers + 1], # Sufficient capacity for dummy demands
        max_duration=max_duration_seconds # Maximum route time in seconds
    )

    # --- Use sliced matrices from global cache if available, otherwise compute on-the-fly ---
    distance_matrix = None
    duration_matrix = None

    # Check if cache is populated
    cache_ready = (
        _matrix_cache['distance_matrix'] is not None and
        _matrix_cache['duration_matrix'] is not None and
        _matrix_cache['customer_id_to_idx'] is not None
    )

    if cache_ready:
        logger.debug(f"Using cached global matrices for cluster TSP (Size: {num_customers})")
        # Get the global distance and duration matrices and mapping
        global_distance_matrix = _matrix_cache['distance_matrix']
        global_duration_matrix = _matrix_cache['duration_matrix']
        customer_id_to_idx = _matrix_cache['customer_id_to_idx']
        depot_idx = _matrix_cache['depot_idx'] # Should be 0

        # Get indices for this specific cluster (Depot + Cluster Customers)
        cluster_indices = [depot_idx]
        missing_ids = []
        for customer_id in cluster_customers['Customer_ID']:
            idx = customer_id_to_idx.get(customer_id)
            if idx is not None:
                cluster_indices.append(idx)
            else:
                # This case should ideally not happen if build_distance_duration_matrices
                # was called with the full customer set. Log a warning if it does.
                missing_ids.append(customer_id)
                
        if missing_ids:
             logger.warning(f"Customer IDs {missing_ids} not found in global matrix cache map. TSP matrix will be incomplete.")
             # Decide how to handle this - Option 1: Fallback, Option 2: Proceed with warning
             # Fallback to on-the-fly computation if critical IDs are missing
             cache_ready = False # Force fallback if any ID is missing

        if cache_ready:
            # Slice the global matrices efficiently using numpy indexing
            n_locations = len(cluster_indices)
            # Use ix_ to select rows and columns based on index list
            distance_matrix = global_distance_matrix[np.ix_(cluster_indices, cluster_indices)]
            duration_matrix = global_duration_matrix[np.ix_(cluster_indices, cluster_indices)]
            
            # Validate dimensions
            if distance_matrix.shape != (n_locations, n_locations) or \
               duration_matrix.shape != (n_locations, n_locations):
                 logger.error("Matrix slicing resulted in unexpected dimensions. Fallback needed.")
                 cache_ready = False # Force fallback


    # Fallback: Compute matrices on-the-fly if cache wasn't ready or slicing failed
    if not cache_ready:
        logger.debug(f"Cache not ready or slicing failed. Computing matrices on-the-fly for cluster TSP (Size: {num_customers})")
        n_locations = num_customers + 1 # Customers + Depot
        locations_coords = [(depot['latitude'], depot['longitude'])] + \
                           list(zip(cluster_customers['Latitude'], cluster_customers['Longitude']))

        distance_matrix = np.zeros((n_locations, n_locations), dtype=int)
        duration_matrix = np.zeros((n_locations, n_locations), dtype=int)

        # Speed in km/s for duration calculation
        avg_speed_kps = avg_speed / 3600 if avg_speed > 0 else 0

        for i in range(n_locations):
            for j in range(i + 1, n_locations):
                # Distance using Haversine (km)
                dist_km = haversine(locations_coords[i], locations_coords[j])

                # Store distance in meters
                distance_matrix[i, j] = distance_matrix[j, i] = int(dist_km * 1000)

                # Duration in seconds
                duration_seconds = (dist_km / avg_speed_kps) if avg_speed_kps > 0 else np.inf
                duration_matrix[i, j] = duration_matrix[j, i] = int(duration_seconds)

    # --- Create Problem Data and Model ---
    # Ensure matrices were actually created (either via cache or on-the-fly)
    if distance_matrix is None or duration_matrix is None:
        logger.error("Distance/Duration matrices could not be obtained for TSP.")
        # Return a large value indicating failure/infeasibility
        return (max_route_time or 24 * 7) * 1.1 # Return > max time

    problem_data = ProblemData(
        clients=pyvrp_clients,
        depots=[pyvrp_depot],
        vehicle_types=[vehicle_type],
        distance_matrices=[distance_matrix],
        duration_matrices=[duration_matrix]
    )
    model = Model.from_data(problem_data)

    # --- Solve the TSP using PyVRP's Genetic Algorithm ---
    # Use fewer iterations suitable for smaller TSP instances
    ga_params = GeneticAlgorithmParams(
        repair_probability=0.8, # Standard default
        nb_iter_no_improvement=500 # Reduced iterations
    )
    pop_params = PopulationParams(
        min_pop_size=10, # Smaller population
        generation_size=20,
        nb_elite=2,
        nb_close=3
    )
    # Reduce max iterations for faster solving on small problems
    stop = MaxIterations(max_iterations=1000) 
    
    result = model.solve(
        stop=stop, 
        params=SolveParams(genetic=ga_params, population=pop_params), 
        display=False # No verbose output during estimation
    )
    
    # --- Extract Result ---
    if result.best.is_feasible():
        # PyVRP duration includes travel and service time in seconds
        total_duration_seconds = result.best.duration() 
        # Convert total duration to hours
        return total_duration_seconds / 3600.0
    else:
        # If max_route_time is provided, return that value (or slightly higher)
        # Otherwise use 24*7 (1 week) as the default max
        return (max_route_time or 24*7) * 1.01  # Return slightly over max_route_time