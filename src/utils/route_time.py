"""Route time estimation methods for vehicle routing."""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from haversine import haversine

def estimate_route_time(
    cluster_customers: pd.DataFrame,
    depot: Dict[str, float],
    service_time: float,
    avg_speed: float,
    method: str = 'Legacy'
) -> float:
    """
    Estimate route time using different methods.
    
    Args:
        cluster_customers: DataFrame containing customer data
        depot: Depot location coordinates
        service_time: Service time per customer (minutes)
        avg_speed: Average vehicle speed (km/h)
        method: Route time estimation method
        
    Returns:
        Estimated route time in hours
    """
    if method == 'Legacy':
        return _legacy_estimation(len(cluster_customers), service_time)
    
    elif method == 'Clarke-Wright':
        return _clarke_wright_estimation(
            cluster_customers, depot, service_time, avg_speed
        )
    
    elif method == 'BHH':
        return _bhh_estimation(
            cluster_customers, depot, service_time, avg_speed
        )
    
    elif method == 'CA':
        return _continuous_approximation(
            cluster_customers, depot, service_time, avg_speed
        )
    
    elif method == 'VRPSolver':
        return _vrp_solver_estimation(
            cluster_customers, depot, service_time, avg_speed
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

# TODO: Implement other estimation methods
def _clarke_wright_estimation():
    raise NotImplementedError

def _continuous_approximation():
    raise NotImplementedError

def _vrp_solver_estimation():
    raise NotImplementedError 