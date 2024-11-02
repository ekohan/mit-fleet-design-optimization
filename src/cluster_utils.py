from haversine import haversine
import numpy as np

def distance(coord1, coord2):
    return haversine(coord1, coord2)

def calculate_cluster_distance(cluster, depot):
    """Calculate round-trip distance for a cluster"""
    if isinstance(cluster, dict):
        cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
    else:  # DataFrame
        cluster_coord = (cluster['Latitude'].mean(), cluster['Longitude'].mean())
    
    depot_coord = (depot['Latitude'], depot['Longitude']) if isinstance(depot, dict) else depot
    return 2 * distance(depot_coord, cluster_coord)  # Round trip distance

def calculate_cluster_time(cluster, depot, avg_speed, service_time_per_customer):
    """Calculate total route time in hours for a cluster
    
    Args:
        cluster: Either a DataFrame of customers or a dict with cluster info
        depot: Either a dict with 'Latitude'/'Longitude' or a tuple of (lat, lon)
        avg_speed: Average speed in km/h
        service_time_per_customer: Service time per customer in hours
    
    Returns:
        float: Total route time in hours
    """
    # Calculate travel distance
    dist = calculate_cluster_distance(cluster, depot)
    travel_time = dist / avg_speed  # hours
    
    # Calculate number of customers
    num_customers = len(cluster['Customers']) if isinstance(cluster, dict) else len(cluster)
    total_service_time = num_customers * service_time_per_customer  # hours
    
    return travel_time + total_service_time

# Function to estimate the initial number of clusters
max_route_time = 10

def estimate_initial_clusters(customers_subset, config, depot, avg_speed, service_time_per_customer, overestimate_factor=1):
    total_customers = len(customers_subset)
    if total_customers == 0:
        return 1
        
    # Total demand is already calculated in the process_configuration function
    # so we can use it directly
    total_demand = customers_subset['Total_Demand'].sum()
    num_clusters_capacity = total_demand / config['Capacity']
    
    # Estimate based on time
    # Simple estimation: assume 1 hour travel time + service time per customer
    total_service_time = total_customers * service_time_per_customer
    num_clusters_time = total_service_time / max_route_time
    
    # Take the maximum of both estimates
    num_clusters = max(num_clusters_capacity, num_clusters_time)
    num_clusters = int(np.ceil(num_clusters * overestimate_factor))
    num_clusters = min(num_clusters, total_customers)
    num_clusters = max(1, num_clusters)
    
    return num_clusters