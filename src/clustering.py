"""
Module for generating clusters from customer data.
"""

import logging
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from haversine import haversine

from config import (
    MAX_SPLIT_DEPTH,
    SERVICE_TIME_PER_CUSTOMER,
    MAX_ROUTE_TIME,
    AVG_SPEED
)

logger = logging.getLogger(__name__)

def generate_clusters_for_configurations(
    customers: pd.DataFrame,
    configurations_df: pd.DataFrame,
    goods: List[str],
    depot: Dict[str, float]
) -> pd.DataFrame:
    """
    Generate clusters for each vehicle configuration in parallel.
    
    Args:
        customers: DataFrame containing customer data
        configurations_df: DataFrame containing vehicle configurations
        goods: List of goods types
        depot: Dictionary containing depot coordinates
    
    Returns:
        DataFrame containing all generated clusters
    """
    # Generate feasibility mapping
    feasible_customers = _generate_feasibility_mapping(
        customers, 
        configurations_df,
        goods
    )
    
    # Process configurations in parallel
    clusters_by_config = Parallel(n_jobs=-1)(
        delayed(process_configuration)(
            config=config,
            customers=customers,
            goods=goods,
            depot=depot,
            avg_speed=AVG_SPEED,
            service_time_per_customer=SERVICE_TIME_PER_CUSTOMER,
            max_route_time=MAX_ROUTE_TIME,
            feasible_customers=feasible_customers,
            max_split_depth=MAX_SPLIT_DEPTH
        )
        for _, config in configurations_df.iterrows()
    )
    
    # Combine all clusters
    clusters_list = []
    for config_clusters in clusters_by_config:
        clusters_list.extend(config_clusters)
    
    return pd.DataFrame(clusters_list)

def process_configuration(
    config: pd.Series,
    customers: pd.DataFrame,
    goods: List[str],
    depot: Dict[str, float],
    avg_speed: float,
    service_time_per_customer: float,
    max_route_time: float,
    feasible_customers: Dict,
    max_split_depth: int
) -> List[Dict]:
    """
    Process a single vehicle configuration to generate feasible clusters.
    
    Args:
        config: Vehicle configuration
        customers: Customer demand data
        goods: List of goods types
        depot: Depot location
        avg_speed: Average vehicle speed
        service_time_per_customer: Service time per customer
        max_route_time: Maximum route time
        feasible_customers: Mapping of customers to feasible configurations
        max_split_depth: Maximum depth for cluster splitting
    
    Returns:
        List of cluster dictionaries
    """
    config_id = config['Config_ID']
    clusters = []
    
    # Get feasible customers for this configuration
    customers_subset = customers[
        customers['Customer_ID'].isin([
            cid for cid, configs in feasible_customers.items() 
            if config_id in configs
        ])
    ].copy()

    if customers_subset.empty:
        return []

    # Calculate total demand
    total_demand = customers_subset[[f'{g}_Demand' for g in goods]].sum(axis=1)
    customers_subset['Total_Demand'] = total_demand
    
    # Initial clustering
    num_clusters = estimate_num_initial_clusters(
        customers_subset, 
        config, 
        depot, 
        avg_speed, 
        service_time_per_customer,
        goods
    )

    coords = customers_subset[['Latitude', 'Longitude']]
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=10000, n_init='auto')
    customers_subset['Cluster'] = kmeans.fit_predict(coords)

    # Process each initial cluster
    clusters_to_check: List[Tuple[pd.DataFrame, int]] = [
        (customers_subset[customers_subset['Cluster'] == c], 0)
        for c in customers_subset['Cluster'].unique()
    ]

    cluster_id_base = int(str(config_id) + "000")
    current_cluster_id = 0

    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()
        cluster_demand = cluster_customers['Total_Demand'].sum()
        route_time = 1 + len(cluster_customers) * service_time_per_customer

        if (cluster_demand > config['Capacity'] or route_time > max_route_time) and depth < max_split_depth:
            if len(cluster_customers) > 1:
                # Split cluster
                coords = cluster_customers[['Latitude', 'Longitude']].to_numpy()
                sub_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=10000, n_init='auto')
                sub_labels = sub_kmeans.fit_predict(coords)

                for label in [0, 1]:
                    sub_cluster = cluster_customers[sub_labels == label]
                    if len(sub_cluster) > 0:
                        clusters_to_check.append((sub_cluster, depth + 1))
        else:
            # Add valid cluster
            current_cluster_id += 1
            clusters.append({
                'Cluster_ID': cluster_id_base + current_cluster_id,
                'Config_ID': config['Config_ID'],
                'Customers': cluster_customers['Customer_ID'].tolist(),
                'Total_Demand': {
                    g: float(cluster_customers[f'{g}_Demand'].sum()) 
                    for g in goods
                },
                'Centroid_Latitude': float(cluster_customers['Latitude'].mean()),
                'Centroid_Longitude': float(cluster_customers['Longitude'].mean()),
                'Goods_In_Config': [g for g in goods if config[g] == 1],
                'Route_Time': float(route_time)
            })

    return clusters

def estimate_num_initial_clusters(
    customers: pd.DataFrame,
    config: pd.Series,
    depot: Dict[str, float],
    avg_speed: float,
    service_time: float,
    goods: List[str]
) -> int:
    """
    Estimate the number of initial clusters needed based on capacity and time constraints.
    
    Args:
        customers: DataFrame containing customer data
        config: Vehicle configuration
        depot: Depot location coordinates
        avg_speed: Average vehicle speed (km/h)
        service_time: Service time per customer (hours)
        goods: List of goods types
    
    Returns:
        Estimated number of clusters needed
    """
    if customers.empty:
        return 0

    # Calculate total demand for relevant goods
    total_demand = 0
    for good in goods:
        if config[good]:  # Only consider goods this vehicle can carry
            total_demand += customers[f'{good}_Demand'].sum()

    # Estimate clusters needed based on capacity
    clusters_by_capacity = np.ceil(total_demand / config['Capacity'])

    # Calculate average distance from depot to customers
    depot_coord = (depot['Latitude'], depot['Longitude'])
    avg_distance = np.mean([
        haversine(depot_coord, (lat, lon))
        for lat, lon in zip(customers['Latitude'], customers['Longitude'])
    ])

    # Estimate time for an average route
    avg_customers_per_cluster = len(customers) / clusters_by_capacity
    avg_route_time = (
        2 * avg_distance / avg_speed +  # Round trip from depot
        service_time * avg_customers_per_cluster  # Service time for customers
    )

    # Estimate clusters needed based on time
    clusters_by_time = np.ceil(
        avg_route_time * len(customers) / 
        (MAX_ROUTE_TIME * avg_customers_per_cluster)
    )

    # Take the maximum of the two estimates
    num_clusters = int(max(clusters_by_capacity, clusters_by_time, 1))
    
    logger.debug(
        f"Estimated clusters: {num_clusters} "
        f"(capacity: {clusters_by_capacity}, time: {clusters_by_time})"
    )
    
    return num_clusters

def _generate_feasibility_mapping(
    customers: pd.DataFrame,
    configurations_df: pd.DataFrame,
    goods: List[str]
) -> Dict:
    """
    Generate mapping of feasible configurations for each customer.
    """
    feasible_customers = {}
    
    for _, customer in customers.iterrows():
        customer_id = customer['Customer_ID']
        feasible_configs = []
        
        for _, config in configurations_df.iterrows():
            if _is_customer_feasible(customer, config, goods):
                feasible_configs.append(config['Config_ID'])
        
        if feasible_configs:
            feasible_customers[customer_id] = feasible_configs
    
    return feasible_customers

def _is_customer_feasible(
    customer: pd.Series,
    config: pd.Series,
    goods: List[str]
) -> bool:
    """
    Check if a customer's demands can be served by a configuration.
    """
    for good in goods:
        if customer[f'{good}_Demand'] > 0 and not config[good]:
            return False
        if customer[f'{good}_Demand'] > config['Capacity']:
            return False
    return True