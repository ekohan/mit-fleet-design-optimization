"""
Module for generating clusters from customer data.
"""

import logging
from typing import Dict, List
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed
from haversine import haversine
from config.parameters import Parameters
from utils.route_time import estimate_route_time
from scipy.spatial import distance
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Cluster:
    """Represents a cluster of customers that can be served by a vehicle configuration."""
    cluster_id: int
    config_id: int
    customers: List[str]
    total_demand: Dict[str, float]
    centroid_latitude: float
    centroid_longitude: float
    goods_in_config: List[str]
    route_time: float

    def to_dict(self) -> Dict:
        """Convert cluster to dictionary format."""
        return {
            'Cluster_ID': self.cluster_id,
            'Config_ID': self.config_id,
            'Customers': self.customers,
            'Total_Demand': self.total_demand,
            'Centroid_Latitude': self.centroid_latitude,
            'Centroid_Longitude': self.centroid_longitude,
            'Goods_In_Config': self.goods_in_config,
            'Route_Time': self.route_time
        }

    @classmethod
    def from_customers(
        cls,
        customers: pd.DataFrame,
        config: pd.Series,
        cluster_id: int,
        goods: List[str],
        depot: Dict[str, float],
        service_time: float,
        avg_speed: float,
        route_time_estimation: str
    ) -> 'Cluster':
        """Create a cluster from customer data."""
        return cls(
            cluster_id=cluster_id,
            config_id=config['Config_ID'],
            customers=customers['Customer_ID'].tolist(),
            total_demand={g: float(customers[f'{g}_Demand'].sum()) for g in goods},
            centroid_latitude=float(customers['Latitude'].mean()),
            centroid_longitude=float(customers['Longitude'].mean()),
            goods_in_config=[g for g in goods if config[g] == 1],
            route_time=float(estimate_route_time(
                cluster_customers=customers,
                depot=depot,
                service_time=service_time,
                avg_speed=avg_speed,
                method=route_time_estimation
            ))
        )

def get_clustering_input(
    customers: pd.DataFrame, 
    goods: List[str], 
    method: str,
    geo_weight: float,
    demand_weight: float,
    distance_metric: str = 'euclidean'
) -> np.ndarray:
    """Get appropriate input for clustering algorithm."""
    # Add small random noise to coordinates for all methods
    coords = customers[['Latitude', 'Longitude']].values
    epsilon = 1e-4
    coords = coords + np.random.uniform(-epsilon, epsilon, size=coords.shape)
    
    if method != 'agglomerative':
        return coords
    
    # For agglomerative clustering, we need to return a distance matrix
    if distance_metric == 'euclidean':
        return pairwise_distances(coords, metric='euclidean')
    
    # For composite distance
    demands = customers[[f'{g}_Demand' for g in goods]].fillna(0).values
    product_weights = {
        'Frozen': 0.5,    # Highest priority - temperature sensitive
        'Chilled': 0.3,   # Medium priority - temperature controlled
        'Dry': 0.2        # Lower priority - no temperature control
    }
    
    # Apply weights to each product type
    weighted_demands = np.zeros_like(demands)
    for i, good in enumerate(goods):
        weighted_demands[:, i] = demands[:, i] * product_weights[good]
    
    # Compute distances
    geo_dist = pairwise_distances(coords)
    demand_dist = pairwise_distances(weighted_demands)
    
    # Normalize distances only if they have non-zero values
    if geo_dist.max() > 0:
        geo_dist = geo_dist / geo_dist.max()
    if demand_dist.max() > 0:
        demand_dist = demand_dist / demand_dist.max()
    
    # Combine with configurable weights
    return (geo_weight * geo_dist + 
            demand_weight * demand_dist)

def get_clustering_model(n_clusters: int, method: str):
    """Get clustering model based on method name."""
    # For small n_clusters, force KMeans which works with any size
    if n_clusters < 2:
        return MiniBatchKMeans(n_clusters=1, random_state=42, batch_size=10000, n_init=3)
        
    if method == 'minibatch_kmeans':
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=10000, n_init=3)
    elif method == 'kmedoids':
        return KMedoids(n_clusters=n_clusters, random_state=42)
    elif method == 'agglomerative':
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='complete'
        )
    raise ValueError(f"Unknown clustering method: {method}")

def generate_clusters_for_configurations(
    customers: pd.DataFrame,
    configurations_df: pd.DataFrame,
    params: Parameters,
) -> pd.DataFrame:
    """
    Generate clusters for each vehicle configuration in parallel.
    
    Args:
        customers: DataFrame containing customer data
        configurations_df: DataFrame containing vehicle configurations
        params: Parameters object containing vehicle configuration parameters
    
    Returns:
        DataFrame containing all generated clusters
    """
    # Generate feasibility mapping
    feasible_customers = _generate_feasibility_mapping(
        customers, 
        configurations_df,
        params.goods
    )
    
    # Process configurations in parallel
    clusters_by_config = Parallel(n_jobs=-1)(
        delayed(process_configuration)(
            config,
            customers,
            params.goods,
            params.depot,
            params.avg_speed,
            params.service_time,
            params.max_route_time,
            feasible_customers,
            params.clustering['max_depth'],
            params.clustering['method'],
            params.clustering['route_time_estimation'],
            params.clustering['geo_weight'],
            params.clustering['demand_weight'],
            params.clustering['distance']
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
    service_time: float,
    max_route_time: float,
    feasible_customers: Dict,
    max_split_depth: int,
    clustering_method: str,
    route_time_estimation: str,
    geo_weight: float,
    demand_weight: float,
    distance_metric: str
) -> List[Dict]:
    """
    Process a single vehicle configuration to generate feasible clusters.
    
    Args:
        config: Vehicle configuration
        customers: Customer demand data
        goods: List of goods types
        depot: Depot location
        avg_speed: Average vehicle speed
        service_time: Service time per customer (minutes)
        max_route_time: Maximum route time (hours)
        feasible_customers: Mapping of customers to feasible configurations
        max_split_depth: Maximum depth for cluster splitting
        clustering_method: Clustering method
        route_time_estimation: Route time estimation method
        geo_weight: Geographical distance weight
        demand_weight: Demand distance weight
        distance_metric: Distance metric for clustering
    
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
        service_time,  # in minutes
        goods,
        max_route_time,  # in hours
        route_time_estimation
    )

    # Get input data and cluster
    data = get_clustering_input(
        customers_subset, 
        goods, 
        clustering_method, 
        geo_weight,
        demand_weight,
        distance_metric
    )
    model = get_clustering_model(num_clusters, clustering_method)
    customers_subset['Cluster'] = model.fit_predict(data)

    # Process each initial cluster
    clusters_to_check = [
        (customers_subset[customers_subset['Cluster'] == c], 0)
        for c in customers_subset['Cluster'].unique()
    ]

    cluster_id_base = int(str(config_id) + "000")
    current_cluster_id = 0

    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()
        cluster_demand = cluster_customers['Total_Demand'].sum()
        route_time = estimate_route_time(
            cluster_customers=cluster_customers,
            depot=depot,
            service_time=service_time,
            avg_speed=avg_speed,
            method=route_time_estimation
        )

        if (cluster_demand > config['Capacity'] or route_time > max_route_time) and depth < max_split_depth:
            if len(cluster_customers) > 1:
                # Split cluster using same approach
                data = get_clustering_input(
                    cluster_customers, 
                    goods, 
                    clustering_method,
                    geo_weight,
                    demand_weight,
                    distance_metric
                )
                # Split over-capacity clusters into two
                model = get_clustering_model(2, clustering_method)
                sub_labels = model.fit_predict(data)

                for label in [0, 1]:
                    sub_cluster = cluster_customers[sub_labels == label]
                    if len(sub_cluster) > 0:
                        clusters_to_check.append((sub_cluster, depth + 1))
        else:
            # Add valid cluster
            current_cluster_id += 1
            cluster = Cluster.from_customers(
                cluster_customers,
                config,
                cluster_id_base + current_cluster_id,
                goods,
                depot,
                service_time,
                avg_speed,
                route_time_estimation
            )
            clusters.append(cluster.to_dict())

    return clusters

def _generate_feasibility_mapping(
    customers: pd.DataFrame,
    configurations_df: pd.DataFrame,
    goods: List[str]
) -> Dict:
    """Generate mapping of feasible configurations for each customer."""
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
    """Check if a customer's demands can be served by a configuration."""
    for good in goods:
        if customer[f'{good}_Demand'] > 0 and not config[good]:
            return False
        if customer[f'{good}_Demand'] > config['Capacity']:
            return False
    return True

def estimate_num_initial_clusters(
    customers: pd.DataFrame,
    config: pd.Series,
    depot: Dict[str, float],
    avg_speed: float,
    service_time: float,
    goods: List[str],
    max_route_time: float,
    route_time_estimation: str
) -> int:
    """
    Estimate the number of initial clusters needed based on capacity and time constraints.
    
    Args:
        customers: DataFrame containing customer data
        config: Vehicle configuration
        depot: Depot location coordinates
        avg_speed: Average vehicle speed (km/h)
        service_time: Service time per customer (minutes)
        goods: List of goods types
        max_route_time: Maximum route time (hours)
        route_time_estimation: Method to estimate route times 
                             (Legacy, Clarke-Wright, BHH, CA, VRPSolver)
    
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
    depot_coord = (depot['latitude'], depot['longitude'])
    avg_distance = np.mean([
        haversine(depot_coord, (lat, lon))
        for lat, lon in zip(customers['Latitude'], customers['Longitude'])
    ])

    # Estimate time for an average route
    avg_customers_per_cluster = len(customers) / clusters_by_capacity
    avg_cluster = customers.sample(n=min(int(avg_customers_per_cluster), len(customers)))
    avg_route_time = estimate_route_time(
        cluster_customers=avg_cluster,
        depot=depot,
        service_time=service_time,  # in minutes
        avg_speed=avg_speed,
        method=route_time_estimation
    )

    # Estimate clusters needed based on time
    clusters_by_time = np.ceil(
        avg_route_time * len(customers) / 
        (max_route_time * avg_customers_per_cluster)
    )

    # Take the maximum of the two estimates
    num_clusters = int(max(clusters_by_capacity, clusters_by_time, 1))
    
    logger.debug(
        f"Estimated clusters: {num_clusters} "
        f"(capacity: {clusters_by_capacity}, time: {clusters_by_time})"
    )
    
    return num_clusters

def compute_composite_distance(customers: pd.DataFrame) -> np.ndarray:
    """Compute composite distance matrix combining geographical and demand distances"""
    # Compute geographical distance
    coords = customers[['Latitude', 'Longitude']].values
    geo_dist = distance.cdist(coords, coords, metric='euclidean')
    
    # Compute demand distance
    demand = customers[['Dry', 'Chilled', 'Frozen']].fillna(0).values  # Handle NaN
    demand_dist = distance.cdist(demand, demand, metric='euclidean')
    
    # Normalize distances to [0,1] range
    if geo_dist.max() > 0:  # Prevent division by zero
        geo_dist /= geo_dist.max()
    if demand_dist.max() > 0:  # Prevent division by zero
        demand_dist /= demand_dist.max()
    
    # Return weighted combination
    return 0.7 * geo_dist + 0.3 * demand_dist
