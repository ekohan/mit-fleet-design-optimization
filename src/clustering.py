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
import re
from functools import lru_cache

logger = logging.getLogger(__name__)

# Add after other imports
class Symbols:
    """Unicode symbols for logging."""
    CHECKMARK = "✓"
    CROSS = "✗"

PRODUCT_WEIGHTS = {'Frozen': 0.5, 'Chilled': 0.3, 'Dry': 0.2}

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
    method: str = ''

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
            'Route_Time': self.route_time,
            'Method': self.method
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
        route_time_estimation: str,
        method: str = ''
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
            )),
            method=method
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
    if not method.startswith('agglomerative'):
        logger.debug(f"Using feature-based input for method: {method}")
        return customers[['Latitude', 'Longitude']].values
    
    logger.debug(f"Using precomputed distance matrix for method: {method} with geo_weight={geo_weight}, demand_weight={demand_weight}")
    return compute_composite_distance(customers, goods, geo_weight, demand_weight)

def compute_composite_distance(
    customers: pd.DataFrame,
    goods: List[str],
    geo_weight: float,
    demand_weight: float
) -> np.ndarray:
    """Compute composite distance matrix combining geographical and demand distances."""
    # Compute geographical distance
    coords = customers[['Latitude', 'Longitude']].values
    geo_dist = pairwise_distances(coords, metric='euclidean')
    
    # Get weighted demands
    product_weights = {
        'Frozen': 0.5,    # Highest priority - temperature sensitive
        'Chilled': 0.3,   # Medium priority - temperature controlled
        'Dry': 0.2        # Lower priority - no temperature control
    }
    
    demands = customers[[f'{g}_Demand' for g in goods]].fillna(0).values
    weighted_demands = np.zeros_like(demands)
    for i, good in enumerate(goods):
        weighted_demands[:, i] = demands[:, i] * product_weights.get(good, 1.0)  # Default weight if not specified
    
    # Compute demand distance
    demand_dist = pairwise_distances(weighted_demands)
    
    # Normalize distances only if they have non-zero values
    if geo_dist.max() > 0:
        geo_dist = geo_dist / geo_dist.max()
    if demand_dist.max() > 0:
        demand_dist = demand_dist / demand_dist.max()
    
    # Return weighted combination
    composite_distance = (geo_weight * geo_dist) + (demand_weight * demand_dist)
    
    # Ensure the distance matrix is square
    if composite_distance.shape[0] != composite_distance.shape[1]:
        logger.error(f"Composite distance matrix is not square: shape={composite_distance.shape}")
        raise ValueError(f"Composite distance matrix is not square: shape={composite_distance.shape}")
    
    return composite_distance

def get_clustering_model(n_clusters: int, method: str):
    """Return the clustering model based on the method name."""
    if method == 'minibatch_kmeans':
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'kmedoids':
        return KMedoids(n_clusters=n_clusters, random_state=42)
    elif method.startswith('agglomerative'):
        # All agglomerative methods use precomputed distances
        return AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    else:
        logger.error(f"❌ Unknown clustering method: {method}")
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
    
    # Add info logging with distinctive symbols
    logger.info(f"🔍 Clustering method received: '{params.clustering['method']}'")
    
    if params.clustering['method'] == 'combine':
        logger.info("🔄 Using combine method")
        methods = ['minibatch_kmeans', 'kmedoids']
        
        # Define weight combinations (geo_weight, demand_weight)
        weight_combinations = [
            (1.0, 0.0),
            (0.8, 0.2),
            (0.6, 0.4),
            (0.4, 0.6),
            (0.2, 0.8),
            (0.0, 1.0)
        ]
        
        for geo_w, demand_w in weight_combinations:
            method_name = f'agglomerative_geo_{geo_w}_demand_{demand_w}'
            methods.append(method_name)
    else:
        logger.info(f"📍 Using single method: {params.clustering['method']}")
        methods = [params.clustering['method']]

    # Initialize a global cluster ID counter to ensure uniqueness
    cluster_id_counter = 0

    # Process configurations in parallel for each method
    all_clusters = []
    for method in methods:
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
                method,  # Using the current method
                params.clustering['route_time_estimation'],
                params.clustering['geo_weight'],
                params.clustering['demand_weight'],
                params.clustering['distance']
            )
            for _, config in configurations_df.iterrows()
        )
        
        # Combine all clusters from this method with unique Cluster_IDs
        for config_clusters in clusters_by_config:
            for cluster in config_clusters:
                # Assign unique Cluster_ID
                cluster['Cluster_ID'] = cluster_id_counter
                cluster_id_counter += 1
                all_clusters.append(cluster)
    
    # Convert to DataFrame
    combined_clusters_df = pd.DataFrame(all_clusters)
    
    # Remove duplicate clusters based on customer sets
    combined_clusters_df['Customer_Set'] = combined_clusters_df['Customers'].apply(lambda x: frozenset(x))
    combined_clusters_df = combined_clusters_df.drop_duplicates(subset=['Customer_Set']).drop(columns=['Customer_Set'])

    # Validate cluster coverage
    validate_cluster_coverage(combined_clusters_df, customers)

    logger.info(f"{Symbols.CHECKMARK} Generated a total of {len(combined_clusters_df)} unique clusters using '{params.clustering['method']}' method.")

    return combined_clusters_df

def compute_demands(customers: pd.DataFrame, goods: List[str]) -> Dict[str, np.ndarray]:
    """Compute and cache demand calculations for customers."""
    return {
        'total': customers[[f'{g}_Demand' for g in goods]].sum(axis=1),
        'by_good': customers[[f'{g}_Demand' for g in goods]].fillna(0).values,
        'weighted': customers[[f'{g}_Demand' for g in goods]].fillna(0).values * \
                   np.array([PRODUCT_WEIGHTS.get(g, 1.0) for g in goods])
    }

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
    """Process a single vehicle configuration to generate feasible clusters."""
    config_id = config['Config_ID']
    clusters = []
    
    # Initialize adjusted weights with default values
    adjusted_geo_weight = geo_weight
    adjusted_demand_weight = demand_weight
    
    # Get feasible customers for this configuration
    customers_subset = customers[
        customers['Customer_ID'].isin([
            cid for cid, configs in feasible_customers.items() 
            if config_id in configs
        ])
    ].copy()

    if customers_subset.empty:
        return []

    # Always compute total demand first
    demands = compute_demands(customers_subset, goods)
    customers_subset['Total_Demand'] = demands['total']

    # For small subsets, always use MiniBatchKMeans
    if len(customers_subset) <= 2:
        logger.debug(f"Using MiniBatchKMeans for small cluster with {len(customers_subset)} customers.")
        data = customers_subset[['Latitude', 'Longitude']].values
        model = MiniBatchKMeans(n_clusters=1, random_state=42)
        customers_subset['Cluster'] = model.fit_predict(data)
    else:
        # Initial clustering
        num_clusters = estimate_num_initial_clusters(
            customers_subset, 
            config, 
            depot, 
            avg_speed, 
            service_time,
            goods,
            max_route_time,
            route_time_estimation
        )

        # Handle agglomerative variants
        if '_' in clustering_method:
            pattern = r'agglomerative_geo_(\d+\.\d+)_demand_(\d+\.\d+)'
            if match := re.match(pattern, clustering_method):
                adjusted_geo_weight = float(match.group(1))
                adjusted_demand_weight = float(match.group(2))

        # Get input data and cluster using adjusted weights
        data = get_clustering_input(
            customers_subset, 
            goods, 
            clustering_method, 
            adjusted_geo_weight,
            adjusted_demand_weight,
            distance_metric
        )

        # Ensure the number of clusters does not exceed the number of customers
        num_clusters = min(num_clusters, len(customers_subset))

        model = get_clustering_model(num_clusters, clustering_method)
        customers_subset['Cluster'] = model.fit_predict(data)

    # Process clusters (same for both cases)
    clusters_to_check = [
        (customers_subset[customers_subset['Cluster'] == c], 0)
        for c in customers_subset['Cluster'].unique()
    ]
    cluster_id_base = int(str(config_id) + "000")
    current_cluster_id = 0

    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()
        # Ensure Total_Demand is computed for the subset
        if 'Total_Demand' not in cluster_customers.columns:
            cluster_customers['Total_Demand'] = compute_demands(cluster_customers, goods)['total']
            
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
                logger.debug(f"Splitting cluster ID {current_cluster_id} at depth {depth}.")

                # Split cluster using the same approach
                split_data = get_clustering_input(
                    cluster_customers, 
                    goods, 
                    clustering_method,
                    adjusted_geo_weight,
                    adjusted_demand_weight,
                    distance_metric
                )
                # Split over-capacity clusters into two
                split_model = get_clustering_model(2, clustering_method)
                sub_labels = split_model.fit_predict(split_data)

                for label in [0, 1]:
                    sub_cluster = cluster_customers[sub_labels == label]
                    if not sub_cluster.empty:
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
                route_time_estimation,
                clustering_method
            )
            cluster_dict = cluster.to_dict()
            # Add the clustering method to the cluster information
            cluster_dict['Method'] = clustering_method
            clusters.append(cluster_dict)

    logger.info(f"🔧 Clustering Method: {clustering_method} | geo_weight={adjusted_geo_weight}, demand_weight={adjusted_demand_weight}")

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

def validate_cluster_coverage(clusters_df, customers_df):
    customer_coverage = {cid: False for cid in customers_df['Customer_ID']}
    for customers in clusters_df['Customers']:
        for cid in customers:
            customer_coverage[cid] = True
    uncovered = [cid for cid, covered in customer_coverage.items() if not covered]

@lru_cache(maxsize=128)
def compute_distance_matrix(customer_ids, method):
    """Cache distance computations for frequently accessed customer sets."""
