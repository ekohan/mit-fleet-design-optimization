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
from src.config.parameters import Parameters
from src.utils.route_time import estimate_route_time
from scipy.spatial import distance
from dataclasses import dataclass, replace
import re
from functools import lru_cache
from sklearn.mixture import GaussianMixture
import warnings
from sklearn.exceptions import ConvergenceWarning
import itertools

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
        settings: 'ClusteringSettings',
        method: str = ''
    ) -> 'Cluster':
        """Create a cluster from customer data using settings."""
        return cls(
            cluster_id=cluster_id,
            config_id=config['Config_ID'],
            customers=customers['Customer_ID'].tolist(),
            total_demand={g: float(customers[f'{g}_Demand'].sum()) for g in settings.goods},
            centroid_latitude=float(customers['Latitude'].mean()),
            centroid_longitude=float(customers['Longitude'].mean()),
            goods_in_config=[g for g in settings.goods if config[g] == 1],
            route_time=float(estimate_route_time(
                cluster_customers=customers,
                depot=settings.depot,
                service_time=settings.service_time,
                avg_speed=settings.avg_speed,
                method=settings.route_time_estimation
            )),
            method=settings.method
        )

@dataclass
class ClusteringSettings:
    """Encapsulates all settings required for a clustering run."""
    method: str
    goods: List[str]
    depot: Dict[str, float]
    avg_speed: float
    service_time: float
    max_route_time: float
    max_depth: int
    route_time_estimation: str
    geo_weight: float
    demand_weight: float
    distance_metric: str # 'euclidean' or 'composite'

def compute_cluster_metric_input(
    customers: pd.DataFrame,
    settings: ClusteringSettings
) -> np.ndarray:
    """Get appropriate input for clustering algorithm."""
    # Methods that need precomputed distance matrix
    needs_precomputed = (
        settings.method.startswith('agglomerative') 
    )
    
    if needs_precomputed:
        logger.debug(f"Using precomputed distance matrix for method: {settings.method} with geo_weight={settings.geo_weight}, demand_weight={settings.demand_weight}")
        return compute_composite_distance(customers, settings.goods, settings.geo_weight, settings.demand_weight)
    else:
        logger.debug(f"Using feature-based input for method: {settings.method}")
        return customers[['Latitude', 'Longitude']].values

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
    
    # Compute demand profiles
    demands = customers[[f'{g}_Demand' for g in goods]].fillna(0).values
    demand_profiles = np.zeros_like(demands, dtype=float)
    
    # Convert to proportions (fixing the broadcasting issue)
    total_demands = demands.sum(axis=1)
    nonzero_mask = total_demands > 0
    for i in range(len(goods)):
        demand_profiles[nonzero_mask, i] = demands[nonzero_mask, i] / total_demands[nonzero_mask]
    
    # Apply temperature sensitivity weights
    product_weights = {
        'Frozen': 0.5,    # Highest priority
        'Chilled': 0.3,   # Medium priority
        'Dry': 0.2        # Lower priority
    }
    for i, good in enumerate(goods):
        demand_profiles[:, i] *= product_weights.get(good, 1.0)
    
    # Compute demand similarity using cosine distance
    demand_dist = pairwise_distances(demand_profiles, metric='cosine')
    demand_dist = np.nan_to_num(demand_dist, nan=1.0)
    
    # Normalize distances
    if geo_dist.max() > 0:
        geo_dist = geo_dist / geo_dist.max()
    if demand_dist.max() > 0:
        demand_dist = demand_dist / demand_dist.max()
    
    # Combine distances with weights
    composite_distance = (geo_weight * geo_dist) + (demand_weight * demand_dist)
    
    return composite_distance

def get_clustering_model(n_clusters: int, method: str):
    """Return the clustering model based on the method name."""
    if method == 'minibatch_kmeans':
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'kmedoids':
        return KMedoids(n_clusters=n_clusters, random_state=42)
    elif method.startswith('agglomerative'):
        return AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    elif method == 'gaussian_mixture':
        return GaussianMixture(
            n_components=n_clusters,
            random_state=42,
            covariance_type='full'
        )
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
    logger.info("--- Starting Cluster Generation Process ---")
    if customers.empty or configurations_df.empty:
        logger.warning("Input customers or configurations are empty. Returning empty DataFrame.")
        return pd.DataFrame()

    # 1. Generate feasibility mapping
    logger.info("Generating feasibility mapping...")
    feasible_customers = _generate_feasibility_mapping(
        customers, 
        configurations_df,
        params.goods
    )
    if not feasible_customers:
        logger.warning("No customers are feasible for any configuration. Returning empty DataFrame.")
        return pd.DataFrame()
    logger.info(f"Feasibility mapping generated for {len(feasible_customers)} customers.")

    # 2. Generate list of ClusteringSettings objects for all runs
    list_of_settings = _get_clustering_settings_list(params)

    # Use itertools.count for safer ID generation across parallel runs potentially
    cluster_id_generator = itertools.count()

    # 4. Process configurations in parallel for each settings configuration
    all_clusters = []
    for settings_for_run in list_of_settings:
        logger.info(f"--- Running Configuration: {settings_for_run.method} (GeoW: {settings_for_run.geo_weight:.2f}, DemW: {settings_for_run.demand_weight:.2f}) ---")

        # Run clustering for all configurations using these settings in parallel
        # Use threading backend if models aren't releasing GIL effectively, but start with process-based
        clusters_by_config = Parallel(n_jobs=-1, backend='loky')(
            delayed(process_configuration)(
                config=config,
                customers=customers,
                feasible_customers=feasible_customers,
                settings=settings_for_run
            )
            for _, config in configurations_df.iterrows()
        )
        
        # Flatten the list of lists returned by Parallel and assign IDs
        for config_clusters in clusters_by_config:
            for cluster_dict in config_clusters:
                # Assign unique Cluster_ID using the generator
                cluster_dict['Cluster_ID'] = next(cluster_id_generator)
                all_clusters.append(cluster_dict)
        logger.info(f"--- Configuration {settings_for_run.method} completed, generated {len([c for config_clusters in clusters_by_config for c in config_clusters])} raw clusters ---")

    # Convert to DataFrame
    if not all_clusters:
        logger.warning("No clusters were generated by any configuration.")
        return pd.DataFrame()

    # Remove duplicate clusters based on customer sets
    logger.info(f"Combining and deduplicating {len(all_clusters)} raw clusters from all configurations...")
    combined_clusters_df = pd.DataFrame(all_clusters)
    unique_clusters_df = _deduplicate_clusters(combined_clusters_df)

    # Validate cluster coverage
    validate_cluster_coverage(unique_clusters_df, customers)

    logger.info("--- Cluster Generation Complete ---")
    logger.info(f"{Symbols.CHECKMARK} Generated a total of {len(unique_clusters_df)} unique clusters across all configurations.")

    return unique_clusters_df

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
    feasible_customers: Dict,
    settings: ClusteringSettings
) -> List[Dict]:
    """Process a single vehicle configuration to generate feasible clusters."""
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

    # Always compute total demand first
    demands = compute_demands(customers_subset, settings.goods)
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
            settings
        )

        # Get input data and cluster using weights from settings
        data = compute_cluster_metric_input(
            customers_subset,
            settings
        )

        # Ensure the number of clusters does not exceed the number of customers
        num_clusters = min(num_clusters, len(customers_subset))

        model = get_clustering_model(num_clusters, settings.method)
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
            cluster_customers['Total_Demand'] = compute_demands(cluster_customers, settings.goods)['total']
            
        cluster_demand = cluster_customers['Total_Demand'].sum()
        route_time = estimate_route_time(
            cluster_customers=cluster_customers,
            depot=settings.depot,
            service_time=settings.service_time,
            avg_speed=settings.avg_speed,
            method=settings.route_time_estimation
        )

        if (cluster_demand > config['Capacity'] or route_time > settings.max_route_time) and depth < settings.max_depth:
            if len(cluster_customers) > 1:
                logger.debug(f"Splitting cluster for config {config_id} (size {len(cluster_customers)}) at depth {depth}.")

                # Split cluster using the same approach
                split_data = compute_cluster_metric_input(
                    cluster_customers,
                    settings
                )
                # Split over-capacity clusters into two
                split_model = get_clustering_model(2, settings.method)
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
                settings,
                settings.method
            )
            cluster_dict = cluster.to_dict()
            # Add the clustering method to the cluster information
            cluster_dict['Method'] = settings.method
            clusters.append(cluster_dict)

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
    settings: ClusteringSettings
) -> int:
    """Estimate the number of initial clusters needed based on capacity and time constraints."""
    if customers.empty:
        return 0

    # Calculate total demand for relevant goods
    total_demand = 0
    for good in settings.goods:
        if config[good]:  # Only consider goods this vehicle can carry
            total_demand += customers[f'{good}_Demand'].sum()

    # Estimate clusters needed based on capacity
    clusters_by_capacity = np.ceil(total_demand / config['Capacity'])

    # Calculate average distance from depot to customers
    depot_coord = (settings.depot['latitude'], settings.depot['longitude'])
    avg_distance = np.mean([
        haversine(depot_coord, (lat, lon))
        for lat, lon in zip(customers['Latitude'], customers['Longitude'])
    ])

    # Estimate time for an average route
    avg_customers_per_cluster = len(customers) / clusters_by_capacity
    # Ensure sample size doesn't exceed population size and is at least 1 if possible
    sample_size = max(1, min(int(avg_customers_per_cluster), len(customers)))
    avg_cluster = customers.sample(n=sample_size)
    avg_route_time = estimate_route_time(
        cluster_customers=avg_cluster,
        depot=settings.depot,
        service_time=settings.service_time,  # in minutes
        avg_speed=settings.avg_speed,
        method=settings.route_time_estimation
    )

    # Estimate clusters needed based on time
    clusters_by_time = np.ceil(
        avg_route_time * len(customers) / 
        (settings.max_route_time * avg_customers_per_cluster)
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


def _deduplicate_clusters(clusters_df: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate clusters based on the set of customers."""
    if clusters_df.empty:
        return clusters_df
    logger.info(f"Starting deduplication with {len(clusters_df)} clusters.")
    # Create a column of frozensets for efficient duplicate checking
    # The lambda handles cases where 'Customers' might not be a list/tuple, creating an empty set
    clusters_df['Customer_Set'] = clusters_df['Customers'].apply(
        lambda x: frozenset(x) if isinstance(x, (list, tuple)) else frozenset()
    )
    # Drop duplicates based on the frozenset column
    deduplicated_df = clusters_df.drop_duplicates(subset=['Customer_Set'], keep='first').drop(columns=['Customer_Set'])
    
    if len(deduplicated_df) < len(clusters_df):
        logger.info(f"Finished deduplication: Removed {len(clusters_df) - len(deduplicated_df)} duplicate clusters, {len(deduplicated_df)} unique clusters remain.")
    else:
        logger.info(f"Finished deduplication: No duplicate clusters found ({len(deduplicated_df)} clusters).")
    return deduplicated_df

def _get_clustering_settings_list(params: Parameters) -> List[ClusteringSettings]:
    """Generates a list of ClusteringSettings objects for all runs."""
    settings_list = []

    # Create a base settings object with common parameters
    # The weights here will be used ONLY if a single method is specified
    base_settings = ClusteringSettings(
        method=params.clustering['method'],
        goods=params.goods,
        depot=params.depot,
        avg_speed=params.avg_speed,
        service_time=params.service_time,
        max_route_time=params.max_route_time,
        max_depth=params.clustering['max_depth'],
        route_time_estimation=params.clustering['route_time_estimation'],
        geo_weight=params.clustering['geo_weight'],
        demand_weight=params.clustering['demand_weight'],
        distance_metric=params.clustering['distance']
    )

    if base_settings.method == 'combine':
        logger.info("🔄 Generating settings variations for 'combine' method")
        # 1. Base methods (kmeans, kmedoids, gmm) - Assume primarily geographical (Geo=1, Dem=0)
        # This reflects original logic where these likely ignored param weights in combine mode
        base_method_names = ['minibatch_kmeans', 'kmedoids', 'gaussian_mixture']
        for name in base_method_names:
            settings_list.append(replace(
                base_settings, # Start from base
                method=name # Set correct method name
            ))

        # 2. Agglomerative with different explicit weights
        weight_combinations = [
            (1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.4, 0.6), (0.2, 0.8), (0.0, 1.0)
        ]
        for geo_w, demand_w in weight_combinations:
            settings_list.append(replace(
                base_settings,
                method='agglomerative', # Set method to agglomerative
                geo_weight=geo_w, # Use specific weight combo
                demand_weight=demand_w # Use specific weight combo
            ))

    else:
        # Single method specified: Use the base_settings as configured initially
        # (which already includes the method name and default weights from params)
        logger.info(f"📍 Using single method configuration: {base_settings.method}")
        settings_list.append(base_settings)

    logger.info(f"Generated {len(settings_list)} distinct clustering settings configurations.")
    return settings_list
