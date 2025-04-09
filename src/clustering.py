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
from dataclasses import dataclass, replace
from sklearn.mixture import GaussianMixture
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
            for cluster in config_clusters:
                # Assign unique Cluster_ID
                cluster.cluster_id = next(cluster_id_generator)
                all_clusters.append(cluster)
        logger.info(f"--- Configuration {settings_for_run.method} completed, generated {len([c for config_clusters in clusters_by_config for c in config_clusters])} raw clusters ---")

    if not all_clusters:
        logger.warning("No clusters were generated by any configuration.")
        return pd.DataFrame()

    # Convert to dictionary for DataFrame
    all_clusters_dicts = [cluster.to_dict() for cluster in all_clusters]
    
    # Remove duplicate clusters based on customer sets
    logger.info(f"Combining and deduplicating {len(all_clusters_dicts)} raw clusters from all configurations...")
    combined_clusters_df = pd.DataFrame(all_clusters_dicts)
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
) -> List[Cluster]:
    """Process a single vehicle configuration to generate feasible clusters."""
    # 1. Get customers that can be served by the configuration
    customers_subset = get_feasible_customers_subset(customers, feasible_customers, config['Config_ID'])
    if customers_subset.empty:
        return []
    
    # 2. Create initial clusters based on size
    labeled_customers = create_initial_clusters(customers_subset, config, settings)
    
    # 3. Process clusters recursively until constraints are satisfied
    return process_clusters_recursively(labeled_customers, config, settings)

def get_feasible_customers_subset(
    customers: pd.DataFrame, 
    feasible_customers: Dict, 
    config_id: int
) -> pd.DataFrame:
    """Extract feasible customers for a given configuration."""
    return customers[
        customers['Customer_ID'].isin([
            cid for cid, configs in feasible_customers.items() 
            if config_id in configs
        ])
    ].copy()

def add_demand_information(
    customers_subset: pd.DataFrame, 
    goods: List[str]
) -> pd.DataFrame:
    """Add total demand information to the customer dataframe."""
    customers_copy = customers_subset.copy()
    demands = compute_demands(customers_copy, goods)
    customers_copy['Total_Demand'] = demands['total']
    return customers_copy

def create_initial_clusters(
    customers_subset: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings
) -> pd.DataFrame:
    """Create initial clusters based on dataset size and settings."""
    # Add total demand
    customers_with_demand = add_demand_information(customers_subset, settings.goods)
    
    if len(customers_with_demand) <= 2:
        return create_small_dataset_clusters(customers_with_demand)
    else:
        return create_normal_dataset_clusters(customers_with_demand, config, settings)

def create_small_dataset_clusters(customers_subset: pd.DataFrame) -> pd.DataFrame:
    """Create clusters for small datasets (≤2 customers)."""
    customers_copy = customers_subset.copy()
    data = customers_copy[['Latitude', 'Longitude']].values
    model = MiniBatchKMeans(n_clusters=1, random_state=42)
    customers_copy['Cluster'] = model.fit_predict(data)
    return customers_copy

def create_normal_dataset_clusters(
    customers_subset: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings
) -> pd.DataFrame:
    """Create clusters for normal-sized datasets."""
    customers_copy = customers_subset.copy()
    
    # Determine number of clusters
    num_clusters = estimate_num_initial_clusters(
        customers_copy,
        config,
        settings
    )
    
    # Get input data and cluster using weights from settings
    data = compute_cluster_metric_input(
        customers_copy,
        settings
    )
    
    # Ensure the number of clusters doesn't exceed the number of customers
    num_clusters = min(num_clusters, len(customers_copy))
    
    model = get_clustering_model(num_clusters, settings.method)
    customers_copy['Cluster'] = model.fit_predict(data)
    return customers_copy

def generate_cluster_id_base(config_id: int) -> int:
    """Generate a base cluster ID from the configuration ID."""
    return int(str(config_id) + "000")

def check_constraints(
    cluster_customers: pd.DataFrame,
    config: pd.Series,
    settings: ClusteringSettings
) -> tuple[bool, bool]:
    """
    Check if cluster violates capacity or time constraints.
    
    Returns:
        tuple: (capacity_violated, time_violated)
    """
    # Ensure Total_Demand is computed
    if 'Total_Demand' not in cluster_customers.columns:
        cluster_customers = add_demand_information(cluster_customers, settings.goods)
    
    cluster_demand = cluster_customers['Total_Demand'].sum()
    route_time = estimate_route_time(
        cluster_customers=cluster_customers,
        depot=settings.depot,
        service_time=settings.service_time,
        avg_speed=settings.avg_speed,
        method=settings.route_time_estimation
    )
    
    capacity_violated = cluster_demand > config['Capacity']
    time_violated = route_time > settings.max_route_time
    
    return capacity_violated, time_violated

def should_split_cluster(
    cluster_customers: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings, 
    depth: int
) -> bool:
    """Determine if a cluster should be split based on constraints."""
    capacity_violated, time_violated = check_constraints(cluster_customers, config, settings)
    is_singleton_cluster = len(cluster_customers) <= 1
    
    # Log warning for single-customer constraints
    if (capacity_violated or time_violated) and is_singleton_cluster:
        logger.warning(f"⚠️ Can't split further (singleton cluster) but constraints still violated: "
                     f"capacity={capacity_violated}, time={time_violated}")
    
    # Return True if we need to split (constraints violated and we can split)
    return (capacity_violated or time_violated) and not is_singleton_cluster

def split_cluster(
    cluster_customers: pd.DataFrame, 
    settings: ClusteringSettings
) -> List[pd.DataFrame]:
    """Split an oversized cluster into smaller ones."""
    # Prepare data for splitting
    split_data = compute_cluster_metric_input(
        cluster_customers,
        settings
    )
    
    # Split into two clusters
    cluster_model = get_clustering_model(2, settings.method)
    sub_labels = cluster_model.fit_predict(split_data)
    
    # Create sub-clusters
    sub_clusters = []
    sub_cluster_sizes = []
    for label in [0, 1]:
        sub_cluster = cluster_customers[sub_labels == label]
        if not sub_cluster.empty:
            sub_clusters.append(sub_cluster)
            sub_cluster_sizes.append(len(sub_cluster))
    
    logger.debug(f"Split cluster of size {len(cluster_customers)} into {len(sub_clusters)} "
                f"sub-clusters of sizes {sub_cluster_sizes}")
    
    return sub_clusters

def create_cluster(
    cluster_customers: pd.DataFrame, 
    config: pd.Series, 
    cluster_id: int, 
    settings: ClusteringSettings
) -> Cluster:
    """Create a Cluster object from customer data."""
    cluster = Cluster.from_customers(
        cluster_customers,
        config,
        cluster_id,
        settings,
        settings.method
    )
    return cluster

def process_clusters_recursively(
    labeled_customers: pd.DataFrame, 
    config: pd.Series, 
    settings: ClusteringSettings
) -> List[Cluster]:
    """Process clusters recursively to ensure constraints are satisfied."""
    config_id = config['Config_ID']
    cluster_id_base = generate_cluster_id_base(config_id)
    current_cluster_id = 0
    clusters = [] 
    
    # Process clusters until all constraints are satisfied
    clusters_to_check = [
        (labeled_customers[labeled_customers['Cluster'] == c], 0)
        for c in labeled_customers['Cluster'].unique()
    ]
    
    logger.info(f"Starting recursive processing for config {config_id} with {len(clusters_to_check)} initial clusters")
    
    split_count = 0
    skipped_count = 0
    
    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()
        max_depth_reached = depth >= settings.max_depth
        
        # Check if max depth reached
        if max_depth_reached:
            # Check if constraints violated
            capacity_violated, time_violated = check_constraints(cluster_customers, config, settings)
            
            if capacity_violated or time_violated:
                logger.warning(f"⚠️ Max depth {settings.max_depth} reached but constraints still violated: "
                              f"capacity={capacity_violated}, time={time_violated}, "
                              f"method={settings.method}, config_id={config['Config_ID']}")
                skipped_count += 1
                continue  # Skip this cluster
        
        # Not at max depth, check if we should split
        if not max_depth_reached and should_split_cluster(cluster_customers, config, settings, depth):
            split_count += 1
            logger.debug(f"Splitting cluster for config {config_id} (size {len(cluster_customers)}) at depth {depth}/{settings.max_depth}")
            # Split oversized clusters
            for sub_cluster in split_cluster(cluster_customers, settings):
                clusters_to_check.append((sub_cluster, depth + 1))
        else:
            # Add valid cluster (either constraints satisfied or could not be split further)
            current_cluster_id += 1
            cluster = create_cluster(
                cluster_customers, 
                config,
                cluster_id_base + current_cluster_id, 
                settings
            )
            clusters.append(cluster)
    
    if skipped_count > 0:
        logger.warning(f"⚠️ Skipped {skipped_count} clusters that exceeded capacity at max depth for config {config_id}.")
    
    logger.info(f"Completed recursive processing for config {config_id}: {len(clusters)} final clusters, "
               f"{split_count} splits performed")
    
    return clusters

def validate_cluster_coverage(clusters_df, customers_df):
    """Validate that all customers are covered by at least one cluster."""
    customer_coverage = {cid: False for cid in customers_df['Customer_ID']}
    for customers in clusters_df['Customers']:
        for cid in customers:
            customer_coverage[cid] = True
    uncovered = [cid for cid, covered in customer_coverage.items() if not covered]
    
    if uncovered:
        logger.warning(f"Found {len(uncovered)} customers not covered by any cluster: {uncovered[:5]}...")
    else:
        logger.info(f"{Symbols.CHECKMARK} All {len(customer_coverage)} customers are covered by at least one cluster.")

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
