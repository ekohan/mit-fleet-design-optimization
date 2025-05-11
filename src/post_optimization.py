"""Post-optimization improvement module for the Fleet Size and Mix problem."""

import logging
from typing import Dict, Tuple
import pandas as pd

from src.utils.route_time import estimate_route_time
from src.config.parameters import Parameters

from src.utils.logging import Colors, Symbols

logger = logging.getLogger(__name__)

# Cache for merged cluster route times
_merged_route_time_cache: Dict[Tuple[str, ...], float] = {}

def _get_merged_route_time(
    customers: pd.DataFrame,
    params: Parameters
) -> float:
    """
    Estimate (and cache) the route time for a merged cluster of customers.
    Always uses the same method & max_route_time from params.
    """
    key: Tuple[str, ...] = tuple(sorted(customers['Customer_ID']))
    if key in _merged_route_time_cache:
        return _merged_route_time_cache[key]
    
    time, _sequence = estimate_route_time(
        cluster_customers=customers,
        depot=params.depot,
        service_time=params.service_time,
        avg_speed=params.avg_speed,
        method='BHH', # TODO: Remove hardcoded values on route time estimation --- params.route_time_estimation,
        max_route_time=params.max_route_time
    )
    _merged_route_time_cache[key] = time
    return time

SMALL_CLUSTER_SIZE = 7  # Only merge clusters with 1-6 customers
MERGED_CLUSTER_TSP_MSG = "Merged cluster, no TSP computed"  # Placeholder for merged clusters

def improve_solution(
    initial_solution: Dict,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    params: Parameters
) -> Dict:
    """Improve solution by merging small clusters and re-optimizing."""
    # Import here to avoid circular dependency
    from src.fsm_optimizer import solve_fsm_problem
    
    MAX_ITERATIONS = 4
    best_solution = initial_solution
    
    # Add a static counter to track total calls
    if not hasattr(improve_solution, 'total_calls'):
        improve_solution.total_calls = 0
    improve_solution.total_calls += 1
    
    # If we've exceeded max total calls, return immediately
    if improve_solution.total_calls > MAX_ITERATIONS:
        return best_solution
        
    logger.info(f"\n{Symbols.CHECK} Attempting post-optimization improvements (call {improve_solution.total_calls}/{MAX_ITERATIONS})...")
    
    # Get original selected clusters
    selected_clusters = best_solution.get('selected_clusters', best_solution.get('clusters'))
    if selected_clusters is None:
        logger.error("Cannot find clusters in solution.")
        return best_solution
    
    # Ensure goods columns exist in selected_clusters
    for good in params.goods:
        if good not in selected_clusters.columns:
            selected_clusters[good] = selected_clusters['Config_ID'].map(
                lambda x: configurations_df[configurations_df['Config_ID'] == x].iloc[0][good]
            )
    
    # Generate merged versions
    merged_clusters = generate_post_optimization_merges(
        selected_clusters,
        configurations_df,
        customers_df,
        params
    )
    
    if merged_clusters.empty:
        logger.info("→ No valid merged clusters generated")
        return best_solution
    
    logger.info(f"→ Generated {len(merged_clusters)} merged cluster options")
    
    # Get all columns from the master DataFrame (selected_clusters)
    all_columns = selected_clusters.columns.tolist()
    
    # Ensure merged_clusters has all the same columns as selected_clusters
    # This preserves the structure including TSP_Sequence if it exists
    for col in all_columns:
        if col not in merged_clusters.columns:
            if col == 'TSP_Sequence':
                merged_clusters[col] = MERGED_CLUSTER_TSP_MSG
            else:
                merged_clusters[col] = None
    
    # Combine clusters while preserving all columns from the master DataFrame
    combined_clusters = pd.concat([
        selected_clusters,
        merged_clusters[all_columns]  # Ensure same column order
    ], ignore_index=True)
    
    # Re-run optimization with combined set
    improved_solution = solve_fsm_problem(
        combined_clusters,
        configurations_df,
        customers_df,
        params
    )
    
    return improved_solution if improved_solution['total_cost'] < best_solution['total_cost'] else best_solution

def generate_post_optimization_merges(
    selected_clusters: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    params: Parameters
) -> pd.DataFrame:
    """Generate merged clusters from selected small clusters."""
    new_clusters = []
    stats = {
        'attempted': 0, 
        'valid': 0, 
        'invalid_time': 0, 
        'invalid_capacity': 0,
        'invalid_compatibility': 0
    }
    
    # Create an indexed DataFrame for efficient configuration lookups
    configs_indexed = configurations_df.set_index('Config_ID')
    # Index customers for fast lookup
    customers_indexed = customers_df.set_index('Customer_ID')
    
    # Get all small clusters
    small_clusters = selected_clusters[
        selected_clusters['Customers'].apply(len) <= SMALL_CLUSTER_SIZE
    ].copy()
    
    # Get potential target clusters (all clusters)
    target_clusters = selected_clusters.copy()
    
    if len(small_clusters) == 0:
        return pd.DataFrame()
        
    logger.info(f"→ Found {len(small_clusters)} small clusters")
    
    # Try merging small clusters with potential targets
    for _, small_cluster in small_clusters.iterrows():
        small_config = configs_indexed.loc[small_cluster['Config_ID']]
        
        # Get goods used by small cluster
        small_goods = {
            g: small_cluster['Total_Demand'][g] 
            for g in params.goods 
            if small_cluster['Total_Demand'][g] > 0
        }
        
        for _, target_cluster in target_clusters.iterrows():
            if target_cluster['Cluster_ID'] == small_cluster['Cluster_ID']:
                continue
                
            stats['attempted'] += 1
            
            # Get target configuration
            target_config = configs_indexed.loc[target_cluster['Config_ID']]
            
            # Check capacity compatibility
            if target_config['Capacity'] < small_config['Capacity']:
                stats['invalid_compatibility'] += 1
                continue
                
            # Check if target config can handle small cluster's goods
            if not all(target_config[g] == 1 for g in small_goods.keys()):
                stats['invalid_compatibility'] += 1
                continue
            
            is_valid, route_time, demands = validate_merged_cluster(
                small_cluster,
                target_cluster,
                target_config,
                customers_df,
                params
            )
            
            if is_valid:
                stats['valid'] += 1
                
                # Get merged customers data for centroid calculation
                merged_customers = customers_indexed.loc[
                    target_cluster['Customers'] + small_cluster['Customers']
                ].reset_index()
                
                # Calculate new centroid
                centroid_lat = merged_customers['Latitude'].mean()
                centroid_lon = merged_customers['Longitude'].mean()
                
                # Create new cluster with core required fields
                new_cluster = {
                    'Cluster_ID': f"{target_cluster['Cluster_ID']}_{small_cluster['Cluster_ID']}",
                    'Config_ID': target_cluster['Config_ID'],  # Keep target config
                    'Customers': target_cluster['Customers'] + small_cluster['Customers'],
                    'Route_Time': route_time,
                    'Total_Demand': demands,
                    'Method': f"merged_{target_cluster['Method']}",
                    'Centroid_Latitude': centroid_lat,
                    'Centroid_Longitude': centroid_lon,
                }
                
                # Always set TSP_Sequence for merged clusters to our placeholder
                new_cluster['TSP_Sequence'] = MERGED_CLUSTER_TSP_MSG
                
                # Add goods configuration from target vehicle
                for good in params.goods:
                    new_cluster[good] = target_config[good]
                
                new_clusters.append(new_cluster)
    
    if not new_clusters:
        return pd.DataFrame()
        
    # Create barebones DataFrame with the minimal required columns
    # The improve_solution function will handle adding any missing columns
    minimal_columns = [
        'Cluster_ID', 'Config_ID', 'Customers', 'Route_Time', 
        'Total_Demand', 'Method', 'Centroid_Latitude', 'Centroid_Longitude',
        'TSP_Sequence'  # Always include TSP_Sequence with our placeholder
    ] + list(params.goods)
    
    return pd.DataFrame(new_clusters, columns=minimal_columns)

def validate_merged_cluster(
    cluster1: pd.Series,
    cluster2: pd.Series,
    config: pd.Series,
    customers_df: pd.DataFrame,
    params: Parameters
) -> Tuple[bool, float, Dict]:
    """Validate if two clusters can be merged."""
    # Index customers for fast lookup
    if customers_df.index.name != 'Customer_ID':
        customers_indexed = customers_df.set_index('Customer_ID', drop=False)
    else:
        customers_indexed = customers_df
    # Check compartment compatibility
    merged_goods = {}
    for g in params.goods:
        # Handle case where Total_Demand might be a dict or series
        demand1 = cluster1['Total_Demand'][g] if isinstance(cluster1['Total_Demand'], (dict, pd.Series)) else cluster1[g]
        demand2 = cluster2['Total_Demand'][g] if isinstance(cluster2['Total_Demand'], (dict, pd.Series)) else cluster2[g]
        merged_goods[g] = demand1 + demand2
    
    # Validate capacity
    if any(demand > config['Capacity'] for demand in merged_goods.values()):
        return False, 0, {}

    # Get all customers from both clusters
    cluster1_customers = cluster1['Customers'] if isinstance(cluster1['Customers'], list) else [cluster1['Customers']]
    cluster2_customers = cluster2['Customers'] if isinstance(cluster2['Customers'], list) else [cluster2['Customers']]
    
    merged_customers = customers_indexed.loc[
        cluster1_customers + cluster2_customers
    ]
    
    # Validate customer locations
    if (merged_customers['Latitude'].isna().any() or 
        merged_customers['Longitude'].isna().any()):
        return False, 0, {}

    # Estimate (and cache) new route time using the general estimator
    new_route_time = _get_merged_route_time(merged_customers, params)

    if new_route_time > params.max_route_time:
        return False, 0, {}

    return True, new_route_time, merged_goods 