"""Post-optimization improvement module for the Fleet Size and Mix problem."""

import logging
from typing import Dict, Tuple
import pandas as pd
import numpy as np
from haversine import haversine_vector, Unit

from src.utils.route_time import estimate_route_time, calculate_total_service_time_hours
from src.config.parameters import Parameters

from src.utils.logging import Colors, Symbols

logger = logging.getLogger(__name__)

# Cache for merged cluster route times
_merged_route_time_cache: Dict[Tuple[str, ...], Tuple[float, list | None]] = {}

# Proximity-based filtering constants and utilities

def _get_merged_route_time(
    customers: pd.DataFrame,
    params: Parameters
) -> Tuple[float, list | None]:
    """
    Estimate (and cache) the route time and sequence for a merged cluster of customers.
    Always uses the same method & max_route_time from params.
    """
    key: Tuple[str, ...] = tuple(sorted(customers['Customer_ID']))
    if key in _merged_route_time_cache:
        return _merged_route_time_cache[key]
    
    time, sequence = estimate_route_time(
        cluster_customers=customers,
        depot=params.depot,
        service_time=params.service_time,
        avg_speed=params.avg_speed,
        method=params.clustering['route_time_estimation'],
        max_route_time=params.max_route_time
    )
    _merged_route_time_cache[key] = (time, sequence)
    return time, sequence

SMALL_CLUSTER_SIZE = 7  # Only merge clusters with 1-6 customers
NEAREST_MERGE_CANDIDATES = 10  # Only consider the 10 nearest clusters
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
        merged_clusters  # Allow merged_clusters to contribute all its columns
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
    
    # Start from selected_clusters (which already has goods columns) and add capacity
    cluster_meta = selected_clusters.copy()
    cluster_meta['Capacity'] = cluster_meta['Config_ID'].map(configs_indexed['Capacity'])
    small_meta = cluster_meta[
        cluster_meta['Customers'].apply(len) <= SMALL_CLUSTER_SIZE
    ]
    if small_meta.empty:
        return pd.DataFrame()
    target_meta = cluster_meta

    logger.info(f"→ Found {len(small_meta)} small clusters")

    # Precompute numpy arrays for vectorized capacity & goods checks
    goods_arr = target_meta[params.goods].to_numpy()
    cap_arr   = target_meta['Capacity'].to_numpy()
    ids       = target_meta['Cluster_ID'].to_numpy()
    lat_arr   = target_meta['Centroid_Latitude'].to_numpy()
    lon_arr   = target_meta['Centroid_Longitude'].to_numpy()

    # Vectorized filtering loop
    for _, small in small_meta.iterrows():
        sd = np.array([small['Total_Demand'][g] for g in params.goods])
        peak = sd.max()
        needs = sd > 0
        if needs.any():
            goods_ok = (goods_arr[:, needs] == 1).all(axis=1)
        else:
            goods_ok = np.ones_like(cap_arr, dtype=bool)
        cap_ok   = cap_arr >= peak
        not_self = ids != small['Cluster_ID']

        # Proximity-based filtering: compute distances and pick nearest candidates
        small_point = (small['Centroid_Latitude'], small['Centroid_Longitude'])
        target_points = np.column_stack((lat_arr, lon_arr))
        
        distances = haversine_vector(small_point, target_points, unit=Unit.KILOMETERS, comb=True)
        distances = distances.flatten() # Ensure distances is a 1D array
        
        valid_mask = cap_ok & goods_ok & not_self & ~np.isnan(distances)
        valid_idxs = np.where(valid_mask)[0]
        if valid_idxs.size == 0:
            continue
        nearest_idxs = valid_idxs[np.argsort(distances[valid_idxs])[:NEAREST_MERGE_CANDIDATES]]
        for idx in nearest_idxs:
            # Quick lower-bound time prune before costly route-time estimation
            target = target_meta.iloc[idx]
            rt_target = target['Route_Time']
            rt_small  = small['Route_Time']

            # Compute service time for the cluster not contributing the max route_time (avoid double count)
            if rt_small > rt_target:
                svc_time_other = calculate_total_service_time_hours(len(target['Customers']), params.service_time)
            else:
                svc_time_other = calculate_total_service_time_hours(len(small['Customers']), params.service_time)

            # Quick lower-bound time prune before costly route-time estimation (no proximity term)
            lb = max(rt_target, rt_small) + svc_time_other
            if lb > params.max_route_time:
                stats['invalid_time'] = stats.get('invalid_time', 0) + 1
                logger.debug(f"Lower-bound prune: merge {small['Cluster_ID']} + {target['Cluster_ID']} lb={lb:.2f} > max={params.max_route_time:.2f}")
                continue
            stats['attempted'] += 1
            target_config = configs_indexed.loc[target['Config_ID']]
            is_valid, route_time, demands, tsp_sequence = validate_merged_cluster(
                small, target, target_config, customers_indexed, params
            )
            if not is_valid:
                # Assuming validate_merged_cluster now logs reasons for invalidity if needed
                # or updates stats for invalid_capacity, invalid_compatibility
                continue
            stats['valid'] += 1

            # Build merged cluster
            merged_customer_ids = target['Customers'] + small['Customers']
            # Ensure merged_customers are fetched correctly for centroid calculation
            # It's crucial that customers_indexed contains all relevant customers.
            # If validate_merged_cluster already did this, we might optimize, but safety first.
            current_merged_customers_df = customers_indexed.loc[merged_customer_ids].reset_index()

            centroid_lat = current_merged_customers_df['Latitude'].mean()
            centroid_lon = current_merged_customers_df['Longitude'].mean()
            new_cluster = {
                'Cluster_ID': f"{target['Cluster_ID']}_{small['Cluster_ID']}",
                'Config_ID': target['Config_ID'],
                'Customers': merged_customer_ids,
                'Route_Time': route_time,
                'Total_Demand': demands,
                'Method': f"merged_{target['Method']}"
            ,
                'Centroid_Latitude': centroid_lat,
                'Centroid_Longitude': centroid_lon
            }
            new_cluster['TSP_Sequence'] = tsp_sequence if tsp_sequence is not None else MERGED_CLUSTER_TSP_MSG
            for good in params.goods:
                new_cluster[good] = target_config[good]
            new_clusters.append(new_cluster)
    
    # Log prune statistics before returning
    logger.info(
        f"→ Merge prune stats: attempted={stats['attempted']}, "
        f"invalid_time={stats['invalid_time']}, "
        f"invalid_capacity={stats['invalid_capacity']}, "
        f"invalid_compatibility={stats['invalid_compatibility']}, "
        f"valid={stats['valid']}"
    )
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
) -> Tuple[bool, float, Dict, list | None]:
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
        return False, 0, {}, None

    # Get all customers from both clusters
    cluster1_customers = cluster1['Customers'] if isinstance(cluster1['Customers'], list) else [cluster1['Customers']]
    cluster2_customers = cluster2['Customers'] if isinstance(cluster2['Customers'], list) else [cluster2['Customers']]
    
    merged_customers_ids = cluster1_customers + cluster2_customers
    # Validate that all customer IDs are present in customers_indexed
    # Check if customers_indexed has 'Customer_ID' as its index
    if customers_indexed.index.name != 'Customer_ID':
        # This case should ideally not happen if indexing is consistent
        logger.error("customers_indexed is not indexed by 'Customer_ID' in validate_merged_cluster.")
        # Fallback or raise error, for now, assume it's an issue and return invalid
        return False, 0, {}, None

    missing_ids = [cid for cid in merged_customers_ids if cid not in customers_indexed.index]
    if missing_ids:
        logger.warning(f"Missing customer IDs {missing_ids} during merge validation for potential merge of clusters involving {cluster1.get('Cluster_ID', 'Unknown')} and {cluster2.get('Cluster_ID', 'Unknown')}.")
        return False, 0, {}, None

    merged_customers = customers_indexed.loc[
        merged_customers_ids
    ].reset_index()
    
    # Validate customer locations
    if (merged_customers['Latitude'].isna().any() or 
        merged_customers['Longitude'].isna().any()):
        return False, 0, {}, None

    # Estimate (and cache) new route time using the general estimator
    new_route_time, new_sequence = _get_merged_route_time(merged_customers, params)

    if new_route_time > params.max_route_time:
        return False, 0, {}, None

    return True, new_route_time, merged_goods, new_sequence 