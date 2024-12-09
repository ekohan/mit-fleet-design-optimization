"""Post-optimization improvement module for the Fleet Size and Mix problem."""

import logging
from typing import Dict, Tuple
import pandas as pd

from utils.route_time import _bhh_estimation
from config.parameters import Parameters
from fsm_optimizer import solve_fsm_problem
from utils.logging import Colors, Symbols

logger = logging.getLogger(__name__)

SMALL_CLUSTER_SIZE = 4  # Only merge clusters with 1-3 customers

def improve_solution(
    initial_solution: Dict,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    params: Parameters
) -> Dict:
    """
    Improve solution by merging small clusters and re-optimizing.
    
    Args:
        initial_solution: Original optimization result
        configurations_df: Vehicle configurations
        customers_df: Customer demands
        params: Parameters object
    
    Returns:
        Improved solution or original if no improvement found
    """
    logger.info(f"\n{Symbols.CHECK} Attempting post-optimization improvements...")
    
    # Get original selected clusters
    selected_clusters = initial_solution.get('selected_clusters', initial_solution.get('clusters'))
    if selected_clusters is None:
        logger.error("Cannot find clusters in solution.")
        return initial_solution
    
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
        return initial_solution
    
    logger.info(f"→ Generated {len(merged_clusters)} merged cluster options")
    
    # Define required columns including goods
    required_columns = [
        'Cluster_ID', 'Config_ID', 'Customers', 'Route_Time', 
        'Total_Demand', 'Method', 'Centroid_Latitude', 'Centroid_Longitude'
    ] + list(params.goods)
    
    # Combine clusters with explicit column ordering
    combined_clusters = pd.concat([
        selected_clusters[required_columns],
        merged_clusters[required_columns]
    ], ignore_index=True)
    
    # Re-run optimization with combined set
    improved_solution = solve_fsm_problem(
        combined_clusters,
        configurations_df,
        customers_df,
        params
    )
    
    return improved_solution if improved_solution['total_cost'] < initial_solution['total_cost'] else initial_solution

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
        small_config = configurations_df[
            configurations_df['Config_ID'] == small_cluster['Config_ID']
        ].iloc[0]
        
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
            target_config = configurations_df[
                configurations_df['Config_ID'] == target_cluster['Config_ID']
            ].iloc[0]
            
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
                merged_customers = customers_df[
                    customers_df['Customer_ID'].isin(
                        target_cluster['Customers'] + small_cluster['Customers']
                    )
                ]
                
                # Calculate new centroid
                centroid_lat = merged_customers['Latitude'].mean()
                centroid_lon = merged_customers['Longitude'].mean()
                
                # Get target configuration - we keep its goods and capacity
                target_config = configurations_df[
                    configurations_df['Config_ID'] == target_cluster['Config_ID']
                ].iloc[0]
                
                # Create new cluster with all required fields
                new_cluster = {
                    'Cluster_ID': f"{target_cluster['Cluster_ID']}_{small_cluster['Cluster_ID']}",
                    'Config_ID': target_cluster['Config_ID'],  # Keep target config
                    'Customers': target_cluster['Customers'] + small_cluster['Customers'],
                    'Route_Time': route_time,
                    'Total_Demand': demands,
                    'Method': f"merged_{target_cluster['Method']}",
                    'Centroid_Latitude': centroid_lat,
                    'Centroid_Longitude': centroid_lon,
                    'Capacity': target_config['Capacity']  # Preserve target capacity
                }
                
                # Add goods configuration from target vehicle
                for good in params.goods:
                    new_cluster[good] = target_config[good]
                
                new_clusters.append(new_cluster)
    
    if not new_clusters:
        return pd.DataFrame()
        
    # Create DataFrame with explicit column ordering
    columns = [
        'Cluster_ID', 'Config_ID', 'Customers', 'Route_Time', 
        'Total_Demand', 'Method', 'Centroid_Latitude', 'Centroid_Longitude'
    ] + list(params.goods)
    
    return pd.DataFrame(new_clusters, columns=columns)

def validate_merged_cluster(
    cluster1: pd.Series,
    cluster2: pd.Series,
    config: pd.Series,
    customers_df: pd.DataFrame,
    params: Parameters
) -> Tuple[bool, float, Dict]:
    """Validate if two clusters can be merged."""
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
    
    merged_customers = customers_df[
        customers_df['Customer_ID'].isin(
            cluster1_customers + cluster2_customers
        )
    ]
    
    # Validate customer locations
    if (merged_customers['Latitude'].isna().any() or 
        merged_customers['Longitude'].isna().any()):
        return False, 0, {}

    # Calculate new route time using BHH
    new_route_time = _bhh_estimation(
        merged_customers,
        params.depot,
        params.service_time,
        params.avg_speed
    )

    if new_route_time > params.max_route_time:
        return False, 0, {}

    return True, new_route_time, merged_goods 