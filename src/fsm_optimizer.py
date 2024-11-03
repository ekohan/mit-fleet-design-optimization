"""
Fleet Size and Mix (FSM) Optimizer module.
Handles the optimization model creation, solving, and solution validation
for the vehicle routing problem with multiple compartments.
"""

import logging
import time
from typing import Dict, Tuple, Set
import pandas as pd
import pulp
from haversine import haversine
import sys

from config import (
    DEPOT,
    GOODS,
    VARIABLE_COST_PER_KM
)

logger = logging.getLogger(__name__)

def solve_fsm_problem(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    verbose: bool = False
) -> Dict:
    """
    Solve the Fleet Size and Mix optimization problem.
    
    Args:
        clusters_df: DataFrame containing generated clusters
        configurations_df: DataFrame containing vehicle configurations
        customers_df: DataFrame containing customer demands
        verbose: Whether to enable verbose output to screen
    
    Returns:
        Dictionary containing optimization results
    """
    # Create optimization model and get variables
    model, y_vars = _create_optimization_model(clusters_df, configurations_df)
    
    # Solve the model
    solver = pulp.GUROBI_CMD(msg=1 if verbose else 0)
    
    start_time = time.time()
    model.solve(solver)
    end_time = time.time()
    
    if verbose:
        print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
    
    # Check solution status
    if model.status != pulp.LpStatusOptimal:
        print(f"Optimization status: {pulp.LpStatus[model.status]}")
        print("The model is infeasible. Please check for customers not included in any cluster or other constraint issues.")
        sys.exit(1)

    # Extract and validate solution
    selected_clusters = _extract_solution(clusters_df, y_vars)
    missing_customers = _validate_solution(
        selected_clusters, 
        customers_df,
        configurations_df
    )
    
    # Calculate statistics
    solution_stats = _calculate_solution_statistics(
        selected_clusters,
        configurations_df
    )
    
    # Print detailed solution if profiling is enabled
    if verbose:
        _print_solution_details(
            selected_clusters,
            configurations_df,
            solution_stats
        )
    
    return {
        'solver_status': pulp.LpStatus[model.status],
        'solver_name': 'GUROBI',
        'selected_clusters': selected_clusters,
        'missing_customers': missing_customers,
        'execution_time': end_time - start_time,
        **solution_stats
    }

def _create_optimization_model(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame
) -> Tuple[pulp.LpProblem, Dict]:
    """
    Create the optimization model with decision variables and constraints.
    """
    model = pulp.LpProblem("FSM-MVC-CD", pulp.LpMinimize)
    
    # Create decision variables
    y_vars = {
        cluster['Cluster_ID']: pulp.LpVariable(
            f"y_{cluster['Cluster_ID']}", 
            cat='Binary'
        )
        for _, cluster in clusters_df.iterrows()
    }
    
    # Objective Function
    total_cost = _build_objective_function(
        clusters_df,
        configurations_df,
        y_vars
    )
    model += total_cost, "Total_Cost"
    
    # Add constraints
    _add_customer_coverage_constraints(model, clusters_df, y_vars)
    
    return model, y_vars

def _build_objective_function(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    y_vars: Dict
) -> pulp.LpAffineExpression:
    """
    Build the objective function for minimizing total cost.
    """
    total_cost = 0
    for _, cluster in clusters_df.iterrows():
        config_id = cluster['Config_ID']
        config = configurations_df[
            configurations_df['Config_ID'] == config_id
        ].iloc[0]
        
        # Calculate costs
        fixed_cost = config['Fixed_Cost']
        cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
        depot_coord = (DEPOT['Latitude'], DEPOT['Longitude'])
        dist = 2 * haversine(depot_coord, cluster_coord)
        variable_cost = dist * VARIABLE_COST_PER_KM
        
        cluster_cost = fixed_cost + variable_cost
        total_cost += cluster_cost * y_vars[cluster['Cluster_ID']]
    
    return total_cost

def _add_customer_coverage_constraints(
    model: pulp.LpProblem,
    clusters_df: pd.DataFrame,
    y_vars: Dict
) -> None:
    """
    Add constraints ensuring each customer is served exactly once.
    """
    # Create customer to cluster mapping
    customer_cluster_map = {}
    for _, cluster in clusters_df.iterrows():
        cluster_id = cluster['Cluster_ID']
        for customer_id in cluster['Customers']:
            if customer_id not in customer_cluster_map:
                customer_cluster_map[customer_id] = []
            customer_cluster_map[customer_id].append(cluster_id)
    
    # Add constraints
    for customer_id, cluster_ids in customer_cluster_map.items():
        model += (
            pulp.lpSum([y_vars[cid] for cid in cluster_ids]) == 1,
            f"Serve_Customer_{customer_id}"
        )

def _extract_solution(
    clusters_df: pd.DataFrame,
    y_vars: Dict
) -> pd.DataFrame:
    """
    Extract the selected clusters from the optimization solution.
    """
    return clusters_df[
        clusters_df['Cluster_ID'].isin([
            cid for cid, var in y_vars.items() 
            if var.varValue and var.varValue > 0.5
        ])
    ]

def _validate_solution(
    selected_clusters: pd.DataFrame,
    customers_df: pd.DataFrame,
    configurations_df: pd.DataFrame
) -> Set:
    """
    Validate that all customers are served in the solution.
    """
    all_customers_set = set(customers_df['Customer_ID'])
    served_customers = set()
    for _, cluster in selected_clusters.iterrows():
        served_customers.update(cluster['Customers'])

    missing_customers = all_customers_set - served_customers
    if missing_customers:
        print(f"\nWARNING: {len(missing_customers)} customers are not served:")
        print(missing_customers)
        # Print unserved customer demands
        unserved = customers_df[customers_df['Customer_ID'].isin(missing_customers)]
        print("\nUnserved customer demands:")
        print(unserved[['Customer_ID', 'Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']])
        
    return missing_customers

def _print_solution_details(
    selected_clusters: pd.DataFrame,
    configurations_df: pd.DataFrame,
    solution_stats: Dict
) -> None:
    """
    Print summarized information about the solution.
    """
    # Warnings first (if any)
    if solution_stats.get('missing_customers'):
        print("\n⚠️  WARNING: Some customers are not served!")
        print(f"Number of unserved customers: {len(solution_stats['missing_customers'])}")
        print("Run with enable_profiling=True to see detailed customer information")
    
    # Cost Summary
    print("\nSolution Summary:")
    print("-" * 50)
    print(f"Total Cost:     ${(solution_stats['total_fixed_cost'] + solution_stats['total_variable_cost']):>10,.2f}")
    print(f"Total Vehicles: {solution_stats['total_vehicles']}")

    # Vehicle Usage Summary
    print("\nVehicles by Type:")
    print(solution_stats['vehicles_used'])

    # Calculate cluster statistics
    cluster_stats = selected_clusters.copy()
    
    # Customer statistics
    customers_per_cluster = cluster_stats['Customers'].apply(len)
    print("\nCustomers per Cluster:")
    print(f"  Min:    {customers_per_cluster.min():>4.0f}")
    print(f"  Max:    {customers_per_cluster.max():>4.0f}")
    print(f"  Avg:    {customers_per_cluster.mean():>4.1f}")
    print(f"  Median: {customers_per_cluster.median():>4.1f}")

    # Calculate truck load percentages
    load_percentages = []
    for _, cluster in cluster_stats.iterrows():
        config = configurations_df[
            configurations_df['Config_ID'] == cluster['Config_ID']
        ].iloc[0]
        # Calculate maximum load percentage across all goods
        max_load_pct = max(
            cluster['Total_Demand'][good] / config['Capacity'] * 100 
            for good in GOODS
        )
        load_percentages.append(max_load_pct)
    
    load_percentages = pd.Series(load_percentages)
    print("\nTruck Load Percentages:")
    print(f"  Min:    {load_percentages.min():>4.1f}%")
    print(f"  Max:    {load_percentages.max():>4.1f}%")
    print(f"  Avg:    {load_percentages.mean():>4.1f}%")
    print(f"  Median: {load_percentages.median():>4.1f}%")

    # Print warnings if any cluster exceeds capacity
    overloaded = load_percentages[load_percentages > 100]
    if not overloaded.empty:
        print(f"\n⚠️  WARNING: {len(overloaded)} clusters exceed vehicle capacity")

def _calculate_solution_statistics(
    selected_clusters: pd.DataFrame,
    configurations_df: pd.DataFrame
) -> Dict:
    """
    Calculate various statistics about the solution.
    """
    # Calculate fixed costs
    total_fixed_cost = selected_clusters.merge(
        configurations_df[["Config_ID", "Fixed_Cost"]], 
        on="Config_ID"
    )["Fixed_Cost"].sum()
    
    # Calculate distances and variable costs
    selected_clusters = selected_clusters.copy()
    selected_clusters['Estimated_Distance'] = selected_clusters.apply(
        _calculate_cluster_distance, axis=1
    )
    
    total_variable_cost = (
        selected_clusters['Estimated_Distance'] * VARIABLE_COST_PER_KM
    ).sum()
    
    # Calculate vehicles used by type
    vehicles_used = (
        selected_clusters.merge(
            configurations_df[["Config_ID", "Vehicle_Type"]], 
            on="Config_ID"
        )["Vehicle_Type"]
        .value_counts()
        .sort_index()
    )
    
    return {
        'total_fixed_cost': total_fixed_cost,
        'total_variable_cost': total_variable_cost,
        'vehicles_used': vehicles_used,
        'total_vehicles': len(selected_clusters),
        'selected_clusters': selected_clusters  # Include updated clusters with distances
    }

def _calculate_cluster_distance(cluster: pd.Series) -> float:
    """
    Calculate the round-trip distance for a cluster.
    """
    cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
    depot_coord = (DEPOT['Latitude'], DEPOT['Longitude'])
    return 2 * haversine(depot_coord, cluster_coord) 