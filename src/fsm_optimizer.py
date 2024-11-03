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

from utils.logging import Colors, Symbols

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
    logger = logging.getLogger(__name__)
    from utils.logging import Colors, Symbols

    all_customers_set = set(customers_df['Customer_ID'])
    served_customers = set()
    for _, cluster in selected_clusters.iterrows():
        served_customers.update(cluster['Customers'])

    missing_customers = all_customers_set - served_customers
    if missing_customers:
        logger.warning(
            f"\n{Symbols.CROSS} {len(missing_customers)} customers are not served!"
        )
        
        # Print unserved customer demands
        unserved = customers_df[customers_df['Customer_ID'].isin(missing_customers)]
        logger.warning(
            f"{Colors.YELLOW}→ Unserved Customers:{Colors.RESET}\n"
            f"{Colors.GRAY}  Customer ID  Dry  Chilled  Frozen{Colors.RESET}"
        )
        
        for _, customer in unserved.iterrows():
            logger.warning(
                f"{Colors.YELLOW}  {customer['Customer_ID']:>10}  "
                f"{customer['Dry_Demand']:>3.0f}  "
                f"{customer['Chilled_Demand']:>7.0f}  "
                f"{customer['Frozen_Demand']:>6.0f}{Colors.RESET}"
            )
        
    return missing_customers

def _print_solution_details(
    selected_clusters: pd.DataFrame,
    configurations_df: pd.DataFrame,
    solution_stats: Dict
) -> None:
    """Print summarized information about the solution."""
    logger = logging.getLogger(__name__)
    from utils.logging import Colors, Symbols
    
    # Warnings first (if any)
    if solution_stats.get('missing_customers'):
        logger.warning(
            f"{Symbols.CROSS} Some customers are not served!\n"
            f"{Colors.YELLOW}→ Unserved customers: {len(solution_stats['missing_customers'])}{Colors.RESET}"
        )
    
    # Cost Summary
    logger.info(f"\n{Symbols.CHART} Solution Summary")
    logger.info("=" * 50)
    logger.info(
        f"{Colors.CYAN}Total Cost:     ${Colors.BOLD}"
        f"{(solution_stats['total_fixed_cost'] + solution_stats['total_variable_cost']):>10,.2f}"
        f"{Colors.RESET}"
    )
    logger.info(
        f"{Colors.CYAN}Total Vehicles: {Colors.BOLD}"
        f"{solution_stats['total_vehicles']}{Colors.RESET}"
    )

    # Vehicle Usage Summary
    logger.info(f"\n{Symbols.TRUCK} Vehicles by Type")
    for vehicle_type, count in solution_stats['vehicles_used'].items():
        logger.info(
            f"{Colors.BLUE}→ Type {vehicle_type}:{Colors.BOLD}"
            f"{count:>4}{Colors.RESET}"
        )

    # Calculate cluster statistics
    cluster_stats = selected_clusters.copy()
    
    # Customer statistics
    customers_per_cluster = cluster_stats['Customers'].apply(len)
    logger.info(f"\n{Symbols.PACKAGE} Customers per Cluster")
    logger.info(
        f"{Colors.MAGENTA}  Min:    {Colors.BOLD}{customers_per_cluster.min():>4.0f}{Colors.RESET}\n"
        f"{Colors.MAGENTA}  Max:    {Colors.BOLD}{customers_per_cluster.max():>4.0f}{Colors.RESET}\n"
        f"{Colors.MAGENTA}  Avg:    {Colors.BOLD}{customers_per_cluster.mean():>4.1f}{Colors.RESET}\n"
        f"{Colors.MAGENTA}  Median: {Colors.BOLD}{customers_per_cluster.median():>4.1f}{Colors.RESET}"
    )

    # Calculate truck load percentages
    load_percentages = []
    for _, cluster in cluster_stats.iterrows():
        config = configurations_df[
            configurations_df['Config_ID'] == cluster['Config_ID']
        ].iloc[0]
        max_load_pct = max(
            cluster['Total_Demand'][good] / config['Capacity'] * 100 
            for good in GOODS
        )
        load_percentages.append(max_load_pct)
    
    load_percentages = pd.Series(load_percentages)
    logger.info(f"\n{Symbols.GEAR} Truck Load Percentages")
    logger.info(
        f"{Colors.CYAN}  Min:    {Colors.BOLD}{load_percentages.min():>4.1f}%{Colors.RESET}\n"
        f"{Colors.CYAN}  Max:    {Colors.BOLD}{load_percentages.max():>4.1f}%{Colors.RESET}\n"
        f"{Colors.CYAN}  Avg:    {Colors.BOLD}{load_percentages.mean():>4.1f}%{Colors.RESET}\n"
        f"{Colors.CYAN}  Median: {Colors.BOLD}{load_percentages.median():>4.1f}%{Colors.RESET}"
    )

    # Print warnings if any cluster exceeds capacity
    overloaded = load_percentages[load_percentages > 100]
    if not overloaded.empty:
        logger.warning(
            f"\n{Symbols.CROSS} {len(overloaded)} clusters exceed vehicle capacity!\n"
            f"{Colors.YELLOW}→ Maximum overload: {Colors.BOLD}"
            f"{overloaded.max():.1f}%{Colors.RESET}"
        )

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