"""
Fleet Size and Mix (FSM) Optimizer module.
Handles the optimization model creation, solving, and solution validation
for the vehicle routing problem with multiple compartments.
"""

import logging
import time
from typing import Dict, Tuple, Set
import pandas as pd
from haversine import haversine
import sys
import gurobipy as gp
from gurobipy import GRB

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
    model, y_vars = _create_optimization_model(
        clusters_df, 
        configurations_df,
        customers_df
    )
    
    # Set output level
    if not verbose:
        model.setParam('OutputFlag', 0)
    
    # Solve the model
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    
    if verbose:
        print(f"Optimization completed in {end_time - start_time:.2f} seconds.")
    
    # Get solver status
    status_map = {
        GRB.OPTIMAL: 'Optimal',
        GRB.INFEASIBLE: 'Infeasible',
        GRB.UNBOUNDED: 'Unbounded',
        GRB.INF_OR_UNBD: 'Infeasible or Unbounded',
        GRB.TIME_LIMIT: 'Time Limit Reached'
    }
    
    solver_status = status_map.get(model.status, f'Other Status: {model.status}')
    
    if model.status == GRB.OPTIMAL:
        selected_clusters = _extract_solution(clusters_df, y_vars)
        missing_customers = _validate_solution(
            selected_clusters, 
            customers_df,
            configurations_df
        )
        
        # Calculate vehicles used by type
        vehicles_used = (
            selected_clusters
            .merge(configurations_df[['Config_ID', 'Vehicle_Type']], on='Config_ID')
            ['Vehicle_Type']
            .value_counts()
        )
        
        # Calculate costs
        solution_stats = _calculate_solution_statistics(selected_clusters, configurations_df)
        
        return {
            'solver_name': 'Gurobi',
            'solver_status': solver_status,
            'selected_clusters': selected_clusters,
            'total_fixed_cost': solution_stats['total_fixed_cost'],
            'total_variable_cost': solution_stats['total_variable_cost'],
            'vehicles_used': vehicles_used,  # Now this is a pandas Series
            'missing_customers': missing_customers
        }
    else:
        raise Exception(f"Optimization failed with status: {solver_status}")

def _create_optimization_model(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame
) -> Tuple[gp.Model, Dict]:
    """Create the optimization model with decision variables and constraints."""
    # Create a new model
    model = gp.Model("FSM-MVC-CD")
    
    # Calculate vehicle-cluster compatibility and costs
    V_k = {}
    cost_matrix = {}
    
    with open('results/compatibility_and_costs.txt', 'w') as f:
        f.write("=== Vehicle-Cluster Compatibility and Costs ===\n\n")
        
        for _, v in configurations_df.iterrows():
            for _, k in clusters_df.iterrows():
                # Check compatibility
                is_compatible = True
                for good in k['Goods_In_Config']:
                    if v[good] != 1:
                        is_compatible = False
                        break
                
                if is_compatible:
                    total_demand = sum(k['Total_Demand'].values())
                    if total_demand > v['Capacity']:
                        is_compatible = False
                
                V_k[(v['Config_ID'], k['Cluster_ID'])] = 1 if is_compatible else 0
                
                # Calculate costs
                fixed_cost = v['Fixed_Cost']
                # Calculate distance-based variable cost
                cluster_coord = (k['Centroid_Latitude'], k['Centroid_Longitude'])
                depot_coord = (DEPOT['Latitude'], DEPOT['Longitude'])
                dist = 2 * haversine(depot_coord, cluster_coord)  # Round trip distance
                variable_cost = dist * VARIABLE_COST_PER_KM  # Use vehicle's variable cost rate
                
                total_cost = fixed_cost + variable_cost
                cost_matrix[(v['Config_ID'], k['Cluster_ID'])] = total_cost
                
                # Write detailed information to file
                f.write(f"\nVehicle Config {v['Config_ID']} - Cluster {k['Cluster_ID']}:\n")
                f.write(f"Compatible: {is_compatible}\n")
                f.write(f"Cluster needs: {k['Goods_In_Config']}\n")
                f.write(f"Vehicle has: Dry={v['Dry']}, Chilled={v['Chilled']}, Frozen={v['Frozen']}\n")
                f.write(f"Costs:\n")
                f.write(f"  Fixed Cost: ${fixed_cost:,.2f}\n")
                f.write(f"  Variable Cost: ${variable_cost:,.2f}\n")
                f.write(f"  Total Cost: ${total_cost:,.2f}\n")
                
                if not is_compatible:
                    f.write("Failed because: ")
                    incompatibility_reasons = []
                    for good in k['Goods_In_Config']:
                        if v[good] != 1:
                            incompatibility_reasons.append(f"Missing {good} compartment")
                    if 'total_demand' in locals() and total_demand > v['Capacity']:
                        incompatibility_reasons.append(f"Demand ({total_demand}) exceeds capacity ({v['Capacity']})")
                    f.write(", ".join(incompatibility_reasons) + "\n")
                f.write("-" * 50 + "\n")
        
        # Add cost matrix summary at the end
        f.write("\n\n=== Cost Matrix Summary ===\n")
        f.write("\nConfig_ID, Cluster_ID, Cost\n")
        for (config_id, cluster_id), cost in sorted(cost_matrix.items()):
            f.write(f"{config_id}, {cluster_id}, ${cost:,.2f}\n")

    # Create decision variables only for compatible vehicle-cluster pairs
    y_vars = {}
    for (config_id, cluster_id), is_compatible in V_k.items():
        if is_compatible == 1:
            y_vars[(config_id, cluster_id)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"y_{config_id}_{cluster_id}",
                obj=cost_matrix[(config_id, cluster_id)]  # Set cost coefficient directly
            )
    
    # Set objective function (minimizing total cost)
    model.setObjective(
        gp.quicksum(
            y_vars[key] * cost_matrix[key]
            for key in y_vars.keys()
        ),
        GRB.MINIMIZE
    )
    
    # Add constraints
    _add_customer_coverage_constraints(
        model, 
        clusters_df, 
        customers_df, 
        y_vars, 
        V_k, 
        configurations_df
    )
    _add_vehicle_allocation_constraints(
        model, 
        clusters_df, 
        configurations_df, 
        y_vars, 
        V_k
    )
    
    return model, y_vars

def _add_customer_coverage_constraints(
    model: gp.Model,
    clusters_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    assign_vars: Dict,
    V_k: Dict,
    configurations_df: pd.DataFrame
) -> None:
    """Add constraints ensuring each customer is visited at least once."""
    
    for _, customer in customers_df.iterrows():
        customer_id = customer['Customer_ID']
        
        # Sum over all compatible vehicle-cluster pairs that can serve this customer
        model.addConstr(
            gp.quicksum(
                assign_vars[v['Config_ID'], k['Cluster_ID']]
                for _, k in clusters_df.iterrows()
                if customer_id in k['Customers']  # Check if cluster can serve this customer
                for _, v in configurations_df.iterrows()
                if V_k[(v['Config_ID'], k['Cluster_ID'])] == 1  # Check if vehicle-cluster pair is compatible
            ) >= 1,  # Changed to >= 1 to match original logic
            name=f"Customer_{customer_id}_Visited_Once"
        )

def _add_vehicle_allocation_constraints(
    model: gp.Model,
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    assign_vars: Dict,
    V_k: Dict
) -> None:
    """Add constraints ensuring each cluster is served by at most one vehicle."""
    
    for _, cluster in clusters_df.iterrows():
        cluster_id = cluster['Cluster_ID']
        
        # Sum over all compatible vehicles that can serve this cluster
        model.addConstr(
            gp.quicksum(
                assign_vars[v['Config_ID'], cluster_id]
                for _, v in configurations_df.iterrows()
                if V_k[(v['Config_ID'], cluster_id)] == 1  # Only consider compatible vehicles
            ) <= 1,  # At most one vehicle per cluster
            name=f"Cluster_{cluster_id}_Served"
        )

def _extract_solution(
    clusters_df: pd.DataFrame,
    y_vars: Dict[Tuple[int, int], gp.Var]
) -> pd.DataFrame:
    """Extract the selected clusters from the optimization solution."""
    return clusters_df[
        clusters_df['Cluster_ID'].isin([
            cid for cid, var in y_vars.items() 
            if var.X > 0.5
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