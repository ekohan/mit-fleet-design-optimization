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

from utils.logging import Colors, Symbols
from config.parameters import Parameters

logger = logging.getLogger(__name__)

def solve_fsm_problem(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    parameters: Parameters,
    verbose: bool = False
) -> Dict:
    """
    Solve the Fleet Size and Mix optimization problem.
    
    Args:
        clusters_df: DataFrame containing generated clusters
        configurations_df: DataFrame containing vehicle configurations
        customers_df: DataFrame containing customer demands
        parameters: Parameters object containing optimization parameters
        verbose: Whether to enable verbose output to screen
    
    Returns:
        Dictionary containing optimization results
    """
    # Create optimization model
    model, y_vars = _create_model(clusters_df, configurations_df, parameters)
    
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
    
    # Verbose output for selected clusters
    if verbose:
        print("Selected Clusters and their Vehicle Configurations:")
        print(selected_clusters[['Cluster_ID', 'Config_ID', 'Total_Demand']])  # Adjust columns as needed

    missing_customers = _validate_solution(
        selected_clusters, 
        customers_df,
        configurations_df
    )
    
    # Calculate statistics
    solution_stats = _calculate_solution_statistics(
        selected_clusters,
        configurations_df,
        parameters
    )
    
    # Print detailed solution if profiling is enabled
    if verbose:
        print("\nSolver Results:")
        print(f"Solver Status: {pulp.LpStatus[model.status]}")
        print(f"Total Fixed Cost: ${solution_stats['total_fixed_cost']:,.2f}")
        print(f"Total Variable Cost: ${solution_stats['total_variable_cost']:,.2f}")
        print(f"Total Cost: ${solution_stats['total_fixed_cost'] + solution_stats['total_variable_cost']:,.2f}")
        print(f"Total Vehicles Used: {solution_stats['total_vehicles']}")
        print(f"Missing Customers: {len(missing_customers)}")
        if missing_customers:
            print(f"Unserved Customers: {', '.join(map(str, missing_customers))}")

    # Create a DataFrame for decision variables
    decision_vars = pd.DataFrame({
        'Variable': [var.name for var in model.variables()],
        'Value': [var.varValue for var in model.variables()]
    })

    # Save decision variables to a CSV file
    decision_vars.to_csv('decision_variables.csv', index=False)

    return {
        'solver_status': pulp.LpStatus[model.status],
        'solver_name': 'GUROBI',
        'selected_clusters': selected_clusters,
        'missing_customers': missing_customers,
        'execution_time': end_time - start_time,
        **solution_stats
    }

def _create_model(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    parameters: Parameters
) -> Tuple[pulp.LpProblem, Dict]:
    """
    Create the optimization model (Model 2) aligning with the mathematical formulation,
    with penalization for lightly loaded trucks below a specified threshold.
    """
    import pulp

    # Create the optimization model
    model = pulp.LpProblem("FSM-MCV_Model2", pulp.LpMinimize)

    # Sets
    N = set(clusters_df['Customers'].explode().unique())  # Customers
    K = set(clusters_df['Cluster_ID'])  # Clusters
    V = set(configurations_df['Config_ID'])  # Vehicle configurations

    # K_i: clusters containing customer i
    K_i = {
        i: set(clusters_df[clusters_df['Customers'].apply(lambda x: i in x)]['Cluster_ID'])
        for i in N
    }

    # V_k: vehicle configurations that can serve cluster k
    V_k = {}
    for k in K:
        V_k[k] = set()
        cluster = clusters_df.loc[clusters_df['Cluster_ID'] == k].iloc[0]
        cluster_goods_required = set(g for g in parameters.goods if cluster['Total_Demand'][g] > 0)
        q_k = sum(cluster['Total_Demand'].values())

        for _, config in configurations_df.iterrows():
            v = config['Config_ID']
            # Check capacity
            if q_k > config['Capacity']:
                continue  # Vehicle cannot serve this cluster

            # Check product compatibility
            compatible = all(
                config[g] == 1 for g in cluster_goods_required
            )

            if compatible:
                V_k[k].add(v)

        # If V_k[k] is empty, handle accordingly
        if not V_k[k]:
            logger.warning(f"Cluster {k} cannot be served by any vehicle configuration.")
            # Force y_k to 0 (cluster cannot be selected)
            V_k[k].add('NoVehicle')  # Placeholder
            x_vars['NoVehicle', k] = pulp.LpVariable(f"x_NoVehicle_{k}", cat='Binary')
            model += x_vars['NoVehicle', k] == 0
            c_vk['NoVehicle', k] = 0  # Cost is zero as it's not selected

    # Decision Variables
    x_vars = {}
    y_vars = {}
    c_vk = pd.DataFrame(index=list(K), columns=list(V))  # Initialize c_vk as a DataFrame

    for k in K:
        y_vars[k] = pulp.LpVariable(f"y_{k}", cat='Binary')
        for v in V_k[k]:
            x_vars[v, k] = pulp.LpVariable(f"x_{v}_{k}", cat='Binary')

    # Parameters
    for k in K:
        cluster = clusters_df.loc[clusters_df['Cluster_ID'] == k].iloc[0]
        for v in V_k[k]:
            if v != 'NoVehicle':
                config = configurations_df.loc[configurations_df['Config_ID'] == v].iloc[0]
                # Calculate the base cost
                base_cost = _calculate_cluster_cost(
                    cluster=cluster,
                    config=config,
                    parameters=parameters
                )

                # Calculate load percentage
                total_demand = sum(cluster['Total_Demand'][g] for g in parameters.goods)
                capacity = config['Capacity']
                load_percentage = total_demand / capacity

                # Penalize if load percentage is less than threshold
                if load_percentage < parameters.light_load_threshold:
                    penalty_amount = parameters.light_load_penalty * (parameters.light_load_threshold - load_percentage)
                    c_vk.at[k, v] = base_cost + penalty_amount
                else:
                    c_vk.at[k, v] = base_cost
            else:
                c_vk.at[k, v] = 0  # Cost is zero for placeholder

    # Objective Function
    model += pulp.lpSum(
        c_vk.at[k, v] * x_vars[v, k]
        for k in K for v in V_k[k]
    ), "Total_Cost"

    # Constraints

    # 1. Customer Allocation Constraint
    for i in N:
        model += pulp.lpSum(
            x_vars[v, k]
            for k in K_i[i]
            for v in V_k[k]
            if v != 'NoVehicle'
        ) >= 1, f"Customer_Coverage_{i}"

    # 2. Vehicle Configuration Assignment Constraint
    for k in K:
        model += (
            pulp.lpSum(x_vars[v, k] for v in V_k[k]) == y_vars[k]
        ), f"Vehicle_Assignment_{k}"

    # 3. Unserviceable Clusters Constraint
    for k in K:
        if 'NoVehicle' in V_k[k]:
            model += y_vars[k] == 0, f"Unserviceable_Cluster_{k}"

    # Example of adding a diversity constraint
    max_allowed_same_configurations = 99

    for k in K:
        model += pulp.lpSum(y_vars[k] for v in V_k[k]) <= max_allowed_same_configurations, f"Max_Same_Config_{k}"

    # Export c_vk to a CSV file
    c_vk.to_csv('cost_matrix.csv', index=True)  # Save the cost matrix to a CSV file

    return model, y_vars

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
    solution_stats: Dict,
    parameters: Parameters
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
            for good in parameters.goods
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
    configurations_df: pd.DataFrame,
    parameters: Parameters
) -> Dict:
    """
    Calculate various statistics about the solution.
    """
    
    # Calculate fixed costs
    total_fixed_cost = selected_clusters.merge(
        configurations_df[["Config_ID", "Fixed_Cost"]], 
        on="Config_ID"
    )["Fixed_Cost"].sum()
    
    # Calculate variable costs based on route time
    selected_clusters = selected_clusters.copy()
    
    total_variable_cost = (
        selected_clusters['Route_Time'] * parameters.variable_cost_per_hour
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

def _calculate_cluster_cost(
    cluster: pd.Series,
    config: pd.Series,
    parameters: Parameters
) -> float:
    """
    Calculate the total cost (fixed + variable) for serving a cluster with a vehicle configuration.
    """
    # Fixed cost from vehicle configuration
    fixed_cost = config['Fixed_Cost']
    
    # Variable cost based on route time (in hours)
    route_time = cluster['Route_Time']  # Already in hours
    variable_cost = parameters.variable_cost_per_hour * route_time

    total_cost = fixed_cost + variable_cost
    
    # Debugging log
    logger.debug(f"Cluster {cluster['Cluster_ID']} - Vehicle {config['Config_ID']}: Fixed Cost = {fixed_cost}, Variable Cost = {variable_cost}, Total Cost = {total_cost}")
    
    return total_cost