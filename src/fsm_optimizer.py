"""
Fleet Size and Mix (FSM) Optimizer module.
Handles the optimization model creation, solving, and solution validation
for the vehicle routing problem with multiple compartments.
"""

import logging
import time
from typing import Dict, Tuple, Set, Any
import pandas as pd
import pulp
from haversine import haversine
import sys

from src.utils.logging import Colors, Symbols
from src.config.parameters import Parameters
from src.post_optimization import improve_solution
from src.utils.solver import pick_solver

logger = logging.getLogger(__name__)

def solve_fsm_problem(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    parameters: Parameters,
    solver=None,
    verbose: bool = False
) -> Dict:
    """
    Solve the Fleet Size and Mix optimization problem.
    
    Args:
        clusters_df: DataFrame containing generated clusters
        configurations_df: DataFrame containing vehicle configurations
        customers_df: DataFrame containing customer demands
        parameters: Parameters object containing optimization parameters
        solver: Optional solver to use instead of pick_solver
        verbose: Whether to enable verbose output to screen
    
    Returns:
        Dictionary containing optimization results
    """
    # Create optimization model
    model, y_vars, x_vars, c_vk = _create_model(clusters_df, configurations_df, parameters)
    
    # Select solver: use provided or pick based on FSM_SOLVER env
    solver = solver or pick_solver(verbose)
    logger.info(f"Using solver: {solver.name}")
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
    selected_clusters = _extract_solution(clusters_df, y_vars, x_vars)
    missing_customers = _validate_solution(
        selected_clusters, 
        customers_df,
        configurations_df
    )
    
    # Add goods columns from configurations before calculating statistics
    for good in parameters.goods:
        selected_clusters[good] = selected_clusters['Config_ID'].map(
            lambda x: configurations_df[configurations_df['Config_ID'] == x].iloc[0][good]
        )

    # Calculate statistics using the actual optimization costs
    solution = _calculate_solution_statistics(
        selected_clusters,
        configurations_df,
        parameters,
        model,
        x_vars,
        c_vk
    )
    
    # Add additional solution data
    solution.update({
        'selected_clusters': selected_clusters,
        'missing_customers': missing_customers,
        'solver_name': model.solver.name,
        'solver_status': pulp.LpStatus[model.status]
    })
    
    # Try to improve solution
    if parameters.post_optimization:
        solution = improve_solution(
            solution,
            configurations_df,
            customers_df,
            parameters
        )
    
    return solution

def _create_model(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    parameters: Parameters
) -> Tuple[pulp.LpProblem, Dict[str, pulp.LpVariable], Dict[Tuple[str, Any], pulp.LpVariable], Dict[Tuple[str, str], float]]:
    """
    Create the optimization model (Model 2) aligning with the mathematical formulation.
    """
    import pulp

    # Create the optimization model
    model = pulp.LpProblem("FSM-MCV_Model2", pulp.LpMinimize)

    # Sets
    N = set(clusters_df['Customers'].explode().unique())  # Customers
    K = set(clusters_df['Cluster_ID'])  # Clusters
    V = set(configurations_df['Config_ID'])  # Vehicle configurations

    # Initialize decision variables dictionaries
    x_vars = {}
    y_vars = {}
    c_vk = {}

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

    # Create remaining decision variables
    for k in K:
        y_vars[k] = pulp.LpVariable(f"y_{k}", cat='Binary')
        for v in V_k[k]:
            if (v, k) not in x_vars:  # Only create if not already created
                x_vars[v, k] = pulp.LpVariable(f"x_{v}_{k}", cat='Binary')

    # Parameters
    for k in K:
        cluster = clusters_df.loc[clusters_df['Cluster_ID'] == k].iloc[0]
        for v in V_k[k]:
            if v != 'NoVehicle':
                config = configurations_df.loc[configurations_df['Config_ID'] == v].iloc[0]
                # Calculate load percentage
                total_demand = sum(cluster['Total_Demand'][g] for g in parameters.goods)
                capacity = config['Capacity']
                load_percentage = total_demand / capacity

                # Apply fixed penalty if under threshold
                penalty_amount = parameters.light_load_penalty if load_percentage < parameters.light_load_threshold else 0
                base_cost = _calculate_cluster_cost(
                    cluster=cluster,
                    config=config,
                    parameters=parameters
                )
                
                c_vk[v, k] = base_cost + penalty_amount
                logger.debug(
                    f"Cluster {k}, vehicle {v}: Load Percentage = {load_percentage:.2f}, "
                    f"Penalty = {penalty_amount}"
                )
            else:
                c_vk[v, k] = 0  # Cost is zero for placeholder

    # Objective Function
    model += pulp.lpSum(
        c_vk[v, k] * x_vars[v, k]
        for k in K for v in V_k[k]
    ), "Total_Cost"

    # Constraints

    # 1. Customer Allocation Constraint (Exact Assignment)
    for i in N:
        model += pulp.lpSum(
            x_vars[v, k]
            for k in K_i[i]
            for v in V_k[k]
            if v != 'NoVehicle'
        ) == 1, f"Customer_Coverage_{i}"

    # 2. Vehicle Configuration Assignment Constraint
    for k in K:
        model += (
            pulp.lpSum(x_vars[v, k] for v in V_k[k]) == y_vars[k]
        ), f"Vehicle_Assignment_{k}"

    # 3. Unserviceable Clusters Constraint
    for k in K:
        if 'NoVehicle' in V_k[k]:
            model += y_vars[k] == 0, f"Unserviceable_Cluster_{k}"

    return model, y_vars, x_vars, c_vk

def _extract_solution(
    clusters_df: pd.DataFrame,
    y_vars: Dict,
    x_vars: Dict
) -> pd.DataFrame:
    """Extract the selected clusters and their assigned configurations."""
    selected_cluster_ids = [
        cid for cid, var in y_vars.items() 
        if var.varValue and var.varValue > 0.5
    ]

    cluster_config_map = {}
    for (v, k), var in x_vars.items():
        if var.varValue and var.varValue > 0.5 and k in selected_cluster_ids:
            cluster_config_map[k] = v

    # Get selected clusters with ALL columns from input DataFrame
    # This preserves the goods columns that were set during merging
    selected_clusters = clusters_df[
        clusters_df['Cluster_ID'].isin(selected_cluster_ids)
    ].copy()

    # Update Config_ID while keeping existing columns
    selected_clusters['Config_ID'] = selected_clusters['Cluster_ID'].map(cluster_config_map)

    return selected_clusters

def _validate_solution(
    selected_clusters: pd.DataFrame,
    customers_df: pd.DataFrame,
    configurations_df: pd.DataFrame
) -> Set:
    """
    Validate that all customers are served in the solution.
    """
    logger = logging.getLogger(__name__)

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
        f"{Colors.CYAN}Total Fixed Cost:  ${Colors.BOLD}"
        f"{solution_stats['total_fixed_cost']:>10,.2f}{Colors.RESET}"
    )
    logger.info(
        f"{Colors.CYAN}Total Variable Cost:${Colors.BOLD}"
        f"{solution_stats['total_variable_cost']:>10,.2f}{Colors.RESET}"
    )
    logger.info(
        f"{Colors.CYAN}Total Light Load Penalties:${Colors.BOLD}"
        f"{solution_stats['total_light_load_penalties']:>10,.2f}{Colors.RESET}"
    )
    logger.info(
        f"{Colors.CYAN}Total Compartment Penalties:${Colors.BOLD}"
        f"{solution_stats['total_compartment_penalties']:>10,.2f}{Colors.RESET}"
    )
    logger.info(
        f"{Colors.CYAN}Total Cost:         ${Colors.BOLD}"
        f"{solution_stats['total_cost']:>10,.2f}{Colors.RESET}"
    )

    # Vehicle Usage Summary
    logger.info(f"\n{Symbols.TRUCK} Vehicles by Type")
    for vehicle_type in sorted(solution_stats['vehicles_used']):
        vehicle_count = solution_stats['vehicles_used'][vehicle_type]
        logger.info(
            f"{Colors.BLUE}→ Type {vehicle_type}:{Colors.BOLD}"
            f"{vehicle_count:>4}{Colors.RESET}"
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
            if config[good] == 1
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
    parameters: Parameters,
    model: pulp.LpProblem,
    x_vars: Dict,
    c_vk: Dict
) -> Dict:
    """Calculate solution statistics using the optimization results."""
    
    # Get selected assignments and their actual costs from the optimization
    selected_assignments = {
        (v, k): c_vk[(v, k)] 
        for (v, k), var in x_vars.items() 
        if var.varValue == 1
    }
    
    # Calculate compartment penalties
    total_compartment_penalties = sum(
        parameters.compartment_setup_cost * (
            sum(1 for g in parameters.goods 
                if row[g] == 1) - 1
        )
        for _, row in selected_clusters.iterrows()
        if sum(1 for g in parameters.goods 
              if row[g] == 1) > 1
    )
    
    # Get vehicle statistics and fixed costs
    selected_clusters = selected_clusters.merge(
        configurations_df, 
        on="Config_ID",
        how='left'
    )
    
    # Calculate base costs (without penalties)
    total_fixed_cost = selected_clusters["Fixed_Cost"].sum()
    total_variable_cost = (
        selected_clusters['Route_Time'] * parameters.variable_cost_per_hour
    ).sum()
    
    # Total cost from optimization
    total_cost = sum(selected_assignments.values())
    
    # Light load penalties are the remaining difference
    total_light_load_penalties = (
        total_cost - 
        (total_fixed_cost + total_variable_cost + total_compartment_penalties)
    )
    
    # Total penalties
    total_penalties = total_light_load_penalties + total_compartment_penalties
    
    return {
        'total_fixed_cost': total_fixed_cost,
        'total_variable_cost': total_variable_cost,
        'total_light_load_penalties': total_light_load_penalties,
        'total_compartment_penalties': total_compartment_penalties,
        'total_penalties': total_penalties,
        'total_cost': total_cost,
        'vehicles_used': selected_clusters['Vehicle_Type'].value_counts().sort_index().to_dict(),
        'total_vehicles': len(selected_clusters)
    }

def _calculate_cluster_cost(
    cluster: pd.Series,
    config: pd.Series,
    parameters: Parameters
) -> float:
    """
    Calculate the base cost for serving a cluster with a vehicle configuration.
    Includes:
    - Fixed cost
    - Variable cost (time-based)
    - Compartment setup cost
    
    Note: Light load penalties are handled separately in the model creation.
    Args:
        cluster: The cluster data as a Pandas Series.
        config: The vehicle configuration data as a Pandas Series.
        parameters: Parameters object containing optimization parameters.

    Returns:
        Base cost of serving the cluster with the given vehicle configuration.
    """
    # Base costs
    fixed_cost = config['Fixed_Cost']
    route_time = cluster['Route_Time']
    variable_cost = parameters.variable_cost_per_hour * route_time

    # Compartment setup cost
    num_compartments = sum(1 for g in parameters.goods if config[g])
    compartment_cost = 0.0
    if num_compartments > 1:
        compartment_cost = parameters.compartment_setup_cost * (num_compartments - 1)

    # Total cost
    total_cost = fixed_cost + variable_cost + compartment_cost
    
    return total_cost