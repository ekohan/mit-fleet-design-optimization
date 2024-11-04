"""
Fleet Size and Mix (FSM) Optimizer using Gurobi.
Implements the multi-compartment vehicle routing optimization model.
"""

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from typing import Dict, Tuple
import logging
from utils.logging import Colors, Symbols
from haversine import haversine
from utils.cluster_utils import calculate_cluster_time

logger = logging.getLogger(__name__)

INFO_SYMBOL = "ℹ"  # Replace Symbols.INFO with a direct symbol

def solve_fsm_problem(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    parameters: 'Parameters',  # Changed to use parameters object
    verbose: bool = False
) -> Dict:
    """
    Solve the Fleet Size and Mix optimization problem.
    configurations_df represents vehicle types, not individual vehicles.
    Each type can be used multiple times.
    """
    try:
        # Create optimization model
        model = gp.Model("Fleet_Size_Mix")
        
        # Sets
        K = clusters_df.index.tolist()  # Set of clusters
        V = configurations_df.index.tolist()  # Set of vehicle types
        N = customers_df['Customer_ID'].tolist()  # Use actual Customer_ID instead of index
        
        if verbose:
            print("\nSets Debug:")
            print(f"Number of Clusters: {len(K)}")
            print(f"Number of Vehicle Types: {len(V)}")
            print(f"Number of Customers: {len(N)}")
        
        # Create K_i: clusters containing each customer
        K_i = {}
        for i in N:  # Use actual customer IDs
            customer_str = str(i)
            K_i[customer_str] = []
            for k in K:
                cluster_customers = [str(c) for c in clusters_df.loc[k, 'Customers']]
                if customer_str in cluster_customers:
                    K_i[customer_str].append(k)
        
        # Create V_k: vehicles that can serve each cluster
        V_k = {}
        for k in K:
            V_k[k] = []
            cluster_total_demand = sum(clusters_df.loc[k, 'Total_Demand'].values())
            for v in V:
                if (configurations_df.loc[v, 'Capacity'] >= cluster_total_demand and 
                    _is_compatible(configurations_df.loc[v], clusters_df.loc[k])):
                    V_k[k].append(v)
        
        # Debug vehicle compatibility (moved after V_k creation)
        if verbose:
            print("\nChecking vehicle compatibility:")
            for k in K:
                if not V_k[k]:
                    print(f"⚠️ Cluster {k} has no compatible vehicles!")
                    cluster = clusters_df.loc[k]
                    print(f"  Total Demand: {cluster['Total_Demand']}")
                    print(f"  Goods Required: {cluster['Goods_In_Config']}")
                    print(f"  Number of customers: {len(cluster['Customers'])}")

        # Debug customer assignments
        if verbose:
            print("\nChecking customer assignments:")
            customers_without_options = []
            for i in N:
                if not K_i[str(i)]:
                    customers_without_options.append(i)
            if customers_without_options:
                print(f"⚠️ Found {len(customers_without_options)} customers with no valid cluster options!")
                print(f"First few: {customers_without_options[:5]}")

        # Validate customer assignments
        unassigned_customers = []
        for i in N:  # Use actual customer IDs
            if not K_i[str(i)]:
                unassigned_customers.append(i)
        
        if unassigned_customers:
            if verbose:
                print("\n⚠️ Warning: Found unassigned customers!")
                print(f"Number of unassigned customers: {len(unassigned_customers)}")
                print(f"First few unassigned: {unassigned_customers[:5]}")
                print("\nFirst cluster customers:", clusters_df.loc[0, 'Customers'])
                print("First few customer IDs:", N[:5])  # Show actual customer IDs
            raise ValueError(f"{len(unassigned_customers)} customers are not assigned to any cluster")
        
        if verbose:
            print("\nCustomer-Cluster Assignment Stats:")
            assignments_per_customer = [len(clusters) for clusters in K_i.values()]
            print(f"Average clusters per customer: {sum(assignments_per_customer)/len(assignments_per_customer):.1f}")
            print(f"Min clusters per customer: {min(assignments_per_customer)}")
            print(f"Max clusters per customer: {max(assignments_per_customer)}")
        
        # Calculate costs
        c_vk = {
            (v, k): _calculate_cost(
                configurations_df.loc[v],
                clusters_df.loc[k],
                parameters
            )
            for k in K
            for v in V_k[k]
        }
        
        # Decision Variables: x[v,k] = 1 if vehicle type v serves cluster k
        x = model.addVars(
            [(v, k) for k in K for v in V_k[k]],
            vtype=GRB.BINARY,
            name="x"
        )
        
        # Decision Variables: y[k] = 1 if cluster k is selected
        y = model.addVars(K, vtype=GRB.BINARY, name="y")
        
        # Objective Function: Minimize total cost
        model.setObjective(
            gp.quicksum(
                x[v, k] * c_vk[v, k]
                for k in K
                for v in V_k[k]
            ),
            GRB.MINIMIZE
        )
        
        # Constraints
        # 1. Each customer must be served exactly once
        for i in N:
            model.addConstr(
                gp.quicksum(y[k] for k in K_i[str(i)]) == 1,
                name=f"customer_allocation[{i}]"
            )
        
        # 2. If cluster is selected, it must have exactly one vehicle
        for k in K:
            model.addConstr(
                gp.quicksum(x[v, k] for v in V_k[k]) == y[k],
                name=f"cluster_allocation[{k}]"
            )
        
        # Solve the model
        if not verbose:
            model.setParam('OutputFlag', 0)
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Get selected clusters with their vehicle assignments
            selected_indices = []
            vehicle_assignments = {}
            cluster_distances = {}
            
            for k in K:
                if sum(x[v, k].x for v in V_k[k]) > 0.5:
                    selected_indices.append(k)
                    # Find which vehicle was assigned to this cluster
                    for v in V_k[k]:
                        if x[v, k].x > 0.5:
                            vehicle_assignments[k] = v
                            cluster_coord = (clusters_df.loc[k, 'Centroid_Latitude'], 
                                          clusters_df.loc[k, 'Centroid_Longitude'])
                            depot_coord = (parameters.depot['latitude'], 
                                         parameters.depot['longitude'])
                            cluster_distances[k] = 2 * haversine(depot_coord, cluster_coord)
            
            # Create selected_clusters DataFrame and preserve all columns
            selected_clusters = clusters_df.loc[selected_indices].copy()
            
            if verbose:
                print("\nDebug - Selected Clusters DataFrame columns:")
                print(selected_clusters.columns.tolist())
                print("\nFirst row sample:")
                print(selected_clusters.iloc[0])
            
            # Add vehicle assignments and distances
            selected_clusters['Vehicle_Type'] = selected_clusters.index.map(vehicle_assignments)
            selected_clusters['Estimated_Distance'] = selected_clusters.index.map(cluster_distances)
            
            # Ensure Customers column is present and in the right format
            if 'Customers' not in selected_clusters.columns:
                logger.error("Missing 'Customers' column in clusters_df")
                selected_clusters['Customers'] = [[]] * len(selected_clusters)
            elif not isinstance(selected_clusters['Customers'].iloc[0], list):
                selected_clusters['Customers'] = selected_clusters['Customers'].apply(lambda x: list(x) if isinstance(x, (list, tuple, set)) else [])

            # Check if all customers are served
            served_customers = set()
            for _, cluster in selected_clusters.iterrows():
                served_customers.update(str(c) for c in cluster['Customers'])
            
            total_customers = set(str(c) for c in N)
            missing_customers = total_customers - served_customers
            
            if verbose:
                print(f"\n{INFO_SYMBOL} Customer Coverage Analysis:")
                print(f"  Total Customers: {len(total_customers)}")
                print(f"  Served Customers: {len(served_customers)}")
                print(f"  Missing Customers: {len(missing_customers)}")
                if missing_customers:
                    print(f"  First few missing: {list(missing_customers)[:5]}")
                print(f"\n  Selected Clusters: {len(selected_indices)}")
                print(f"  Average Customers per Cluster: {len(served_customers)/len(selected_indices):.1f}")
            
            # Calculate vehicle usage
            vehicles_used = pd.Series({
                v: sum(x[v, k].x for k in K if (v, k) in x.keys())
                for v in V
            }).astype(int)

            # Calculate costs
            fixed_cost = sum(configurations_df.loc[v, 'Fixed_Cost'] * x[v, k].x 
                           for v, k in x.keys())
            variable_cost = sum(_calculate_cost(configurations_df.loc[v], clusters_df.loc[k], parameters) * x[v, k].x 
                              for v, k in x.keys())
            total_cost = fixed_cost + variable_cost

            # Calculate average distance
            avg_distance = selected_clusters['Estimated_Distance'].mean()

            return {
                'solver_name': 'Gurobi',
                'solver_status': 'Optimal',
                'total_cost': total_cost,
                'total_fixed_cost': fixed_cost,
                'total_variable_cost': variable_cost,
                'vehicles_used': vehicles_used,
                'selected_clusters': selected_clusters,
                'missing_customers': list(missing_customers),  # Add missing customers to output
                'avg_distance': avg_distance
            }
        else:
            # For infeasible or error cases, return empty DataFrame with correct columns
            empty_df = pd.DataFrame(columns=[
                'Cluster_ID', 'Config_ID', 'Customers', 'Total_Demand',
                'Centroid_Latitude', 'Centroid_Longitude', 'Goods_In_Config',
                'Route_Time', 'Vehicle_Type', 'Estimated_Distance'
            ])
            
            return {
                'solver_name': 'Gurobi',
                'solver_status': 'Infeasible',
                'total_cost': 0,
                'total_fixed_cost': 0,
                'total_variable_cost': 0,
                'vehicles_used': pd.Series(dtype='int64'),
                'selected_clusters': empty_df,
                'missing_customers': [str(c) for c in N]
            }
        
    except Exception as e:
        if verbose:
            print(f"Error during optimization: {str(e)}")
        
        # Create empty DataFrame with required columns
        empty_df = pd.DataFrame(columns=[
            'Cluster_ID', 'Config_ID', 'Customers', 'Total_Demand',
            'Centroid_Latitude', 'Centroid_Longitude', 'Goods_In_Config',
            'Route_Time', 'Vehicle_Type', 'Estimated_Distance'
        ])
        
        return {
            'solver_name': 'Gurobi',
            'solver_status': 'Error',
            'total_cost': 0,
            'total_fixed_cost': 0,
            'total_variable_cost': 0,
            'vehicles_used': pd.Series(dtype='int64'),
            'selected_clusters': empty_df,  # Use DataFrame with correct columns
            'missing_customers': [str(c) for c in N]
        }
def _is_compatible(vehicle_config: pd.Series, cluster: pd.Series) -> bool:
    """Check if vehicle configuration is compatible with cluster requirements."""
    # Check compartment compatibility
    for compartment in cluster['Goods_In_Config']:
        if not vehicle_config[compartment]:
            return False
    return True

def _has_capacity(vehicle_config: pd.Series, cluster: pd.Series) -> bool:
    """Check if vehicle has enough capacity for cluster total demand."""
    # Sum up all demands in the cluster
    total_cluster_demand = sum(cluster['Total_Demand'].values())
    
    # Check if vehicle capacity is sufficient
    return vehicle_config['Capacity'] >= total_cluster_demand

def _calculate_cost(vehicle: pd.Series, cluster: pd.Series, parameters: 'Parameters') -> float:
    """
    Calculate total cost (fixed + variable) for serving a cluster with a vehicle.
    
    Args:
        vehicle: Vehicle configuration series
        cluster: Cluster series with centroid coordinates
        parameters: Parameters object containing depot location and costs
    
    Returns:
        float: Total cost (fixed + variable) for the route
    """
    # Calculate distance
    cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
    depot_coord = (parameters.depot['latitude'], parameters.depot['longitude'])
    distance = 2 * haversine(depot_coord, cluster_coord)
    
    return vehicle['Fixed_Cost'] + (distance * parameters.variable_cost_per_km)

def _calculate_solution_stats(
    selected: list,
    configurations_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    customers_df: pd.DataFrame
) -> Dict:
    """Calculate detailed statistics about the solution."""
    # Implementation details...
    pass

def _validate_input_data(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame
) -> bool:
    """Validate input data for feasibility."""
    valid = True
    
    # Check vehicle capacities vs cluster demands
    for _, cluster in clusters_df.iterrows():
        max_demand = max(cluster['Total_Demand'].values())
        if max_demand > configurations_df['Capacity'].max():
            logger.error(
                f"Cluster {cluster['Cluster_ID']} has demand {max_demand} > "
                f"max vehicle capacity {configurations_df['Capacity'].max()}"
            )
            valid = False
    
    # Check compartment availability
    required_compartments = set()
    for _, cluster in clusters_df.iterrows():
        required_compartments.update(cluster['Goods_In_Config'])
    
    available_compartments = set()
    for _, vehicle in configurations_df.iterrows():
        available_compartments.update(
            good for good in ['Dry', 'Chilled', 'Frozen'] 
            if vehicle[good]
        )
    
    missing_compartments = required_compartments - available_compartments
    if missing_compartments:
        logger.error(f"No vehicles available with compartments: {missing_compartments}")
        valid = False
    
    return valid
