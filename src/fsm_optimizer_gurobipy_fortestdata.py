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
from config import DEPOT, VARIABLE_COST_PER_KM
from haversine import haversine

logger = logging.getLogger(__name__)

def solve_fsm_problem(
    clusters_df: pd.DataFrame,
    configurations_df: pd.DataFrame,
    customers_df: pd.DataFrame,
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
        N = customers_df.index.tolist()  # Set of customers
        
        if verbose:
            print("\nSets Debug:")
            print(f"Clusters (K): {K}")
            print(f"Vehicle Types (V): {V}")
            print(f"Customers (N): {N}")
        
        # Create V_k: vehicles that can serve each cluster
        V_k = {}
        for k in K:
            V_k[k] = []
            cluster_total_demand = sum(clusters_df.loc[k, 'Total_Demand'].values())
            for v in V:
                if (configurations_df.loc[v, 'Capacity'] >= cluster_total_demand and 
                    _is_compatible(configurations_df.loc[v], clusters_df.loc[k])):
                    V_k[k].append(v)
        
        if verbose:
            print("\nCompatibility Check:")
            for k in K:
                total_demand = sum(clusters_df.loc[k, 'Total_Demand'].values())
                print(f"\nCluster {k} (Total Demand: {total_demand}):")
                print("Compatible vehicles:")
                for v in V_k[k]:
                    print(f"  Vehicle {v} (Capacity: {configurations_df.loc[v, 'Capacity']})")
        
        # Create K_i: clusters containing each customer
        K_i = {
            str(i): [
                k for k in K 
                if str(i) in [str(c) for c in clusters_df.loc[k, 'Customers']]
            ]
            for i in N
        }
        
        if verbose:
            print("\nCustomer-Cluster Assignments:")
            for i in N:
                if not K_i[i]:
                    print(f"WARNING: Customer {i} is not assigned to any cluster!")
                else:
                    print(f"Customer {i} is in clusters: {K_i[i]}")
        
        # Calculate costs
        c_vk = {
            (v, k): _calculate_cost(
                configurations_df.loc[v],
                clusters_df.loc[k]
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
                gp.quicksum(
                    x[v, k]
                    for k in K_i[i]
                    for v in V_k[k]
                ) == 1,
                name=f"customer_allocation[{i}]"
            )
        
        # 2. Each cluster must be served by exactly one vehicle type
        for k in K:
            model.addConstr(
                gp.quicksum(
                    x[v, k]
                    for v in V_k[k]
                ) == 1,
                name=f"cluster_allocation[{k}]"
            )
        
        # Solve the model
        if not verbose:
            model.setParam('OutputFlag', 0)
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            # Get selected assignments
            selected = [
                (v, k) for v, k in x.keys()
                if x[v, k].X > 0.5
            ]
            
            # Calculate costs
            total_fixed_cost = sum(
                configurations_df.loc[v, 'Fixed_Cost']
                for v, k in selected
            )
            
            total_variable_cost = sum(
                c_vk[v, k] - configurations_df.loc[v, 'Fixed_Cost']
                for v, k in selected
            )
            
            # Get vehicles used
            vehicles_used = pd.Series(
                [v for v, k in selected]
            ).value_counts()
            
            # Get cluster assignments with details
            cluster_assignments = []
            for v, k in selected:
                assignment = {
                    'cluster_id': k,
                    'vehicle_type': configurations_df.loc[v, 'Vehicle_Type'],
                    'vehicle_capacity': configurations_df.loc[v, 'Capacity'],
                    'total_demand': sum(clusters_df.loc[k, 'Total_Demand'].values()),
                    'demands': clusters_df.loc[k, 'Total_Demand'],
                    'route_time': clusters_df.loc[k, 'Route_Time'],
                    'customers': clusters_df.loc[k, 'Customers']
                }
                cluster_assignments.append(assignment)
            
            # Sort assignments by cluster ID for better readability
            cluster_assignments.sort(key=lambda x: x['cluster_id'])
            
            if verbose:
                print("\n📊 Detailed Solution")
                print("=" * 50)
                
                print("\nVehicle Allocation:")
                for v_type, count in vehicles_used.items():
                    vehicle = configurations_df.loc[v_type]
                    print(f"→ Type {vehicle['Vehicle_Type']}: {count:3d} vehicles (Capacity: {vehicle['Capacity']})")
                
                print("\nCluster Assignments:")
                for assignment in cluster_assignments:
                    print(f"\nCluster {assignment['cluster_id']}:")
                    print(f"  → Assigned Vehicle Type: {assignment['vehicle_type']} (Capacity: {assignment['vehicle_capacity']})")
                    print(f"  → Total Demand: {assignment['total_demand']} units")
                    print(f"  → Demands by Type: {assignment['demands']}")
                    print(f"  → Route Time: {assignment['route_time']} minutes")
                    print(f"  → Customers: {assignment['customers']}")
                    print(f"  → Utilization: {(assignment['total_demand'] / assignment['vehicle_capacity'] * 100):.1f}%")
            
            return {
                'solver_name': 'Gurobi',
                'solver_status': 'Optimal',
                'total_cost': total_fixed_cost + total_variable_cost,
                'total_fixed_cost': total_fixed_cost,
                'total_variable_cost': total_variable_cost,
                'vehicles_used': vehicles_used,
                'cluster_assignments': cluster_assignments
            }
            
        elif model.status == GRB.INFEASIBLE:
            model.computeIIS()
            if verbose:
                print("\nInfeasibility Analysis:")
                for c in model.getConstrs():
                    if c.IISConstr:
                        print(f"Constraint {c.ConstrName} is part of the conflict")
            
            return {
                'solver_name': 'Gurobi',
                'solver_status': 'Infeasible',
                'error': "No feasible solution exists",
                'total_fixed_cost': 0,
                'total_variable_cost': 0,
                'vehicles_used': pd.Series(dtype='int64'),
                'missing_customers': N
            }
        
    except Exception as e:
        if verbose:
            print(f"Error during optimization: {str(e)}")
        return {
            'solver_name': 'Gurobi',
            'solver_status': 'Error',
            'error': str(e),
            'total_fixed_cost': 0,
            'total_variable_cost': 0,
            'vehicles_used': pd.Series(dtype='int64'),
            'missing_customers': N
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

def _calculate_cost(vehicle: pd.Series, cluster: pd.Series) -> float:
    """Calculate total cost (fixed + variable) for serving a cluster with a vehicle."""
    # Calculate distance
    cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
    depot_coord = (DEPOT['Latitude'], DEPOT['Longitude'])
    distance = 2 * haversine(depot_coord, cluster_coord)
    
    return vehicle['Fixed_Cost'] + (distance * VARIABLE_COST_PER_KM)

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