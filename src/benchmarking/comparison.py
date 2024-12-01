from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import time
from pathlib import Path
import sys
from copy import deepcopy

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.benchmarking.cvrp_adapter import parse_vrptw_file, adapt_to_mcv, VRPTWInstance
from src.config.parameters import Parameters
from src.clustering import generate_clusters_for_configurations
from src.utils.vehicle_configurations import generate_vehicle_configurations
from src.fsm_optimizer import solve_fsm_problem
import numpy as np
from src.utils.metrics import calculate_vehicle_utilization, count_vehicles_by_type

@dataclass
class ComparisonResult:
    """Stores comparison metrics between CVRP and MCV solutions"""
    instance_name: str
    original_capacity: float
    cvrp_vehicles: int
    cvrp_distance: float
    mcv_vehicles: int
    mcv_vehicle_mix: Dict[int, int]
    mcv_total_cost: float
    execution_time: float
    avg_vehicle_utilization: float
    
    def to_dict(self) -> dict:
        return {
            'Instance': self.instance_name,
            'CVRP Vehicles': self.cvrp_vehicles,
            'MCV Vehicles': self.mcv_vehicles,
            'Vehicle Difference': self.mcv_vehicles - self.cvrp_vehicles,
            'MCV Total Cost': self.mcv_total_cost,
            'Execution Time (s)': self.execution_time,
            'Avg Vehicle Utilization': self.avg_vehicle_utilization
        }

def calculate_vehicle_utilization(selected_clusters: pd.DataFrame, configs_df: pd.DataFrame, params: Parameters) -> float:
    """Calculate average vehicle utilization across selected clusters."""
    utilizations = []
    for _, cluster in selected_clusters.iterrows():
        config_id = cluster.get('Config_ID')
        if config_id is None:
            continue
            
        config = configs_df[configs_df['Config_ID'] == config_id].iloc[0]
        capacity = config['Capacity']
        
        # Get total demand based on the structure of the cluster
        if 'Total_Demand' in cluster:
            if isinstance(cluster['Total_Demand'], dict):
                total_demand = sum(cluster['Total_Demand'].values())
            else:
                total_demand = cluster['Total_Demand']
        else:
            demands = [cluster.get(f'{product}_Demand', 0) for product in params.goods]
            total_demand = sum(demands)
        
        utilization = total_demand / capacity
        utilizations.append(utilization)
        
    return np.mean(utilizations) if utilizations else 0.0

def count_vehicles_by_type(selected_clusters: pd.DataFrame) -> Dict[str, int]:
    """Count number of vehicles used by configuration type."""
    return selected_clusters['Config_ID'].value_counts().to_dict()

def run_comparison(
    instance_path: str,
    config_path: str,
    time_limit: int = 300
) -> ComparisonResult:
    """Compare MCV solution against VRPTW benchmark"""
    try:
        # Parse instance and convert to MCV format
        instance = parse_vrptw_file(instance_path)
        params = Parameters.from_yaml(config_path)
        params.max_route_time = 12
        params.clustering['method'] = "combine"
        
        # Override service time from VRPTW instance
        params.service_time = instance.service_time
        
        # Calculate speed to maintain benchmark travel times
        # Bogotá area dimensions
        BOGOTA_NS = 66.6  # km
        BOGOTA_EW = 44.4  # km
        # Original grid was 500x500
        GRID_SIZE = 500
        
        # Calculate km per grid unit
        km_per_unit = (BOGOTA_NS + BOGOTA_EW) / (2 * GRID_SIZE)
        
        # In benchmark: 1 unit distance = 1 unit time
        # So speed should be km_per_unit km per time unit
       # params.avg_speed = km_per_unit * 60  # Convert to km/h
        
        print(f"\nSpeed calculation:")
        print(f"  Grid unit to km ratio: {km_per_unit:.3f} km/unit")
        print(f"  Adjusted speed: {params.avg_speed:.1f} km/h")
        
        customers_df, original_capacity = adapt_to_mcv(instance, config_path)
        
        # Apply original capacity to all vehicles
        for vehicle in params.vehicles.values():
            vehicle['capacity'] = original_capacity
        
        print(f"\nComparing solutions for {instance.name}")
        print("=" * 50)
        print(f"Parameters:")
        print(f"  Max Route Time: {params.max_route_time} hours")
        print(f"  Service Time: {params.service_time} minutes")
        print(f"  Average Speed: {params.avg_speed} km/h")
        
        # Solve MCV version
        print("\nSolving multi-compartment version...")
        print(f"CVRP Best Known Vehicles: {instance.best_known_vehicles}")
        start_time = time.time()
        
        # Step 1: Generate vehicle configurations
        configs_df = generate_vehicle_configurations(
            params.vehicles,
            params.goods
        )
        print(f"Generated {len(configs_df)} vehicle configurations")
        
        # Step 2: Generate clusters
        clusters_df = generate_clusters_for_configurations(
            customers=customers_df,
            configurations_df=configs_df,
            params=params
        )
        
        # Debug cluster demands vs vehicle capacities
        if len(clusters_df) > 0:
            print("\nCluster Demand Analysis:")
            max_cluster_demand = max(
                clusters_df['Total_Demand'].apply(lambda x: sum(x.values()) if isinstance(x, dict) else x)
            )
            min_vehicle_capacity = min(vehicle['capacity'] for vehicle in params.vehicles.values())
            print(f"Maximum cluster demand: {max_cluster_demand}")
            print(f"Minimum vehicle capacity: {min_vehicle_capacity}")
            
            # Round demands to handle floating point precision issues
            clusters_df['Total_Demand'] = clusters_df['Total_Demand'].apply(
                lambda x: {k: round(v, 6) for k, v in x.items()} if isinstance(x, dict) else round(x, 6)
            )
            
        print(f"Created {len(clusters_df)} feasible clusters")
        print(f"Average customers per cluster: {clusters_df['Customers'].apply(len).mean():.1f}")
        print(f"Average route time: {clusters_df['Route_Time'].mean():.1f} hours")
        
        # Step 3: Solve FSM problem
        solution = solve_fsm_problem(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=customers_df,
            parameters=params,
            verbose=False
        )
        execution_time = time.time() - start_time
        
        # Calculate metrics
        vehicle_utilization = calculate_vehicle_utilization(
            solution['selected_clusters'],
            configs_df,
            params
        )
        vehicle_mix = count_vehicles_by_type(solution['selected_clusters'])
        
        # Print comparison
        print("\nSolution Comparison:")
        print("-" * 50)
        print(f"CVRP Benchmark:")
        print(f"  Vehicles: {instance.best_known_vehicles}")
        print(f"  Vehicle Capacity: {original_capacity}")
        
        print(f"\nMCV Solution:")
        print(f"  Total Vehicles: {len(solution['selected_clusters'])}")
        print("  Vehicle Mix:")
        for config_id, count in vehicle_mix.items():
            config = configs_df[configs_df['Config_ID'] == config_id].iloc[0]
            capacity = config['Capacity']
            vehicle_type = config['Vehicle_Type']
            compartments = [g for g in params.goods if config[g] == 1]
            print(f"    Type {vehicle_type} Config {config_id} (Capacity: {capacity}, Compartments: {compartments}): {count} vehicles")
        
        total_cost = (
            solution['total_fixed_cost'] + 
            solution['total_variable_cost'] + 
            solution['total_light_load_penalties'] +
            solution['total_compartment_penalties']
        )
        
        print(f"\nCosts:")
        print(f"  Fixed Cost: ${solution['total_fixed_cost']:,.2f}")
        print(f"  Variable Cost: ${solution['total_variable_cost']:,.2f}")
        print(f"  Light Load Penalties: ${solution['total_light_load_penalties']:,.2f}")
        print(f"  Compartment Penalties: ${solution['total_compartment_penalties']:,.2f}")
        print(f"  Total Cost: ${total_cost:,.2f}")
        print(f"  Avg Vehicle Utilization: {vehicle_utilization:.1%}")
        print(f"  Execution Time: {execution_time:.2f}s")
        print("-" * 50)
        
        result = ComparisonResult(
            instance_name=instance.name,
            original_capacity=original_capacity,
            cvrp_vehicles=instance.best_known_vehicles,
            mcv_vehicles=len(solution['selected_clusters']),
            mcv_vehicle_mix=vehicle_mix,
            mcv_total_cost=total_cost,
            execution_time=execution_time,
            avg_vehicle_utilization=vehicle_utilization
        )
        
        return result
        
    except Exception as e:
        print(f"\nError processing {instance_path}: {str(e)}")
        raise