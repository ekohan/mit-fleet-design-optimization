"""
Script to convert CVRP instances to FSM format and run optimization.
"""

import logging
from pathlib import Path
import sys
import time
from enum import Enum
from typing import List, Dict, Union
import argparse

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
from src.benchmarking.cvrp_parser import CVRPParser
from src.config.parameters import Parameters
from src.utils.logging import setup_logging
from src.main import solve_fsm_problem
from src.utils.vehicle_configurations import generate_vehicle_configurations
from src.clustering import generate_clusters_for_configurations
from src.utils.coordinate_converter import CoordinateConverter, GeoBounds
from src.utils.save_results import save_optimization_results

class CVRPBenchmarkType(Enum):
    NORMAL = "normal"  # Type 1: Single instance, single good
    SPLIT = "split"    # Type 2: Single instance, split demand
    SCALED = "scaled"  # Type 3: Single instance scaled
    COMBINED = "combined"  # Type 4: Multiple instances combined

def convert_cvrp_to_fsm(
    instance_names: Union[str, List[str]],
    benchmark_type: CVRPBenchmarkType,
    num_goods: int = 3,
    split_ratios: Dict[str, float] = None
) -> tuple:
    """
    Convert CVRP instance(s) to FSM format based on benchmark type.
    
    Args:
        instance_names: Single instance name or list of instance names
        benchmark_type: Type of benchmark conversion
        num_goods: Number of goods to consider (2 or 3)
        split_ratios: Dictionary of ratios for splitting demand (for SPLIT type)
    """
    if isinstance(instance_names, str):
        instance_names = [instance_names]
        
    if benchmark_type == CVRPBenchmarkType.COMBINED and len(instance_names) < 2:
        raise ValueError(f"Combined benchmark type requires at least 2 instances")
        
    # Default split ratios if not provided
    if split_ratios is None:
        if num_goods == 2:
            split_ratios = {'dry': 0.6, 'chilled': 0.4}
        else:
            split_ratios = {'dry': 0.5, 'chilled': 0.3, 'frozen': 0.2}
            
    # Parse instances
    instances = []
    for name in instance_names:
        instance_path = Path(__file__).parent / 'cvrp_instances' / f'{name}.vrp'
        parser = CVRPParser(str(instance_path))
        instances.append(parser.parse())
        
    # Convert based on benchmark type
    if benchmark_type == CVRPBenchmarkType.NORMAL:
        return _convert_normal(instances[0])
    elif benchmark_type == CVRPBenchmarkType.SPLIT:
        return _convert_split(instances[0], split_ratios)
    elif benchmark_type == CVRPBenchmarkType.SCALED:
        return _convert_scaled(instances[0], num_goods)
    else:  # COMBINED
        return _convert_combined(instances)

def _convert_normal(instance) -> tuple:
    """Type 1: Normal conversion - single good (dry)"""
    # Print total demand for debugging
    total_demand = sum(instance.demands.values())
    print(f"Total CVRP demand: {total_demand}")
    print(f"CVRP capacity per vehicle: {instance.capacity}")
    print(f"Minimum theoretical vehicles needed: {total_demand / instance.capacity:.2f}")
    
    customers_data = _create_customer_data(
        instance,
        lambda demand: {'Dry_Demand': demand, 'Chilled_Demand': 0, 'Frozen_Demand': 0}
    )
    
    # Verify converted demand
    df = pd.DataFrame(customers_data)
    total_converted = df['Dry_Demand'].sum()
    print(f"Total converted demand: {total_converted}")
    
    params = _create_base_params(instance)
    params.expected_vehicles = instance.num_vehicles
    
    # Override the default vehicles with just our CVRP vehicle
    params.vehicles = {
        'CVRP': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {'Dry': True, 'Chilled': False, 'Frozen': False}
        }
    }
    
    print(f"\nVehicle Configuration:")
    print(f"Capacity: {instance.capacity}")
    print(f"Fixed Cost: {params.vehicles['CVRP']['fixed_cost']}")
    print(f"Compartments: {params.vehicles['CVRP']['compartments']}")
    
    return pd.DataFrame(customers_data), params

def _convert_split(instance, split_ratios: Dict[str, float]) -> tuple:
    """Type 2: Split demand across goods"""
    # Convert split_ratios keys to match DataFrame column names
    df_split_ratios = {
        f'{good.capitalize()}_Demand': ratio 
        for good, ratio in split_ratios.items()
    }
    
    customers_data = _create_customer_data(
        instance,
        lambda demand: {
            column: demand * ratio 
            for column, ratio in df_split_ratios.items()
        }
    )
    
    params = _create_base_params(instance)
    params.expected_vehicles = instance.num_vehicles
    
    # Override vehicles with just the multi-compartment CVRP vehicle
    params.vehicles = {
        'CVRP_Multi': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {good: True for good in split_ratios}
        }
    }
    
    return pd.DataFrame(customers_data), params

def _convert_scaled(instance, num_goods: int) -> tuple:
    """Type 3: Scale instance for multiple goods - only scale dry goods"""
    customers_data = _create_customer_data(
        instance,
        lambda demand: {
            'Dry_Demand': demand * num_goods,
            'Chilled_Demand': 0,
            'Frozen_Demand': 0
        }
    )
    
    params = _create_base_params(instance)
    params.expected_vehicles = instance.num_vehicles * num_goods
    
    # Override vehicles with scaled CVRP vehicle
    params.vehicles = {
        'CVRP_Scaled': {
            'capacity': instance.capacity * num_goods,
            'fixed_cost': 1000,
            'compartments': {'Dry': True, 'Chilled': False, 'Frozen': False}
        }
    }
    
    return pd.DataFrame(customers_data), params

def _convert_combined(instances: List) -> tuple:
    """Type 4: Combine multiple instances"""
    # Only use as many goods as we have instances
    goods = ['Dry', 'Chilled', 'Frozen'][:len(instances)]
    goods_columns = [f'{good}_Demand' for good in goods]
    
    customers_data = []
    for idx, (instance, good, column) in enumerate(zip(instances, goods, goods_columns)):
        instance_data = _create_customer_data(
            instance,
            lambda demand: {col: demand if col == column else 0 for col in goods_columns}
        )
        for customer in instance_data:
            customer['Customer_ID'] = f"{idx+1}_{customer['Customer_ID']}"
        customers_data.extend(instance_data)
    
    params = _create_base_params(instances[0])  # Use first instance for depot
    params.expected_vehicles = sum(inst.num_vehicles for inst in instances)
    
    # Create a vehicle type for each instance with its specific capacity and good
    params.vehicles = {
        f'CVRP_{idx+1}': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {g: (g == good) for g in goods}
        }
        for idx, (instance, good) in enumerate(zip(instances, goods))
    }
    
    return pd.DataFrame(customers_data), params

def _create_customer_data(instance, demand_func) -> List[Dict]:
    """Helper to create customer data with given demand function"""
    # Calculate bounds from coordinates
    converter = CoordinateConverter(instance.coordinates)
    geo_coords = converter.convert_all_coordinates(instance.coordinates)
    
    customers_data = []
    for cust_id, coords in geo_coords.items():
        if cust_id != instance.depot_id:
            customer = {
                'Customer_ID': str(cust_id),
                'Latitude': coords[0],
                'Longitude': coords[1],
                'Dry_Demand': 0,
                'Chilled_Demand': 0,
                'Frozen_Demand': 0  # Initialize all demands to 0
            }
            # Update with any non-zero demands from the demand_func
            customer.update(demand_func(instance.demands.get(cust_id, 0)))
            customers_data.append(customer)
            
    return customers_data

def _create_base_params(instance) -> Parameters:
    """Helper to create base parameters"""
    params = Parameters.from_yaml()
    converter = CoordinateConverter(instance.coordinates)
    geo_coords = converter.convert_all_coordinates(instance.coordinates)
    depot_coords = geo_coords[instance.depot_id]
    
    params.depot = {
        'latitude': depot_coords[0],
        'longitude': depot_coords[1]
    }

    params.max_route_time = float('inf')
    
    return params

def main():
    """Main function to run CVRP to FSM conversion and optimization."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Convert CVRP instance to FSM and optimize')
    parser.add_argument('--instance', 
                       default='X-n106-k14',   
                       nargs='+',  # Accept one or more instances
                       help='Name of instance file(s) without extension. For combined type, provide multiple instances.')
    parser.add_argument('--format',
                       default='excel',
                       choices=['excel', 'json'],
                       help='Output format for results')
    parser.add_argument('--benchmark-type',
                       choices=['normal', 'split', 'scaled', 'combined'],
                       help='Type of benchmark conversion')
    parser.add_argument('--num-goods',
                       type=int,
                       default=3,
                       choices=[2, 3],
                       help='Number of goods to consider')
    parser.add_argument('--info',
                       action='store_true',
                       help='Show detailed information about the tool and exit')
    
    args = parser.parse_args()
    
    if args.info:
        print("\nCVRP to FSM Conversion Tool")
        print("=" * 80)
        
        print("\nDescription:")
        print("  Converts CVRP benchmark instances to FSM format and runs optimization")
        print("  Supports different conversion types for benchmarking FSM performance")
        
        print("\nBenchmark Types:")
        print("-" * 80)
        print("normal:")
        print("  - Single instance converted to single good (dry)")
        print("  - Uses original vehicle capacity and number of vehicles")
        print("  - Best for direct comparison with CVRP results")
        print("\nsplit:")
        print("  - Single instance with demand split across multiple goods")
        print("  - Maintains original total demand but distributes across compartments")
        print("  - Tests multi-compartment optimization with correlated demands")
        print("\nscaled:")
        print("  - Single instance scaled for multiple goods")
        print("  - Multiplies capacity and vehicles by number of goods")
        print("  - Tests scalability of the FSM solver")
        print("\ncombined:")
        print("  - Multiple instances combined into one problem")
        print("  - Each instance represents a different good type")
        print("  - Tests handling of independent demand patterns")
        
        print("\nUsage Examples:")
        print("-" * 80)
        print("1. Basic conversion with normal type:")
        print("   python src/benchmarking/cvrp_to_fsm.py --instance X-n106-k14 --benchmark-type normal")
        print("\n2. Split demand across 2 goods:")
        print("   python src/benchmarking/cvrp_to_fsm.py --instance X-n106-k14 --benchmark-type split --num-goods 2")
        print("\n3. Save results as JSON:")
        print("   python src/benchmarking/cvrp_to_fsm.py --instance X-n106-k14 --benchmark-type normal --format json")
        
        print("\nAvailable Instances:")
        print("-" * 80)
        instance_dir = Path(__file__).parent / 'cvrp_instances'
        instances = sorted([f.stem for f in instance_dir.glob('*.vrp')])
        print("  " + "\n  ".join(instances))
        
        print("\nOutput:")
        print("-" * 80)
        print("- Generates solution files in the results directory")
        print("- File naming: cvrp_<instance>_<benchmark-type>.<format>")
        print("- Includes detailed metrics and solution statistics")
        print("=" * 80)
        return
        
    # Now check for required arguments
    if not args.benchmark_type:
        parser.error("argument --benchmark-type is required")
    if not args.instance:
        parser.error("argument --instance is required")

    
    start_time = time.time()
    
    # Convert CVRP to FSM format
    logger.info(f"Converting CVRP instance: {args.instance}")
    customers_df, params = convert_cvrp_to_fsm(
        instance_names=args.instance,
        benchmark_type=CVRPBenchmarkType(args.benchmark_type),
        num_goods=args.num_goods
    )
    
    # Generate vehicle configurations
    configs_df = generate_vehicle_configurations(params.vehicles, params.goods)
    logger.info(f"Generated {len(configs_df)} vehicle configurations")
    
    # Generate clusters
    clusters_df = generate_clusters_for_configurations(
        customers=customers_df,
        configurations_df=configs_df,
        params=params
    )
    logger.info(f"Generated {len(clusters_df)} clusters")
    
    # Solve optimization problem
    solution = solve_fsm_problem(
        clusters_df=clusters_df,
        configurations_df=configs_df,
        customers_df=customers_df,
        parameters=params,
        verbose=True
    )
    
    # Print results
    print("\nOptimization Results:")
    print(f"Total Cost: ${solution['total_cost']:,.2f}")
    print(f"Number of Vehicles Used: {sum(solution['vehicles_used'].values())}")
    print(f"Expected Vehicles: {params.expected_vehicles}")
    
    if solution['missing_customers']:
        print(f"\nWarning: {len(solution['missing_customers'])} customers not served!")
    
    # Save results
    file_name = f"cvrp_{args.instance}_{args.benchmark_type}"
    results_dir = Path(__file__).parent.parent.parent / 'results'
    results_path = results_dir / f"{file_name}.{'xlsx' if args.format == 'excel' else 'json'}"
    params.demand_file = file_name
    
    save_optimization_results(
        execution_time=time.time() - start_time,
        solver_name=solution['solver_name'],
        solver_status=solution['solver_status'],
        configurations_df=configs_df,
        selected_clusters=solution['selected_clusters'],
        total_fixed_cost=solution['total_fixed_cost'],
        total_variable_cost=solution['total_variable_cost'],
        total_light_load_penalties=solution['total_light_load_penalties'],
        total_compartment_penalties=solution['total_compartment_penalties'],
        total_penalties=solution['total_penalties'],
        vehicles_used=solution['vehicles_used'],
        missing_customers=solution['missing_customers'],
        parameters=params,
        filename=results_path,
        format=args.format,
        is_benchmark=True
    )

if __name__ == "__main__":
    main() 