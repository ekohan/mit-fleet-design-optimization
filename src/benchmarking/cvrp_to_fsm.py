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
import math  # Add this at the top with other imports

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
from src.utils.coordinate_converter import CoordinateConverter
from src.utils.save_results import save_optimization_results

class CVRPBenchmarkType(Enum):
    NORMAL = "normal"  
    SPLIT = "split"    
    SCALED = "scaled"  
    COMBINED = "combined"  
    SPATIAL = "spatial"  # New Type 5: Spatial Differentiation

def convert_cvrp_to_fsm(
    instance_names: Union[str, List[str]],
    benchmark_type: CVRPBenchmarkType,
    num_goods: int = 3,
    split_ratios: Dict[str, float] = None
) -> tuple:
    """
    Convert CVRP instance(s) to FSM format based on benchmark type.
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
    elif benchmark_type == CVRPBenchmarkType.COMBINED:
        return _convert_combined(instances)
    elif benchmark_type == CVRPBenchmarkType.SPATIAL:
        return _convert_spatial(instances[0], num_goods)
    else:
        raise ValueError(f"Unsupported benchmark type: {benchmark_type}")

def _convert_normal(instance) -> tuple:
    """Type 1: Normal conversion - single good (dry)"""
    total_demand = sum(instance.demands.values())
    print(f"Total CVRP demand: {total_demand}")
    print(f"CVRP capacity per vehicle: {instance.capacity}")
    print(f"Minimum theoretical vehicles needed: {total_demand / instance.capacity:.2f}")
    
    customers_data = _create_customer_data(
        instance,
        lambda demand: {'Dry_Demand': demand, 'Chilled_Demand': 0, 'Frozen_Demand': 0}
    )
    
    df = pd.DataFrame(customers_data)
    print(f"Total converted demand: {df['Dry_Demand'].sum()}")
    
    params = _create_base_params(instance)
    params.expected_vehicles = instance.num_vehicles
    
    params.vehicles = {
        'CVRP': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {'Dry': True, 'Chilled': False, 'Frozen': False}
        }
    }
    return pd.DataFrame(customers_data), params

def _convert_split(instance, split_ratios: Dict[str, float]) -> tuple:
    """Type 2: Split demand across goods"""
    df_split_ratios = {
        f'{good.capitalize()}_Demand': ratio 
        for good, ratio in split_ratios.items()
    }
    
    customers_data = _create_customer_data(
        instance,
        lambda demand: {col: math.ceil(demand * ratio) for col, ratio in df_split_ratios.items()}
    )
    
    params = _create_base_params(instance)
    params.expected_vehicles = instance.num_vehicles
    
    params.vehicles = {
        'CVRP_Multi': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {good: True for good in split_ratios}
        }
    }
    
    return pd.DataFrame(customers_data), params

def _convert_scaled(instance, num_goods: int) -> tuple:
    """Type 3: Scale instance for multiple goods"""
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
    
    params = _create_base_params(instances[0])
    params.expected_vehicles = sum(inst.num_vehicles for inst in instances)
    
    params.vehicles = {
        f'CVRP_{idx+1}': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {g: (g == good) for g in goods}
        }
        for idx, (instance, good) in enumerate(zip(instances, goods))
    }
    
    return pd.DataFrame(customers_data), params

def _convert_spatial(instance, num_goods: int) -> tuple:
    """Type 5: Spatial Differentiation
    Divide service area into quadrants and assign product mixes accordingly.
    For simplicity:
    - Quadrant 1 (top-left): Mostly Chilled
    - Quadrant 2 (top-right): Mostly Dry
    - Quadrant 3 (bottom-left): Mostly Frozen
    - Quadrant 4 (bottom-right): Mixed (Dry & Chilled)
    """
    # Convert coordinates and find midpoints
    converter = CoordinateConverter(instance.coordinates)
    geo_coords = converter.convert_all_coordinates(instance.coordinates)
    
    all_lats = [c[0] for cid, c in geo_coords.items() if cid != instance.depot_id]
    all_lons = [c[1] for cid, c in geo_coords.items() if cid != instance.depot_id]
    
    if not all_lats or not all_lons:
        raise ValueError("No customer coordinates found.")
    
    mid_lat = (min(all_lats) + max(all_lats)) / 2.0
    mid_lon = (min(all_lons) + max(all_lons)) / 2.0
    
    def demand_func(demand, lat, lon):
        if lat > mid_lat and lon < mid_lon:
            # Quadrant 1: Mostly Chilled
            return {
                'Dry_Demand': math.ceil(demand * 0.1),
                'Chilled_Demand': math.ceil(demand * 0.7),
                'Frozen_Demand': math.ceil(demand * 0.2)
            }
        elif lat > mid_lat and lon >= mid_lon:
            # Quadrant 2: Mostly Dry
            return {
                'Dry_Demand': math.ceil(demand * 0.7),
                'Chilled_Demand': math.ceil(demand * 0.2),
                'Frozen_Demand': math.ceil(demand * 0.1)
            }
        elif lat <= mid_lat and lon < mid_lon:
            # Quadrant 3: Mostly Frozen
            return {
                'Dry_Demand': math.ceil(demand * 0.1),
                'Chilled_Demand': math.ceil(demand * 0.2),
                'Frozen_Demand': math.ceil(demand * 0.7)
            }
        else:
            # Quadrant 4: Mixed (Dry & Chilled)
            return {
                'Dry_Demand': math.ceil(demand * 0.4),
                'Chilled_Demand': math.ceil(demand * 0.4),
                'Frozen_Demand': math.ceil(demand * 0.2)
            }

    customers_data = []
    for cid, coords in geo_coords.items():
        if cid != instance.depot_id:
            d = instance.demands.get(cid, 0)
            dm = demand_func(d, coords[0], coords[1])
            customer = {
                'Customer_ID': str(cid),
                'Latitude': coords[0],
                'Longitude': coords[1],
                'Dry_Demand': dm['Dry_Demand'],
                'Chilled_Demand': dm['Chilled_Demand'],
                'Frozen_Demand': dm['Frozen_Demand']
            }
            customers_data.append(customer)

    params = _create_base_params(instance)
    # Assume the expected vehicles is roughly the same as original
    params.expected_vehicles = instance.num_vehicles
    
    # Vehicle can handle all compartments since we have multiple goods now
    params.vehicles = {
        'CVRP_Spatial': {
            'capacity': instance.capacity,
            'fixed_cost': 1000,
            'compartments': {'Dry': True, 'Chilled': True, 'Frozen': True}
        }
    }
    
    return pd.DataFrame(customers_data), params

def _create_customer_data(instance, demand_func) -> List[Dict]:
    converter = CoordinateConverter(instance.coordinates)
    geo_coords = converter.convert_all_coordinates(instance.coordinates)
    
    customers_data = []
    for cust_id, coords in geo_coords.items():
        if cust_id != instance.depot_id:
            base = {
                'Customer_ID': str(cust_id),
                'Latitude': coords[0],
                'Longitude': coords[1],
                'Dry_Demand': 0,
                'Chilled_Demand': 0,
                'Frozen_Demand': 0
            }
            base.update(demand_func(instance.demands.get(cust_id, 0)))
            customers_data.append(base)
            
    return customers_data

def _create_base_params(instance) -> Parameters:
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
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='Convert CVRP instance to FSM and optimize')
    parser.add_argument('--instance', 
                       default='X-n106-k14',   
                       nargs='+', 
                       help='Name of instance file(s) without extension.')
    parser.add_argument('--format',
                       default='excel',
                       choices=['excel', 'json'],
                       help='Output format for results')
    parser.add_argument('--benchmark-type',
                       choices=['normal', 'split', 'scaled', 'combined', 'spatial'],
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
        print("  - Single instance, single good")
        print("split:")
        print("  - Single instance, split demand across multiple goods")
        print("scaled:")
        print("  - Single instance scaled by number of goods")
        print("combined:")
        print("  - Multiple instances combined, each representing a good")
        print("spatial:")
        print("  - Single instance with product mix varying by geographic quadrant")
        
        return
        
    if not args.benchmark_type:
        parser.error("argument --benchmark-type is required")
    if not args.instance:
        parser.error("argument --instance is required")

    start_time = time.time()
    
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
