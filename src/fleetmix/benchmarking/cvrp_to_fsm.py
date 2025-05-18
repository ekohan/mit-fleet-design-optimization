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
from fleetmix.benchmarking.parsers.cvrp_parser import CVRPParser
from fleetmix.config.parameters import Parameters
from fleetmix.utils.logging import setup_logging
from fleetmix.main import solve_fsm_problem
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.clustering import generate_clusters_for_configurations
from fleetmix.utils.coordinate_converter import CoordinateConverter, GeoBounds
from fleetmix.utils.save_results import save_optimization_results
from fleetmix.benchmarking.converters.cvrp import convert_cvrp_to_fsm, CVRPBenchmarkType

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
        instance_dir = Path(__file__).parent / 'datasets' / 'cvrp'
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
        solver_runtime_sec=solution['solver_runtime_sec'],
        post_optimization_runtime_sec=solution['post_optimization_runtime_sec'],
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
        is_benchmark=True,
        expected_vehicles=params.expected_vehicles
    )

if __name__ == "__main__":
    main() 