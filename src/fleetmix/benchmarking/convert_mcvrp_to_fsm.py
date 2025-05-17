"""
CLI tool to convert and optimize MCVRP instances (3 compartments) to FSM format.
Includes expected vehicles in output summary and saved results.
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Ensure project root is on PYTHONPATH when executed as script
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from fleetmix.utils.logging import setup_logging
from fleetmix.benchmarking.mcvrp_to_fsm import convert_mcvrp_to_fsm
from fleetmix.utils.vehicle_configurations import generate_vehicle_configurations
from fleetmix.clustering import generate_clusters_for_configurations
from fleetmix.main import solve_fsm_problem
from fleetmix.utils.save_results import save_optimization_results

DEFAULT_INSTANCE = "10_3_3_3_(01)"

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert an MCVRP instance (3 compartments) to FSM and optimize"
    )
    p.add_argument(
        "--instance",
        default=DEFAULT_INSTANCE,
        help="Instance stem under src/benchmarking/datasets/mcvrp/ (omit .dat)",
    )
    p.add_argument(
        "--format",
        default="excel",
        choices=["excel", "json"],
        help="Output file format",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose solver output",
    )
    p.add_argument(
        "--info",
        action="store_true",
        help="Show tool information and exit",
    )
    return p

def _print_info() -> None:
    print("\nMCVRP-to-FSM Conversion Tool")
    print("=" * 80)
    print("• Fixed to 3 product types: Dry, Chilled, Frozen")
    print("• No scaling / splitting logic – 1:1 mapping to FSM")
    print("\nExample:")
    print(
        "python src/benchmarking/convert_mcvrp_to_fsm.py "
        f"--instance {DEFAULT_INSTANCE}"
    )
    print("\nAvailable instances:")
    inst_dir = Path(__file__).parent / "datasets" / "mcvrp"
    for p in sorted(inst_dir.glob("*.dat")):
        print(f"  {p.stem}")
    print("=" * 80)

def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    args = _build_arg_parser().parse_args()

    if args.info:
        _print_info()
        return

    # Convert the instance
    instance_path = (
        Path(__file__).parent   
        / "datasets"
        / "mcvrp"
        / f"{args.instance}.dat"
    )
    logger.info("Converting instance %s", instance_path.name)
    customers_df, params = convert_mcvrp_to_fsm(instance_path)

    # Generate vehicle configurations & clusters
    configs_df = generate_vehicle_configurations(params.vehicles, params.goods)
    clusters_df = generate_clusters_for_configurations(
        customers=customers_df,
        configurations_df=configs_df,
        params=params,
    )

    # Solve FSM optimization
    start_time = time.time()
    solution = solve_fsm_problem(
        clusters_df=clusters_df,
        configurations_df=configs_df,
        customers_df=customers_df,
        parameters=params,
        verbose=args.verbose,
    )

    # Console output
    print("\nOptimization Results:")
    print(f"Total Cost: ${solution['total_cost']:,.2f}")
    print(f"Vehicles Used: {sum(solution['vehicles_used'].values())}")
    print(f"Expected Vehicles: {params.expected_vehicles}")

    # Save results
    file_stub = f"mcvrp_{args.instance}"
    results_dir = Path(__file__).parent.parent.parent / "results"
    results_path = (
        results_dir / f"{file_stub}.{ 'xlsx' if args.format=='excel' else 'json'}"
    )
    params.demand_file = file_stub

    save_optimization_results(
        execution_time=time.time() - start_time,
        solver_name=solution["solver_name"],
        solver_status=solution["solver_status"],
        solver_runtime_sec=solution["solver_runtime_sec"],
        post_optimization_runtime_sec=solution["post_optimization_runtime_sec"],
        configurations_df=configs_df,
        selected_clusters=solution["selected_clusters"],
        total_fixed_cost=solution["total_fixed_cost"],
        total_variable_cost=solution["total_variable_cost"],
        total_light_load_penalties=solution["total_light_load_penalties"],
        total_compartment_penalties=solution["total_compartment_penalties"],
        total_penalties=solution["total_penalties"],
        vehicles_used=solution["vehicles_used"],
        missing_customers=solution["missing_customers"],
        parameters=params,
        filename=results_path,
        format=args.format,
        is_benchmark=True,
        expected_vehicles=params.expected_vehicles,
    )
    logger.info("Saved results to %s", results_path)

if __name__ == "__main__":
    main() 