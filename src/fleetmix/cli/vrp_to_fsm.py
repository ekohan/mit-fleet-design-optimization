"""
Unified CLI tool for converting CVRP or MCVRP instances to FSM and optimizing.
"""
from __future__ import annotations
import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on PYTHONPATH when executed as script
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fleetmix.utils.logging import setup_logging
from fleetmix.pipeline.vrp_interface import VRPType, convert_to_fsm, run_optimization
from fleetmix.benchmarking.converters.cvrp import CVRPBenchmarkType
from fleetmix.cli.convert_mcvrp_to_fsm import DEFAULT_INSTANCE as DEFAULT_MCVRP_INSTANCE
from fleetmix.utils.save_results import save_optimization_results


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified VRP-to-FSM Conversion Tool"
    )
    p.add_argument(
        "--vrp-type",
        choices=[t.value for t in VRPType],
        help="VRP type: 'cvrp' or 'mcvrp'",
    )
    p.add_argument(
        "--instance",
        nargs='+',
        default=[DEFAULT_MCVRP_INSTANCE],
        help=(
            "Instance name(s) (without extension). For mcvrp, stems under datasets/mcvrp; "
            "for cvrp, stems under datasets/cvrp."
        ),
    )
    p.add_argument(
        "--benchmark-type",
        choices=[t.value for t in CVRPBenchmarkType],
        help="Benchmark type (CVRP only): normal, split, scaled, combined",
    )
    p.add_argument(
        "--num-goods",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of goods (CVRP only)",
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
    print("\nUnified VRP-to-FSM Conversion Tool")
    print("=" * 80)
    print("Supported VRP types:")
    print("  mcvrp: Multicompartment VRP (3 goods)")
    print(
        "    Example: python src/benchmarking/vrp_to_fsm.py \
        --vrp-type mcvrp --instance 10_3_3_3_(01)"
    )
    print("  cvrp: Classic VRP (2 or 3 goods) with benchmark variants")
    print(
        "    Example: python src/benchmarking/vrp_to_fsm.py \
        --vrp-type cvrp --instance X-n106-k14 --benchmark-type normal"
    )
    print("\nUse --help to see all available options.")


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    parser = _build_parser()
    args = parser.parse_args()

    if args.info:
        _print_info()
        return

    if not args.vrp_type:
        parser.error("argument --vrp-type is required")

    vrp_type = VRPType(args.vrp_type)
    instances = args.instance

    try:
        if vrp_type == VRPType.CVRP:
            if not args.benchmark_type:
                parser.error("argument --benchmark-type is required for CVRP")
            benchmark_type = CVRPBenchmarkType(args.benchmark_type)
            customers_df, params = convert_to_fsm(
                vrp_type,
                instance_names=instances,
                benchmark_type=benchmark_type,
                num_goods=args.num_goods,
            )
            instance_stub = "_".join(instances)
            filename_stub = f"vrp_{vrp_type.value}_{instance_stub}_{benchmark_type.value}"
        else:  # MCVRP
            instance = instances[0]
            # Path resolution for MCVRP is handled here as it's simpler (always one file)
            # We use FileExistsError for CVRP files since we do explicit check above
            # The actual FileNotFoundError for MCVRP will be raised by mcvrp.py if file doesn't exist
            instance_path = Path(__file__).parent.parent / "benchmarking" / "datasets" / "mcvrp" / f"{instance}.dat"
            customers_df, params = convert_to_fsm(
                vrp_type,
                instance_path=instance_path, # mcvrp_to_fsm expects instance_path
            )
            filename_stub = f"vrp_{vrp_type.value}_{instance}"
    except FileNotFoundError as e:
        parser.error(str(e))
        return # Should not be strictly necessary as parser.error exits, but good for clarity

    results_dir = Path(__file__).parent.parent.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    ext = "xlsx" if args.format == "excel" else "json"
    results_path = results_dir / f"{filename_stub}.{ext}"

    start_time = time.time() # Start timer for execution_time
    solution, configs_df = run_optimization(
        customers_df=customers_df,
        params=params,
        verbose=args.verbose,
    )

    save_optimization_results(
        execution_time=time.time() - start_time,
        solver_name=solution["solver_name"],
        solver_status=solution["solver_status"],
        solver_runtime_sec=solution["solver_runtime_sec"],
        post_optimization_runtime_sec=solution["post_optimization_runtime_sec"],
        configurations_df=configs_df, # Pass configs_df received from run_optimization
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
        is_benchmark=True, # All calls from this CLI are considered benchmarks
        expected_vehicles=params.expected_vehicles,
    )

    logger.info("Saved results to %s", results_path)

if __name__ == "__main__":
    main() 