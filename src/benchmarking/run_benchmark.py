"""
Benchmark runner script for single-compartment VRP solutions.
"""
import sys
from pathlib import Path
import time
import argparse

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.config.parameters import Parameters
from src.utils.logging import setup_logging, ProgressTracker, Colors, Symbols
from src.utils.data_processing import load_customer_demand
from src.benchmarking.vrp_solver import VRPSolver

def parse_benchmark_args():
    """Parse command line arguments for benchmarking."""
    parser = argparse.ArgumentParser(
        description='Run VRP benchmarking'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/config/default_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=300,
        help='Time limit for VRP solver in seconds'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    return parser.parse_args()

def print_solution_details(solution):
    """Print VRP solution details to screen."""
    print(f"\n{Symbols.INFO} VRP Solution Summary:")
    print(f"{Colors.BLUE}→ Total Cost: ${Colors.BOLD}{solution.total_cost:,.2f}{Colors.RESET}")
    print(f"{Colors.BLUE}→ Total Distance: {Colors.BOLD}{solution.total_distance:.1f} km{Colors.RESET}")
    print(f"{Colors.BLUE}→ Vehicles Used: {Colors.BOLD}{solution.num_vehicles}{Colors.RESET}")
    print(f"{Colors.BLUE}→ Avg Vehicle Utilization: {Colors.BOLD}{sum(solution.vehicle_utilization)/len(solution.vehicle_utilization)*100:.1f}%{Colors.RESET}")
    print(f"{Colors.BLUE}→ Solution Status: {Colors.BOLD}{solution.solver_status}{Colors.RESET}")
    print(f"{Colors.BLUE}→ Execution Time: {Colors.BOLD}{solution.execution_time:.1f}s{Colors.RESET}")

def main():
    """Run VRP benchmarking."""
    args = parse_benchmark_args()
    setup_logging()
    
    # Load parameters
    params = Parameters.from_yaml(args.config)
    
    # Define benchmark steps
    steps = [
        'Load Data',
        'Run VRP Solver'
    ]
    
    progress = ProgressTracker(steps)
    
    # Step 1: Load customer data
    customers = load_customer_demand(params.demand_file)
    progress.advance(
        f"Loaded {Colors.BOLD}{len(customers)}{Colors.RESET} customers"
    )
    
    # Step 2: Run VRP solver
    vrp_solver = VRPSolver(
        customers=customers,
        params=params,
        time_limit=args.time_limit
    )
    solution = vrp_solver.solve(verbose=args.verbose)
    progress.advance(
        f"Completed VRP: {Colors.BOLD}${solution.total_cost:,.2f}{Colors.RESET} "
        f"total cost, {Colors.BOLD}{solution.num_vehicles}{Colors.RESET} vehicles"
    )
    
    # Print detailed results
    print_solution_details(solution)
    progress.close()

if __name__ == "__main__":
    main() 