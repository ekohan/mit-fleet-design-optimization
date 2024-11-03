"""
Main module for the vehicle routing optimization problem.
"""
import time
from utils.logging import setup_logging, ProgressTracker, Colors, Symbols
from utils.data_processing import load_customer_demand
from utils.config_utils import generate_vehicle_configurations
from utils.save_results import save_optimization_results
from clustering import generate_clusters_for_configurations
from fsm_optimizer import solve_fsm_problem
from config import VEHICLE_TYPES, GOODS, DEPOT

def main():
    """Run the FSM optimization pipeline."""
    setup_logging()
    
    # Define optimization steps
    steps = [
        'Load Data',
        'Generate Configs',
        'Create Clusters',
        'Optimize Fleet',
        'Save Results'
    ]
    
    progress = ProgressTracker(steps)
    start_time = time.time()

    # Step 1: Load customer data
    customers = load_customer_demand()
    progress.advance(f"Loaded {Colors.BOLD}{len(customers)}{Colors.RESET} customers")

    # Step 2: Generate vehicle configurations
    configs_df = generate_vehicle_configurations(VEHICLE_TYPES, GOODS)
    progress.advance(f"Generated {Colors.BOLD}{len(configs_df)}{Colors.RESET} vehicle configurations")

    # Step 3: Generate clusters
    clusters_df = generate_clusters_for_configurations(
        customers=customers,
        configurations_df=configs_df,
        goods=GOODS,
        depot=DEPOT
    )
    progress.advance(f"Created {Colors.BOLD}{len(clusters_df)}{Colors.RESET} clusters")

    # Step 4: Solve optimization problem
    solution = solve_fsm_problem(
        clusters_df=clusters_df,
        configurations_df=configs_df,
        customers_df=customers,
        verbose=True
    )
    progress.advance(
        f"Optimized fleet: {Colors.BOLD}${solution['total_fixed_cost'] + solution['total_variable_cost']:,.2f}{Colors.RESET} total cost"
    )

    # Step 5: Save results
    save_optimization_results(
        execution_time=time.time() - start_time,
        solver_name=solution['solver_name'],
        solver_status=solution['solver_status'],
        configurations_df=configs_df,
        selected_clusters=solution['selected_clusters'],
        total_fixed_cost=solution['total_fixed_cost'],
        total_variable_cost=solution['total_variable_cost'],
        vehicles_used=solution['vehicles_used'],
        missing_customers=solution['missing_customers']
    )
    progress.advance(f"Results saved {Colors.GRAY}(execution time: {time.time() - start_time:.1f}s){Colors.RESET}")
    progress.close()

if __name__ == "__main__":
    main() 