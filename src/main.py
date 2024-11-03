"""
Main module for the vehicle routing optimization problem.
"""
import logging
from utils.data_processing import load_customer_demand
from utils.config_utils import generate_vehicle_configurations, print_configurations
from utils.save_results import save_optimization_results
from clustering import generate_clusters_for_configurations
from fsm_optimizer import solve_fsm_problem
from config import VEHICLE_TYPES, GOODS, DEPOT

def main():
    """Run the FSM optimization pipeline."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Step 1: Load customer data
    logger.info("Loading customer data...")
    customers = load_customer_demand()

    # Step 2: Generate vehicle configurations
    logger.info("Generating vehicle configurations...")
    configs_df = generate_vehicle_configurations(VEHICLE_TYPES, GOODS)
    # print_configurations(configs_df, GOODS)

    # Step 3: Generate clusters
    logger.info("Generating clusters...")
    clusters_df = generate_clusters_for_configurations(
        customers=customers,
        configurations_df=configs_df,
        goods=GOODS,
        depot=DEPOT
    )

    # Step 4: Solve optimization problem
    logger.info("Solving optimization problem...")
    # TODO: revisar lo del profiling
    solution = solve_fsm_problem(
        clusters_df=clusters_df,
        configurations_df=configs_df,
        customers_df=customers,
        verbose=True
    )

    # Step 5: Save results
    save_optimization_results(
        execution_time=solution['execution_time'],
        solver_name=solution['solver_name'],
        solver_status=solution['solver_status'],
        configurations_df=configs_df,
        selected_clusters=solution['selected_clusters'],
        total_fixed_cost=solution['total_fixed_cost'],
        total_variable_cost=solution['total_variable_cost'],
        vehicles_used=solution['vehicles_used'],
        missing_customers=solution['missing_customers']
    )

if __name__ == "__main__":
    main() 