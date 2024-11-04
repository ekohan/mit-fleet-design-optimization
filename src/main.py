"""
Main module for the vehicle routing optimization problem.
"""
import time
import os
from datetime import datetime
import pandas as pd

# Add this section to define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

from utils.logging import setup_logging, ProgressTracker, Colors, Symbols
from utils.data_processing import load_customer_demand
from utils.config_utils import generate_vehicle_configurations
from utils.save_results import save_optimization_results
from clustering import generate_clusters_for_configurations
from fsm_optimizer_gurobipy import solve_fsm_problem

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

    # Process solution
    if solution['solver_status'] == 'Optimal':
        print(f"✓ Optimized fleet: ${solution['total_cost']:.2f} total cost")
        
        # Print results
        print(f"\n=== Optimization Results ===")
        print(f"✓ Total Cost: ${solution['total_cost']:.2f}")
        print(f"✓ Execution Time: {time.time() - start_time:.1f}s")

        # Calculate and print statistics
        if not solution['selected_clusters'].empty:
            customers_per_cluster = solution['selected_clusters']['Customers'].apply(len)
            
            # Calculate load percentages
            load_percentages = []
            for _, cluster in solution['selected_clusters'].iterrows():
                total_demand = sum(cluster['Total_Demand'].values())
                vehicle_type = solution['vehicles_used'].index[0]  # Assuming one vehicle type for simplicity
                capacity = configs_df.loc[vehicle_type, 'Capacity']
                load_percentages.append((total_demand / capacity) * 100)
            load_percentages = pd.Series(load_percentages)

            print("\n=== Cluster Statistics ===")
            print(f"Customers per Cluster:")
            print(f"  Min: {customers_per_cluster.min():.0f}")
            print(f"  Max: {customers_per_cluster.max():.0f}")
            print(f"  Avg: {customers_per_cluster.mean():.1f}")
            print(f"  Median: {customers_per_cluster.median():.1f}")
            
            print(f"\nTruck Load Percentages:")
            print(f"  Min: {load_percentages.min():.1f}%")
            print(f"  Max: {load_percentages.max():.1f}%")
            print(f"  Avg: {load_percentages.mean():.1f}%")
            print(f"  Median: {load_percentages.median():.1f}%")

        # Save results
        results_file = os.path.join(RESULTS_DIR, f"optimization_results_{datetime.now():%Y%m%d_%H%M%S}.xlsx")
        save_optimization_results(
            execution_time=time.time() - start_time,
            solver_name=solution['solver_name'],
            solver_status=solution['solver_status'],
            configurations_df=configs_df,
            selected_clusters=solution['selected_clusters'],
            total_fixed_cost=solution['total_fixed_cost'],
            total_variable_cost=solution['total_variable_cost'],
            vehicles_used=solution['vehicles_used'],
            missing_customers=set(solution['missing_customers'])  # Convert list to set
        )

        print("\n🚀 Optimization completed!")

    else:
        print(f"✗ No optimal solution found. Status: {solution['solver_status']}")

    progress.advance(f"Results saved {Colors.GRAY}(execution time: {time.time() - start_time:.1f}s){Colors.RESET}")
    progress.close()

if __name__ == "__main__":
    main() 