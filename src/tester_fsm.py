"""
Test program for FSM optimization using simplified data.
"""

import logging
import sys
from test_data import generate_test_data
from fsm_optimizer_gurobipy_fortestdata import solve_fsm_problem
from utils.logging import Colors, Symbols
from typing import Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

def print_solution(solution: Dict) -> None:
    """Print the optimization solution in a formatted way."""
    print("\n📊 Solution Summary")
    print("=" * 50)
    
    # Print costs
    print("\nCost Breakdown:")
    print(f"Total Cost:     $ {solution['total_cost']:>9.2f}")
    print(f"Fixed Cost:     $ {solution['total_fixed_cost']:>9.2f}")
    print(f"Variable Cost:  $ {solution['total_variable_cost']:>9.2f}")
    
    # Print vehicle allocation
    print("\nVehicle Allocation:")
    for v_type, count in solution['vehicles_used'].items():
        print(f"→ Type {v_type}:   {count} vehicles")
    
    # Print cluster assignments
    print("\nCluster Assignments:")
    if solution.get('cluster_assignments'):
        for assignment in solution['cluster_assignments']:
            print(f"\nCluster {assignment['cluster_id']}:")
            print(f"  → Vehicle Type: {assignment['vehicle_type']}")
            print(f"  → Total Demand: {assignment['total_demand']} units")
            print(f"  → Route Time: {assignment['route_time']} minutes")
            print(f"  → Customers: {assignment['customers']}")
            print(f"  → Utilization: {(assignment['total_demand'] / assignment['vehicle_capacity'] * 100):.1f}%")
    else:
        print("  No assignments available")

def run_optimization():
    """Run FSM optimization with test data."""
    
    # Generate test data
    logger.info(f"\n{Symbols.ROCKET} Generating Test Data")
    logger.info("=" * 50)
    configs_df, clusters_df, customers_df = generate_test_data()
    
    # Print data summary
    logger.info(f"{Colors.CYAN}Configurations: {Colors.BOLD}{len(configs_df)} vehicles{Colors.RESET}")
    logger.info(f"{Colors.CYAN}Clusters: {Colors.BOLD}{len(clusters_df)} clusters{Colors.RESET}")
    logger.info(f"{Colors.CYAN}Customers: {Colors.BOLD}{len(customers_df)} customers{Colors.RESET}")
    
    # Print cluster details
    logger.info(f"\n{Colors.CYAN}Cluster Details:{Colors.RESET}")
    for idx, cluster in clusters_df.iterrows():
        logger.info(f"\nCluster {idx}:")
        logger.info(f"  Customers: {cluster['Customers']}")
        logger.info(f"  Total Demand: {cluster['Total_Demand']}")
        logger.info(f"  Required Compartments: {cluster['Goods_In_Config']}")
        logger.info(f"  Route Time: {cluster['Route_Time']} minutes")
        logger.info(f"  Location: ({cluster['Centroid_Latitude']}, {cluster['Centroid_Longitude']})")
    
    # Print configuration details
    logger.info(f"\n{Colors.CYAN}Vehicle Configurations:{Colors.RESET}")
    for _, config in configs_df.iterrows():
        logger.info(
            f"  Type {config['Vehicle_Type']}: "
            f"Capacity={config['Capacity']}, "
            f"Cost=${config['Fixed_Cost']}, "
            f"Compartments: Dry={config['Dry']}, "
            f"Chilled={config['Chilled']}, "
            f"Frozen={config['Frozen']}"
        )
    
    # Run optimization
    logger.info(f"\n{Symbols.GEAR} Running Optimization")
    logger.info("=" * 50)
    
    try:
        solution = solve_fsm_problem(
            clusters_df=clusters_df,
            configurations_df=configs_df,
            customers_df=customers_df,
            verbose=True
        )
        
        # Print solution
        print_solution(solution)
        
    except Exception as e:
        logger.error(f"\n{Symbols.CROSS} Optimization Error:")
        logger.error(f"  {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = run_optimization()
    sys.exit(0 if success else 1)