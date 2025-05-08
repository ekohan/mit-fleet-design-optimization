from enum import Enum
from pathlib import Path
import time
import pandas as pd

from src.benchmarking.mcvrp_to_fsm import convert_mcvrp_to_fsm
from src.benchmarking.cvrp_converter import convert_cvrp_to_fsm, CVRPBenchmarkType
from src.config.parameters import Parameters
from src.utils.vehicle_configurations import generate_vehicle_configurations
from src.clustering import generate_clusters_for_configurations
from src.main import solve_fsm_problem

class VRPType(Enum):
    CVRP = 'cvrp'
    MCVRP = 'mcvrp'


def convert_to_fsm(vrp_type: VRPType, **kwargs) -> tuple[pd.DataFrame, Parameters]:
    """
    Library facade to convert VRP instances to FSM format.
    """
    if vrp_type == VRPType.MCVRP:
        return convert_mcvrp_to_fsm(kwargs["instance_path"])
    elif vrp_type == VRPType.CVRP:
        return convert_cvrp_to_fsm(
            instance_names=kwargs["instance_names"],
            benchmark_type=kwargs["benchmark_type"],
            num_goods=kwargs.get("num_goods", 3),
            split_ratios=kwargs.get("split_ratios")
        )
    else:
        raise ValueError(f"Unsupported VRP type: {vrp_type}")



def run_optimization(
    customers_df: pd.DataFrame,
    params: Parameters,
    verbose: bool = False
) -> tuple[dict, pd.DataFrame]:
    """
    Run the common FSM optimization pipeline.
    Returns the solution dictionary and the configurations DataFrame.
    """
    # Generate vehicle configurations and clusters
    configs_df = generate_vehicle_configurations(params.vehicles, params.goods)
    clusters_df = generate_clusters_for_configurations(
        customers=customers_df,
        configurations_df=configs_df,
        params=params
    )

    # Solve FSM optimization
    solution = solve_fsm_problem(
        clusters_df=clusters_df,
        configurations_df=configs_df,
        customers_df=customers_df,
        parameters=params,
        verbose=verbose,
    )

    # Console output
    print("\nOptimization Results:")
    print(f"Total Cost: ${solution['total_cost']:,.2f}")
    print(f"Vehicles Used: {sum(solution['vehicles_used'].values())}")
    print(f"Expected Vehicles: {params.expected_vehicles}")

    return solution, configs_df 