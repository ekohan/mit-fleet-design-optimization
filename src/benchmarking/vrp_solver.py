"""
Single-compartment VRP solver module using PyVRP.
Provides baseline comparison for multi-compartment vehicle solutions.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pyvrp import (
    Model, GeneticAlgorithm, GeneticAlgorithmParams,
    Depot, Client, VehicleType, ProblemData,
    PopulationParams, SolveParams
)
from pyvrp.stop import MaxIterations
from haversine import haversine

from src.config.parameters import Parameters
from src.utils.logging import Colors, Symbols

@dataclass
class VRPSolution:
    """Results from VRP solver."""
    total_cost: float
    total_distance: float
    num_vehicles: int
    routes: List[List[int]]
    vehicle_loads: List[float]
    execution_time: float
    solver_status: str
    customer_assignments: Dict[str, int]  # customer_id -> route_id mapping
    route_sequences: List[List[str]]  # List of customer sequences per route
    vehicle_utilization: List[float]  # Capacity utilization per route

class VRPSolver:
    """Single-compartment VRP solver implementation."""
    
    def __init__(
        self,
        customers: pd.DataFrame,
        params: Parameters,
        time_limit: int = 300
    ):
        self.customers = customers
        self.params = params
        self.time_limit = time_limit
        self.model = self._prepare_model()
    
    def _prepare_model(self) -> Model:
        """Prepare PyVRP model from customer data."""
        # Combine multi-product demands
        total_demands = self.customers[
            [f'{g}_Demand' for g in self.params.goods]
        ].sum(axis=1)
        
        # Get coordinates
        coords = self.customers[['Latitude', 'Longitude']].values
        
        # Calculate distance matrix
        n = len(self.customers)
        distance_matrix = np.zeros((n+1, n+1))  # +1 for depot
        
        # Add depot coordinates at index 0
        depot_coords = np.array([
            [self.params.depot['latitude'], 
             self.params.depot['longitude']]
        ])
        all_coords = np.vstack([depot_coords, coords])
        
        for i in range(n+1):
            for j in range(i+1, n+1):
                dist = haversine(
                    (all_coords[i,0], all_coords[i,1]),
                    (all_coords[j,0], all_coords[j,1])
                )
                distance_matrix[i,j] = distance_matrix[j,i] = dist
        
        # Create vehicle type with realistic capacity
        vehicle_type = VehicleType(
            num_available=len(self.customers),  # Upper bound on number of vehicles
            capacity=[min(v['capacity'] for v in self.params.vehicles.values())]  # Use smallest vehicle capacity
        )
        
        # Create depot and client objects
        depot = Depot(
            x=int(self.params.depot['latitude'] * 10000),
            y=int(self.params.depot['longitude'] * 10000)
        )
        
        clients = [
            Client(
                x=int(row['Latitude'] * 10000),
                y=int(row['Longitude'] * 10000),
                delivery=[int(total_demands.iloc[idx])],  # Use actual demand
                service_time=self.params.service_time * 60  # Convert minutes to seconds
            )
            for idx, (_, row) in enumerate(self.customers.iterrows())
        ]
        
        # Create problem data with time windows and duration constraints
        self.data = ProblemData(
            clients=clients,
            depots=[depot],
            vehicle_types=[vehicle_type],
            distance_matrices=[distance_matrix],
            duration_matrices=[distance_matrix / self.params.avg_speed],  # Convert distance to duration
            release_times=[0],  # Start time
            due_dates=[self.params.max_route_time * 3600]  # Convert hours to seconds
        )
        
        return Model.from_data(self.data)
    
    def solve(self, verbose: bool = False) -> VRPSolution:
        """Solve the VRP instance."""
        # Create genetic algorithm parameters
        ga_params = GeneticAlgorithmParams(
            repair_probability=0.8,  # Default value
            nb_iter_no_improvement=20000  # Default value
        )
        
        # Create population parameters
        pop_params = PopulationParams(
            min_pop_size=25,  # Default value
            generation_size=40,  # Default value
            nb_elite=4,  # Default value
            nb_close=5,  # Default value
            lb_diversity=0.1,  # Default value
            ub_diversity=0.5  # Default value
        )
        
        # Create stopping criterion (e.g., max iterations)
        stop = MaxIterations(max_iterations=1000)
        
        # Solve and return best solution
        result = self.model.solve(
            stop=stop,
            params=SolveParams(),
            display=verbose
        )
        
        # Extract solution details
        solution = result.best
        
        # Convert to our solution format
        return VRPSolution(
            total_cost=solution.distance_cost() + solution.duration_cost(),
            total_distance=solution.distance(),
            num_vehicles=len(solution.routes()),
            routes=[list(route) for route in solution.routes()],
            vehicle_loads=[
                sum(self.data.clients()[i-1].delivery[0] for i in route if i > 0)  # Adjust index by -1 and skip depot
                for route in solution.routes()
            ],
            execution_time=result.runtime,
            solver_status="Optimal" if solution.is_feasible else "Infeasible",
            customer_assignments={},  # TODO: Fill this from routes
            route_sequences=[[str(i) for i in route] for route in solution.routes()],
            vehicle_utilization=[
                sum(self.data.clients()[i-1].delivery[0] for i in route if i > 0) / self.data.vehicle_types()[0].capacity[0]
                for route in solution.routes()
            ]
        )
    
    def _print_solution(
        self,
        total_cost: float,
        total_distance: float,
        num_vehicles: int,
        execution_time: float,
        utilization: List[float]
    ) -> None:
        """Print solution details."""
        print(f"\n{Symbols.INFO} VRP Solution Summary:")
        print(f"{Colors.BLUE}→ Total Cost: ${Colors.BOLD}{total_cost:,.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}→ Total Distance: {Colors.BOLD}{total_distance:.1f} km{Colors.RESET}")
        print(f"{Colors.BLUE}→ Vehicles Used: {Colors.BOLD}{num_vehicles}{Colors.RESET}")
        print(f"{Colors.BLUE}→ Avg Vehicle Utilization: {Colors.BOLD}{np.mean(utilization)*100:.1f}%{Colors.RESET}")
        print(f"{Colors.BLUE}→ Execution Time: {Colors.BOLD}{execution_time:.1f}s{Colors.RESET}") 