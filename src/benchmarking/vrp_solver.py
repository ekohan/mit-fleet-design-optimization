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
from joblib import Parallel, delayed
import logging

from src.config.parameters import Parameters
from src.utils.logging import Colors, Symbols
from src.utils.route_time import estimate_route_time

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
        self.route_time_estimation = params.clustering['route_time_estimation']
        self.model = self._prepare_model()
    
    def _prepare_model(self) -> Model:
        """Prepare PyVRP model from customer data."""
        # Track which clients belong to which product type
        self.client_products = []  # Store product type for each client
        expanded_clients = []
        
        for _, row in self.customers.iterrows():
            for good in self.params.goods:
                demand = row[f'{good}_Demand']
                if demand > 0:
                    expanded_clients.append(Client(
                        x=int(row['Latitude'] * 10000),
                        y=int(row['Longitude'] * 10000),
                        delivery=[int(demand)],
                        service_duration=self.params.service_time * 60
                    ))
                    self.client_products.append(good)
        
        # Print key parameters
        print("\nProblem Parameters:")
        print(f"Number of customers: {len(self.customers)}")
        print(f"Service time per customer: {self.params.service_time} minutes")
        print(f"Max route time: {self.params.max_route_time} hours")
        print(f"Average speed: {self.params.avg_speed} km/h")
        
        # Create vehicle types with proper capacities
        vehicle_types = []
        for vt_name, vt_info in self.params.vehicles.items():
            vehicle_types.append(VehicleType(
                num_available=len(expanded_clients),  # Allow maximum flexibility
                capacity=[vt_info['capacity']],
                fixed_cost=vt_info['fixed_cost'],
                max_duration=self.params.max_route_time * 3600  # hours to seconds
            ))
        
        # Print demand analysis
        print(f"\nDemand Analysis:")
        print(f"Total delivery points: {len(expanded_clients)}")
        for good in self.params.goods:
            demands = self.customers[f'{good}_Demand']
            customers_with_demand = demands[demands > 0]
            min_capacity = min(vt['capacity'] for vt in self.params.vehicles.values())
            print(f"\n{good}:")
            print(f"Customers requiring {good}: {len(customers_with_demand)}")
            print(f"Total demand: {demands.sum():,.0f} kg")
            print(f"Average demand per customer: {customers_with_demand.mean():.1f} kg")
            print(f"Min demand: {customers_with_demand.min():.1f} kg")
            print(f"Max demand: {customers_with_demand.max():.1f} kg")
            print(f"Minimum vehicles needed (capacity): {np.ceil(demands.sum() / min_capacity):.0f}")
            print(f"Minimum vehicles needed (time): {np.ceil(len(customers_with_demand) * self.params.service_time / (self.params.max_route_time * 60)):.0f}")
        
        # Calculate base distance matrix
        base_distance_matrix = self._calculate_distance_matrix(len(expanded_clients))
        
        # Convert distances to durations (in seconds)
        duration_matrix = (base_distance_matrix / self.params.avg_speed) * 3600
        
        # Create problem data
        self.data = ProblemData(
            clients=expanded_clients,
            depots=[Depot(
                x=int(self.params.depot['latitude'] * 10000),
                y=int(self.params.depot['longitude'] * 10000)
            )],
            vehicle_types=vehicle_types,
            distance_matrices=[base_distance_matrix],
            duration_matrices=[duration_matrix]
        )
        
        return Model.from_data(self.data)
    
    def _calculate_distance_matrix(self, n_clients: int) -> np.ndarray:
        """Calculate distance matrix for expanded client list."""
        distance_matrix = np.zeros((n_clients + 1, n_clients + 1))  # +1 for depot
        
        # Add depot coordinates at index 0
        depot_coords = (self.params.depot['latitude'], self.params.depot['longitude'])
        
        # Calculate distances
        client_idx = 0
        for _, row in self.customers.iterrows():
            client_coords = (row['Latitude'], row['Longitude'])
            # Depot to/from client
            dist = haversine(depot_coords, client_coords)
            for good in self.params.goods:
                if row[f'{good}_Demand'] > 0:
                    client_idx += 1
                    distance_matrix[0, client_idx] = dist
                    distance_matrix[client_idx, 0] = dist
        
        # Calculate client-to-client distances
        for i in range(1, n_clients + 1):
            for j in range(i + 1, n_clients + 1):
                dist = distance_matrix[i, 0]  # If same location, distance is 0
                if dist > 0:
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def solve_parallel(self, verbose: bool = False) -> Dict[str, VRPSolution]:
        """Solve VRP instances for each product type in parallel."""
        # Split customers by product type
        product_instances = []
        for good in self.params.goods:
            mask = self.customers[f'{good}_Demand'] > 0
            if mask.any():
                # Create a copy with only this product's demand
                product_customers = self.customers.copy()
                # Zero out other products' demands
                for other_good in self.params.goods:
                    if other_good != good:
                        product_customers[f'{other_good}_Demand'] = 0
                product_instances.append((good, product_customers))
        
        # Solve in parallel using existing solve method
        solutions = Parallel(n_jobs=-1)(
            delayed(VRPSolver(
                customers=customers,
                params=self.params,
                time_limit=self.time_limit
            ).solve)(verbose=verbose)
            for _, customers in product_instances
        )
        
        return {
            product: solution 
            for (product, _), solution in zip(product_instances, solutions)
        }
    
    def solve(self, verbose: bool = False) -> VRPSolution:
        """Solve the VRP instance."""
        # Create genetic algorithm parameters
        ga_params = GeneticAlgorithmParams(
            repair_probability=0.8,
            nb_iter_no_improvement=20000
        )
        
        # Create population parameters
        pop_params = PopulationParams(
            min_pop_size=25,
            generation_size=40,
            nb_elite=4,
            nb_close=5,
            lb_diversity=0.1,
            ub_diversity=0.5
        )
        
        # Create stopping criterion
        stop = MaxIterations(max_iterations=5000)
        
        # Solve and return best solution
        result = self.model.solve(
            stop=stop,
            params=SolveParams(genetic=ga_params, population=pop_params),
            display=verbose
        )
        
        # Extract solution details
        solution = result.best
        
        # Get routes
        routes = [list(route) for route in solution.routes()]
        
        # Calculate route times and check feasibility
        feasible_routes = []
        route_times = []
        client_idx = 0
        client_to_customer = {}  # Map expanded client indices to original customer indices
        
        # Build mapping of expanded client indices to original customer indices
        for cust_idx, row in self.customers.iterrows():
            for good in self.params.goods:
                if row[f'{good}_Demand'] > 0:
                    client_to_customer[client_idx] = cust_idx
                    client_idx += 1

        for route in routes:
            if len(route) > 1:  # Skip empty routes
                # Get customers in this route (excluding depot)
                route_customers = pd.DataFrame([
                    {
                        'Latitude': self.customers.iloc[client_to_customer[i-1]]['Latitude'],
                        'Longitude': self.customers.iloc[client_to_customer[i-1]]['Longitude']
                    }
                    for i in route[1:]  # Skip depot (index 0)
                ])
                
                # Calculate route time using BHH (returns minutes)
                route_time = estimate_route_time(
                    cluster_customers=route_customers,
                    depot=self.params.depot,
                    service_time=self.params.service_time,
                    avg_speed=self.params.avg_speed,
                    method='BHH'
                )
                
                # Convert route_time from minutes to hours for comparison
                if route_time / 60 <= self.params.max_route_time:  # Both in hours now
                    feasible_routes.append(route)
                    route_times.append(route_time)
        
        if not feasible_routes:
            return VRPSolution(
                total_cost=float('inf'),
                total_distance=float('inf'),
                num_vehicles=0,
                routes=[],
                vehicle_loads=[],
                execution_time=result.runtime,
                solver_status="Infeasible",
                customer_assignments={},
                route_sequences=[],
                vehicle_utilization=[]
            )
            
        # Use feasible_routes for solution
        routes = feasible_routes
        
        # Convert to our solution format
        return VRPSolution(
            total_cost=solution.distance_cost() + solution.duration_cost(),
            total_distance=solution.distance(),
            num_vehicles=len(routes),
            routes=routes,
            vehicle_loads=[
                sum(self.data.clients()[i-1].delivery[0] for i in route if i > 0)
                for route in routes
            ],
            execution_time=result.runtime,
            solver_status="Optimal" if solution.is_feasible else "Infeasible",
            customer_assignments={},  # TODO: Fill this from routes
            route_sequences=[[str(i) for i in route] for route in routes],
            vehicle_utilization=[
                sum(self.data.clients()[i-1].delivery[0] for i in route if i > 0) / self.data.vehicle_types()[0].capacity[0]
                for route in routes
            ]
        )
    
    def _print_solution(
        self,
        total_cost: float,
        total_distance: float,
        num_vehicles: int,
        routes: List[List[int]],
        execution_time: float,
        utilization: List[float]
    ) -> None:
        """Print solution details."""
        print(f"\nℹ️ VRP Solution Summary:")
        print(f"{Colors.BLUE}→ Total Cost: ${Colors.BOLD}{total_cost:,.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}→ Total Distance: {Colors.BOLD}{total_distance:.1f} km{Colors.RESET}")
        print(f"{Colors.BLUE}→ Total Vehicles Used: {Colors.BOLD}{num_vehicles}{Colors.RESET}")
        
        # Count vehicles by product type
        vehicles_by_product = {good: 0 for good in self.params.goods}
        for route in routes:
            if route:  # If route is not empty
                product = self.client_products[route[1]-1]  # Get product type of first client in route (skip depot)
                vehicles_by_product[product] += 1
        
        # Print vehicle breakdown
        print(f"\nVehicles by Product Type:")
        for product, count in vehicles_by_product.items():
            print(f"{Colors.BLUE}→ {product}: {Colors.BOLD}{count}{Colors.RESET}")
        
        print(f"{Colors.BLUE}→ Avg Vehicle Utilization: {Colors.BOLD}{np.mean(utilization)*100:.1f}%{Colors.RESET}")
        print(f"{Colors.BLUE}→ Execution Time: {Colors.BOLD}{execution_time:.1f}s{Colors.RESET}") 