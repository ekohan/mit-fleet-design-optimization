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
from src.benchmarking.benchmark_types import BenchmarkType

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
        time_limit: int = 300,
        benchmark_type: BenchmarkType = BenchmarkType.SINGLE_COMPARTMENT
    ):
        self.customers = customers
        self.params = params
        self.time_limit = time_limit
        self.benchmark_type = benchmark_type
        self.route_time_estimation = params.clustering['route_time_estimation']
        self.model = self._prepare_model()
    
    def _prepare_model(self) -> Model:
        """Prepare PyVRP model from customer data."""
        expanded_clients = []
        
        if self.benchmark_type == BenchmarkType.SINGLE_COMPARTMENT:
            # For single product solving, create a client for each product demand
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
        else:
            # For multi-compartment, use total demand
            for _, row in self.customers.iterrows():
                total_demand = sum(row[f'{good}_Demand'] for good in self.params.goods)
                if total_demand > 0:
                    expanded_clients.append(Client(
                        x=int(row['Latitude'] * 10000),
                        y=int(row['Longitude'] * 10000),
                        delivery=[int(total_demand)],
                        service_duration=self.params.service_time * 60
                    ))

        # Create vehicle types with proper capacities
        vehicle_types = [
            VehicleType(
                num_available=len(expanded_clients),
                capacity=[vt_info['capacity']],
                fixed_cost=vt_info['fixed_cost'],
                max_duration=self.params.max_route_time * 3600
            )
            for vt_name, vt_info in self.params.vehicles.items()
        ]

        # Calculate base distance matrix
        base_distance_matrix = self._calculate_distance_matrix(len(expanded_clients))
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
            dist = haversine(depot_coords, client_coords)
            
            # Check if this customer should be included based on benchmark type
            if self.benchmark_type == BenchmarkType.MULTI_COMPARTMENT:
                total_demand = sum(row[f'{good}_Demand'] for good in self.params.goods)
                if total_demand > 0:
                    client_idx += 1
                    distance_matrix[0, client_idx] = dist
                    distance_matrix[client_idx, 0] = dist
            else:  # SINGLE_COMPARTMENT
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
    
    def solve_scv_parallel(self, verbose: bool = False) -> Dict[str, VRPSolution]:
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
            delayed(self._solve_single_product)(customers, verbose)
            for _, customers in product_instances
        )
        
        return {
            product: solution 
            for (product, _), solution in zip(product_instances, solutions)
        }
    
    def _solve_single_product(self, customers: pd.DataFrame, verbose: bool) -> VRPSolution:
        """Helper method to solve a single product instance."""
        solver = VRPSolver(
            customers=customers,
            params=self.params,
            time_limit=self.time_limit,
            benchmark_type=BenchmarkType.SINGLE_COMPARTMENT
        )
        return solver.solve_scv(verbose=verbose)
    
    def solve_scv(self, verbose: bool = False) -> VRPSolution:
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
        utilization: List[float],
        benchmark_type: BenchmarkType,
        compartment_configs: Optional[List[Dict[str, float]]] = None
    ) -> None:
        """Print solution details."""
        print(f"\nℹ️ VRP Solution Summary:")
        print(f"{Colors.BLUE}→ Total Cost: ${Colors.BOLD}{total_cost:,.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}→ Total Distance: {Colors.BOLD}{total_distance:.1f} km{Colors.RESET}")
        print(f"{Colors.BLUE}→ Total Vehicles Used: {Colors.BOLD}{num_vehicles}{Colors.RESET}")
        print(f"{Colors.BLUE}→ Avg Vehicle Utilization: {Colors.BOLD}{np.mean(utilization)*100:.1f}%{Colors.RESET}")
        print(f"{Colors.BLUE}→ Execution Time: {Colors.BOLD}{execution_time:.1f}s{Colors.RESET}")

        if benchmark_type == BenchmarkType.SINGLE_COMPARTMENT:
            # Count vehicles by product type
            vehicles_by_product = {good: 0 for good in self.params.goods}
            for route in routes:
                if route:  # If route is not empty
                    # Assuming client_products is a list mapping client index to product type
                    product = self.client_products[route[1]-1]  # Get product type of first client in route (skip depot)
                    vehicles_by_product[product] += 1
            
            # Print vehicle breakdown
            print(f"\nVehicles by Product Type:")
            for product, count in vehicles_by_product.items():
                print(f"{Colors.BLUE}→ {product}: {Colors.BOLD}{count}{Colors.RESET}")

        # Print compartment configurations if available
        if compartment_configs:
            print(f"\nCompartment Configurations:")
            for i, config in enumerate(compartment_configs, 1):
                # Calculate total used capacity
                total_used_capacity = sum(config.values())
                # Calculate empty capacity
                empty_capacity = 1.0 - total_used_capacity
                
                print(f"\n{Colors.BLUE}Route {i}:{Colors.RESET}")
                for product, percentage in config.items():
                    if percentage >= 0.01:  # Only show if >= 1%
                        print(f"{Colors.BLUE}{product}: {Colors.BOLD}{percentage*100:.1f}%{Colors.RESET}")
                
                # Always print empty capacity
                print(f"{Colors.BLUE}Empty: {Colors.BOLD}{empty_capacity*100:.1f}%{Colors.RESET}")
    
    def _prepare_multi_compartment_data(self) -> pd.DataFrame:
        """
        Prepare customer data for multi-compartment solving by aggregating demands.
        Also stores original demand breakdown for post-processing.
        """
        # Create a copy of customer data
        mc_customers = self.customers.copy()
        
        # Store original demands per product type for later use
        self.original_demands = {}
        
        # Use index as customer ID if CustomerID column doesn't exist
        id_column = 'CustomerID' if 'CustomerID' in self.customers.columns else self.customers.index
        
        for idx, row in self.customers.iterrows():
            customer_id = str(row[id_column]) if 'CustomerID' in self.customers.columns else str(idx)
            self.original_demands[customer_id] = {
                good: row.get(f'{good}_Demand', 0)
                for good in self.params.goods
            }
        
        # Calculate total demand for each customer
        mc_customers['Total_Demand'] = mc_customers.apply(
            lambda row: sum(row.get(f'{good}_Demand', 0) for good in self.params.goods),
            axis=1
        )
        
        return mc_customers

    def _determine_compartment_configuration(
        self,
        route_customers: List[str],
        vehicle_type_idx: int
    ) -> Dict[str, float]:
        """
        Determine optimal compartment configuration for a route.
        Returns dict mapping product types to their required capacity percentage.
        """
        # Calculate total demand per product type for this route
        route_demands = {good: 0.0 for good in self.params.goods}
        
        # Get the vehicle type from the solution
        vehicle_type = list(self.params.vehicles.values())[vehicle_type_idx]
        total_vehicle_capacity = vehicle_type['capacity']
        
        for customer_id in route_customers:
            for good in self.params.goods:
                route_demands[good] += self.original_demands[customer_id][good]
        
        # Calculate percentage for each product type relative to vehicle capacity
        compartments = {
            good: demand / total_vehicle_capacity
            for good, demand in route_demands.items()
            if demand > 0
        }
        
        return compartments

    def solve_mcv(self, verbose: bool = False) -> Dict[str, VRPSolution]:
        """
        Solve multi-compartment VRP instance.
        Returns a single solution with compartment configurations.
        """
        # Prepare aggregated data
        mc_customers = self._prepare_multi_compartment_data()
        
        # Create and solve VRP with aggregated demands
        solver = VRPSolver(
            customers=mc_customers,
            params=self.params,
            time_limit=self.time_limit,
            benchmark_type=BenchmarkType.MULTI_COMPARTMENT
        )
        
        # Get base solution
        base_solution = solver.solve_scv(verbose=verbose)
        
        # Post-process routes to determine compartment configurations
        route_configurations = []
        customer_assignments = {}
        route_sequences = []
        
        for route_idx, route in enumerate(base_solution.routes):
            if not route:  # Skip empty routes
                continue
            
            # Get customer IDs for this route (excluding depot)
            route_customers = [
                str(mc_customers.index[client_idx - 1])  # Adjust for 0-based indexing
                for client_idx in route[1:]  # Skip depot
            ]
            
            # Get vehicle type index from the solution
            vehicle_type_idx = int(base_solution.vehicle_utilization[route_idx])  # Ensure it's an integer
            
            # Determine compartment configuration using original demands
            compartments = self._determine_compartment_configuration(
                route_customers,
                vehicle_type_idx
            )
            route_configurations.append(compartments)
            
            # Update customer assignments
            for customer_id in route_customers:
                customer_assignments[customer_id] = route_idx
            
            route_sequences.append(route_customers)
        
        # Create multi-compartment solution
        mc_solution = VRPSolution(
            total_cost=base_solution.total_cost,
            total_distance=base_solution.total_distance,
            num_vehicles=base_solution.num_vehicles,
            routes=base_solution.routes,
            vehicle_loads=base_solution.vehicle_loads,
            execution_time=base_solution.execution_time,
            solver_status=base_solution.solver_status,
            customer_assignments=customer_assignments,
            route_sequences=route_sequences,
            vehicle_utilization=base_solution.vehicle_utilization
        )
        
        # Store compartment configurations for later use
        mc_solution.compartment_configurations = route_configurations
        
        if verbose:
            self._print_solution(
                total_cost=mc_solution.total_cost,
                total_distance=mc_solution.total_distance,
                num_vehicles=mc_solution.num_vehicles,
                routes=mc_solution.routes,
                execution_time=mc_solution.execution_time,
                utilization=mc_solution.vehicle_utilization,
                benchmark_type=self.benchmark_type,
                compartment_configs=route_configurations
            )
        
        return {"multi_compartment": mc_solution}
    
    def solve(self, verbose: bool = False) -> Dict[str, VRPSolution]:
        """
        Solve VRP instance based on benchmark type.
        Returns solutions dictionary mapping product types to their solutions.
        """
        if self.benchmark_type == BenchmarkType.SINGLE_COMPARTMENT:
            return self.solve_scv_parallel(verbose=verbose)
        elif self.benchmark_type == BenchmarkType.MULTI_COMPARTMENT:
            return self.solve_mcv(verbose=verbose)
        else:
            raise ValueError(f"Unknown benchmark type: {self.benchmark_type}")