"""
Single-compartment VRP solver module using PyVRP.
Provides baseline comparison for multi-compartment vehicle solutions.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from pyvrp import (
    Model, GeneticAlgorithmParams,
    Depot, Client, VehicleType, ProblemData,
    PopulationParams, SolveParams
)
from pyvrp.stop import MaxIterations
from haversine import haversine
from joblib import Parallel, delayed
import logging

from fleetmix.config.parameters import Parameters
from fleetmix.utils.logging import Colors, Symbols
from fleetmix.utils.route_time import estimate_route_time
from fleetmix.core_types import BenchmarkType, VRPSolution

# Add logging to track utilization
logging.basicConfig(level=logging.WARNING)

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
        client_coords_list = [(0, 0)] * (n_clients + 1) # Store coords (lat, lon) for depot + clients
        
        # Add depot coordinates at index 0
        depot_coords = (self.params.depot['latitude'], self.params.depot['longitude'])
        client_coords_list[0] = depot_coords
        
        # Calculate distances and store client coordinates
        client_idx = 0
        for _, row in self.customers.iterrows():
            coords = (row['Latitude'], row['Longitude'])
            
            # Check if this customer should be included based on benchmark type
            if self.benchmark_type == BenchmarkType.MULTI_COMPARTMENT:
                total_demand = sum(row[f'{good}_Demand'] for good in self.params.goods)
                if total_demand > 0:
                    client_idx += 1
                    dist = haversine(depot_coords, coords)
                    distance_matrix[0, client_idx] = dist
                    distance_matrix[client_idx, 0] = dist
                    client_coords_list[client_idx] = coords # Store coords
            else:  # SINGLE_COMPARTMENT
                for good in self.params.goods:
                    if row[f'{good}_Demand'] > 0:
                        client_idx += 1
                        dist = haversine(depot_coords, coords)
                        distance_matrix[0, client_idx] = dist
                        distance_matrix[client_idx, 0] = dist
                        client_coords_list[client_idx] = coords # Store coords
        
        # Calculate client-to-client distances using stored coordinates
        for i in range(1, n_clients + 1):
            for j in range(i + 1, n_clients + 1):
                coords_i = client_coords_list[i]
                coords_j = client_coords_list[j]
                dist = haversine(coords_i, coords_j)
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
        # Create genetic algorithm parameters with balanced values
        ga_params = GeneticAlgorithmParams(
            repair_probability=0.9,
            nb_iter_no_improvement=10000  # Increased from 2000
        )
        
        # Create population parameters with larger population for better exploration
        pop_params = PopulationParams(
            min_pop_size=40,  # Increased from 25
            generation_size=60,  # Increased from 40
            nb_elite=6,  # Increased from 4
            nb_close=8,  # Increased from 5
            lb_diversity=0.1,
            ub_diversity=0.6  # Increased from 0.5
        )
        
        # Create stopping criterion with more iterations
        stop = MaxIterations(max_iterations=5000)  # Increased from 2000
        
        # Solve and return best solution
        result = self.model.solve(
            stop=stop,
            params=SolveParams(genetic=ga_params, population=pop_params),
            display=verbose
        )
        
        # Extract solution details
        solution = result.best
        
        # Get routes (keep as PyVRP Route objects)
        routes = solution.routes()
        
        # Calculate route times and check feasibility
        feasible_routes = []
        route_times = []
        route_feasibility = []  # Track feasibility of each route
        
        for route_idx, route in enumerate(routes):
            if len(route) <= 1:
                if verbose:
                    print(f"Route {route_idx} skipped: Empty route")
                continue

            # Get total demand for this route
            total_demand = sum(
                self.data.clients()[client - 1].delivery[0] 
                for i in range(1, len(route)-1)
                if (client := route[i]) > 0
            )
            
            # Get vehicle type and capacity
            vehicle_type_idx = route.vehicle_type()
            vehicle_type = self.data.vehicle_types()[vehicle_type_idx]
            vehicle_capacity = vehicle_type.capacity[0]
            
            # Calculate utilization percentage
            utilization = (total_demand / vehicle_capacity) * 100
            
            # Create DataFrame of route customers for time estimation
            route_customers = pd.DataFrame([
                {
                    'Latitude': self.data.clients()[client - 1].x / 10000,
                    'Longitude': self.data.clients()[client - 1].y / 10000
                }
                for i in range(1, len(route)-1)
                if (client := route[i]) > 0
            ])
            
            # Calculate route time
            route_time = estimate_route_time(
                cluster_customers=route_customers,
                depot=self.params.depot,
                service_time=self.params.service_time,
                avg_speed=self.params.avg_speed,
                method='BHH',
                max_route_time=self.params.max_route_time,
                prune_tsp=self.params.prune_tsp
            )
            
            # Check if route is feasible (but include it anyway)
            is_feasible = (utilization <= 100 and route_time <= self.params.max_route_time)
            
            # Log route status
            if not is_feasible:
                if utilization > 100:
                    logging.warning(f"{Colors.RED}Route {route_idx} exceeds capacity (Utilization: {utilization:.1f}%){Colors.RESET}")
                if route_time > self.params.max_route_time:
                    logging.warning(f"{Colors.RED}Route {route_idx} exceeds max time ({route_time:.2f} > {self.params.max_route_time}){Colors.RESET}")
            elif verbose:
                logging.info(f"{Colors.GREEN}Route {route_idx} feasible: Utilization={utilization:.1f}%, Time={route_time:.2f}h{Colors.RESET}")
            
            feasible_routes.append([route[i] for i in range(len(route))])
            route_times.append(route_time)
            route_feasibility.append(is_feasible)
        
        if not feasible_routes:
            return VRPSolution(
                total_cost=float('inf'),
                fixed_cost=0.0,
                variable_cost=0.0,
                total_distance=float('inf'),
                num_vehicles=0,
                routes=[],
                vehicle_loads=[],
                execution_time=result.runtime,
                solver_status="Infeasible",
                route_sequences=[],
                vehicle_utilization=[],
                vehicle_types=[],
                route_times=[],
                route_distances=[],
                route_feasibility=[]
            )
            
        # Use feasible_routes for solution
        routes = feasible_routes
        
        # Calculate costs
        fixed_cost = solution.fixed_vehicle_cost()
        variable_cost = solution.duration_cost() + solution.distance_cost()
        
        # Calculate vehicle utilization correctly for each route
        vehicle_utilizations = []
        vehicle_types = []
        vehicle_loads = []
        route_distances = []
        
        # Process routes before converting to lists
        for route_idx, pyvrp_route in enumerate(solution.routes()):
            if pyvrp_route.visits():  # Skip empty routes
                vehicle_type_idx = pyvrp_route.vehicle_type()
                vehicle_type = self.data.vehicle_types()[vehicle_type_idx]
                vehicle_capacity = vehicle_type.capacity[0]

                # Calculate total load for this route
                load = sum(self.data.clients()[i-1].delivery[0] for i in pyvrp_route.visits())
                
                # Ensure load does not exceed capacity
                if load > vehicle_capacity:
                    logging.warning(f"Route {route_idx} exceeds vehicle capacity: {load}/{vehicle_capacity}")
                    load = vehicle_capacity

                vehicle_loads.append(load)
                vehicle_utilizations.append((load / vehicle_capacity) * 100)  # Store as percentage
                vehicle_types.append(vehicle_type_idx)  # Store the vehicle type index
                route_distances.append(solution.distance())
                
        # Now convert routes to lists for storage
        routes = [[i for i in route] for route in solution.routes()]

        return VRPSolution(
            total_cost=fixed_cost + variable_cost,
            fixed_cost=fixed_cost,
            variable_cost=variable_cost,
            total_distance=solution.distance(),
            num_vehicles=len(feasible_routes),
            routes=feasible_routes,
            vehicle_loads=vehicle_loads,
            execution_time=result.runtime,
            solver_status="Optimal" if solution.is_feasible() else "Infeasible",
            route_sequences=[[str(i) for i in route] for route in feasible_routes],
            vehicle_utilization=vehicle_utilizations,
            vehicle_types=vehicle_types,
            route_times=route_times,
            route_distances=route_distances,
            route_feasibility=route_feasibility  # Add the feasibility information
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
        print(f"\n🚚 VRP Solution Summary:")
        
        # Handle infeasible solutions
        if total_cost == float('inf') or not routes:
            print(f"{Colors.RED}→ Status: INFEASIBLE{Colors.RESET}")
            print(f"{Colors.RED}→ No feasible solution found{Colors.RESET}")
            print(f"{Colors.BLUE}→ Execution Time: {Colors.BOLD}{execution_time:.1f}s{Colors.RESET}")
            return

        # Print solution details for feasible solutions
        print(f"{Colors.BLUE}→ Total Cost: ${Colors.BOLD}{total_cost:,.2f}{Colors.RESET}")
        print(f"{Colors.BLUE}→ Total Distance: {Colors.BOLD}{total_distance:.1f} km{Colors.RESET}")
        print(f"{Colors.BLUE}→ Total Vehicles Used: {Colors.BOLD}{num_vehicles}{Colors.RESET}")
        
        # Only calculate utilization if we have valid routes
        if utilization:
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
        """Solve multi-compartment VRP instance."""
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
        
        # Filter valid routes and their indices
        valid_indices = [i for i, r in enumerate(base_solution.routes) if r and len(r) > 2]
        
        # Calculate compartment configurations for valid routes
        route_configurations = []
        for idx in valid_indices:
            route = base_solution.routes[idx]
            route_customers = [str(mc_customers.index[client - 1]) for client in route[1:]]
            compartments = self._determine_compartment_configuration(
                route_customers,
                base_solution.vehicle_types[idx]
            )
            route_configurations.append(compartments)
        
        # Create filtered solution
        mc_solution = VRPSolution(
            total_cost=base_solution.total_cost,
            fixed_cost=base_solution.fixed_cost,
            variable_cost=base_solution.variable_cost,
            total_distance=base_solution.total_distance,
            num_vehicles=len(valid_indices),
            routes=[base_solution.routes[i] for i in valid_indices],
            vehicle_loads=[base_solution.vehicle_loads[i] for i in valid_indices],
            execution_time=base_solution.execution_time,
            solver_status=base_solution.solver_status,
            route_sequences=[[str(c) for c in base_solution.routes[i][1:]] for i in valid_indices],
            vehicle_utilization=[base_solution.vehicle_utilization[i] for i in valid_indices],
            vehicle_types=[base_solution.vehicle_types[i] for i in valid_indices],
            route_times=base_solution.route_times,  # Keep all route times
            route_distances=[base_solution.route_distances[i] for i in valid_indices],
            route_feasibility=[base_solution.route_feasibility[i] for i in valid_indices]  # Keep all route feasibility
        )
        
        # Store compartment configurations
        mc_solution.compartment_configurations = route_configurations
        
        # Print solution if verbose
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
        
        return {'multi_compartment': mc_solution}
    
    def _print_diagnostic_information(self, customers: pd.DataFrame) -> None:
        """Print diagnostic information about the problem instance."""
        print("\nCustomer Diagnostic Information:")
        print(f"Total customers: {len(customers)}")
        
        # Demand information for each product type
        for good in self.params.goods:
            mask = customers[f'{good}_Demand'] > 0
            demands = customers[f'{good}_Demand']
            print(f"\n{good} demand statistics:")
            print(f"Customers with demand: {mask.sum()}")
            if mask.sum() > 0:
                print(f"Min demand: {demands[demands > 0].min():.2f}")
                print(f"Max demand: {demands.max():.2f}")
                print(f"Mean demand: {demands[demands > 0].mean():.2f}")
                print(f"Total demand: {demands.sum():.2f}")
        
        # Location spread
        print("\nLocation spread:")
        print(f"Latitude range: {customers['Latitude'].min():.4f} to {customers['Latitude'].max():.4f}")
        print(f"Longitude range: {customers['Longitude'].min():.4f} to {customers['Longitude'].max():.4f}")
        
        # Distance to depot
        depot_coords = (self.params.depot['latitude'], self.params.depot['longitude'])
        distances = []
        for _, row in customers.iterrows():
            if any(row[f'{good}_Demand'] > 0 for good in self.params.goods):
                dist = haversine((row['Latitude'], row['Longitude']), depot_coords)
                distances.append(dist)
        
        if distances:
            print("\nDistance to depot (km):")
            print(f"Min distance: {min(distances):.2f}")
            print(f"Max distance: {max(distances):.2f}")
            print(f"Mean distance: {sum(distances)/len(distances):.2f}")
        
        # Vehicle information
        print("\nVehicle Types Available:")
        for vtype, vinfo in self.params.vehicles.items():
            print(f"Type {vtype}:")
            print(f"  Capacity: {vinfo['capacity']}")
            print(f"  Fixed cost: {vinfo['fixed_cost']}")

        print(f"\nMax route time: {self.params.max_route_time} hours")
        print(f"Average speed: {self.params.avg_speed} km/h")
        print(f"Service time: {self.params.service_time} minutes")
        print("\nStarting solver...\n")

    def solve(self, verbose: bool = False) -> Dict[str, VRPSolution]:
        """
        Solve VRP instance based on benchmark type.
        Returns solutions for all product types in parallel for single compartment case.
        """
        if verbose:
            self._print_diagnostic_information(self.customers)
        
        if self.benchmark_type == BenchmarkType.SINGLE_COMPARTMENT:
            return self.solve_scv_parallel(verbose=verbose)
        elif self.benchmark_type == BenchmarkType.MULTI_COMPARTMENT:
            return self.solve_mcv(verbose=verbose)
        else:
            raise ValueError(f"Unknown benchmark type: {self.benchmark_type}")