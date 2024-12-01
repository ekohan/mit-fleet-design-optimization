"""
Module for generating clusters from customer data.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed
from haversine import haversine
from config.parameters import Parameters
from utils.route_time import estimate_route_time
from scipy.spatial import distance
from dataclasses import dataclass
import re
from functools import lru_cache
from pulp import *
import itertools
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

# Add after other imports
class Symbols:
    """Unicode symbols for logging."""
    CHECKMARK = "✓"
    CROSS = "✗"

PRODUCT_WEIGHTS = {'Frozen': 0.5, 'Chilled': 0.3, 'Dry': 0.2}

@dataclass
class Cluster:
    """Represents a cluster of customers that can be served by a vehicle configuration."""
    cluster_id: int
    config_id: int
    customers: List[str]
    total_demand: Dict[str, float]
    centroid_latitude: float
    centroid_longitude: float
    goods_in_config: List[str]
    route_time: float

    def to_dict(self) -> Dict:
        """Convert cluster to dictionary format."""
        return {
            'Cluster_ID': self.cluster_id,
            'Config_ID': self.config_id,
            'Customers': self.customers,
            'Total_Demand': self.total_demand,
            'Centroid_Latitude': self.centroid_latitude,
            'Centroid_Longitude': self.centroid_longitude,
            'Goods_In_Config': self.goods_in_config,
            'Route_Time': self.route_time
        }

    @classmethod
    def from_customers(
        cls,
        customers: pd.DataFrame,
        config: pd.Series,
        cluster_id: int,
        goods: List[str],
        depot: Dict[str, float],
        service_time: float,
        avg_speed: float,
        route_time_estimation: str
    ) -> 'Cluster':
        """Create a cluster from customer data."""
        return cls(
            cluster_id=cluster_id,
            config_id=config['Config_ID'],
            customers=customers['Customer_ID'].tolist(),
            total_demand={g: float(customers[f'{g}_Demand'].sum()) for g in goods},
            centroid_latitude=float(customers['Latitude'].mean()),
            centroid_longitude=float(customers['Longitude'].mean()),
            goods_in_config=[g for g in goods if config[g] == 1],
            route_time=float(estimate_route_time(
                cluster_customers=customers,
                depot=depot,
                service_time=service_time,
                avg_speed=avg_speed,
                method=route_time_estimation
            ))
        )

def get_clustering_input(
    customers: pd.DataFrame, 
    goods: List[str], 
    method: str,
    geo_weight: float,
    demand_weight: float,
    distance_metric: str = 'euclidean'
) -> np.ndarray:
    """Get appropriate input for clustering algorithm."""
    if not method.startswith('agglomerative'):
        logger.debug(f"Using feature-based input for method: {method}")
        return customers[['Latitude', 'Longitude']].values
    
    logger.debug(f"Using precomputed distance matrix for method: {method} with geo_weight={geo_weight}, demand_weight={demand_weight}")
    return compute_composite_distance(customers, goods, geo_weight, demand_weight)

def compute_composite_distance(
    customers: pd.DataFrame,
    goods: List[str],
    geo_weight: float,
    demand_weight: float
) -> np.ndarray:
    """Compute composite distance matrix combining geographical and demand distances."""
    # Compute geographical distance
    coords = customers[['Latitude', 'Longitude']].values
    geo_dist = pairwise_distances(coords, metric='euclidean')
    
    # Get weighted demands
    product_weights = {
        'Frozen': 0.5,    # Highest priority - temperature sensitive
        'Chilled': 0.3,   # Medium priority - temperature controlled
        'Dry': 0.2        # Lower priority - no temperature control
    }
    
    demands = customers[[f'{g}_Demand' for g in goods]].fillna(0).values
    weighted_demands = np.zeros_like(demands)
    for i, good in enumerate(goods):
        weighted_demands[:, i] = demands[:, i] * product_weights.get(good, 1.0)  # Default weight if not specified
    
    # Compute demand distance
    demand_dist = pairwise_distances(weighted_demands)
    
    # Normalize distances only if they have non-zero values
    if geo_dist.max() > 0:
        geo_dist = geo_dist / geo_dist.max()
    if demand_dist.max() > 0:
        demand_dist = demand_dist / demand_dist.max()
    
    # Return weighted combination
    composite_distance = (geo_weight * geo_dist) + (demand_weight * demand_dist)
    
    # Ensure the distance matrix is square
    if composite_distance.shape[0] != composite_distance.shape[1]:
        logger.error(f"Composite distance matrix is not square: shape={composite_distance.shape}")
        raise ValueError(f"Composite distance matrix is not square: shape={composite_distance.shape}")
    
    return composite_distance

def get_clustering_model(n_clusters: int, method: str):
    """Return the clustering model based on the method name."""
    if method == 'minibatch_kmeans':
        return MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'kmedoids':
        return KMedoids(n_clusters=n_clusters, random_state=42)
    elif method.startswith('agglomerative'):
        # All agglomerative methods use precomputed distances
        return AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='average')
    else:
        logger.error(f"❌ Unknown clustering method: {method}")
        raise ValueError(f"Unknown clustering method: {method}")

def generate_clusters_for_configurations(
    customers: pd.DataFrame,
    configurations_df: pd.DataFrame,
    params: Parameters,
) -> pd.DataFrame:
    """Generate clusters for each vehicle configuration."""
    logger.info(f"📍 Clustering method received: '{params.clustering['method']}'")
    
    if params.clustering['method'] == 'ostermeier':
        logger.info("🔄 Using Ostermeier MCVRP clustering method")
        all_clusters = []
        cluster_id_counter = 0
        
        # Generate feasibility mapping first
        feasible_customers = _generate_feasibility_mapping(
            customers=customers,
            configurations_df=configurations_df,
            goods=params.goods
        )
        
        for _, config in configurations_df.iterrows():
            # Get feasible customers for this configuration
            feasible_mask = [
                cid for cid, configs in feasible_customers.items() 
                if config['Config_ID'] in configs
            ]
            customers_subset = customers[customers['Customer_ID'].isin(feasible_mask)]
            
            if customers_subset.empty:
                logger.info(f"ℹ️ Skipping configuration {config['Config_ID']}: No feasible customers")
                continue
                
            # Initialize and solve Ostermeier clustering
            ostermeier = OstermeierCluster(
                customers_df=customers_subset,
                goods=params.goods,
                vehicle_capacity=config['Capacity'],
                depot=params.depot
            )
            
            try:
                config_clusters = ostermeier.solve()
                
                # Convert to standard cluster format
                for cluster_data in config_clusters:
                    cluster_customers = customers_subset[
                        customers_subset['Customer_ID'].isin(cluster_data['customers'])
                    ]
                    
                    if not cluster_customers.empty:
                        cluster = Cluster.from_customers(
                            customers=cluster_customers,
                            config=config,
                            cluster_id=cluster_id_counter,
                            goods=params.goods,
                            depot=params.depot,
                            service_time=params.service_time,
                            avg_speed=params.avg_speed,
                            route_time_estimation=params.clustering['route_time_estimation']
                        )
                        all_clusters.append(cluster.to_dict())
                        cluster_id_counter += 1
                        
            except Exception as e:
                logger.warning(f"⚠️ Could not generate clusters for configuration {config['Config_ID']}: {str(e)}")
                continue  # Skip this configuration and move to the next
        
        # Convert to DataFrame
        combined_clusters_df = pd.DataFrame(all_clusters)
        
        if combined_clusters_df.empty:
            logger.error("❌ No valid clusters were generated for any configuration")
            raise RuntimeError("No valid clusters were generated")
            
        logger.info(f"✓ Generated {len(combined_clusters_df)} clusters across {len(configurations_df)} configurations")
        return combined_clusters_df
        
    else:
        raise ValueError(f"Unknown clustering method: {params.clustering['method']}")

def compute_demands(customers: pd.DataFrame, goods: List[str]) -> Dict[str, np.ndarray]:
    """Compute and cache demand calculations for customers."""
    return {
        'total': customers[[f'{g}_Demand' for g in goods]].sum(axis=1),
        'by_good': customers[[f'{g}_Demand' for g in goods]].fillna(0).values,
        'weighted': customers[[f'{g}_Demand' for g in goods]].fillna(0).values * \
                   np.array([PRODUCT_WEIGHTS.get(g, 1.0) for g in goods])
    }

def process_configuration(
    config: pd.Series,
    customers: pd.DataFrame,
    goods: List[str],
    depot: Dict[str, float],
    avg_speed: float,
    service_time: float,
    max_route_time: float,
    feasible_customers: Dict,
    max_split_depth: int,
    clustering_method: str,
    route_time_estimation: str,
    geo_weight: float,
    demand_weight: float,
    distance_metric: str
) -> List[Dict]:
    """Process a single vehicle configuration to generate feasible clusters."""
    config_id = config['Config_ID']
    clusters = []
    
    # Initialize adjusted weights with default values
    adjusted_geo_weight = geo_weight
    adjusted_demand_weight = demand_weight
    
    # Get feasible customers for this configuration
    customers_subset = customers[
        customers['Customer_ID'].isin([
            cid for cid, configs in feasible_customers.items() 
            if config_id in configs
        ])
    ].copy()

    if customers_subset.empty:
        return []

    # Always compute total demand first
    demands = compute_demands(customers_subset, goods)
    customers_subset['Total_Demand'] = demands['total']

    # For small subsets, always use MiniBatchKMeans
    if len(customers_subset) <= 2:
        logger.debug(f"Using MiniBatchKMeans for small cluster with {len(customers_subset)} customers.")
        data = customers_subset[['Latitude', 'Longitude']].values
        model = MiniBatchKMeans(n_clusters=1, random_state=42)
        customers_subset['Cluster'] = model.fit_predict(data)
    else:
        # Initial clustering
        num_clusters = estimate_num_initial_clusters(
            customers_subset, 
            config, 
            depot, 
            avg_speed, 
            service_time,
            goods,
            max_route_time,
            route_time_estimation
        )

        # Handle agglomerative variants
        if '_' in clustering_method:
            pattern = r'agglomerative_geo_(\d+\.\d+)_demand_(\d+\.\d+)'
            if match := re.match(pattern, clustering_method):
                adjusted_geo_weight = float(match.group(1))
                adjusted_demand_weight = float(match.group(2))

        # Get input data and cluster using adjusted weights
        data = get_clustering_input(
            customers_subset, 
            goods, 
            clustering_method, 
            adjusted_geo_weight,
            adjusted_demand_weight,
            distance_metric
        )

        # Ensure the number of clusters does not exceed the number of customers
        num_clusters = min(num_clusters, len(customers_subset))

        model = get_clustering_model(num_clusters, clustering_method)
        customers_subset['Cluster'] = model.fit_predict(data)

    # Process clusters (same for both cases)
    clusters_to_check = [
        (customers_subset[customers_subset['Cluster'] == c], 0)
        for c in customers_subset['Cluster'].unique()
    ]
    cluster_id_base = int(str(config_id) + "000")
    current_cluster_id = 0

    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()
        # Ensure Total_Demand is computed for the subset
        if 'Total_Demand' not in cluster_customers.columns:
            cluster_customers['Total_Demand'] = compute_demands(cluster_customers, goods)['total']
            
        cluster_demand = cluster_customers['Total_Demand'].sum()
        route_time = estimate_route_time(
            cluster_customers=cluster_customers,
            depot=depot,
            service_time=service_time,
            avg_speed=avg_speed,
            method=route_time_estimation
        )

        if (cluster_demand > config['Capacity'] or route_time > max_route_time) and depth < max_split_depth:
            if len(cluster_customers) > 1:
                logger.debug(f"Splitting cluster ID {current_cluster_id} at depth {depth}.")

                # Split cluster using the same approach
                split_data = get_clustering_input(
                    cluster_customers, 
                    goods, 
                    clustering_method,
                    adjusted_geo_weight,
                    adjusted_demand_weight,
                    distance_metric
                )
                # Split over-capacity clusters into two
                split_model = get_clustering_model(2, clustering_method)
                sub_labels = split_model.fit_predict(split_data)

                for label in [0, 1]:
                    sub_cluster = cluster_customers[sub_labels == label]
                    if not sub_cluster.empty:
                        clusters_to_check.append((sub_cluster, depth + 1))
        else:
            # Add valid cluster
            current_cluster_id += 1
            cluster = Cluster.from_customers(
                cluster_customers,
                config,
                cluster_id_base + current_cluster_id,
                goods,
                depot,
                service_time,
                avg_speed,
                route_time_estimation
            )
            clusters.append(cluster.to_dict())

    logger.info(f"🔧 Clustering Method: {clustering_method} | geo_weight={adjusted_geo_weight}, demand_weight={adjusted_demand_weight}")

    return clusters

def _generate_feasibility_mapping(
    customers: pd.DataFrame,
    configurations_df: pd.DataFrame,
    goods: List[str]
) -> Dict:
    """Generate mapping of feasible configurations for each customer."""
    feasible_customers = {}
    
    for _, customer in customers.iterrows():
        customer_id = customer['Customer_ID']
        feasible_configs = []
        
        for _, config in configurations_df.iterrows():
            if _is_customer_feasible(customer, config, goods):
                feasible_configs.append(config['Config_ID'])
        
        if feasible_configs:
            feasible_customers[customer_id] = feasible_configs
    
    return feasible_customers

def _is_customer_feasible(
    customer: pd.Series,
    config: pd.Series,
    goods: List[str]
) -> bool:
    """Check if a customer's demands can be served by a configuration."""
    for good in goods:
        if customer[f'{good}_Demand'] > 0 and not config[good]:
            return False
        if customer[f'{good}_Demand'] > config['Capacity']:
            return False
    return True

def estimate_num_initial_clusters(
    customers: pd.DataFrame,
    config: pd.Series,
    depot: Dict[str, float],
    avg_speed: float,
    service_time: float,
    goods: List[str],
    max_route_time: float,
    route_time_estimation: str
) -> int:
    """
    Estimate the number of initial clusters needed based on capacity and time constraints.
    
    Args:
        customers: DataFrame containing customer data
        config: Vehicle configuration
        depot: Depot location coordinates
        avg_speed: Average vehicle speed (km/h)
        service_time: Service time per customer (minutes)
        goods: List of goods types
        max_route_time: Maximum route time (hours)
        route_time_estimation: Method to estimate route times 
                             (Legacy, Clarke-Wright, BHH, CA, VRPSolver)
    
    Returns:
        Estimated number of clusters needed
    """
    if customers.empty:
        return 0

    # Calculate total demand for relevant goods
    total_demand = 0
    for good in goods:
        if config[good]:  # Only consider goods this vehicle can carry
            total_demand += customers[f'{good}_Demand'].sum()

    # Estimate clusters needed based on capacity
    clusters_by_capacity = np.ceil(total_demand / config['Capacity'])

    # Calculate average distance from depot to customers
    depot_coord = (depot['latitude'], depot['longitude'])
    avg_distance = np.mean([
        haversine(depot_coord, (lat, lon))
        for lat, lon in zip(customers['Latitude'], customers['Longitude'])
    ])

    # Estimate time for an average route
    avg_customers_per_cluster = len(customers) / clusters_by_capacity
    avg_cluster = customers.sample(n=min(int(avg_customers_per_cluster), len(customers)))
    avg_route_time = estimate_route_time(
        cluster_customers=avg_cluster,
        depot=depot,
        service_time=service_time,  # in minutes
        avg_speed=avg_speed,
        method=route_time_estimation
    )

    # Estimate clusters needed based on time
    clusters_by_time = np.ceil(
        avg_route_time * len(customers) / 
        (max_route_time * avg_customers_per_cluster)
    )

    # Take the maximum of the two estimates
    num_clusters = int(max(clusters_by_capacity, clusters_by_time, 1))
    
    logger.debug(
        f"Estimated clusters: {num_clusters} "
        f"(capacity: {clusters_by_capacity}, time: {clusters_by_time})"
    )
    
    return num_clusters

def validate_cluster_coverage(clusters_df, customers_df):
    customer_coverage = {cid: False for cid in customers_df['Customer_ID']}
    for customers in clusters_df['Customers']:
        for cid in customers:
            customer_coverage[cid] = True
    uncovered = [cid for cid, covered in customer_coverage.items() if not covered]

@lru_cache(maxsize=128)
def compute_distance_matrix(customer_ids, method):
    """Cache distance computations for frequently accessed customer sets."""

class OstermeierCluster:
    """Implementation of Ostermeier's MCVRP clustering approach."""
    
    def __init__(self, customers_df: pd.DataFrame, goods: List[str], vehicle_capacity: float, depot: Dict[str, float]):
        self.customers = customers_df
        self.goods = goods
        self.vehicle_capacity = vehicle_capacity
        self.num_customers = len(customers_df)
        self.M = range(len(goods))
        self.allow_splits = False
        self.depot = depot
        
        try:
            import gurobipy as gp
            from gurobipy import GRB
            self.gp = gp
            self.GRB = GRB
        except ImportError:
            raise ImportError("Gurobi is required for Ostermeier clustering")

    def abs_(self, model, x):
        """Helper function to create absolute value in Gurobi."""
        z = model.addVar(name=f"abs_{model.NumVars}")
        model.addConstr(z >= x)
        model.addConstr(z >= -x)
        return z

    def estimate_cluster_distance_bhh(self, cluster_customers):
        """Estimate intra-cluster distance using BHH approximation."""
        if len(cluster_customers) <= 1:
            return 0
            
        # Calculate area of minimum bounding rectangle
        min_lat = cluster_customers['Latitude'].min()
        max_lat = cluster_customers['Latitude'].max()
        min_lon = cluster_customers['Longitude'].min()
        max_lon = cluster_customers['Longitude'].max()
        
        # Convert to km using approximate conversion factors
        width = haversine((min_lat, min_lon), (min_lat, max_lon))
        height = haversine((min_lat, min_lon), (max_lat, min_lon))
        area = width * height
        
        # BHH constant (0.75 is an empirical approximation)
        k = 0.75
        
        # BHH formula: k * sqrt(area * n)
        return k * math.sqrt(area * len(cluster_customers))

    def solve(self, max_vehicles: Optional[int] = None) -> List[Dict]:
        """
        Solve the Ostermeier clustering problem using Gurobi with BHH approximation.
        """
        if max_vehicles is None:
            max_vehicles = min(self.num_customers, 10)
            
        try:
            model = self.gp.Model("OstermeierClustering")
            
            # Binary variables for customer-vehicle assignment
            u = model.addVars(
                [(i, k) for i in range(self.num_customers) for k in range(max_vehicles)],
                vtype=self.GRB.BINARY,
                name="u"
            )
            
            # Binary variables for product-compartment assignment
            y = model.addVars(
                [(p, k, m) for p in self.goods 
                 for k in range(max_vehicles) 
                 for m in self.M],
                vtype=self.GRB.BINARY,
                name="y"
            )
            
            # Relaxed customer assignment - allow unassigned customers
            for i in range(self.num_customers):
                model.addConstr(
                    self.gp.quicksum(u[i,k] for k in range(max_vehicles)) <= 1,
                    name=f"customer_assignment_{i}"
                )
            
            # Vehicle capacity constraints
            for k in range(max_vehicles):
                model.addConstr(
                    self.gp.quicksum(
                        self.customers[f'{p}_Demand'].iloc[i] * u[i,k]
                        for i in range(self.num_customers)
                        for p in self.goods
                        if self.customers[f'{p}_Demand'].iloc[i] > 0
                    ) <= self.vehicle_capacity,
                    name=f"capacity_{k}"
                )
                
                # Product assignment constraints
                for p in self.goods:
                    model.addConstr(
                        self.gp.quicksum(y[p,k,m] for m in self.M) >= 1,
                        name=f"product_assignment_{p}_{k}"
                    )
            
            # Simplified distance objective
            obj = 0
            for k in range(max_vehicles):
                cluster_size = self.gp.quicksum(u[i,k] for i in range(self.num_customers))
                
                # Add penalty for unassigned customers
                obj += -100 * cluster_size  # Encourage assigning customers
                
                # Add simple distance term
                for i in range(self.num_customers):
                    lat = self.customers['Latitude'].iloc[i]
                    lon = self.customers['Longitude'].iloc[i]
                    depot_dist = abs(lat - self.depot['latitude']) + abs(lon - self.depot['longitude'])
                    obj += depot_dist * u[i,k]
            
            model.setObjective(obj, self.GRB.MINIMIZE)
            model.optimize()
            
            return self._extract_solution(model, u, y, max_vehicles)
            
        except Exception as e:
            logger.error(f"❌ Error in Ostermeier clustering: {str(e)}")
            raise RuntimeError(f"Ostermeier clustering failed: {str(e)}")

    def _extract_solution(self, model, u, y, max_vehicles: int) -> List[Dict]:
        """Extract solution from Gurobi model."""
        clusters = []
        if model.status in [self.GRB.OPTIMAL, self.GRB.TIME_LIMIT, self.GRB.SOLUTION_LIMIT]:
            for k in range(max_vehicles):
                cluster_customers = set()
                cluster_products = defaultdict(list)
                
                # Get customers assigned to this vehicle
                for i in range(self.num_customers):
                    if u[i,k].X > 0.5:
                        cluster_customers.add(self.customers.iloc[i]['Customer_ID'])
                
                # Get product-compartment assignments
                for p in self.goods:
                    for m in self.M:
                        if y[p,k,m].X > 0.5:
                            cluster_products[p].append(m)
                
                if cluster_customers:  # Only add non-empty clusters
                    clusters.append({
                        'customers': list(cluster_customers),
                        'vehicle': k,
                        'products': dict(cluster_products)
                    })
        
        return clusters
