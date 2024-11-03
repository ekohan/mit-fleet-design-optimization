import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import MiniBatchKMeans
import time
from utils.data_processing import load_customer_demand
from haversine import haversine
import sys
from utils.config_utils import generate_vehicle_configurations, print_configurations
from utils.cluster_utils import estimate_initial_clusters
import cProfile
import io
import pstats
from joblib import Parallel, delayed
from utils.save_results import save_optimization_results



pr = cProfile.Profile()
pr.enable()


# TODO: hacer una version con flags que tome opcion profiling,
# etc. Parece que hay algo llamado pandas profiling.

# Step 1: Load Customer Data
customers = load_customer_demand()
num_customers = customers['Customer_ID'].nunique()

# Depot location
depot = {'Latitude': 4.7, 'Longitude': -74.1}

# Step 2: Define Vehicle Types and Configurations

vehicle_types = {
    'A': {'Capacity': 2000, 'Fixed_Cost': 100},
    'B': {'Capacity': 3000, 'Fixed_Cost': 130},
    'C': {'Capacity': 4000, 'Fixed_Cost': 150}
}

variable_cost_per_km = 0.01  # Same for all vehicles
goods = ['Dry', 'Chilled', 'Frozen']

configurations_df = generate_vehicle_configurations(vehicle_types, goods)
print_configurations(configurations_df, goods)

# Time-related parameters
avg_speed = 40  # km/h
max_route_time = 10  # Maximum route time in hours
service_time_per_customer = 10/60  # 10 minutes converted to hours

# Step 3: Generate Clusters for Each Vehicle Configuration

## Heuristics to prune clustering
# Check if customer's demands can be served by this configuration
def is_customer_feasible(customer, config):
    for good in goods:
        if customer[f'{good}_Demand'] > 0 and not config[good]:
            return False
        if customer[f'{good}_Demand'] > config['Capacity']:
            return False
    return True

# Pre-filter customers by feasibility for each configuration
# TODO: check naming feasible_customers
feasible_customers = {}
for _, customer in customers.iterrows():
    customer_id = customer['Customer_ID']
    feasible_configs = []
    for _, config in configurations_df.iterrows():
        if is_customer_feasible(customer, config):
            feasible_configs.append(config['Config_ID'])
    if feasible_configs:
        feasible_customers[customer_id] = feasible_configs


## TODO: Acá empieza el clustering main loop

def process_configuration(config, customers, goods, depot, avg_speed, service_time_per_customer, max_route_time, feasible_customers):
    config_id = config['Config_ID']
    clusters = []
    MAX_SPLIT_DEPTH = 10  # Add this line to limit recursion
    split_depth = 0       # Add this line to track depth
    
    # Get feasible customers for this configuration
    customers_subset = customers[
        customers['Customer_ID'].isin([
            cid for cid, configs in feasible_customers.items() 
            if config_id in configs
        ])
    ].copy()

    if customers_subset.empty:
        return []

    # Calculate total demand
    total_demand = customers_subset[[f'{g}_Demand' for g in goods]].sum(axis=1)
    customers_subset['Total_Demand'] = total_demand
    
    # Use estimate_initial_clusters
    num_clusters = estimate_initial_clusters(
        customers_subset, 
        config, 
        depot, 
        avg_speed, 
        service_time_per_customer
    )

    # Initial clustering
    coords = customers_subset[['Latitude', 'Longitude']]
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=10000)
    customers_subset['Cluster'] = kmeans.fit_predict(coords)

    # Process each initial cluster
    clusters_to_check = []
    for c in customers_subset['Cluster'].unique():
        mask = customers_subset['Cluster'] == c
        clusters_to_check.append((customers_subset[mask], 0))  # Add depth tracking

    cluster_id_base = int(str(config_id) + "000")
    current_cluster_id = 0

    while clusters_to_check:
        cluster_customers, depth = clusters_to_check.pop()  # Unpack depth
        cluster_demand = cluster_customers['Total_Demand'].sum()
        route_time = 1 + len(cluster_customers) * service_time_per_customer

        if (cluster_demand > config['Capacity'] or route_time > max_route_time) and depth < MAX_SPLIT_DEPTH:
            if len(cluster_customers) <= 1:
                continue
            else:
                # Split cluster
                coords = cluster_customers[['Latitude', 'Longitude']].to_numpy()
                sub_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=10000)
                sub_labels = sub_kmeans.fit_predict(coords)

                for label in [0, 1]:
                    sub_cluster = cluster_customers[sub_labels == label]
                    if len(sub_cluster) > 0:
                        clusters_to_check.append((sub_cluster, depth + 1))  # Increment depth
        else:
            # Add valid cluster
            current_cluster_id += 1
            clusters.append({
                'Cluster_ID': cluster_id_base + current_cluster_id,
                'Config_ID': config['Config_ID'],
                'Customers': cluster_customers['Customer_ID'].tolist(),
                'Total_Demand': {
                    g: float(cluster_customers[f'{g}_Demand'].sum()) for g in goods
                },
                'Centroid_Latitude': float(cluster_customers['Latitude'].mean()),
                'Centroid_Longitude': float(cluster_customers['Longitude'].mean()),
                'Goods_In_Config': [g for g in goods if config[g] == 1],
                'Route_Time': float(route_time)
            })

    return clusters

# Parallel processing of configurations
clusters_by_config = Parallel(n_jobs=-1)(
    delayed(process_configuration)(
        config,
        customers,
        goods,
        depot,
        avg_speed,
        service_time_per_customer,
        max_route_time,
        feasible_customers
    )
    for _, config in configurations_df.iterrows()
)

# Combine all clusters
clusters_list = []
for config_clusters in clusters_by_config:
    clusters_list.extend(config_clusters)

# Create DataFrame of clusters
clusters_df = pd.DataFrame(clusters_list)

# Step 4: Build the Optimization Model

# Decision variables:
# y[c] = 1 if cluster c is selected, 0 otherwise

model = pulp.LpProblem("Vehicle_Routing", pulp.LpMinimize)

# Create decision variables
y_vars = {}
for idx, cluster in clusters_df.iterrows():
    y_vars[cluster['Cluster_ID']] = pulp.LpVariable(f"y_{cluster['Cluster_ID']}", cat='Binary')

# Objective Function: Minimize total cost (fixed cost + variable cost)

def distance(coord1, coord2):
    return haversine(coord1, coord2)

total_cost = 0
for idx, cluster in clusters_df.iterrows():
    config_id = cluster['Config_ID']
    config = configurations_df[configurations_df['Config_ID'] == config_id].iloc[0]
    fixed_cost = config['Fixed_Cost']
    # Estimate distance (from depot to centroid and back)
    cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
    depot_coord = (depot['Latitude'], depot['Longitude'])
    dist = 2 * distance(depot_coord, cluster_coord)  # Simplified estimation
    variable_cost = dist * variable_cost_per_km
    cluster_cost = fixed_cost + variable_cost
    total_cost += cluster_cost * y_vars[cluster['Cluster_ID']]

model += total_cost, "Total_Cost"

# Constraints:

# Each customer must be served exactly once

# Create a mapping from customer to clusters they are in
customer_cluster_map = {}
for idx, cluster in clusters_df.iterrows():
    cluster_id = cluster['Cluster_ID']
    for customer_id in cluster['Customers']:
        if customer_id not in customer_cluster_map:
            customer_cluster_map[customer_id] = []
        customer_cluster_map[customer_id].append(cluster_id)

# Add constraints to ensure each customer is served exactly once
for customer_id, cluster_ids in customer_cluster_map.items():
    model += pulp.lpSum([y_vars[cid] for cid in cluster_ids]) == 1, f"Serve_Customer_{customer_id}"

# Solve the model using Gurobi

solver = pulp.GUROBI_CMD(msg=1)

start_time = time.time()
model.solve(solver)
end_time = time.time()

print(f"Optimization completed in {end_time - start_time:.2f} seconds.")

# Check the status
if model.status != pulp.LpStatusOptimal:
    print(f"Optimization status: {pulp.LpStatus[model.status]}")
    print("The model is infeasible. Please check for customers not included in any cluster or other constraint issues.")
    sys.exit(1)

# If we get here, we have a valid solution
selected_clusters = clusters_df[
    clusters_df['Cluster_ID'].isin([
        cid for cid, var in y_vars.items() 
        if var.varValue and var.varValue > 0.5
    ])
]

# Validate the solution
all_customers_set = set(customers['Customer_ID'])
served_customers = set()
for _, cluster in selected_clusters.iterrows():
    served_customers.update(cluster['Customers'])

missing_customers = all_customers_set - served_customers
if missing_customers:
    print(f"\nWARNING: {len(missing_customers)} customers are not served:")
    print(missing_customers)
    # Optionally print their demands
    unserved = customers[customers['Customer_ID'].isin(missing_customers)]
    print("\nUnserved customer demands:")
    print(unserved[['Customer_ID', 'Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']])

# Step 5: Extract and Summarize the Solution

# Summarize the number of vehicles used (integer)
total_vehicles_used = len(selected_clusters)

# Number of clusters
num_clusters_used = len(selected_clusters)

# Number of customers by cluster
customers_per_cluster = selected_clusters[['Cluster_ID', 'Customers']].copy()
customers_per_cluster['Num_Customers'] = customers_per_cluster['Customers'].apply(len)

# Calculate total fixed cost
total_fixed_cost = selected_clusters.merge(
    configurations_df[["Config_ID", "Fixed_Cost"]], 
    on="Config_ID"
)["Fixed_Cost"].sum()

# Calculate distances and variable cost for each cluster
def calculate_cluster_distance(cluster):
    cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
    depot_coord = (depot['Latitude'], depot['Longitude'])
    return 2 * haversine(depot_coord, cluster_coord)  # Round trip distance

# Add estimated distances to selected_clusters
selected_clusters = selected_clusters.copy()
selected_clusters['Estimated_Distance'] = selected_clusters.apply(
    calculate_cluster_distance, axis=1
)

# Calculate total variable cost
total_variable_cost = (
    selected_clusters['Estimated_Distance'] * variable_cost_per_km
).sum()

print("\nCost Breakdown:")
print("-" * 50)
print(f"Fixed Cost:     ${total_fixed_cost:>10,.2f}")
print(f"Variable Cost:  ${total_variable_cost:>10,.2f}")
print(f"Total Cost:     ${(total_fixed_cost + total_variable_cost):>10,.2f}")

# Calculate vehicles used by type
vehicles_used = (
    selected_clusters.merge(
        configurations_df[["Config_ID", "Vehicle_Type"]], on="Config_ID"
    )["Vehicle_Type"]
    .value_counts()
    .sort_index()
)

print("\nVehicles Used:")
print(vehicles_used)

print("\nCluster Details:")
print("=" * 50)
for idx, cluster in selected_clusters.iterrows():
    config_id = cluster['Config_ID']
    config = configurations_df[
        configurations_df['Config_ID'] == config_id
    ].iloc[0]
    vehicle_type = config['Vehicle_Type']
    goods_carried = [g for g in goods if config[g] == 1]
    
    print(f"\nCluster {cluster['Cluster_ID']}")
    print("-" * 25)
    print(f"Configuration:")
    print(f"  ID: {config_id}")    
    print(f"  Vehicle Type: {vehicle_type}")
    print(f"  Capacity: {config['Capacity']}")
    print(f"  Fixed Cost: ${config['Fixed_Cost']}")
    print(f"  Compartments: {', '.join(goods_carried)}")
    
    print("\nDemand:")
    print(f"  Customers: {len(cluster['Customers'])}")
    for good in goods:
        demand = cluster['Total_Demand'][good]
        print(f"  {good}:     {demand:>8,.2f}")
        # Check if demand exceeds capacity
        if demand > config['Capacity']:
            print(f"    WARNING: Demand exceeds vehicle capacity of {config['Capacity']}")
    
    if 'Estimated_Distance' in cluster:
        print(f"  Distance:{cluster['Estimated_Distance']:>8,.2f} km")

    print(f"  Route Time: {cluster['Route_Time']:>8,.2f} hours")



# Print the profiling results
pr.disable()
s = io.StringIO()
sortby = 'cumulative'  # You can also sort by 'time' or 'calls'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)  # Adjust the number to see more or fewer lines
print(s.getvalue())

save_optimization_results(
    execution_time=end_time - start_time,
    solver_name='GUROBI',
    solver_status=pulp.LpStatus[model.status],
    configurations_df=configurations_df,
    selected_clusters=selected_clusters,
    total_fixed_cost=total_fixed_cost,
    total_variable_cost=total_variable_cost,
    vehicles_used=vehicles_used,
    missing_customers=missing_customers
)