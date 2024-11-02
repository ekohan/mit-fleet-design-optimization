import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import MiniBatchKMeans
import itertools
import time
from haversine import haversine_vector, Unit
import cProfile
import pstats
import io
from joblib import Parallel, delayed
from customer_data_loader import get_customer_demand
from config_utils import generate_vehicle_configurations, print_configurations


# TODO: DEPRECATED.

pr = cProfile.Profile()
pr.enable()

# Step 1: Read customer data from CSV file

customers = get_customer_demand()
num_customers = customers["Customer_ID"].nunique()

# Depot location
depot = {"Latitude": 4.7, "Longitude": -74.1}

# Step 2: Define Vehicle Types and Configurations

vehicle_types = {
    "A": {"Capacity": 2000, "Fixed_Cost": 100},
    "B": {"Capacity": 3000, "Fixed_Cost": 130},
    "C": {"Capacity": 4000, "Fixed_Cost": 150},
}

variable_cost_per_km = 0.01  # Same for all vehicles
goods = ["Dry", "Chilled", "Frozen"]

configurations_df = generate_vehicle_configurations(vehicle_types, goods)
print_configurations(configurations_df, goods)

# Step 3: Generate Clusters for Each Vehicle Configuration

# Define the time constraint parameters
avg_speed = 50  # Average speed in km/h
max_route_time = 8  # Maximum allowed time per route in hours
service_time_per_customer = 10 / 60  # Service time per customer in hours (10 minutes)

# Initialize clusters list and customers_included set
clusters_list = []
cluster_id = 1

# Keep track of customers who have been included in clusters
customers_included = set()

# Precompute distances from depot to all customers
depot_coord = (depot['Latitude'], depot['Longitude'])
customer_coords = customers[['Latitude', 'Longitude']].to_numpy()
depot_coords_array = np.array([depot_coord])  # Shape (1, 2)
depot_coords_repeated = np.repeat(depot_coords_array, len(customer_coords), axis=0)
distances = haversine_vector(depot_coords_repeated, customer_coords, unit=Unit.KILOMETERS)
customers['Distance_To_Depot'] = distances

# Function to compute estimated time for a cluster
def compute_estimated_time(cluster_customers, avg_speed, service_time_per_customer):
    max_distance = cluster_customers['Distance_To_Depot'].max()
    total_travel_distance = 2 * max_distance
    travel_time = total_travel_distance / avg_speed
    total_service_time = len(cluster_customers) * service_time_per_customer
    estimated_time = travel_time + total_service_time
    return estimated_time, total_travel_distance

# Adjust initial number of clusters
def get_initial_num_clusters(customers_subset, config, goods_in_config):
    total_demand = customers_subset[[f"{g}_Demand" for g in goods_in_config]].sum()
    base_num_clusters = max(1, int(np.ceil(total_demand.sum() / config["Capacity"])))
    num_clusters = int(base_num_clusters * 1.5)
    num_clusters = min(num_clusters, len(customers_subset))
    return num_clusters

# Function to process clusters
def process_cluster(cluster_customers, config, avg_speed, service_time_per_customer, max_route_time, cluster_id_start, goods_in_config):
    clusters = []
    cluster_id = cluster_id_start
    split_depth = 0
    MAX_SPLIT_DEPTH = 20
    clusters_to_process = [cluster_customers]

    while clusters_to_process:
        cluster_customers = clusters_to_process.pop(0)
        cluster_demand = cluster_customers[[f"{g}_Demand" for g in goods_in_config]].sum()
        estimated_time, total_distance = compute_estimated_time(
            cluster_customers, avg_speed, service_time_per_customer
        )
        if (
            cluster_demand.sum() <= config["Capacity"]
            and estimated_time <= max_route_time
        ):
            clusters.append(
                {
                    "Cluster_ID": cluster_id,
                    "Config_ID": config["Config_ID"],
                    "Customers": cluster_customers["Customer_ID"].tolist(),
                    "Total_Demand": cluster_demand.to_dict(),
                    "Centroid_Latitude": cluster_customers["Latitude"].mean(),
                    "Centroid_Longitude": cluster_customers["Longitude"].mean(),
                    "Goods_In_Config": goods_in_config,
                    "Estimated_Time": estimated_time,
                    "Estimated_Distance": total_distance,
                }
            )
            cluster_id += 1
        else:
            if len(cluster_customers) <= 1 or split_depth >= MAX_SPLIT_DEPTH:
                clusters.append(
                    {
                        "Cluster_ID": cluster_id,
                        "Config_ID": config["Config_ID"],
                        "Customers": cluster_customers["Customer_ID"].tolist(),
                        "Total_Demand": cluster_demand.to_dict(),
                        "Centroid_Latitude": cluster_customers["Latitude"].mean(),
                        "Centroid_Longitude": cluster_customers["Longitude"].mean(),
                        "Goods_In_Config": goods_in_config,
                        "Estimated_Time": estimated_time,
                        "Estimated_Distance": total_distance,
                    }
                )
                cluster_id += 1
            else:
                # Split cluster
                split_depth += 1
                if len(cluster_customers) >= 2:
                    sub_kmeans = MiniBatchKMeans(
                        n_clusters=2, random_state=42, batch_size=10000
                    )
                    cluster_customers["Sub_Cluster"] = sub_kmeans.fit_predict(
                        cluster_customers[["Latitude", "Longitude"]]
                    )
                    for sc in cluster_customers["Sub_Cluster"].unique():
                        sub_cluster_customers = cluster_customers[
                            cluster_customers["Sub_Cluster"] == sc
                        ].copy()
                        clusters_to_process.append(sub_cluster_customers)
                else:
                    # Cannot split further
                    clusters.append(
                        {
                            "Cluster_ID": cluster_id,
                            "Config_ID": config["Config_ID"],
                            "Customers": cluster_customers["Customer_ID"].tolist(),
                            "Total_Demand": cluster_demand.to_dict(),
                            "Centroid_Latitude": cluster_customers["Latitude"].mean(),
                            "Centroid_Longitude": cluster_customers["Longitude"].mean(),
                            "Goods_In_Config": goods_in_config,
                            "Estimated_Time": estimated_time,
                            "Estimated_Distance": total_distance,
                        }
                    )
                    cluster_id += 1
        split_depth += 1
    return clusters, cluster_id

# Main loop over configurations
for idx, config in configurations_df.iterrows():
    goods_in_config = [g for g in goods if config[g] == 1]
    if not goods_in_config:
        continue

    customers_subset = customers[
        (customers[[f"{g}_Demand" for g in goods_in_config]] > 0).any(axis=1)
    ].copy()
    customers_subset = customers_subset[
        ~customers_subset["Customer_ID"].isin(customers_included)
    ].copy()

    if customers_subset.empty:
        continue

    num_clusters = get_initial_num_clusters(customers_subset, config, goods_in_config)

    # Generate clusters using MiniBatchKMeans
    coords = customers_subset[["Latitude", "Longitude"]]
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, random_state=42, batch_size=10000
    )
    customers_subset["Cluster"] = kmeans.fit_predict(coords)

    cluster_id_start = cluster_id

    # Process clusters in parallel
    clusters_to_process = [customers_subset[customers_subset["Cluster"] == c] for c in customers_subset["Cluster"].unique()]
    results = Parallel(n_jobs=-1)(
        delayed(process_cluster)(
            cluster_customers=cluster_customers.reset_index(drop=True),
            config=config,
            avg_speed=avg_speed,
            service_time_per_customer=service_time_per_customer,
            max_route_time=max_route_time,
            cluster_id_start=cluster_id_start + idx,
            goods_in_config=goods_in_config
        )
        for idx, cluster_customers in enumerate(clusters_to_process)
    )

    for clusters_result, new_cluster_id in results:
        clusters_list.extend(clusters_result)
        cluster_id = max(cluster_id, new_cluster_id)
        for cluster in clusters_result:
            customers_included.update(cluster["Customers"])

    # Re-process customers not included due to constraint violations
    remaining_customers = set(customers_subset["Customer_ID"]) - customers_included
    customers_subset = customers_subset[
        customers_subset["Customer_ID"].isin(remaining_customers)
    ]

# Check for customers not included in any cluster
all_customers_set = set(customers["Customer_ID"])
missing_customers = all_customers_set - customers_included
if missing_customers:
    print(
        f"Number of customers not included in any cluster: {len(missing_customers)}"
    )
    # For missing customers, try to assign them to configurations that can serve them
    for customer_id in missing_customers:
        customer = customers[customers["Customer_ID"] == customer_id].iloc[0]
        customer_demands = customer[[f"{g}_Demand" for g in goods]].to_dict()
        # Find a configuration that can serve the customer's demands
        assigned = False
        for idx, config in configurations_df.iterrows():
            goods_in_config = [
                g
                for g in goods
                if config[g] == 1 and customer_demands[f"{g}_Demand"] > 0
            ]
            if goods_in_config:
                if (
                    sum(
                        [customer_demands[f"{g}_Demand"] for g in goods_in_config]
                    )
                    <= config["Capacity"]
                ):
                    # Compute estimated time for the singleton cluster
                    singleton_cluster_customers = pd.DataFrame([customer])
                    estimated_time, total_distance = compute_estimated_time(
                        singleton_cluster_customers, avg_speed, service_time_per_customer
                    )
                    if estimated_time <= max_route_time:
                        clusters_list.append(
                            {
                                "Cluster_ID": cluster_id,
                                "Config_ID": config["Config_ID"],
                                "Customers": [customer_id],
                                "Total_Demand": {
                                    g: customer_demands[f"{g}_Demand"]
                                    for g in goods_in_config
                                },
                                "Centroid_Latitude": customer["Latitude"],
                                "Centroid_Longitude": customer["Longitude"],
                                "Goods_In_Config": goods_in_config,
                                "Estimated_Time": estimated_time,
                                "Estimated_Distance": total_distance,
                            }
                        )
                        customers_included.add(customer_id)
                        cluster_id += 1
                        assigned = True
                        break
        if not assigned:
            print(
                f"Customer {customer_id} could not be assigned to any configuration."
            )

# Create DataFrame of clusters
clusters_df = pd.DataFrame(clusters_list)

# Step 4: Build the Optimization Model

# Decision variables:
# y[c] = 1 if cluster c is selected, 0 otherwise

model = pulp.LpProblem("Vehicle_Routing", pulp.LpMinimize)

# Create decision variables
y_vars = {}
for idx, cluster in clusters_df.iterrows():
    y_vars[cluster["Cluster_ID"]] = pulp.LpVariable(
        f"y_{cluster['Cluster_ID']}", cat="Binary"
    )

# Objective Function: Minimize total cost (fixed cost + variable cost)

total_cost = 0
for idx, cluster in clusters_df.iterrows():
    config_id = cluster["Config_ID"]
    config = configurations_df[
        configurations_df["Config_ID"] == config_id
    ].iloc[0]
    fixed_cost = config["Fixed_Cost"]
    # Use the precomputed estimated distance
    variable_cost = cluster["Estimated_Distance"] * variable_cost_per_km
    cluster_cost = fixed_cost + variable_cost
    total_cost += cluster_cost * y_vars[cluster["Cluster_ID"]]

model += total_cost, "Total_Cost"

# Constraints:

# Each customer must be served exactly once

# Create a mapping from customer to clusters they are in
customer_cluster_map = {}
for idx, cluster in clusters_df.iterrows():
    cluster_id = cluster["Cluster_ID"]
    for customer_id in cluster["Customers"]:
        if customer_id not in customer_cluster_map:
            customer_cluster_map[customer_id] = []
        customer_cluster_map[customer_id].append(cluster_id)

# Add constraints to ensure each customer is served exactly once
for customer_id, cluster_ids in customer_cluster_map.items():
    model += (
        pulp.lpSum([y_vars[cid] for cid in cluster_ids]) == 1,
        f"Serve_Customer_{customer_id}",
    )

# Solve the model using Gurobi

solver = pulp.GUROBI_CMD(msg=1)

start_time = time.time()
model.solve(solver)
end_time = time.time()

print(f"Optimization completed in {end_time - start_time:.2f} seconds.")

# Check the status
if model.status != 1:
    print("Solver did not find an optimal solution.")
else:
    print("Optimization solved optimally.")

# Step 5: Extract and Summarize the Solution

selected_clusters = clusters_df[
    clusters_df["Cluster_ID"].isin(
        [cid for cid, var in y_vars.items() if var.varValue > 0.5]
    )
]

# Summarize the number of vehicles used (integer)
total_vehicles_used = len(selected_clusters)

# Number of clusters
num_clusters_used = len(selected_clusters)

# Number of customers by cluster
customers_per_cluster = selected_clusters[["Cluster_ID", "Customers"]].copy()
customers_per_cluster["Num_Customers"] = customers_per_cluster["Customers"].apply(
    len
)

print("\nSummary of Results:")
print(f"Total Number of Vehicles Used: {total_vehicles_used}")
print(f"Number of Clusters Used: {num_clusters_used}")
print("\nNumber of Customers by Cluster:")
print(customers_per_cluster[["Cluster_ID", "Num_Customers"]])

# Additional metrics for business stakeholders

# Total fixed cost
total_fixed_cost = selected_clusters.merge(
    configurations_df[["Config_ID", "Fixed_Cost"]], on="Config_ID"
)["Fixed_Cost"].sum()

# Total variable cost
total_variable_cost = (
    selected_clusters["Estimated_Distance"].sum() * variable_cost_per_km
)

print(f"\nTotal Fixed Cost: ${total_fixed_cost:.2f}")
print(f"Total Variable Cost: ${total_variable_cost:.2f}")
print(f"Total Cost: ${total_fixed_cost + total_variable_cost:.2f}")

# Vehicle types used
vehicles_used = (
    selected_clusters.merge(
        configurations_df[["Config_ID", "Vehicle_Type"]], on="Config_ID"
    )["Vehicle_Type"]
    .value_counts()
    .sort_index()
)
print("\nVehicles Used:")
print(vehicles_used)

# Validate that each customer is visited once
customers_served = set()
for idx, cluster in selected_clusters.iterrows():
    customers_served.update(cluster["Customers"])

if len(customers_served) == num_customers:
    print("\nAll customers are served exactly once.")
else:
    missing_customers = all_customers_set - customers_served
    print(
        f"\nError: Not all customers are served. Missing customers: {missing_customers}"
    )

# Additional details per cluster
print("\nDetails per Cluster:")
for idx, cluster in selected_clusters.iterrows():
    config_id = cluster["Config_ID"]
    config = configurations_df[
        configurations_df["Config_ID"] == config_id
    ].iloc[0]
    vehicle_type = config["Vehicle_Type"]
    goods_carried = [g for g in goods if config[g] == 1]
    print(f"Cluster ID: {cluster['Cluster_ID']}")
    print(f"  Vehicle Type: {vehicle_type}")
    print(f"  Goods Carried: {goods_carried}")
    print(f"  Number of Customers: {len(cluster['Customers'])}")
    print(f"  Total Demand: {cluster['Total_Demand']}")
    print(f"  Vehicle Capacity: {config['Capacity']}")
    print(f"  Estimated Time: {cluster['Estimated_Time']:.2f} hours")
    print(f"  Estimated Distance: {cluster['Estimated_Distance']:.2f} km")
    print("")

pr.disable()
s = io.StringIO()
sortby = 'cumulative'  # You can also sort by 'time' or 'calls'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats(20)  # Adjust the number to see more or fewer lines
print(s.getvalue())
