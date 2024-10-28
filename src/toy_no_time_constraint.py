import numpy as np
import pandas as pd
import pulp
from sklearn.cluster import MiniBatchKMeans
import itertools
import time
from haversine import haversine

# Step 1: Read customer data from CSV file
# Read the CSV file without headers
df = pd.read_csv("../data/sales_2023_avg_daily_demand.csv", header=None, names=['Customer_ID', 'Latitude', 'Longitude', 'Units_Demand', 'Demand_Type'], encoding="latin-1")

# Pivot the data to get separate columns for each type of demand, filling missing values with 0
df_pivot = df.pivot_table(index=['Customer_ID', 'Latitude', 'Longitude'],
                          columns='Demand_Type',
                          values='Units_Demand',
                          fill_value=0).reset_index()

# Rename the columns to match the desired output
df_pivot.columns.name = None  # Remove the pivot table's column grouping name
df_pivot = df_pivot.rename(columns={'Dry': 'Dry_Demand', 'Chilled': 'Chilled_Demand', 'Frozen': 'Frozen_Demand'})

# Ensure no customer has zero demand for all goods
no_demand = (df_pivot['Dry_Demand'] == 0) & (df_pivot['Chilled_Demand'] == 0) & (df_pivot['Frozen_Demand'] == 0)
if no_demand.any():
    # Assign a minimal demand to Dry_Demand for customers with zero demand
    df_pivot.loc[no_demand, 'Dry_Demand'] = 1

customers = df_pivot
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

# Generate all possible compartment configurations (binary options)
compartment_options = list(itertools.product([0, 1], repeat=len(goods)))
compartment_configs = []
config_id = 1
for vt_name, vt_info in vehicle_types.items():
    for option in compartment_options:
        # Skip configuration if no compartments are selected (i.e., all zeros)
        if sum(option) == 0:
            continue
        compartment = dict(zip(goods, option))
        compartment['Vehicle_Type'] = vt_name
        compartment['Config_ID'] = config_id
        compartment_configs.append(compartment)
        config_id += 1

configurations_df = pd.DataFrame(compartment_configs)

# Merge with vehicle types to get capacities and costs
configurations_df = configurations_df.merge(
    pd.DataFrame(vehicle_types).T.reset_index().rename(columns={'index': 'Vehicle_Type'}),
    on='Vehicle_Type'
)

# Step 3: Generate Clusters for Each Vehicle Configuration

clusters_list = []
cluster_id = 1

# Keep track of customers who have been included in clusters
customers_included = set()

for idx, config in configurations_df.iterrows():
    # Get the goods this configuration can carry
    goods_in_config = [g for g in goods if config[g] == 1]
    if not goods_in_config:
        continue  # Skip configurations that don't carry any goods

    # Customers needing at least one of the goods in the configuration
    customers_subset = customers[
        (customers[[f'{g}_Demand' for g in goods_in_config]] > 0).any(axis=1)
    ].copy()

    # Exclude customers already included in clusters
    customers_subset = customers_subset[~customers_subset['Customer_ID'].isin(customers_included)].copy()

    if customers_subset.empty:
        continue  # All customers already included

    # Determine the number of clusters needed
    total_demand = customers_subset[[f'{g}_Demand' for g in goods_in_config]].sum(axis=1)
    num_clusters = max(1, int(np.ceil(total_demand.sum() / config['Capacity'])))
    num_clusters = min(num_clusters, len(customers_subset))  # Can't have more clusters than customers

    # Generate clusters using MiniBatchKMeans
    coords = customers_subset[['Latitude', 'Longitude']]
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=10000)
    customers_subset['Cluster'] = kmeans.fit_predict(coords)

    # For each cluster, ensure capacity constraints are met
    for c in customers_subset['Cluster'].unique():
        cluster_customers = customers_subset[customers_subset['Cluster'] == c].copy()
        cluster_demand = cluster_customers[[f'{g}_Demand' for g in goods_in_config]].sum()

        # Split cluster if over capacity
        while cluster_demand.sum() > config['Capacity']:
            # Split into two clusters
            sub_kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=10000)
            cluster_customers['Sub_Cluster'] = sub_kmeans.fit_predict(cluster_customers[['Latitude', 'Longitude']])

            for sc in cluster_customers['Sub_Cluster'].unique():
                sub_cluster_customers = cluster_customers[cluster_customers['Sub_Cluster'] == sc].copy()
                sub_cluster_demand = sub_cluster_customers[[f'{g}_Demand' for g in goods_in_config]].sum()

                if sub_cluster_demand.sum() <= config['Capacity']:
                    clusters_list.append({
                        'Cluster_ID': cluster_id,
                        'Config_ID': config['Config_ID'],
                        'Customers': sub_cluster_customers['Customer_ID'].tolist(),
                        'Total_Demand': sub_cluster_demand.to_dict(),
                        'Centroid_Latitude': sub_cluster_customers['Latitude'].mean(),
                        'Centroid_Longitude': sub_cluster_customers['Longitude'].mean(),
                        'Goods_In_Config': goods_in_config
                    })
                    customers_included.update(sub_cluster_customers['Customer_ID'])
                    cluster_id += 1
                else:
                    cluster_customers = sub_cluster_customers.copy()
                    cluster_demand = sub_cluster_demand
                    break  # Re-evaluate the new cluster
            else:
                break  # All sub-clusters are within capacity

        else:
            clusters_list.append({
                'Cluster_ID': cluster_id,
                'Config_ID': config['Config_ID'],
                'Customers': cluster_customers['Customer_ID'].tolist(),
                'Total_Demand': cluster_demand.to_dict(),
                'Centroid_Latitude': cluster_customers['Latitude'].mean(),
                'Centroid_Longitude': cluster_customers['Longitude'].mean(),
                'Goods_In_Config': goods_in_config
            })
            customers_included.update(cluster_customers['Customer_ID'])
            cluster_id += 1

# Check for customers not included in any cluster
all_customers_set = set(customers['Customer_ID'])
missing_customers = all_customers_set - customers_included
if missing_customers:
    print(f"Number of customers not included in any cluster: {len(missing_customers)}")
    # For missing customers, create singleton clusters
    for customer_id in missing_customers:
        customer = customers[customers['Customer_ID'] == customer_id].iloc[0]
        customer_demands = customer[[f'{g}_Demand' for g in goods]].to_dict()
        # Find a configuration that can serve the customer's demands
        for idx, config in configurations_df.iterrows():
            goods_in_config = [g for g in goods if config[g] == 1 and customer_demands[f'{g}_Demand'] > 0]
            if goods_in_config:
                if sum([customer_demands[f'{g}_Demand'] for g in goods_in_config]) <= config['Capacity']:
                    clusters_list.append({
                        'Cluster_ID': cluster_id,
                        'Config_ID': config['Config_ID'],
                        'Customers': [customer_id],
                        'Total_Demand': {g: customer_demands[f'{g}_Demand'] for g in goods_in_config},
                        'Centroid_Latitude': customer['Latitude'],
                        'Centroid_Longitude': customer['Longitude'],
                        'Goods_In_Config': goods_in_config
                    })
                    customers_included.add(customer_id)
                    cluster_id += 1
                    break
        else:
            print(f"Customer {customer_id} could not be assigned to any configuration.")

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
if model.status != 1:
    print("Solver did not find an optimal solution.")
else:
    print("Optimization solved optimally.")

# Step 5: Extract and Summarize the Solution

selected_clusters = clusters_df[clusters_df['Cluster_ID'].isin([cid for cid, var in y_vars.items() if var.varValue > 0.5])]

# Summarize the number of vehicles used (integer)
total_vehicles_used = len(selected_clusters)

# Number of clusters
num_clusters_used = len(selected_clusters)

# Number of customers by cluster
customers_per_cluster = selected_clusters[['Cluster_ID', 'Customers']].copy()
customers_per_cluster['Num_Customers'] = customers_per_cluster['Customers'].apply(len)

print("\nSummary of Results:")
print(f"Total Number of Vehicles Used: {total_vehicles_used}")
print(f"Number of Clusters Used: {num_clusters_used}")
print("\nNumber of Customers by Cluster:")
print(customers_per_cluster[['Cluster_ID', 'Num_Customers']])

# Additional metrics for business stakeholders

# Total fixed cost
total_fixed_cost = selected_clusters.merge(configurations_df[['Config_ID', 'Fixed_Cost']], on='Config_ID')['Fixed_Cost'].sum()

# Total variable cost
total_variable_cost = 0
for idx, cluster in selected_clusters.iterrows():
    cluster_coord = (cluster['Centroid_Latitude'], cluster['Centroid_Longitude'])
    depot_coord = (depot['Latitude'], depot['Longitude'])
    dist = 2 * distance(depot_coord, cluster_coord)
    variable_cost = dist * variable_cost_per_km
    total_variable_cost += variable_cost

print(f"\nTotal Fixed Cost: ${total_fixed_cost:.2f}")
print(f"Total Variable Cost: ${total_variable_cost:.2f}")
print(f"Total Cost: ${total_fixed_cost + total_variable_cost:.2f}")

# Vehicle types used
vehicles_used = selected_clusters.merge(configurations_df[['Config_ID', 'Vehicle_Type']], on='Config_ID')['Vehicle_Type'].value_counts()
print("\nVehicles Used:")
print(vehicles_used)

# Validate that each customer is visited once
customers_served = set()
for idx, cluster in selected_clusters.iterrows():
    customers_served.update(cluster['Customers'])

if len(customers_served) == num_customers:
    print("\nAll customers are served exactly once.")
else:
    missing_customers = all_customers_set - customers_served
    print(f"\nError: Not all customers are served. Missing customers: {missing_customers}")

# Additional details per cluster
print("\nDetails per Cluster:")
for idx, cluster in selected_clusters.iterrows():
    config_id = cluster['Config_ID']
    config = configurations_df[configurations_df['Config_ID'] == config_id].iloc[0]
    vehicle_type = config['Vehicle_Type']
    goods_carried = [g for g in goods if config[g] == 1]
    print(f"Cluster ID: {cluster['Cluster_ID']}")
    print(f"  Vehicle Type: {vehicle_type}")
    print(f"  Goods Carried: {goods_carried}")
    print(f"  Number of Customers: {len(cluster['Customers'])}")
    print(f"  Total Demand: {cluster['Total_Demand']}")
    print(f"  Vehicle Capacity: {config['Capacity']}")
    print("")
