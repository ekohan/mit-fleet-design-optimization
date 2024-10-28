import math
import numpy as np
import pandas as pd
import pulp
import time
from gurobipy import GRB
from pathlib import Path

# Read the CSV file without headers
file_name = Path(__file__).resolve().parent / "../data/sales_2023_avg_daily_demand.csv"
df = pd.read_csv(file_name, header=None, names=['Customer_ID', 'Latitude', 'Longitude', 'Units_Demand', 'Demand_Type'], encoding="latin-1")

# Pivot the data to get separate columns for each type of demand, filling missing values with 0
df_pivot = df.pivot_table(index=['Customer_ID', 'Latitude', 'Longitude'], 
                          columns='Demand_Type', 
                          values='Units_Demand', 
                          fill_value=0).reset_index()

# Rename the columns to match the desired output
df_pivot.columns.name = None  # Remove the pivot table's column grouping name
df_pivot = df_pivot.rename(columns={'Dry': 'Dry_Demand', 'Chilled': 'Chilled_Demand', 'Frozen': 'Frozen_Demand'})

# Display the result
customers = df_pivot

num_customers = customers['Customer_ID'].nunique()

print(customers.head())

# Depot location
depot = {'Latitude': 4.7, 'Longitude': -74.1}

# Vehicle capacity and configurations
vehicle_capacity = 3000  # Total capacity units
goods = ['Dry', 'Chilled', 'Frozen']

# Initial configurations (e.g., starting with some basic configurations)
initial_configurations = [
    {'Config_ID': 1, 'Dry': 1500, 'Chilled': 900, 'Frozen': 600, 'Cost': 100},
    {'Config_ID': 2, 'Dry': 1200, 'Chilled': 1200, 'Frozen': 600, 'Cost': 110},
    # Add more initial configurations if needed
]

# Convert to DataFrame
configurations_df = pd.DataFrame(initial_configurations)

# Preprocess: Create initial clusters based on proximity
from sklearn.cluster import MiniBatchKMeans

def create_initial_clusters(customers, num_clusters):
    coords = customers[['Latitude', 'Longitude']]
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=10000)
    customers['Cluster'] = kmeans.fit_predict(coords)
    return customers

# Set the number of clusters to optimize performance
num_initial_clusters = math.ceil(num_customers / 50)  # Approximate cluster size of 50
customers = create_initial_clusters(customers, num_initial_clusters)
clusters = customers['Cluster'].unique()

# Step 2: Define the Master Problem

def solve_master_problem(customers, configurations_df, clusters, integer=False):
    """
    Solves the master problem of assigning configurations to clusters.

    Returns:
    - model: The solved PuLP model
    - x_vars: Decision variables for configuration assignment
    """
    model = pulp.LpProblem("FSMOP_MCV_CD_Master", pulp.LpMinimize)

    # Decision variables
    x_vars = {}

    # Variables: x[k, c] for configurations and clusters
    for idx, config in configurations_df.iterrows():
        k = config['Config_ID']
        for c in clusters:
            x_vars[(k, c)] = pulp.LpVariable(f"x_{k}_{c}", cat='Integer' if integer else 'Continuous', lowBound=0)
            # Set variables as integer if integer=True

    # Objective Function: Minimize total cost
    config_cost = configurations_df.set_index('Config_ID')['Cost'].to_dict()
    model += pulp.lpSum(config_cost[k] * x_vars[(k, c)] for (k, c) in x_vars), "Total_Cost"

    # Constraints:

    # Demand satisfaction per cluster
    for c in clusters:
        cluster_customers = customers[customers['Cluster'] == c]
        for g in goods:
            demand = cluster_customers[f"{g}_Demand"].sum()
            capacity = pulp.lpSum(configurations_df.loc[configurations_df['Config_ID'] == k, g].values[0] * x_vars[(k, c)]
                                  for k in configurations_df['Config_ID'])
            model += capacity >= demand, f"Demand_{g}_{c}"

    # Solve the model using Gurobi
    solver = pulp.GUROBI_CMD(msg=0)
    model.solve(solver)

    # Check the status
    if model.status != 1:
        print("Solver did not find an optimal solution.")
    else:
        print("Master problem solved optimally.")

    return model, x_vars

# Step 3: Define the Subproblem (Configuration Generation)

def generate_new_configurations(model, customers, configurations_df, clusters):
    """
    Generates new configurations based on dual values from the master problem.

    Returns:
    - new_configurations: A list of new configurations with negative reduced cost
    """
    # Get dual variables for demand constraints
    duals = {}
    for constraint in model.constraints.values():
        if constraint.name.startswith("Demand_"):
            duals[constraint.name] = constraint.pi

    # Create a DataFrame for duals
    duals_df = pd.DataFrame([
        {'Constraint': name, 'Dual': pi}
        for name, pi in duals.items()
    ])

    # For each good, calculate average dual value
    avg_duals = {}
    for g in goods:
        duals_g = duals_df[duals_df['Constraint'].str.contains(f"Demand_{g}")]
        avg_dual = duals_g['Dual'].mean()
        avg_duals[g] = avg_dual if not np.isnan(avg_dual) else 0

    # Generate new configurations based on average duals
    new_configurations = []
    config_id_start = configurations_df['Config_ID'].max() + 1

    # Heuristic: Allocate more capacity to goods with higher duals
    total_dual = sum(avg_duals.values())
    if total_dual == 0:
        proportions = {g: 1/3 for g in goods}
    else:
        proportions = {g: avg_duals[g] / total_dual for g in goods}

    # Create new configurations
    for i in range(3):  # Generate a few configurations
        capacities = {g: proportions[g] * vehicle_capacity for g in goods}
        cost = 100 + np.random.randint(-10, 10)  # Random cost for example
        new_configurations.append({
            'Config_ID': config_id_start + i,
            'Dry': capacities['Dry'],
            'Chilled': capacities['Chilled'],
            'Frozen': capacities['Frozen'],
            'Cost': cost
        })

    return new_configurations

# Step 4: Iterative Optimization Process

def column_generation(customers, initial_configurations, clusters):
    configurations_df = pd.DataFrame(initial_configurations)
    iteration = 0
    max_iterations = 5
    convergence = False
    total_cost = None

    while not convergence and iteration < max_iterations:
        print(f"Iteration {iteration + 1}")
        start_time = time.time()

        # Solve the master problem
        model, x_vars = solve_master_problem(customers, configurations_df, clusters)
        prev_total_cost = total_cost
        total_cost = pulp.value(model.objective)
        print(f"Total Cost: {total_cost:.2f}")
        end_time = time.time()
        print(f"Master Problem Solved in {end_time - start_time:.2f} seconds")

        # Generate new configurations
        new_configs = generate_new_configurations(model, customers, configurations_df, clusters)

        # Remove duplicate configurations
        existing_configs = set(configurations_df['Config_ID'])
        new_configs = [config for config in new_configs if config['Config_ID'] not in existing_configs]

        if not new_configs or (prev_total_cost is not None and abs(prev_total_cost - total_cost) < 1e-4):
            convergence = True
            print("No significant improvement in cost. Converged.")
        else:
            # Add new configurations to the DataFrame
            configurations_df = pd.concat([configurations_df, pd.DataFrame(new_configs)], ignore_index=True)
            print(f"Added {len(new_configs)} new configurations")
        iteration += 1

    return configurations_df

# Run the column generation algorithm
start_time = time.time()
configurations_df = column_generation(customers, initial_configurations, clusters)
end_time = time.time()
print(f"Total optimization time: {end_time - start_time:.2f} seconds")

# Step 5: Solve the final master problem as an integer program
print("\nSolving final master problem with integer variables.")
model_int, x_vars_int = solve_master_problem(customers, configurations_df, clusters, integer=True)
if model_int.status != 1:
    print("Integer master problem did not find an optimal solution.")
else:
    print("Integer master problem solved optimally.")

# Step 6: Extract the Solution

def extract_solution(customers, x_vars, configurations_df):
    # Extract configuration assignments
    config_assignments = []
    for (k, c), var in x_vars.items():
        if var.varValue > 1e-6:
            config_assignments.append({
                'Config_ID': k,
                'Cluster': c,
                'Vehicles_Used': var.varValue
            })

    configs_df = pd.DataFrame(config_assignments)

    # Merge with configurations to get capacities
    configs_df = configs_df.merge(configurations_df, on='Config_ID', how='left')

    return configs_df

configs_df = extract_solution(customers, x_vars_int, configurations_df)

print("\nConfiguration Assignments:")
print(configs_df)

# Step 7: Generate Summary

def generate_summary(customers, configs_df, clusters):
    """
    Generates a summary of the results including number of vehicles used, number of clusters,
    number of customers by cluster, and other relevant metrics.
    """
    total_vehicles = configs_df['Vehicles_Used'].sum()
    num_clusters = len(clusters)
    customers_per_cluster = customers.groupby('Cluster').size().reset_index(name='Num_Customers')

    print("\nSummary of Results:")
    print(f"Total Number of Vehicles Used: {int(total_vehicles)}")
    print(f"Number of Clusters: {num_clusters}")
    print("\nNumber of Customers by Cluster:")
    print(customers_per_cluster)

    # Compute total demand and capacity per cluster
    cluster_demands = customers.groupby('Cluster')[['Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']].sum().reset_index()
    configs_df['Total_Capacity_Dry'] = configs_df['Dry'] * configs_df['Vehicles_Used']
    configs_df['Total_Capacity_Chilled'] = configs_df['Chilled'] * configs_df['Vehicles_Used']
    configs_df['Total_Capacity_Frozen'] = configs_df['Frozen'] * configs_df['Vehicles_Used']

    cluster_capacities = configs_df.groupby('Cluster')[['Total_Capacity_Dry', 'Total_Capacity_Chilled', 'Total_Capacity_Frozen']].sum().reset_index()

    # Merge demands and capacities
    cluster_summary = pd.merge(cluster_demands, cluster_capacities, on='Cluster')

    # Compute utilization
    for g in goods:
        demand_col = f"{g}_Demand"
        capacity_col = f"Total_Capacity_{g}"
        utilization_col = f"{g}_Utilization"
        cluster_summary[utilization_col] = cluster_summary[demand_col] / cluster_summary[capacity_col]

    print("\nCluster Demand and Capacity Utilization:")
    print(cluster_summary[['Cluster', 'Dry_Utilization', 'Chilled_Utilization', 'Frozen_Utilization']])

# Generate and print the summary
generate_summary(customers, configs_df, clusters)

# Validate demand satisfaction
def validate_solution(customers, configs_df):
    for c in clusters:
        cluster_customers = customers[customers['Cluster'] == c]
        total_demand = cluster_customers[['Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']].sum()
        assigned_configs = configs_df[configs_df['Cluster'] == c]
        total_capacity = assigned_configs[['Dry', 'Chilled', 'Frozen']].multiply(assigned_configs['Vehicles_Used'], axis=0).sum()
        for g in goods:
            if total_capacity[g] < total_demand[f"{g}_Demand"] - 1e-6:
                print(f"Demand for {g} in cluster {c} not satisfied.")
            else:
                pass  # Demand is satisfied

validate_solution(customers, configs_df)
