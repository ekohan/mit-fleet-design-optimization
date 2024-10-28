import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import time


# Clustering algorithms
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

# Evaluation metrics
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

# For mapping and visualization
import folium
from folium.plugins import MarkerCluster

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Timing decorator
def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds to run.")
        return result
    return wrapper


# TODO: Constraint checking similar to capacitated_kmeans in kmedoids and hierarchical
# TODO: Similar to capacitated_kmeans, we need to check constraints and adjust clusters
# TODO: understand clustering evaluation metrics, specially time and capacity violations

# Global variables
average_speed = 40  # km/h  # TODO: make this a parameter   
# Maximum operating time (in minutes)
max_operating_time = 8 * 60  # 8 hours
# Fixed service time per customer (in minutes)
service_time_per_customer = 10

# Vehicle capacity constraints (example capacities)
vehicle_capacity = {
    'Dry_Demand': 5000,
    'Chilled_Demand': 3000,
    'Frozen_Demand': 2000
}

# Depot location (Assuming central Bogotá)
depot = {'Latitude': 4.7, 'Longitude': -74.1}


# Functions

def haversine_distance(lat1, lon1, lat2, lon2):
    # Haversine distance in kilometers
    R = 6371  # Earth radius in kilometers
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + \
        np.cos(phi1)*np.cos(phi2)*np.sin(delta_lambda/2.0)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def compute_travel_time(lat1, lon1, lat2, lon2):
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    time = (distance / average_speed) * 60  # Convert hours to minutes
    return time

# Capacitated K-Means Clustering
def capacitated_kmeans(customers, vehicle_capacity, max_operating_time, n_clusters):
    # Initialize KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    customer_coords = customers[['Latitude', 'Longitude']]

    # Fit KMeans
    kmeans.fit(customer_coords)
    customers['Cluster'] = kmeans.labels_

    # Initialize variables
    iteration = 0
    max_iterations = 100
    while iteration < max_iterations:
        iteration += 1
        # Check capacity and time constraints for each cluster
        clusters_to_split = []
        for cluster_id in range(n_clusters):
            cluster_customers = customers[customers['Cluster'] == cluster_id]

            # Calculate total demands
            total_dry = cluster_customers['Dry_Demand'].sum()
            total_chilled = cluster_customers['Chilled_Demand'].sum()
            total_frozen = cluster_customers['Frozen_Demand'].sum()
            total_service_time = cluster_customers['Service_Time'].sum()

            # Calculate total travel time (simplified as sum of travel times from depot)
            total_travel_time = cluster_customers['Time_From_Depot'].sum()
            total_time = total_service_time + total_travel_time

            # Check capacity constraints
            if (total_dry > vehicle_capacity['Dry_Demand'] or
                total_chilled > vehicle_capacity['Chilled_Demand'] or
                total_frozen > vehicle_capacity['Frozen_Demand'] or
                total_time > max_operating_time):
                clusters_to_split.append(cluster_id)

        if not clusters_to_split:
            break  # All clusters meet the constraints

        # Split clusters that violate constraints
        for cluster_id in clusters_to_split:
            cluster_customers = customers[customers['Cluster'] == cluster_id]
            # Re-cluster the customers in this cluster into two clusters
            sub_kmeans = KMeans(n_clusters=2, random_state=42)
            sub_kmeans.fit(cluster_customers[['Latitude', 'Longitude']])
            labels = sub_kmeans.labels_
            # Assign new cluster IDs
            new_cluster_id = customers['Cluster'].max() + 1
            customers.loc[cluster_customers.index[labels == 1], 'Cluster'] = new_cluster_id
        n_clusters = customers['Cluster'].nunique()
    return customers

# Hierarchical Clustering with Capacity Constraints
def capacitated_hierarchical(customers, vehicle_capacity, max_operating_time, distance_threshold=0.01):
    # Compute distance matrix
    customer_coords = customers[['Latitude', 'Longitude']].values
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, linkage='ward')
    customers['Cluster'] = model.fit_predict(customer_coords)

    # Due to complexity, we can skip detailed implementation here
    return customers

# K-Medoids Clustering
def capacitated_kmedoids(customers, vehicle_capacity, max_operating_time, n_clusters):
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=42)
    customer_coords = customers[['Latitude', 'Longitude']]
    kmedoids.fit(customer_coords)
    customers['Cluster'] = kmedoids.labels_

    # Implement constraint adjustments as needed
    return customers

def visualize_clusters(customers, algorithm_name):
    # Calculate the mean latitude and longitude to center the map
    map_center = [customers['Latitude'].mean(), customers['Longitude'].mean()]

    # Initialize the Folium map
    m = folium.Map(location=map_center, zoom_start=12, tiles='CartoDB positron')

    # Generate a color palette
    num_clusters = customers['Cluster'].nunique()
    colors = plt.cm.get_cmap('tab20', num_clusters).colors
    color_map = dict(zip(sorted(customers['Cluster'].unique()), [matplotlib.colors.rgb2hex(c) for c in colors]))

    # Add depot marker
    folium.Marker(
        location=(depot['Latitude'], depot['Longitude']),
        icon=folium.Icon(color='red', icon='home'),
        popup='Depot'
    ).add_to(m)

    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(m)

    # Add customer markers to the map
    for _, row in customers.iterrows():
        cluster_id = row['Cluster']
        folium.CircleMarker(
            location=(row['Latitude'], row['Longitude']),
            radius=4,
            color=color_map[cluster_id],
            fill=True,
            fill_color=color_map[cluster_id],
            fill_opacity=0.7,
            popup=folium.Popup(html=f'''
                <b>Customer ID:</b> {row["Customer_ID"]}<br>
                <b>Cluster:</b> {cluster_id}<br>
                <b>Dry Demand:</b> {row["Dry_Demand"]}<br>
                <b>Chilled Demand:</b> {row["Chilled_Demand"]}<br>
                <b>Frozen Demand:</b> {row["Frozen_Demand"]}
            ''', max_width=250)
        ).add_to(marker_cluster)

    # Add a legend to the map
    legend_html = '''
     <div style="
     position: fixed;
     bottom: 50px; left: 50px; width: 200px; height: auto;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color: white; opacity: 0.8;
     ">
     <h4 style="margin:10px;">Cluster Colors</h4>
    '''
    for cluster_id, color in color_map.items():
        legend_html += f'''
         <p style="margin:10px;">
         <span style="background-color:{color};width:15px;height:15px;display:inline-block;border:1px solid #000;"></span>
         &nbsp;Cluster {cluster_id}
         </p>
        '''
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map to an HTML file
    map_filename = f'{algorithm_name.replace(" ", "_")}_clusters_map.html'
    m.save(map_filename)
    print(f"Map has been saved to {map_filename}.")

def evaluate_clustering(customers):
    # Silhouette Score (requires at least 2 clusters)
    if customers['Cluster'].nunique() > 1:
        customer_coords = customers[['Latitude', 'Longitude']]
        silhouette_avg = silhouette_score(customer_coords, customers['Cluster'])
    else:
        silhouette_avg = np.nan

    # Total Within-Cluster Sum of Squares (WCSS)
    wcss = 0
    for cluster_id in customers['Cluster'].unique():
        cluster_customers = customers[customers['Cluster'] == cluster_id]
        centroid = cluster_customers[['Latitude', 'Longitude']].mean()
        distances = cdist(cluster_customers[['Latitude', 'Longitude']], [centroid])
        wcss += (distances**2).sum()

    # Capacity and Time Utilization
    capacity_violations = 0
    time_violations = 0
    for cluster_id in customers['Cluster'].unique():
        cluster_customers = customers[customers['Cluster'] == cluster_id]
        total_dry = cluster_customers['Dry_Demand'].sum()
        total_chilled = cluster_customers['Chilled_Demand'].sum()
        total_frozen = cluster_customers['Frozen_Demand'].sum()
        total_service_time = cluster_customers['Service_Time'].sum()
        total_travel_time = cluster_customers['Time_From_Depot'].sum()
        total_time = total_service_time + total_travel_time
        if (total_dry > vehicle_capacity['Dry_Demand'] or
            total_chilled > vehicle_capacity['Chilled_Demand'] or
            total_frozen > vehicle_capacity['Frozen_Demand']):
            capacity_violations += 1
        if total_time > max_operating_time:
            time_violations += 1
    return silhouette_avg, wcss, capacity_violations, time_violations

def prepare_clusters_for_milp(clustered_customers):
    cluster_summary = clustered_customers.groupby('Cluster').agg({
        'Dry_Demand': 'sum',
        'Chilled_Demand': 'sum',
        'Frozen_Demand': 'sum',
        'Latitude': 'mean',
        'Longitude': 'mean',
        'Service_Time': 'sum',
        'Time_From_Depot': 'sum'
    }).reset_index()
    return cluster_summary


# Main function
@timing
def main():
    # Step 1: Generate Synthetic Customer Data for Bogotá

    np.random.seed(42)  # For reproducibility
    num_customers = 2199

    # Random geographic coordinates around Bogotá
    latitudes = np.random.uniform(4.5, 4.9, size=num_customers)
    longitudes = np.random.uniform(-74.2, -74.0, size=num_customers)

    # Random demand for different product types (dry, chilled, frozen)
    dry_demand = np.random.randint(1, 20, size=num_customers)
    chilled_demand = np.random.randint(1, 15, size=num_customers)
    frozen_demand = np.random.randint(1, 10, size=num_customers)

    # Customer DataFrame
    customers = pd.DataFrame({
        'Customer_ID': np.arange(1, num_customers + 1),
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Dry_Demand': dry_demand,
        'Chilled_Demand': chilled_demand,
        'Frozen_Demand': frozen_demand,
        'Service_Time': service_time_per_customer  # Fixed service time
    })

    # Step 2: Compute Travel Time Matrix

    # Compute travel time from depot to each customer
    customers['Time_From_Depot'] = customers.apply(
        lambda row: compute_travel_time(depot['Latitude'], depot['Longitude'], row['Latitude'], row['Longitude']), axis=1)


    # Step 3: Run Clustering Algorithms, Evaluate, Create Maps, and Save Results for MILP

    cluster_algorithms = {
        'Capacitated K-Means': capacitated_kmeans,
        'Capacitated K-Medoids': capacitated_kmedoids,
        # 'Capacitated Hierarchical': capacitated_hierarchical  # Uncomment if implemented
    }

    results = []

    for name, algorithm in cluster_algorithms.items():
        print(f"Running {name}...")
        start_time = time.time()
        n_clusters_initial = 10  # Starting number of clusters
        clustered_customers = algorithm(customers.copy(), vehicle_capacity, max_operating_time, n_clusters_initial)
        elapsed_time = time.time() - start_time
        silhouette_avg, wcss, capacity_violations, time_violations = evaluate_clustering(clustered_customers)
        results.append({
            'Algorithm': name,
            'Silhouette Score': silhouette_avg,
            'WCSS': wcss,
            'Capacity Violations': capacity_violations,
            'Time Violations': time_violations,
            'Number of Clusters': clustered_customers['Cluster'].nunique(),
            'Elapsed Time (s)': elapsed_time
        })
        # Save clustered data for MILP
        clustered_customers.to_csv(f'{name.replace(" ", "_")}_clusters.csv', index=False)
        # Visualization (Optional)
        visualize_clusters(clustered_customers, name)

    # Step 4: Display Evaluation Results

    results_df = pd.DataFrame(results)
    print("\nClustering Evaluation Results:")
    print(results_df)

    # Step 5: Prepare Clusters for MILP (Choose the best algorithm based on evaluation)

    # For demonstration, let's choose the algorithm with the highest Silhouette Score
    best_algorithm = results_df.sort_values(by='Silhouette Score', ascending=False).iloc[0]['Algorithm']
    print(f"\nBest Algorithm Selected: {best_algorithm}")

    # Load the best clustering result
    clustered_customers = pd.read_csv(f'{best_algorithm.replace(" ", "_")}_clusters.csv')



    cluster_summary = prepare_clusters_for_milp(clustered_customers)
    print("\nCluster Summary for MILP:")
    print(cluster_summary.head())

    # Save the result to a CSV file
    cluster_summary.to_csv('clusters_for_milp.csv', index=False)

# Conditional for standalone execution
if __name__ == "__main__":
    main()
