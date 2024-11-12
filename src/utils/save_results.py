import pandas as pd
from datetime import datetime
from pathlib import Path
import inspect
from config.parameters import Parameters
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import folium
from folium import plugins

def save_optimization_results(
    execution_time: float,
    solver_name: str,
    solver_status: str,
    configurations_df: pd.DataFrame,
    selected_clusters: pd.DataFrame,
    total_fixed_cost: float,
    total_variable_cost: float,
    vehicles_used: pd.Series,
    missing_customers: set,
    parameters: Parameters,
    filename: str = None,
    format: str = 'excel'
) -> None:
    """Save optimization results to a file (Excel or JSON) and create visualization"""
    
    # Create timestamp and filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).resolve().parent.parent.parent / 'results'
        extension = '.xlsx' if format == 'excel' else '.json'
        filename = results_dir / f"optimization_results_{timestamp}{extension}"
    
    # Ensure results directory exists
    results_dir = Path(filename).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Calculate metrics and prepare data
    customers_per_cluster = selected_clusters['Customers'].apply(len)
    load_percentages = []
    for _, cluster in selected_clusters.iterrows():
        config = configurations_df[
            configurations_df['Config_ID'] == cluster['Config_ID']
        ].iloc[0]
        max_load_pct = max(
            cluster['Total_Demand'][good] / config['Capacity'] * 100 
            for good in parameters.goods
        )
        load_percentages.append(max_load_pct)
    load_percentages = pd.Series(load_percentages)
    
    # Prepare summary metrics
    summary_metrics = [
        ('Total Cost ($)', f"{total_fixed_cost + total_variable_cost:,.2f}"),
        ('Total Vehicles', len(selected_clusters)),
    ]
    
    for vehicle_type in sorted(vehicles_used.index):
        summary_metrics.append(
            (f'Vehicles Type {vehicle_type}', vehicles_used[vehicle_type])
        )
    
    summary_metrics.extend([
        ('Customers per Cluster (Min)', f"{customers_per_cluster.min():.0f}"),
        ('Customers per Cluster (Max)', f"{customers_per_cluster.max():.0f}"),
        ('Customers per Cluster (Avg)', f"{customers_per_cluster.mean():.1f}"),
        ('Customers per Cluster (Median)', f"{customers_per_cluster.median():.1f}"),
        ('Truck Load % (Min)', f"{load_percentages.min():.1f}"),
        ('Truck Load % (Max)', f"{load_percentages.max():.1f}"),
        ('Truck Load % (Avg)', f"{load_percentages.mean():.1f}"),
        ('Truck Load % (Median)', f"{load_percentages.median():.1f}"),
        ('---Parameters---', ''),
        ('Demand File', parameters.demand_file),
        ('Variable Cost per Hour', parameters.variable_cost_per_hour),
        ('Average Speed', parameters.avg_speed),
        ('Max Route Time', parameters.max_route_time),
        ('Service Time per Customer', parameters.service_time),
        ('Max Split Depth', parameters.clustering['max_depth']),
        ('Clustering Method', parameters.clustering['method']),
        ('Clustering Distance', parameters.clustering['distance']),
        ('Route Time Estimation Method', parameters.clustering['route_time_estimation']),
        ('Model Formulation (# 1 = "Eric\'s" and 2 = "Fabri\'s")', parameters.model_type),
        ('Light Load Penalty', parameters.light_load_penalty),
        ('Light Load Threshold', parameters.light_load_threshold)
        ])
    
    # Add vehicle types
    for v_type, specs in parameters.vehicles.items():
        for spec_name, value in specs.items():
            metric_name = f'Vehicle {v_type} {spec_name}'
            summary_metrics.append((metric_name, value))

    # Prepare cluster details
    cluster_details = selected_clusters.copy()
    if 'Customers' in cluster_details.columns:
        cluster_details['Num_Customers'] = cluster_details['Customers'].apply(len)
        cluster_details['Customers'] = cluster_details['Customers'].apply(str)
    if 'Total_Demand' in cluster_details.columns:
        cluster_details['Total_Demand'] = cluster_details['Total_Demand'].apply(str)

    # Prepare all data
    data = {
        'summary_metrics': summary_metrics,
        'configurations_df': configurations_df,
        'cluster_details': cluster_details,
        'vehicles_used': vehicles_used,
        'other_considerations': {
            'Total Vehicles Used': len(selected_clusters),
            'Number of Unserved Customers': len(missing_customers),
            'Unserved Customers': str(list(missing_customers)) if missing_customers else "None",
            'Average Customers per Cluster': cluster_details['Num_Customers'].mean() if 'Num_Customers' in cluster_details.columns else 'N/A',
            'Average Distance per Cluster': cluster_details['Estimated_Distance'].mean() if 'Estimated_Distance' in cluster_details.columns else 'N/A'
        },
        'execution_details': {
            'Execution Time (s)': execution_time,
            'Solver': solver_name,
            'Solver Status': solver_status,
            'Total Fixed Cost': total_fixed_cost,
            'Total Variable Cost': total_variable_cost,
            'Total Cost': total_fixed_cost + total_variable_cost,
            'Demand File': parameters.demand_file
        }
    }

    try:
        if format == 'json':
            _write_to_json(filename, data)
        else:
            _write_to_excel(filename, data)
            
        # Add visualization
        depot_coords = (parameters.depot['latitude'], parameters.depot['longitude'])
        visualize_clusters(selected_clusters, depot_coords, filename)
            
    except Exception as e:
        print(f"Error saving results to {filename}: {str(e)}")
        raise

def _write_to_excel(filename: str, data: dict) -> None:
    """Write optimization results to Excel file."""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Sheet 1: Summary
        pd.DataFrame(data['summary_metrics'], columns=['Metric', 'Value']).to_excel(
            writer, sheet_name='Solution Summary', index=False
        )
        
        # Sheet 2: Configurations
        data['configurations_df'].to_excel(
            writer, sheet_name='Configurations', index=False
        )
        
        # Sheet 3: Selected Clusters
        data['cluster_details'].to_excel(
            writer, sheet_name='Selected Clusters', index=False
        )
        
        # Sheet 4: Vehicle Usage
        vehicles_df = pd.DataFrame(data['vehicles_used']).reset_index()
        vehicles_df.columns = ['Vehicle Type', 'Count']
        vehicles_df.to_excel(writer, sheet_name='Vehicle Usage', index=False)
        
        # Sheet 5: Other Considerations
        pd.DataFrame([data['other_considerations']]).to_excel(
            writer, sheet_name='Other Considerations', index=False
        )

        # Sheet 6: Execution Details
        pd.DataFrame([data['execution_details']]).to_excel(
            writer, sheet_name='Execution Details', index=False
        )

def _write_to_json(filename: str, data: dict) -> None:
    """Write optimization results to JSON file."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_data = {
        'Solution Summary': dict(data['summary_metrics']),
        'Configurations': data['configurations_df'].to_dict(orient='records'),
        'Selected Clusters': data['cluster_details'].to_dict(orient='records'),
        'Vehicle Usage': pd.DataFrame(data['vehicles_used']).reset_index().to_dict(orient='records'),
        'Other Considerations': data['other_considerations'],
        'Execution Details': data['execution_details']
    }
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)

def visualize_clusters(
    selected_clusters: pd.DataFrame,
    depot_coords: tuple,
    filename: str
) -> None:
    """
    Create and save an interactive map visualization of the clusters in Bogotá.
    
    Args:
        selected_clusters: DataFrame containing cluster information
        depot_coords: Tuple of (latitude, longitude) coordinates for the depot
        filename: Base filename to save the plot (will append _clusters.html)
    """
    # Initialize the map centered on Bogotá
    m = folium.Map(
        location=[4.65, -74.1],  # Bogotá center
        zoom_start=11,
        tiles='CartoDB positron'
    )
    
    # Create color palette for clusters
    n_clusters = len(selected_clusters)
    colors = sns.color_palette("husl", n_colors=n_clusters).as_hex()
    
    # Add depot marker
    folium.Marker(
        location=depot_coords,
        icon=folium.Icon(color='red', icon='home', prefix='fa'),
        popup='Depot'
    ).add_to(m)
    
    # Plot each cluster
    for idx, (_, cluster) in enumerate(selected_clusters.iterrows()):
        color = colors[idx]
        cluster_id = cluster['Cluster_ID']
        config_id = cluster['Config_ID']
        
        # Calculate total demand in kg
        total_demand = sum(cluster['Total_Demand'].values()) if isinstance(cluster['Total_Demand'], dict) else 0
        if isinstance(cluster['Total_Demand'], str):
            total_demand = sum(ast.literal_eval(cluster['Total_Demand']).values())
        
        # Get number of customers
        num_customers = len(ast.literal_eval(cluster['Customers']) if isinstance(cluster['Customers'], str) else cluster['Customers'])
        
        # Prepare popup content
        popup_content = f"""
            <b>Cluster ID:</b> {cluster_id}<br>
            <b>Config ID:</b> {config_id}<br>
            <b>Customers:</b> {num_customers}<br>
            <b>Route Time:</b> {cluster['Route_Time']:.2f} hrs<br>
            <b>Total Demand:</b> {total_demand:,.0f} kg
        """
        
        # Plot cluster centroid with larger circle
        folium.CircleMarker(
            location=(cluster['Centroid_Latitude'], cluster['Centroid_Longitude']),
            radius=8,
            color=color,
            fill=True,
            popup=folium.Popup(popup_content, max_width=300),
            weight=2,
            fill_opacity=0.7
        ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    viz_filename = str(filename).rsplit('.', 1)[0] + '_clusters.html'
    m.save(viz_filename)