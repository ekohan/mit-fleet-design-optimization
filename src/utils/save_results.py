import pandas as pd
from datetime import datetime
from pathlib import Path
import inspect
from config.parameters import Parameters
import json
import numpy as np

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
    """Save optimization results to a file (Excel or JSON)"""
    
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
        ('Variable Cost per KM', parameters.variable_cost_per_km),
        ('Average Speed', parameters.avg_speed),
        ('Max Route Time', parameters.max_route_time),
        ('Service Time per Customer', parameters.service_time),
        ('Max Split Depth', parameters.clustering['max_depth']),
        ('Clustering Method', parameters.clustering['method']),
        ('Clustering Distance', parameters.clustering['distance']),
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
            'Total Cost': total_fixed_cost + total_variable_cost
        }
    }

    try:
        if format == 'json':
            _write_to_json(filename, data)
        else:
            _write_to_excel(filename, data)
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