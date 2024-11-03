import pandas as pd
from datetime import datetime
from pathlib import Path
import inspect
from config import *

def save_optimization_results(
    execution_time,
    solver_name,
    solver_status,
    configurations_df,
    selected_clusters,
    total_fixed_cost,
    total_variable_cost,
    vehicles_used,
    missing_customers,
    filename=None
):
    """
    Save optimization results to an Excel file with multiple sheets
    
    Parameters:
    -----------
    execution_time : float
        Total execution time in seconds
    solver_name : str
        Name of the solver used (e.g., 'GUROBI')
    solver_status : str
        Status of the optimization (e.g., 'Optimal')
    configurations_df : pd.DataFrame
        DataFrame containing vehicle configurations
    selected_clusters : pd.DataFrame
        DataFrame containing the selected clusters
    total_fixed_cost : float
        Total fixed cost of the solution
    total_variable_cost : float
        Total variable cost of the solution
    vehicles_used : pd.Series
        Series containing count of vehicles used by type
    missing_customers : set
        Set of customer IDs that couldn't be served
    filename : str, optional
        Custom filename for the Excel file
    """
    
    # Create a timestamp for the filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(__file__).resolve().parent.parent.parent / 'results'
        filename = results_dir / f"optimization_results_{timestamp}.xlsx"
    
    # Ensure results directory exists
    results_dir = Path(filename).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create Excel writer object
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        try:
            # Sheet 1: Summary and parameters used
            # Calculate statistics
            customers_per_cluster = selected_clusters['Customers'].apply(len)
            
            # Calculate load percentages
            load_percentages = []
            for _, cluster in selected_clusters.iterrows():
                config = configurations_df[
                    configurations_df['Config_ID'] == cluster['Config_ID']
                ].iloc[0]
                max_load_pct = max(
                    cluster['Total_Demand'][good] / config['Capacity'] * 100 
                    for good in GOODS
                )
                load_percentages.append(max_load_pct)
            load_percentages = pd.Series(load_percentages)
            
            # Create summary metrics
            summary_metrics = [
                ('Total Cost ($)', f"{total_fixed_cost + total_variable_cost:,.2f}"),
                ('Total Vehicles', len(selected_clusters)),
            ]
            
            # Add vehicle counts by type
            for vehicle_type in sorted(vehicles_used.index):
                summary_metrics.append(
                    (f'Vehicles Type {vehicle_type}', vehicles_used[vehicle_type])
                )
            
            # Add cluster statistics
            summary_metrics.extend([
                ('Customers per Cluster (Min)', f"{customers_per_cluster.min():.0f}"),
                ('Customers per Cluster (Max)', f"{customers_per_cluster.max():.0f}"),
                ('Customers per Cluster (Avg)', f"{customers_per_cluster.mean():.1f}"),
                ('Customers per Cluster (Median)', f"{customers_per_cluster.median():.1f}"),
                ('Truck Load % (Min)', f"{load_percentages.min():.1f}"),
                ('Truck Load % (Max)', f"{load_percentages.max():.1f}"),
                ('Truck Load % (Avg)', f"{load_percentages.mean():.1f}"),
                ('Truck Load % (Median)', f"{load_percentages.median():.1f}"),
                ('---Parameters---', '')  # Separator
            ])
            
            # Dynamically add all configuration parameters
            import config
            import inspect
            
            # Get all uppercase variables from config module (these are our parameters)
            config_params = {
                name: value for name, value in inspect.getmembers(config)
                if name.isupper()
            }
            
            # Process each parameter type appropriately
            for param_name, value in sorted(config_params.items()):
                if param_name == 'VEHICLE_TYPES':
                    # Handle vehicle types dictionary
                    for v_type, specs in sorted(value.items()):
                        for spec_name, spec_value in specs.items():
                            metric_name = f'Vehicle {v_type} {spec_name}'
                            summary_metrics.append((metric_name, spec_value))
                elif param_name == 'DEPOT':
                    # Handle depot dictionary
                    for key, coord in value.items():
                        metric_name = f'Depot {key}'
                        summary_metrics.append((metric_name, coord))
                elif isinstance(value, (list, tuple)):
                    # Handle list parameters
                    summary_metrics.append((param_name, ', '.join(map(str, value))))
                else:
                    # Handle simple parameters
                    summary_metrics.append((param_name, value))
            
            # Create and save summary DataFrame
            summary_df = pd.DataFrame(summary_metrics, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, sheet_name='Solution Summary', index=False)
            
            # Sheet 2: Vehicle Configurations
            configurations_df.to_excel(writer, sheet_name='Configurations', index=False)
            
            # Sheet 3: Selected Clusters
            cluster_details = selected_clusters.copy()
            if 'Customers' in cluster_details.columns:
                cluster_details['Num_Customers'] = cluster_details['Customers'].apply(len)
                cluster_details['Customers'] = cluster_details['Customers'].apply(str)
            if 'Total_Demand' in cluster_details.columns:
                cluster_details['Total_Demand'] = cluster_details['Total_Demand'].apply(str)
            
            cluster_details.to_excel(writer, sheet_name='Selected Clusters', index=False)
            
            # Sheet 4: Vehicle Usage
            vehicles_df = pd.DataFrame(vehicles_used).reset_index()
            vehicles_df.columns = ['Vehicle Type', 'Count']
            vehicles_df.to_excel(writer, sheet_name='Vehicle Usage', index=False)
            
            # Sheet 5: Other Considerations
            other_considerations = pd.DataFrame([{
                'Total Vehicles Used': len(selected_clusters),
                'Number of Unserved Customers': len(missing_customers),
                'Unserved Customers': str(list(missing_customers)) if missing_customers else "None",
                'Average Customers per Cluster': cluster_details['Num_Customers'].mean() if 'Num_Customers' in cluster_details.columns else 'N/A',
                'Average Distance per Cluster': cluster_details['Estimated_Distance'].mean() if 'Estimated_Distance' in cluster_details.columns else 'N/A'
            }])
            other_considerations.to_excel(writer, sheet_name='Other Considerations', index=False)

            # Sheet 6: Execution Details
            execution_details = pd.DataFrame([{
                'Execution Time (s)': execution_time,
                'Solver': solver_name,
                'Solver Status': solver_status,
                'Total Fixed Cost': total_fixed_cost,
                'Total Variable Cost': total_variable_cost,
                'Total Cost': total_fixed_cost + total_variable_cost
            }])
            execution_details.to_excel(writer, sheet_name='Execution Details', index=False)
            
        except Exception as e:
            print(f"Error while saving results: {str(e)}")
            raise