import pandas as pd
from datetime import datetime
from pathlib import Path

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
    
    # Create Excel writer object
    with pd.ExcelWriter(filename) as writer:
        # Sheet 1: Execution Details
        execution_details = pd.DataFrame([{
            'Execution Time (s)': execution_time,
            'Solver': solver_name,
            'Solver Status': solver_status,
            'Total Fixed Cost': total_fixed_cost,
            'Total Variable Cost': total_variable_cost,
            'Total Cost': total_fixed_cost + total_variable_cost
        }])
        execution_details.to_excel(writer, sheet_name='Execution Details', index=False)
        
        # Sheet 2: Vehicle Configurations
        configurations_df.to_excel(writer, sheet_name='Configurations', index=False)
        
        # Sheet 3: Selected Clusters
        # Ensure all necessary columns are present
        cluster_details = selected_clusters.copy()
        if 'Customers' in cluster_details.columns:
            cluster_details['Num_Customers'] = cluster_details['Customers'].apply(len)
            cluster_details['Customers'] = cluster_details['Customers'].apply(str)  # Convert list to string
        if 'Total_Demand' in cluster_details.columns:
            cluster_details['Total_Demand'] = cluster_details['Total_Demand'].apply(str)  # Convert dict to string
        
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