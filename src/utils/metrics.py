import numpy as np
import pandas as pd
from typing import Dict
from src.config.parameters import Parameters

def calculate_vehicle_utilization(selected_clusters: pd.DataFrame, configs_df: pd.DataFrame, params: Parameters) -> float:
    """Calculate average vehicle utilization across selected clusters."""
    utilizations = []
    for _, cluster in selected_clusters.iterrows():
        config_id = cluster.get('Config_ID')
        if config_id is None:
            continue
            
        config = configs_df[configs_df['Config_ID'] == config_id].iloc[0]
        capacity = config['Capacity']
        
        # Get total demand based on the structure of the cluster
        if 'Total_Demand' in cluster:
            if isinstance(cluster['Total_Demand'], dict):
                total_demand = sum(cluster['Total_Demand'].values())
            else:
                total_demand = cluster['Total_Demand']
        else:
            demands = [cluster.get(f'{product}_Demand', 0) for product in params.goods]
            total_demand = sum(demands)
        
        utilization = total_demand / capacity
        utilizations.append(utilization)
        
    return np.mean(utilizations) if utilizations else 0.0

def count_vehicles_by_type(selected_clusters: pd.DataFrame) -> Dict[str, int]:
    """Count number of vehicles used by configuration type."""
    return selected_clusters['Config_ID'].value_counts().to_dict() 