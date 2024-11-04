"""
Test data generator for FSM optimization.
"""

import pandas as pd
from typing import Dict, Tuple, List
from config import DEPOT, GOODS

def generate_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate simplified test data for FSM optimization."""
    
    # 1. Vehicle Configurations
    configurations_data = [
        {
            'Config_ID': i,  # Start from 0
            'Vehicle_Type': i+1,
            'Fixed_Cost': cost,
            'Variable_Cost': var_cost,
            'Capacity': cap,
            'Dry': dry,
            'Chilled': ref,
            'Frozen': frozen
        }
        for i, (cost, var_cost, cap, dry, ref, frozen) in enumerate([
            (10, 0.5, 100, 1, 1, 1),
            (15, 0.2, 150, 1, 1, 1),
            (20, 0.3, 200, 1, 1, 1),
            (30, 0.4, 100, 1, 1, 1),
            (40, 0.2, 120, 1, 1, 1)
        ])
    ]
    configurations_df = pd.DataFrame(configurations_data)
    
    # 2. Customers
    customers_data = []
    for i in range(10):
        customer = {
            'Customer_ID': f'CUST_{i}',  # String IDs
            'Latitude': DEPOT['Latitude'] + ((i+1) * 0.05),
            'Longitude': DEPOT['Longitude'] + ((i+1) * 0.05),
            'Dry_Demand': 25,
            'Chilled_Demand': 20,
            'Frozen_Demand': 15
        }
        customers_data.append(customer)
    
    customers_df = pd.DataFrame(customers_data)
    
    # 3. Clusters as fixed input
    clusters_data = [
        {
            'Cluster_ID': 0,
            'Config_ID': None,
            'Centroid_Latitude': DEPOT['Latitude'] + 0.2,
            'Centroid_Longitude': DEPOT['Longitude'] + 0.2,
            'Total_Demand': {
                'Dry': 32,    # 40% of 80
                'Chilled': 28,  # 35% of 80
                'Frozen': 20    # 25% of 80
            },
            'Customers': ['CUST_0', 'CUST_1'],
            'Goods_In_Config': GOODS,
            'Route_Time': 90  # 50 + (2 * 20)
        },
        {
            'Cluster_ID': 1,
            'Config_ID': None,
            'Centroid_Latitude': DEPOT['Latitude'] + 0.3,
            'Centroid_Longitude': DEPOT['Longitude'] + 0.3,
            'Total_Demand': {
                'Dry': 48,    # 40% of 120
                'Chilled': 42,  # 35% of 120
                'Frozen': 30    # 25% of 120
            },
            'Customers': ['CUST_2', 'CUST_3'],
            'Goods_In_Config': GOODS,
            'Route_Time': 100  # 40 + (2 * 30)
        },
        {
            'Cluster_ID': 2,
            'Config_ID': None,
            'Centroid_Latitude': DEPOT['Latitude'] + 0.2,
            'Centroid_Longitude': DEPOT['Longitude'] + 0.2,
            'Total_Demand': {
                'Dry': 60,    # 40% of 150
                'Chilled': 52,  # 35% of 150
                'Frozen': 38    # 25% of 150
            },
            'Customers': ['CUST_4', 'CUST_5'],
            'Goods_In_Config': GOODS,
            'Route_Time': 60  # 20 + (2 * 20)
        },
        {
            'Cluster_ID': 3,
            'Config_ID': None,
            'Centroid_Latitude': DEPOT['Latitude'] + 0.35,
            'Centroid_Longitude': DEPOT['Longitude'] + 0.35,
            'Total_Demand': {
                'Dry': 40,    # 40% of 100
                'Chilled': 35,  # 35% of 100
                'Frozen': 25    # 25% of 100
            },
            'Customers': ['CUST_6', 'CUST_7'],
            'Goods_In_Config': GOODS,
            'Route_Time': 80  # 10 + (2 * 35)
        },
        {
            'Cluster_ID': 4,
            'Config_ID': None,
            'Centroid_Latitude': DEPOT['Latitude'] + 0.4,
            'Centroid_Longitude': DEPOT['Longitude'] + 0.4,
            'Total_Demand': {
                'Dry': 20,    # 40% of 50
                'Chilled': 17,  # 35% of 50
                'Frozen': 13    # 25% of 50
            },
            'Customers': ['CUST_8', 'CUST_9'],
            'Goods_In_Config': GOODS,
            'Route_Time': 110  # 30 + (2 * 40)
        }
    ]
    
    clusters_df = pd.DataFrame(clusters_data)
    
    # Set indexes
    configurations_df.set_index('Config_ID', inplace=True)
    clusters_df.set_index('Cluster_ID', inplace=True)
    customers_df.set_index('Customer_ID', inplace=True)
    
    # Debug prints
    print("\nDebug Information:")
    print("Configurations:", len(configurations_df))
    print("Clusters:", len(clusters_df))
    print("Customers:", len(customers_df))
    print("\nCustomer Assignments:")
    for idx, cluster in clusters_df.iterrows():
        print(f"Cluster {idx}: {cluster['Customers']}")
    
    return configurations_df, clusters_df, customers_df

if __name__ == "__main__":
    # Test the data generation
    configs_df, clusters_df, customers_df = generate_test_data()
    print("\nConfigurations DataFrame:")
    print(configs_df)
    print("\nClusters DataFrame:")
    print(clusters_df)
    print("\nCustomers DataFrame:")
    print(customers_df)
    
    # Verify customer assignments
    print("\nCustomer Assignments:")
    for cluster_id, cluster in clusters_df.iterrows():
        print(f"\nCluster {cluster_id} customers: {cluster['Customers']}")