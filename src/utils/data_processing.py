import pandas as pd
from pathlib import Path

def data_dir():
    return Path(__file__).resolve().parent.parent.parent / "data"

def load_customer_demand():
    csv_file_path = data_dir() /  "sales_2023_avg_daily_demand.csv"
    print(f"Loading customer demand from {csv_file_path}")
    
    df = pd.read_csv(csv_file_path, header=None, names=['Customer_ID', 'Latitude', 'Longitude', 'Units_Demand', 'Demand_Type'], encoding="latin-1")
    df_pivot = df.pivot_table(index=['Customer_ID', 'Latitude', 'Longitude'], columns='Demand_Type', values='Units_Demand', fill_value=0).reset_index()
    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(columns={'Dry': 'Dry_Demand', 'Chilled': 'Chilled_Demand', 'Frozen': 'Frozen_Demand'})
    if (df_pivot[['Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']] == 0).all(axis=1).any():
        df_pivot.loc[(df_pivot['Dry_Demand'] == 0) & (df_pivot['Chilled_Demand'] == 0) & (df_pivot['Frozen_Demand'] == 0), 'Dry_Demand'] = 1
    return df_pivot
