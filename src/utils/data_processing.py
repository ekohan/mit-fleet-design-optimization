import pandas as pd
from pathlib import Path

def data_dir():
    return Path(__file__).resolve().parent.parent.parent / "data"

def load_customer_demand(demand_file: str):
    csv_file_path = data_dir() / demand_file
    print(f"Loading customer demand from {csv_file_path}")
    
    # Read CSV with existing headers
    df = pd.read_csv(
        csv_file_path,
        dtype={
            'ClientID': str,
            'Lat': float,
            'Lon': float,
            'Kg': int,
            'ProductType': str
        },
        encoding="latin-1"
    )
    
    # Rename columns to match our expected format
    df = df.rename(columns={
        'ClientID': 'Customer_ID',
        'Lat': 'Latitude',
        'Lon': 'Longitude',
        'Kg': 'Units_Demand',
        'ProductType': 'Demand_Type'
    })
    
    # Create pivot table
    df_pivot = df.pivot_table(
        index=['Customer_ID', 'Latitude', 'Longitude'],
        columns='Demand_Type',
        values='Units_Demand',
        fill_value=0,
        aggfunc='sum'
    ).reset_index()
    
    df_pivot.columns.name = None
    df_pivot = df_pivot.rename(columns={
        'Dry': 'Dry_Demand',
        'Chilled': 'Chilled_Demand',
        'Frozen': 'Frozen_Demand'
    })
    
    # Ensure all demand columns are integers
    demand_cols = ['Dry_Demand', 'Chilled_Demand', 'Frozen_Demand']
    for col in demand_cols:
        df_pivot[col] = df_pivot[col].astype(int)
    
    # Set zero demand to 1 if all demands are zero
    if (df_pivot[demand_cols] == 0).all(axis=1).any():
        df_pivot.loc[
            (df_pivot['Dry_Demand'] == 0) & 
            (df_pivot['Chilled_Demand'] == 0) & 
            (df_pivot['Frozen_Demand'] == 0),
            'Dry_Demand'
        ] = 1
    
    return df_pivot
