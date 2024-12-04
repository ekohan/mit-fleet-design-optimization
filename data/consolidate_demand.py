import pandas as pd

def consolidate_demand(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Group by all columns except 'Kg' and sum the 'Kg' values
    consolidated_df = df.groupby(['ClientID', 'Lat', 'Lon', 'ProductType'], as_index=False)['Kg'].sum()
    
    # Sort by ClientID to maintain a consistent order
    consolidated_df = consolidated_df.sort_values('ClientID')
    
    # Save the consolidated data to a new CSV file
    output_path = file_path.replace('.csv', '_consolidated.csv')
    consolidated_df.to_csv(output_path, index=False)
    
    print(f"Original number of rows: {len(df)}")
    print(f"Consolidated number of rows: {len(consolidated_df)}")
    print(f"Consolidated file saved as: {output_path}")
    
    return consolidated_df

# Use the function
consolidated_data = consolidate_demand('data\AverageDayGenDemand.csv')