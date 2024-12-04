import pandas as pd
from pathlib import Path

def create_sql_file(csv_path: str, output_path: str):
    """Generate SQL file from CSV data."""
    # Read the CSV file with correct dtypes
    df = pd.read_csv(csv_path, dtype={
        'Date': str,
        'TransportID': str,  # Convert to string to preserve full number
        'ClientID': str,     # Convert to string to preserve full number
        'Material': str,     # Convert to string to preserve full number
        'ProductType': str,
        'Kg': float,
        'Lat': float,
        'Lon': float
    })
    
    # Convert dates and split into components
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['YearMonth'] = df['Date'].dt.strftime('%Y.%m')
    
    # Translate product types
    product_type_map = {
        'Seco': 'Dry',
        'Refrigerado': 'Chilled',
        'Congelado': 'Frozen'
    }
    df['ProductType'] = df['ProductType'].map(product_type_map)
    
    # Round Kg values to integers
    df['Kg'] = df['Kg'].round().astype(int)
    
    # Create the SQL statements
    sql_statements = []
    
    # Create table statement
    sql_statements.append("""
-- Create the sales_2024 table
CREATE TABLE IF NOT EXISTS sales_2024 (
    Date TEXT,
    Day INTEGER,
    Month INTEGER,
    Year INTEGER,
    YearMonth TEXT,
    TransportID TEXT,
    ClientID TEXT,
    Material TEXT,
    ProductType TEXT,
    Kg INTEGER,
    Lat REAL,
    Lon REAL
);

-- Begin transaction for better performance
BEGIN TRANSACTION;
    """)
    
    # Generate insert statements
    insert_template = """INSERT INTO sales_2024 
        (Date, Day, Month, Year, YearMonth, TransportID, ClientID, Material, ProductType, Kg, Lat, Lon) 
    VALUES"""
    batch_size = 1000  # Number of rows per INSERT statement
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        values = []
        
        for _, row in batch.iterrows():
            try:
                # Format each value appropriately and handle potential NULL values
                value = f"('{row['Date'].strftime('%Y-%m-%d')}', {row['Day']}, {row['Month']}, {row['Year']}, '{row['YearMonth']}', '{row['TransportID']}', '{row['ClientID']}', '{row['Material']}', '{row['ProductType']}', {row['Kg']}, {row['Lat']:.6f}, {row['Lon']:.6f})"
                values.append(value)
            except Exception as e:
                print(f"Error processing row: {row}")
                print(f"Error details: {e}")
                continue
        
        if values:  # Only add statement if we have values
            # Join all values and add to SQL statements
            joined_values = ',\n'.join(values)
            sql_statements.append(f"{insert_template}\n{joined_values};")
    
    # Add closing statements
    sql_statements.append("""
-- Commit the transaction
COMMIT;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_sales_2024_clientid ON sales_2024(ClientID);
CREATE INDEX IF NOT EXISTS idx_sales_2024_date ON sales_2024(Date);
CREATE INDEX IF NOT EXISTS idx_sales_2024_yearmonth ON sales_2024(YearMonth);
CREATE INDEX IF NOT EXISTS idx_sales_2024_producttype ON sales_2024(ProductType);
    """)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(sql_statements))
    
    # Verify row count
    total_rows = 0
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Only count lines that are actual data rows (start with parenthesis and end with comma or parenthesis)
            if line.startswith('(') and (line.endswith(',') or line.endswith(')')):
                total_rows += 1
    
    print(f"\nSQL file generated successfully at {output_path}")
    print(f"Total rows in DataFrame: {len(df)}")
    print(f"Total rows in SQL file: {total_rows}")
    if len(df) != total_rows:
        print(f"WARNING: Row count mismatch! {len(df) - total_rows} rows were lost during processing.")

if __name__ == "__main__":
    # Define paths
    data_dir = Path(__file__).resolve().parent
    csv_path = data_dir / '2024Data.csv'
    sql_path = data_dir / 'sales_2024_create_data.sql'
    
    # Generate SQL file
    create_sql_file(csv_path, sql_path) 