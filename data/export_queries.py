import sqlite3
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Query to CSV mapping
QUERY_MAPPING = {
    'export_avg_daily_demand.sql': 'sales_2023_avg_daily_demand.csv',
    'avg_daily_demand__2023_09.sql': 'sales_2023_09_avg_daily_demand.csv',
    'export_avg_daily_demand__high_demand_day.sql': 'sales_2023_high_demand_day.csv',
    'export_avg_daily_demand__uniform_visits_per_week.sql': 'sales_2023_uniform_visits_per_week.csv',
    'export_avg_daily_demand__low_demand_day.sql': 'sales_2023_low_demand_day.csv',
    'export_all_2024_demand.sql': 'sales_2024_all_demand.csv'  # Added this line
}

def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).resolve().parent

def get_demand_profiles_dir() -> Path:
    """Get the demand profiles directory path."""
    profiles_dir = get_data_dir() / 'demand_profiles'
    profiles_dir.mkdir(exist_ok=True)
    return profiles_dir

def get_output_path(csv_file: str) -> Path:
    """Get the output path for a CSV file."""
    return get_demand_profiles_dir() / csv_file

def read_sql_file(sql_path: Path) -> str:
    """Read SQL query from file."""
    try:
        with open(sql_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"SQL file not found: {sql_path}")
        raise

def execute_query_to_csv(
    conn: sqlite3.Connection,
    query: str,
    output_path: Path,
    index: bool = False
) -> None:
    """Execute SQL query and save results to CSV."""
    try:
        # Read SQL query with explicit data types
        df = pd.read_sql_query(query, conn, dtype={
            'ClientID': str,
            'Lat': float,
            'Lon': float,
            'Kg': int,  
            'ProductType': str
        })
        
        # Additional data cleaning if needed
        df['Kg'] = pd.to_numeric(df['Kg'], errors='coerce').astype(int)
        
        # First group by ClientID to ensure same noise per client
        client_noise = pd.DataFrame({
            'ClientID': df['ClientID'].unique(),
            'lat_noise': np.random.uniform(-1e-4, 1e-4, size=len(df['ClientID'].unique())),
            'lon_noise': np.random.uniform(-1e-4, 1e-4, size=len(df['ClientID'].unique()))
        })
        
        # Merge noise back to original dataframe
        df = df.merge(client_noise, on='ClientID', how='left')
        df['Lat'] += df['lat_noise']
        df['Lon'] += df['lon_noise']
        df = df.drop(['lat_noise', 'lon_noise'], axis=1)
        
        # Export to CSV
        df.to_csv(output_path, index=index)
        logger.info(f"Exported {len(df)} rows to {output_path}")
    except Exception as e:
        logger.error(f"Error executing query or saving CSV: {str(e)}")
        raise

def execute_query_for_date(
    conn: sqlite3.Connection,
    base_query: str,
    date: str,
    output_path: Path,
    index: bool = False
) -> None:
    """Execute SQL query for a specific date and save results to CSV."""
    # Replace the date placeholder in the query
    query = base_query.replace('{{DATE}}', date)
    execute_query_to_csv(conn, query, output_path, index)

def export_daily_queries(db_path: Optional[Path] = None) -> None:
    """Execute queries for each day and export results to CSV files."""
    if db_path is None:
        db_path = get_data_dir() / 'opperar.db'

    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        data_dir = get_data_dir()

        # Get all dates
        dates_query = read_sql_file(data_dir / 'queries' / 'get_2024_dates.sql')
        dates_df = pd.read_sql_query(dates_query, conn)
        
        # Initialize empty DataFrames for aggregation
        all_regular_data = []
        all_filtered_data = []
        
        # Read both base queries
        base_query = read_sql_file(data_dir / 'queries' / 'export_avg_day_2024_demand.sql')
        base_query_filtered = read_sql_file(data_dir / 'queries' / 'export_avg_day_2024_demand_filtered.sql')
        base_query_center = read_sql_file(data_dir / 'queries' / 'export_all_days_center.sql')
        
        # Export data for each date for both regular and filtered queries
        for date in dates_df['Date']:
            # Regular version
            csv_file = f'sales_2024_{date}_demand.csv'
            csv_path = get_output_path(csv_file)
            logger.info(f"Processing regular data for {date}...")
            execute_query_for_date(conn, base_query, date, csv_path)
            # Also store for aggregation
            query = base_query.replace('{{DATE}}', date)
            df_regular = pd.read_sql_query(query, conn)
            df_regular['Date'] = date
            all_regular_data.append(df_regular)
            
            # Filtered version
            csv_file_filtered = f'sales_2024_{date}_demand_filtered.csv'
            csv_path_filtered = get_output_path(csv_file_filtered)
            logger.info(f"Processing filtered data for {date}...")
            execute_query_for_date(conn, base_query_filtered, date, csv_path_filtered)
            # Also store for aggregation
            query_filtered = base_query_filtered.replace('{{DATE}}', date)
            df_filtered = pd.read_sql_query(query_filtered, conn)
            df_filtered['Date'] = date
            all_filtered_data.append(df_filtered)

            # Center version
            csv_file_center = f'sales_2024_{date}_demand_center.csv'
            csv_path_center = get_output_path(csv_file_center)
            logger.info(f"Processing center data for {date}...")
            execute_query_for_date(conn, base_query_center, date, csv_path_center)

        # Combine and save aggregate files
        logger.info("Creating aggregate files...")
        pd.concat(all_regular_data).to_csv(
            get_output_path('sales_2024_all_days_demand.csv'), 
            index=False
        )
        pd.concat(all_filtered_data).to_csv(
            get_output_path('sales_2024_all_days_demand_filtered.csv'), 
            index=False
        )

        conn.close()
        logger.info("All daily exports completed successfully")

    except Exception as e:
        logger.error(f"Error during export process: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise

def export_all_queries(db_path: Optional[Path] = None) -> None:
    """Execute all SQL queries and export results to CSV files."""
    if db_path is None:
        db_path = get_data_dir() / 'opperar.db'

    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")

        data_dir = get_data_dir()
        
        for sql_file, csv_file in QUERY_MAPPING.items():
            sql_path = data_dir / 'queries' / sql_file
            csv_path = get_output_path(csv_file)
            
            logger.info(f"Processing {sql_file}...")
            query = read_sql_file(sql_path)
            execute_query_to_csv(conn, query, csv_path)

        conn.close()
        logger.info("All queries exported successfully")

    except Exception as e:
        logger.error(f"Error during export process: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise

if __name__ == "__main__":
    export_all_queries()
    export_daily_queries() 