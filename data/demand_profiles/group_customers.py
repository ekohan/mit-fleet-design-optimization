import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Optional
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from data.export_queries import get_data_dir, get_output_path, read_sql_file

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def group_customers(df: pd.DataFrame, eps_km: float = 2.0, min_samples: int = 2) -> pd.DataFrame:
    """
    Group customers based on location and product type constraints.
    Only groups customers with demand ≤ 20kg and limits clusters to 200kg.
    
    Args:
        df: DataFrame with customer data
        eps_km: Maximum distance in kilometers between points in a cluster
        min_samples: Minimum number of samples in a cluster
    """
    # Convert km to degrees (approximate)
    eps_degrees = eps_km / 111.0  # 1 degree ≈ 111 km
    
    # First, separate by product type due to temperature requirements
    grouped_dfs = []
    
    for product_type in df['ProductType'].unique():
        product_df = df[df['ProductType'] == product_type].copy()
        
        if len(product_df) > 0:
            # Separate large and small orders
            large_orders = product_df[product_df['Kg'] > 30]
            small_orders = product_df[product_df['Kg'] <= 30]
            
            # Keep large orders as is
            if len(large_orders) > 0:
                grouped_dfs.append(large_orders)
            
            if len(small_orders) > 0:
                # Perform DBSCAN clustering on small orders
                coords = small_orders[['Lat', 'Lon']].values
                clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples, metric='haversine').fit(coords)
                
                small_orders['cluster'] = clustering.labels_
                
                # Group within clusters while respecting weight limits
                grouped = []
                for cluster in small_orders['cluster'].unique():
                    cluster_df = small_orders[small_orders['cluster'] == cluster]
                    
                    # If cluster total weight exceeds 200kg, split it
                    total_weight = cluster_df['Kg'].sum()
                    if total_weight > 200:
                        # Process orders sequentially until reaching weight limit
                        current_group = []
                        current_weight = 0
                        
                        for _, row in cluster_df.iterrows():
                            if current_weight + row['Kg'] <= 200:
                                current_group.append(row)
                                current_weight += row['Kg']
                            else:
                                # Aggregate current group
                                if current_group:
                                    group_df = pd.DataFrame(current_group)
                                    agg = group_df.agg({
                                        'Date': lambda x: x.iloc[0],
                                        'Lat': 'mean',
                                        'Lon': 'mean',
                                        'Kg': 'sum',
                                        'ProductType': lambda x: x.iloc[0],
                                        'ClientID': lambda x: x.iloc[0]
                                    }).to_frame().T
                                    grouped.append(agg)
                                # Start new group
                                current_group = [row]
                                current_weight = row['Kg']
                        
                        # Don't forget the last group
                        if current_group:
                            group_df = pd.DataFrame(current_group)
                            agg = group_df.agg({
                                'Date': lambda x: x.iloc[0],
                                'Lat': 'mean',
                                'Lon': 'mean',
                                'Kg': 'sum',
                                'ProductType': lambda x: x.iloc[0],
                                'ClientID': lambda x: x.iloc[0]
                            }).to_frame().T
                            grouped.append(agg)
                    else:
                        # Aggregate cluster into single point
                        agg = cluster_df.agg({
                            'Date': lambda x: x.iloc[0],
                            'Lat': 'mean',
                            'Lon': 'mean',
                            'Kg': 'sum',
                            'ProductType': lambda x: x.iloc[0],
                            'ClientID': lambda x: x.iloc[0]
                        }).to_frame().T
                        grouped.append(agg)
                
                if grouped:
                    grouped_dfs.append(pd.concat(grouped))
    
    return pd.concat(grouped_dfs) if grouped_dfs else pd.DataFrame()

def execute_query_to_csv_grouped(
    conn: sqlite3.Connection,
    query: str,
    output_path: Path,
    index: bool = False
) -> None:
    """Execute SQL query, group customers, and save results to CSV."""
    try:
        # Read SQL query with explicit data types
        df = pd.read_sql_query(query, conn, dtype={
            'ClientID': str,
            'Lat': float,
            'Lon': float,
            'Kg': int,
            'ProductType': str
        })
        
        # Group customers
        logger.info("Grouping customers...")
        grouped_df = group_customers(df)
        
        # Add location noise
        client_noise = pd.DataFrame({
            'ClientID': grouped_df['ClientID'].unique(),
            'lat_noise': np.random.uniform(-1e-4, 1e-4, size=len(grouped_df['ClientID'].unique())),
            'lon_noise': np.random.uniform(-1e-4, 1e-4, size=len(grouped_df['ClientID'].unique()))
        })
        
        grouped_df = grouped_df.merge(client_noise, on='ClientID', how='left')
        grouped_df['Lat'] += grouped_df['lat_noise']
        grouped_df['Lon'] += grouped_df['lon_noise']
        grouped_df = grouped_df.drop(['lat_noise', 'lon_noise'], axis=1)
        
        # Export to CSV
        grouped_df.to_csv(output_path, index=index)
        logger.info(f"Exported {len(grouped_df)} grouped orders to {output_path}")
        
        # Log grouping statistics
        original_count = len(df)
        grouped_count = len(grouped_df)
        reduction = ((original_count - grouped_count) / original_count) * 100
        logger.info(f"Reduced from {original_count} to {grouped_count} orders ({reduction:.1f}% reduction)")
        
    except Exception as e:
        logger.error(f"Error executing query or saving CSV: {str(e)}")
        raise

def execute_query_for_date_grouped(
    conn: sqlite3.Connection,
    base_query: str,
    date: str,
    output_path: Path,
    index: bool = False
) -> None:
    """Execute SQL query for a specific date, group customers, and save results to CSV."""
    query = base_query.replace('{{DATE}}', date)
    execute_query_to_csv_grouped(conn, query, output_path, index)

# Modified export_daily_queries function
def export_daily_queries_grouped(db_path: Optional[Path] = None) -> None:
    """Execute queries for each day, group customers, and export results to CSV files."""
    if db_path is None:
        db_path = get_data_dir() / 'opperar.db'

    try:
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database: {db_path}")
        data_dir = get_data_dir()

        # Get all dates
        dates_query = read_sql_file(data_dir / 'queries' / 'get_2024_dates.sql')
        dates_df = pd.read_sql_query(dates_query, conn)
        
        # Read base query
        base_query_filtered = read_sql_file(data_dir / 'queries' / 'export_avg_day_2024_demand.sql')
        
        # Export grouped data for each date
        for date in dates_df['Date']:
            csv_file_grouped = f'sales_2024_{date}_demand_grouped.csv'
            csv_path_grouped = get_output_path(csv_file_grouped)
            logger.info(f"Processing and grouping data for {date}...")
            execute_query_for_date_grouped(conn, base_query_filtered, date, csv_path_grouped)

        conn.close()
        logger.info("All daily exports completed successfully")

    except Exception as e:
        logger.error(f"Error during export process: {str(e)}")
        if 'conn' in locals():
            conn.close()
        raise

if __name__ == "__main__":
    export_daily_queries_grouped()