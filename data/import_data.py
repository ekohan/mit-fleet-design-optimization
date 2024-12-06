import sqlite3
from pathlib import Path
import argparse

def import_data_to_unified_sales(db_path: Path):
    """Import all sales data into the unified sales table."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create unified table
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS sales (
            Date TEXT,
            Day INTEGER,
            Month INTEGER,
            Year INTEGER,
            YearMonth TEXT,
            TransportID TEXT,
            ClientID TEXT,
            Material TEXT,
            Description TEXT,
            Units REAL,
            Kg REAL,
            Lat REAL,
            Lon REAL,
            ProductType TEXT,
            SourceYear INTEGER
        );
        
        -- Create indexes for better query performance
        CREATE INDEX IF NOT EXISTS idx_sales_clientid ON sales(ClientID);
        CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(Date);
        CREATE INDEX IF NOT EXISTS idx_sales_yearmonth ON sales(YearMonth);
        CREATE INDEX IF NOT EXISTS idx_sales_producttype ON sales(ProductType);
        CREATE INDEX IF NOT EXISTS idx_sales_sourceyear ON sales(SourceYear);
        """)
        
        # Import 2023 data
        print("Importing 2023 data...")
        with open(Path(__file__).parent / 'sales_2023_transformed.sql', 'r', encoding='utf-8') as f:
            cursor.executescript(f"BEGIN TRANSACTION;\n{f.read()}\nCOMMIT;")
        
        # Import 2024 data
        print("Importing 2024 data...")
        with open(Path(__file__).parent / 'sales_2024_transformed.sql', 'r', encoding='utf-8') as f:
            cursor.executescript(f"BEGIN TRANSACTION;\n{f.read()}\nCOMMIT;")
        
        # Verify row counts
        cursor.execute("SELECT SourceYear, COUNT(*) FROM sales GROUP BY SourceYear")
        counts = cursor.fetchall()
        for year, count in counts:
            print(f"Imported {count} rows for year {year}")
        
        conn.commit()
        print("Import completed successfully")
        
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Import sales data into unified table.')
    parser.add_argument('--db', default='opperar.db', help='Path to SQLite database.')
    args = parser.parse_args()
    
    # Get the script's directory relative to where it's being invoked
    script_dir = Path('data')
    db_path = script_dir / args.db
    import_data_to_unified_sales(db_path) 