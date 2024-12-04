import sqlite3
from tqdm import tqdm
import sys
from pathlib import Path


# Define file paths and table name
sql_file_path = Path(__file__).resolve().parent / 'sales_2024_create_data.sql'
db_path = Path(__file__).resolve().parent / 'opperar.db'
table_name = 'sales_2024'  # Set the table name here

# Connect to SQLite database (it creates the file if it doesn't exist)
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if the table already exists
cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
table_exists = cursor.fetchone()

if table_exists:
    print(f"Table '{table_name}' already exists. Aborting script.")
    conn.close()
    sys.exit()

# Read the SQL file as a single statement
with open(sql_file_path, 'r') as sql_file:
    sql_script = sql_file.read()

try:
    # Execute the entire script at once instead of splitting
    cursor.executescript(sql_script)
    
    # Verify the number of rows inserted
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    print(f"Total rows inserted: {row_count}")

except sqlite3.Error as e:
    print(f"An error occurred: {e}")
    conn.rollback()
else:
    conn.commit()
finally:
    conn.close()

print(f"Database '{db_path}' created and data inserted successfully.")
