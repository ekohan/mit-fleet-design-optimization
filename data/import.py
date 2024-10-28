import sqlite3
from tqdm import tqdm
import sys

# Define file paths and table name
sql_file_path = 'sales_2023_create_data.sql'
db_path = 'opperar.db'
table_name = 'sales_2023'  # Set the table name here

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

# Read the SQL file
with open(sql_file_path, 'r') as sql_file:
    sql_script = sql_file.read()

# Split the script into individual statements
statements = sql_script.split(';')
total_statements = len(statements)

# Execute each statement with a progress bar
with tqdm(total=total_statements, desc="Executing SQL Statements") as pbar:
    for statement in statements:
        statement = statement.strip()
        if statement:  # Ensure it's not an empty statement
            cursor.execute(statement + ';')  # Re-add the semicolon and execute
            pbar.update(1)  # Update the progress bar

# Commit and close the connection
conn.commit()
conn.close()

print(f"Database '{db_path}' created and data inserted successfully.")
