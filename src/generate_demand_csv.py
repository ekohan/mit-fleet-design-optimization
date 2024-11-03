import sqlite3
import csv

def execute_query(query_option="average", params=None):
    """
    Executes a given SQL query on the database based on query_option.
    
    Parameters:
        query_option (str): The name of the query to execute ("average" or "monthly_spread").
        params (tuple): Optional parameters to use with the query.
    
    Returns:
        list of tuples: The results of the query.
        list: Column names of the results.
    """
    # Connect to the database
    connection = sqlite3.connect('data/opperar.db')
    cursor = connection.cursor()
    
    # Define the queries as options
    queries = {
        "average": """
                SELECT
                    *
                from (
                    select
                        ClientID,
                        Lat,
                        Lon,
                        round(sum(Kg)/9/30) as 'Kg', -- quick & dirty daily avg
                        ProductType
                    from sales_2023
                    group by ClientID, ProductType
                ) a

                where a.Lat > 0 and a.Lat < 10 -- filter incomplete/dirty records
                and a.Kg >= 0;
        """,
        
        "monthly_spread": """
            WITH ranked_customers AS (
                SELECT 
                    ClientID,
                    Lat,
                    Lon,
                    ProductType,
                    ROUND(SUM(Kg) / 9, 2) AS MonthlyKg,  -- monthly demand
                    ROW_NUMBER() OVER (ORDER BY ClientID) AS RowNum -- rank clients for day assignment
                FROM sales_2023
                GROUP BY ClientID, ProductType
            ),
            distributed_demand AS (
                SELECT
                    ClientID,
                    Lat,
                    Lon,
                    ProductType,
                    ROUND(MonthlyKg / 24, 2) AS DailyKg,  -- spread monthly demand across 24 days
                    ((RowNum - 1) % 24) + 1 AS Day  -- assign each client to one of 24 days
                FROM ranked_customers
            )
            SELECT 
                Day,
                ClientID,
                Lat,
                Lon,
                ProductType,
                DailyKg
            FROM distributed_demand
            WHERE Lat > 0 AND Lat < 10  -- filter incomplete/dirty records
            AND DailyKg >= 0
            ORDER BY Day, ClientID;
        """
    }
    
    # Select the query based on the option
    query = queries.get(query_option.lower())
    if not query:
        print(f"Invalid query option '{query_option}'. Please choose 'average' or 'monthly_spread'.")
        return None, None

    try:
        # Execute the query with parameters if provided
        print(f"Executing '{query_option}' query...")
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        print("Query executed successfully.")
        
        # Fetch and return results if it's a SELECT query
        results = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        return results, columns
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return None, None
    finally:
        # Close the connection
        cursor.close()
        connection.close()

# Choose the query to run
query_option = "monthly_spread"  # Change this to "average" or "monthly_spread" as needed
results, columns = execute_query(query_option)

# Check if any data was returned and export to CSV
if results:
    csv_file = f"data/sales_2023_{query_option}_daily_demand.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write header
        writer.writerows(results) # Write data rows
    print(f"Data exported to {csv_file}")
else:
    print("No data found for the query.")
