# MIT Fleet Design Optimization

This project focuses on optimizing fleet design for a food distribution company using Multi-Compartment Vehicles (MCVs). The goal is to define the optimal fleet composition to serve customer demand efficiently while minimizing costs.

## Project Structure

```
mit-fleet-design-optimization/
├── data/
│   ├── forecasts/
│   ├── raw/
│   ├── export_avg_daily_demand.sql
│   ├── sales_2023_avg_daily_demand.csv
│   └── sales_2023_create_data.sql
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── main.py           # Principal execution script
│   ├── clustering.py     # Customer clustering implementation
│   ├── fsm_optimizer.py  # Fleet Size and Mix optimization
│   ├── utils/
│   │   ├── config_utils.py
│   │   ├── data_processing.py
│   │   └── save_results.py
│   └── config.py         # Configuration parameters
├── init.sh
├── requirements.txt
└── README.md
```

## Directory Structure Details

### Data Directory
- `forecasts/`: Contains demand forecasting models and results
- `raw/`: Original, unmodified data files
- `sales_2023_avg_daily_demand.csv`: Processed customer demand data
- `import.py`: Creates and populates a SQLite database (opperar.db) with sales data from 2023-01 to 2023-09

### Source Directory (src/)
- `main.py`: Principal execution script that runs the complete optimization pipeline
- `clustering.py`: Implements customer clustering algorithms with capacity and time constraints
- `fsm_optimizer.py`: Fleet Size and Mix optimization using integer programming
- `utils/`: Helper modules
  - `config_utils.py`: Vehicle configuration generation and validation
  - `data_processing.py`: Data loading and preprocessing functions
  - `save_results.py`: Functions for saving and exporting results
- `config.py`: Global configuration parameters (vehicle types, goods, depot location)

## Requirements

### Installation Steps

1. Initialize the project environment:
```bash
./init.sh
```

2. Activate the Python environment:

For Mac/Linux:
```bash
source mit-fleet-env/bin/activate
```

For Windows:
```bash
mit-fleet-env\Scripts\activate
```

## Running the Optimization

To run the fleet optimization:
```bash
python src/main.py
```

The script will:
1. Load customer demand data from a csv in the data directory.
2. Generate vehicle configurations based on defined parameters
3. Create customer clusters considering vehicle capacities and time constraints
4. Solve the Fleet Size and Mix optimization problem
5. Output detailed results including:
   - Cost breakdown (fixed and variable costs)
   - Vehicle allocation by type
   - Cluster details with demands and distances
   - Route time estimations
   - Solution statistics

## Methodology

The optimization pipeline follows these steps:

1. Data Processing
   - Load customer demand data
   - Generate feasible vehicle configurations

2. Clustering
   - Group customers based on location and demand
   - Consider vehicle capacities and time constraints
   - Use parallel processing for efficiency

3. Fleet Size and Mix Optimization
   - Solve integer programming model
   - Minimize total cost (fixed + variable)
   - Ensure all customers are served
   - Respect vehicle capacities

4. Solution Validation and Results Export
   - Validate cluster feasibility
   - Calculate detailed costs
   - Generate comprehensive reports

## License

This project is licensed under the MIT License.