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
│   ├── clustering_playground.py
│   ├── column_generation_playground.py
│   ├── toy_no_time_constraint.py
│   └── toy_with_time__parallel.py
├── init.sh
├── requirements.txt
└── README.md
```

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

## Directory Structure Details

### Data Directory
- `forecasts/`: Contains demand forecasting results and models (TODO)
- `raw/`: Contains original, unmodified data files
- `*.sql`: SQL scripts for data extraction and transformation
- `*.csv`: Processed data files containing locations and demand
- `import.py`: Creates and populates a sqlite DB called "opperar.db" with sales data from 2023-01 to 2023-09.

### Source Directory (src/)
- `clustering_playground.py`: Implementation of customer clustering algorithms. Capacitated k-means, k-medeoids, hierachical clustering. Generates an html map with the clusters and a CSV summarizing results.
- `column_generation_playground.py`: Column generation optimization methods. We can do it...!
- `toy_no_time_constraint.py`: Full basic model using capacity but no time constraints for the clusters.
- `toy_with_time__parallel.py`: Enhanced model with parallel processing, incorporates time constraints.

## Example Usage

After activating the environment, you can run any of the optimization scripts. For example, to run the basic optimization without time constraints:

```bash
python src/toy_no_time_constraint.py
```

The script will:
1. Load the data from `data/processed/sales_2023_avg_daily_demand.csv`
2. Run the fleet optimization algorithm
3. Output the results to the console

## Methodology

1. Clustering
2. Fleet Size and Mix Optimization
3. Sensitivity Analysis

## License

This project is licensed under the MIT License.