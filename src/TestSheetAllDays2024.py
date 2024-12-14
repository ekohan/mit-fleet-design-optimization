import os
from pathlib import Path

# Get all CSV files from the demand_profiles directory
demand_dir = Path('data/demand_profiles')
csv_files = [f for f in os.listdir(demand_dir) if f.startswith('sales_2024_') and f.endswith('_demand.csv')]

# Generate commands file
output_file = Path('run_all_days.txt')

with open(output_file, 'w') as f:
    for csv_file in sorted(csv_files):
        command = f'python src/main.py --verbose --demand-file {csv_file}\n'
        f.write(command)

print(f"Generated {len(csv_files)} commands in {output_file}")