import os
from pathlib import Path
import platform

# Get all CSV files from the demand_profiles directory that are filtered daily files
demand_dir = Path('data/demand_profiles')
csv_files = [f for f in os.listdir(demand_dir) if f.startswith('sales_2024_') and f.endswith('_demand_grouped.csv') and '-' in f]

# Determine OS-specific null redirect
if platform.system() == 'Windows':
    null_redirect = '| Out-Null'  # PowerShell syntax
else:
    null_redirect = '> /dev/null 2>&1'  # Unix/Linux/Mac syntax

# Generate commands file
output_file = Path('run_all_days_grouped.txt')

with open(output_file, 'w') as f:
    for csv_file in sorted(csv_files):
        # Main program run
        command = f'python src/main.py --demand-file {csv_file} {null_redirect}\n'
        f.write(command)
        
        # Single compartment benchmark
        command = f'python src/benchmarking/run_benchmark.py --benchmark-type single_compartment --demand-file {csv_file} {null_redirect}\n'
        f.write(command)
        
        # Multi compartment benchmark
        command = f'python src/benchmarking/run_benchmark.py --benchmark-type multi_compartment --demand-file {csv_file} {null_redirect}\n'
        f.write(command)

total_commands = len(csv_files) * 3  # 3 commands per file
print(f"Generated {total_commands} commands in {output_file}") 