import os
from pathlib import Path

# Get all instance files from the cvrp_instances directory
instances_dir = Path('src/benchmarking/cvrp_instances')
instance_files = [Path(f).stem for f in os.listdir(instances_dir) if f.endswith('.vrp')]

# Generate commands file
output_file = Path('run_all_benchmarks.txt')

with open(output_file, 'w') as f:
    for instance in sorted(instance_files):
        # Normal benchmark
        command = f'python src/benchmarking/cvrp_to_fsm.py --instance {instance} --benchmark-type normal --format excel --output cvrp_{instance}_normal.xlsx\n'
        f.write(command)
        
        # Split benchmark with 2 goods
        command = f'python src/benchmarking/cvrp_to_fsm.py --instance {instance} --benchmark-type split --num-goods 2 --format excel --output cvrp_{instance}_split2.xlsx\n'
        f.write(command)
        
        # Split benchmark with 3 goods
        command = f'python src/benchmarking/cvrp_to_fsm.py --instance {instance} --benchmark-type split --num-goods 3 --format excel --output cvrp_{instance}_split3.xlsx\n'
        f.write(command)
        
        # Scaled benchmark
        command = f'python src/benchmarking/cvrp_to_fsm.py --instance {instance} --benchmark-type scaled --format excel --output cvrp_{instance}_scaled.xlsx\n'
        f.write(command)

print(f"Generated {len(instance_files) * 4} commands in {output_file}")