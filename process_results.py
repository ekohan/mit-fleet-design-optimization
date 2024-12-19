#!/usr/bin/env python3

import json
import csv
from pathlib import Path
import logging
import re

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def extract_instance_metrics(filename: str) -> tuple:
    """Extract N and K values from filename."""
    # Pattern matches X-n<number>-k<number> or combined instances
    pattern = r'X-n(\d+)-k(\d+)'
    matches = re.findall(pattern, filename)
    
    if not matches:
        return 0, 0
    
    # Sum up values if multiple instances
    total_n = sum(int(n) for n, _ in matches)
    total_k = sum(int(k) for _, k in matches)
    
    return total_n, total_k

def extract_transformation_type(filename: str) -> str:
    """Extract transformation type from filename."""
    if filename.endswith('_split'):
        return 'Split'
    elif filename.endswith('_scaled'):
        return 'Scaled'
    elif filename.endswith('_spatial'):
        return 'Spatial'
    elif filename.endswith('_combined'):
        return 'Combined'
    elif filename.endswith('_normal'):
        return 'Normal'
    return 'Unknown'

def extract_metrics_from_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    
    # Extract instance metrics
    n_customers, k_min_vehicles = extract_instance_metrics(json_file.stem)
    total_vehicles = data["Solution Summary"]["Total Vehicles"]
    
    # Extract vehicle capacity based on transformation type
    transformation = extract_transformation_type(json_file.stem)
    summary = data["Solution Summary"]
    
    # Find all vehicle capacity entries in the summary
    capacities = []
    for key in summary:
        if key.startswith("Vehicle") and key.endswith("capacity"):
            capacities.append(str(summary[key]))
    
    # Join capacities with '+' if multiple, otherwise take the single value
    vehicle_capacity = '+'.join(capacities) if len(capacities) > 1 else capacities[0]
    
    # Calculate gap percentage
    gap = ((total_vehicles - k_min_vehicles) / k_min_vehicles * 100) if k_min_vehicles > 0 else 0
    # Calculate customers per vehicle
    customers_per_vehicle = n_customers / k_min_vehicles if k_min_vehicles > 0 else 0
    
    return {
        "Filename": json_file.stem,
        "Transformation": transformation,
        "N Customers": n_customers,
        "K Min Vehicles": k_min_vehicles,
        "Total Vehicles": total_vehicles,
        "Gap (%)": round(gap, 2),
        "Customers per vehicle": round(customers_per_vehicle, 2),
        "Vehicle Capacity": vehicle_capacity,
        "Truck Load % (Min)": data["Solution Summary"]["Truck Load % (Min)"],
        "Truck Load % (Max)": data["Solution Summary"]["Truck Load % (Max)"],
        "Truck Load % (Avg)": data["Solution Summary"]["Truck Load % (Avg)"],
        "Truck Load % (Median)": data["Solution Summary"]["Truck Load % (Median)"],
        "Number of Unserved Customers": data["Other Considerations"]["Number of Unserved Customers"],
        "Execution Time (s)": data["Execution Details"]["Execution Time (s)"],
        "Solver Status": data["Execution Details"]["Solver Status"],
    }

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    results_dir = Path(__file__).parent / 'results'
    output_file = results_dir / 'benchmark_results.csv'
    
    # Get all JSON files
    json_files = list(results_dir.glob('cvrp_*.json'))
    
    if not json_files:
        logger.error("No JSON files found in results directory")
        return
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process all files
    all_metrics = []
    for json_file in json_files:
        try:
            metrics = extract_metrics_from_json(json_file)
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            continue
    
    # Write to CSV
    if all_metrics:
        fieldnames = all_metrics[0].keys()
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metrics)
        
        logger.info(f"Results written to {output_file}")
    else:
        logger.error("No results to write")

if __name__ == "__main__":
    main()
