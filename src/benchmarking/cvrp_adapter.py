from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.config.parameters import Parameters

@dataclass
class VRPTWInstance:
    """Stores VRPTW instance data"""
    name: str
    dimension: int
    capacity: float
    depot_coords: Tuple[float, float]
    customer_coords: List[Tuple[float, float]]
    demands: List[float]
    time_windows: List[Tuple[float, float]]
    service_times: List[float]
    best_known_vehicles: int = 0
    service_time: float = 0

def parse_vrptw_file(file_path: str) -> VRPTWInstance:
    """Parse a VRPTW instance file"""
    sol_path = str(Path(file_path)).replace('.vrp', '.sol')
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Initialize variables
    dimension = 0
    capacity = 0
    coords = []
    demands = []
    time_windows = []
    service_times = []
    name = Path(file_path).stem
    service_time = 0
    
    section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if 'DIMENSION' in line:
            dimension = int(line.split(':')[1])
        elif 'CAPACITY' in line:
            capacity = int(line.split(':')[1])
        elif 'SERVICE_TIME' in line:
            service_time = float(line.split(':')[1])
        elif 'NODE_COORD_SECTION' in line:
            section = 'nodes'
        elif 'DEMAND_SECTION' in line:
            section = 'demands'
        elif 'TIME_WINDOW_SECTION' in line:
            section = 'time_windows'
        elif section == 'nodes':
            parts = line.split()
            if len(parts) >= 3:
                x, y = float(parts[1]), float(parts[2])
                coords.append((x, y))
        elif section == 'demands':
            parts = line.split()
            if len(parts) >= 2:
                demand = float(parts[1])
                demands.append(demand)
        elif section == 'time_windows':
            parts = line.split()
            if len(parts) >= 4:
                earliest = float(parts[1])
                latest = float(parts[2])
                service = float(parts[3])
                time_windows.append((earliest, latest))
                service_times.append(service)
    
    # Count routes from solution file
    best_known_vehicles = 0
    if Path(sol_path).exists():
        with open(sol_path, 'r') as f:
            # Count lines that start with "Route #"
            best_known_vehicles = sum(1 for line in f if line.strip().startswith('Route #'))
    
    return VRPTWInstance(
        name=name,
        dimension=dimension,
        capacity=capacity,
        depot_coords=coords[0],
        customer_coords=coords[1:],
        demands=demands[1:],  # Skip depot
        time_windows=time_windows[1:],  # Skip depot
        service_times=service_times[1:],  # Skip depot
        best_known_vehicles=best_known_vehicles,
        service_time=service_time
    )

def adapt_to_mcv(
    instance: VRPTWInstance,
    config_path: str,
    seed: int = 42
) -> Tuple[pd.DataFrame, float]:
    try:
        # Load MCV configuration
        params = Parameters.from_yaml(config_path)
        np.random.seed(seed)
        
        # Scale coordinates to Bogotá bounds
        LAT_MIN, LAT_MAX = 4.4, 5.0
        LON_MIN, LON_MAX = -74.3, -73.9
        
        # Extract and scale coordinates
        x_coords = [c[0] for c in instance.customer_coords]
        y_coords = [c[1] for c in instance.customer_coords]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        scaled_lat = [LAT_MIN + (LAT_MAX - LAT_MIN) * ((y - y_min) / (y_max - y_min)) for y in y_coords]
        scaled_lon = [LON_MIN + (LON_MAX - LON_MIN) * ((x - x_min) / (x_max - x_min)) for x in x_coords]
        
        # Create base customer DataFrame with scaled demands
        customers = pd.DataFrame({
            'Customer_ID': range(1, len(instance.customer_coords) + 1),
            'Latitude': scaled_lat,
            'Longitude': scaled_lon,
            'Total_Demand': instance.demands
        })
        
        print("\nOriginal CVRP demands:")
        print(f"Total demand: {customers['Total_Demand'].sum():.1f}")
        print(f"Max customer demand: {customers['Total_Demand'].max():.1f}")
        
        # Scale demands to maintain the same total capacity utilization
        scale_factor = 1.0 / len(params.goods)  # e.g., 1/3 for three compartments
        customers['Total_Demand'] = customers['Total_Demand'] * scale_factor
        
        # Real-world inspired product distribution
        goods = params.goods
        product_counts = np.random.choice(
            [1, 2, 3],
            size=len(customers),
            p=[0.70, 0.25, 0.05]  # Adjusted based on real data
        )
        
        # Initialize demand columns with zeros
        for product in goods:
            customers[f'{product}_Demand'] = 0.0
        
        # Distribute demands based on real proportions
        for i, total_demand in enumerate(customers['Total_Demand']):
            original_demand = total_demand * len(params.goods)  # Scale back up for verification
            num_products = product_counts[i]
            if num_products == 1:
                # Single product - mostly Dry
                product = np.random.choice(['Dry', 'Chilled', 'Frozen'], p=[0.75, 0.20, 0.05])
                customers.at[i, f'{product}_Demand'] = total_demand
            elif num_products == 2:
                # Two products - usually Dry + Chilled
                dry_ratio = np.random.uniform(0.4, 0.8)
                customers.at[i, 'Dry_Demand'] = total_demand * dry_ratio
                customers.at[i, 'Chilled_Demand'] = total_demand * (1 - dry_ratio)
            else:
                # Three products - balanced but Dry-heavy
                ratios = np.random.dirichlet([3, 2, 1])
                for j, product in enumerate(goods):
                    customers.at[i, f'{product}_Demand'] = total_demand * ratios[j]
            
            # Verify total demand matches original
            new_total = sum(customers.loc[i, f'{g}_Demand'] for g in goods)
            if abs(new_total - total_demand) > 0.01:  # Allow for small rounding errors
                print(f"Warning: Customer {i+1} demand mismatch!")
                print(f"  Original (scaled): {total_demand:.1f}")
                print(f"  New total: {new_total:.1f}")
                print(f"  Product demands: {', '.join(f'{g}: {customers.loc[i, f'{g}_Demand']:.1f}' for g in goods)}")
        
        print("\nAfter distribution:")
        print(f"Total demand: {sum(customers[f'{g}_Demand'].sum() for g in goods):.1f}")
        print(f"Max customer total demand: {max(customers.apply(lambda row: sum(row[f'{g}_Demand'] for g in goods), axis=1)):.1f}")
        
        # Round demands to 1 decimal place
        for product in goods:
            customers[f'{product}_Demand'] = customers[f'{product}_Demand'].round(1)
        
        print("\nCoordinate scaling summary:")
        print(f"Original X range: {x_min:.1f} to {x_max:.1f}")
        print(f"Original Y range: {y_min:.1f} to {y_max:.1f}")
        print(f"Scaled to Bogotá area:")
        print(f"Latitude range: {min(scaled_lat):.4f} to {max(scaled_lat):.4f}")
        print(f"Longitude range: {min(scaled_lon):.4f} to {max(scaled_lon):.4f}")
        
        # Add debug prints for demand distribution
        print("\nDemand Distribution Summary:")
        total_demand = customers['Total_Demand'].sum()
        for product in goods:
            product_demand = customers[f'{product}_Demand'].sum()
            print(f"  {product}: {product_demand:.1f} kg ({(product_demand/total_demand)*100:.1f}%)")
        
        print("\nLargest Customer Demands:")
        for product in goods:
            max_demand = customers[f'{product}_Demand'].max()
            print(f"  Max {product}: {max_demand:.1f} kg")
        
        # Check if any individual customer demands exceed vehicle capacity
        capacity = float(instance.capacity)
        print(f"\nChecking demands vs capacity ({capacity:.1f} kg):")
        for idx, row in customers.iterrows():
            total_customer_demand = sum(row[f'{g}_Demand'] for g in goods)
            if total_customer_demand > capacity:
                print(f"  Warning: Customer {row['Customer_ID']} total demand ({total_customer_demand:.1f} kg) exceeds vehicle capacity!")
        
        return customers, float(instance.capacity)
    except Exception as e:
        print(f"Error in adapt_to_mcv:")
        print(f"Config path: {config_path}")
        print(f"Instance name: {instance.name}")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print(f"Error location:")
        import traceback
        traceback.print_exc()
        raise