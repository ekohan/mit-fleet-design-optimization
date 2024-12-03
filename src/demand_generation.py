import pandas as pd
import numpy as np
import os
import re

def convert_vrp_to_multicompartment(input_file, output_file):
    # Read VRP file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Extract dimension
    dimension = int(re.search(r'DIMENSION\s*:\s*(\d+)', ''.join(lines)).group(1))
    
    # Extract coordinates and demands
    coords = []
    demands = []
    
    in_coords = False
    in_demands = False
    
    for line in lines:
        if 'NODE_COORD_SECTION' in line:
            in_coords = True
            continue
        elif 'DEMAND_SECTION' in line:
            in_coords = False
            in_demands = True
            continue
        elif 'DEPOT_SECTION' in line:
            break
            
        if in_coords:
            parts = line.strip().split()
            if len(parts) == 3:
                coords.append([float(parts[1]), float(parts[2])])
                
        if in_demands:
            parts = line.strip().split()
            if len(parts) == 2:
                demands.append(int(parts[1]))
    
    # Create multi-compartment distribution
    data = []
    client_id_start = 10000000  # Starting ClientID similar to example file
    
    for i in range(1, dimension):  # Skip depot (index 0)
        total_demand = demands[i]
        if total_demand == 0:
            continue
            
        # Randomly decide if this client gets all product types
        num_product_types = np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        
        if num_product_types == 1:
            # Single product type
            product_type = np.random.choice(['Dry'], p=[1.0])
            data.append([client_id_start + i, coords[i][0], coords[i][1], total_demand, product_type])
            
        elif num_product_types == 2:
            # Two product types
            split = np.random.uniform(0.6, 0.9)
            if np.random.random() < 0.9:  # 90% chance of Dry + Chilled
                data.append([client_id_start + i, coords[i][0], coords[i][1], 
                           int(total_demand * (1-split)), 'Chilled'])
                data.append([client_id_start + i, coords[i][0], coords[i][1], 
                           int(total_demand * split), 'Dry'])
            else:  # 10% chance of Dry + Frozen
                data.append([client_id_start + i, coords[i][0], coords[i][1], 
                           int(total_demand * split), 'Dry'])
                data.append([client_id_start + i, coords[i][0], coords[i][1], 
                           int(total_demand * (1-split)), 'Frozen'])
                
        else:
            # All three product types
            split1 = np.random.uniform(0.65, 0.85)
            split2 = np.random.uniform(0.1, 0.25)
            split3 = 1 - split1 - split2
            
            data.append([client_id_start + i, coords[i][0], coords[i][1], 
                       int(total_demand * split1), 'Dry'])
            data.append([client_id_start + i, coords[i][0], coords[i][1], 
                       int(total_demand * split2), 'Chilled'])
            data.append([client_id_start + i, coords[i][0], coords[i][1], 
                       int(total_demand * split3), 'Frozen'])
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=['ClientID', 'Lat', 'Lon', 'Kg', 'ProductType'])
    
    # Verify proportions
    total_demand = df['Kg'].sum()
    frozen_prop = df[df['ProductType'] == 'Frozen']['Kg'].sum() / total_demand
    chilled_prop = df[df['ProductType'] == 'Chilled']['Kg'].sum() / total_demand
    dry_prop = df[df['ProductType'] == 'Dry']['Kg'].sum() / total_demand
    
    # Adjust if proportions are too far off
    while abs(frozen_prop - 0.02) > 0.01 or abs(chilled_prop - 0.23) > 0.02 or abs(dry_prop - 0.75) > 0.02:
        # Adjust demands slightly and recalculate
        if frozen_prop < 0.01:
            df.loc[df['ProductType'] == 'Frozen', 'Kg'] += 1
        elif frozen_prop > 0.03:
            df.loc[df['ProductType'] == 'Frozen', 'Kg'] -= 1
            
        if chilled_prop < 0.21:
            df.loc[df['ProductType'] == 'Chilled', 'Kg'] += 1
        elif chilled_prop > 0.25:
            df.loc[df['ProductType'] == 'Chilled', 'Kg'] -= 1
            
        # Recalculate proportions
        total_demand = df['Kg'].sum()
        frozen_prop = df[df['ProductType'] == 'Frozen']['Kg'].sum() / total_demand
        chilled_prop = df[df['ProductType'] == 'Chilled']['Kg'].sum() / total_demand
        dry_prop = df[df['ProductType'] == 'Dry']['Kg'].sum() / total_demand
    
    # Save to CSV
    df.to_csv(output_file, index=False)

# Process all CMT files
for i in range(1, 15):
    input_file = f'data/CMT{i}.vrp'
    output_file = f'data/CMT{i}_multi_compartment_demand.csv'
    
    if os.path.exists(input_file):
        print(f"Processing {input_file}...")
        convert_vrp_to_multicompartment(input_file, output_file)
        print(f"Created {output_file}")