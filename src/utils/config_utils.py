import itertools
import pandas as pd

def generate_vehicle_configurations(vehicle_types, goods):
    """Generate all possible vehicle configurations with compartments"""
    compartment_options = list(itertools.product([0, 1], repeat=len(goods)))
    compartment_configs = []
    config_id = 1
    
    for vt_name, vt_info in vehicle_types.items():
        for option in compartment_options:
            # Skip configuration if no compartments are selected
            if sum(option) == 0:
                continue
            compartment = dict(zip(goods, option))
            compartment['Vehicle_Type'] = vt_name
            compartment['Config_ID'] = config_id
            compartment_configs.append(compartment)
            config_id += 1

    configurations_df = pd.DataFrame(compartment_configs)
    
    # Merge with vehicle types to get capacities and costs
    configurations_df = configurations_df.merge(
        pd.DataFrame(vehicle_types).T.reset_index().rename(columns={'index': 'Vehicle_Type'}),
        on='Vehicle_Type'
    )
    
    return configurations_df

def print_configurations(configurations_df, goods):
    """Print vehicle configurations in a formatted way"""
    print("\nVehicle Configurations:")
    print("-" * 50)
    for _, config in configurations_df.iterrows():
        print(f"Config ID: {config['Config_ID']}")
        print(f"  Vehicle Type: {config['Vehicle_Type']}")
        print(f"  Capacity: {config['Capacity']}")
        print(f"  Fixed Cost: ${config['Fixed_Cost']}")
        print(f"  Compartments:")
        for good in goods:
            status = "✓" if config[good] == 1 else "✗"
            print(f"    {good}: {status}")
        print("-" * 50)