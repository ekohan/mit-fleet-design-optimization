"""
Configuration parameters for the optimization process.
"""

# Capacity of each vehicle type
VEHICLE_TYPES = {
    'A': {'Capacity': 2000, 'Fixed_Cost': 100},
    'B': {'Capacity': 3000, 'Fixed_Cost': 130},
    'C': {'Capacity': 4000, 'Fixed_Cost': 150}
}

# Variable cost per km for all vehicles
VARIABLE_COST_PER_KM = 0.01

# Average speed of vehicles in km/h    
AVG_SPEED = 40

# Maximum route time in hours
MAX_ROUTE_TIME = 10 

# Service time per customer in hours
SERVICE_TIME_PER_CUSTOMER = 10/60  # 10 minutes converted to hours  

# Depot location
DEPOT = {'Latitude': 4.7, 'Longitude': -74.1}

# Goods
GOODS = ['Dry', 'Chilled', 'Frozen']    

# Clustering parameters
MAX_SPLIT_DEPTH = 10  # Maximum depth for recursive clustering  
CLUSTERING_METHOD = 'MiniBatchKMeans' # Fixed for now
CLUSTERING_DISTANCE = 'euclidean' # Fixed for now
ROUTE_TIME_ESTIMATION = '1hr + SERVICE_TIME_PER_CUSTOMER * num_customers in cluster' # Fixed for now