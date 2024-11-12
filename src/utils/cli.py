from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Dict, Any
from config.parameters import Parameters
import sys
from utils.logging import Colors

def print_parameter_help():
    """Display detailed help information about parameters"""
    help_text = f"""
{Colors.BOLD}Fleet Size and Mix Optimization Parameters{Colors.RESET}
{Colors.CYAN}════════════════════════════════════════{Colors.RESET}

{Colors.YELLOW}Core Parameters:{Colors.RESET}
  --avg-speed FLOAT         Average vehicle speed in km/h
                           Default: Defined in config file
                           Example: --avg-speed 45

  --max-route-time FLOAT   Maximum route time in hours
                           Default: Defined in config file
                           Example: --max-route-time 12

  --service-time FLOAT     Service time per customer in minutes
                           Default: Defined in config file
                           Example: --service-time 15

  --route-time-estimation STR
                           Method to estimate route times
                           Options: 
                             - Legacy (simple service time based)
                             - Clarke-Wright (savings algorithm)
                             - BHH (Beardwood-Halton-Hammersley)
                             - CA (continuous approximation)
                             - VRPSolver (detailed solver-based)
                           Default: Legacy
                           Example: --route-time-estimation BHH

{Colors.YELLOW}Model Configuration:{Colors.RESET}
  --model-type {1,2}       Mathematical model formulation
                           1 = "Eric's formulation"
                           2 = "Fabri's formulation"
                           Default: 2
                           Example: --model-type 1

  --light-load-penalty FLOAT
                           Penalty cost for light loads
                           Set to 0 to disable penalties
                           Default: 1000
                           Example: --light-load-penalty 500

  --light-load-threshold FLOAT
                           Threshold for light load penalty (0.0 to 1.0)
                           Example: 0.2 means penalize loads below 20%
                           Default: 0.20
                           Example: --light-load-threshold 0.3

{Colors.YELLOW}Input/Output:{Colors.RESET}
  --demand-file STR        Name of the demand file to use
                           Must be in the data directory
                           Default: Defined in config file
                           Example: --demand-file sales_2023_high_demand_day.csv

  --config PATH            Path to custom config file
                           Default: src/config/default_config.yaml
                           Example: --config my_config.yaml

{Colors.YELLOW}Other Options:{Colors.RESET}
  --verbose               Enable verbose output
                           Default: False
                           Example: --verbose

{Colors.CYAN}Examples:{Colors.RESET}
  # Use custom config file
  python src/main.py --config my_config.yaml

  # Override specific parameters
  python src/main.py --avg-speed 45 --max-route-time 12 --service-time 15

  # Change model type and penalties
  python src/main.py --model-type 2 --light-load-penalty 500 --light-load-threshold 0.3

  # Use different demand file with verbose output
  python src/main.py --demand-file sales_2023_high_demand_day.csv --verbose
"""
    print(help_text)
    sys.exit(0)

def parse_args() -> ArgumentParser:
    """Parse command line arguments for parameter overrides"""
    parser = ArgumentParser(
        description='Fleet Size and Mix Optimization',
        formatter_class=RawTextHelpFormatter
    )
    
    # Add help-params argument
    parser.add_argument(
        '--help-params',
        action='store_true',
        help='Show detailed parameter information and exit'
    )
    
    # Add arguments for each parameter that can be overridden
    parser.add_argument('--config', type=str, help='Path to custom config file')
    parser.add_argument('--avg-speed', type=float, help='Average vehicle speed in km/h')
    parser.add_argument('--max-route-time', type=float, help='Maximum route time in hours')
    parser.add_argument('--service-time', type=float, help='Service time per customer in minutes')
    parser.add_argument('--demand-file', type=str, help='Name of the demand file to use')
    parser.add_argument(
        '--model-type',
        type=int,
        choices=[1, 2],
        help='Model type (1="Eric\'s", 2="Fabri\'s")'
    )
    parser.add_argument('--light-load-penalty', type=float, help='Penalty for light loads')
    parser.add_argument('--light-load-threshold', type=float, help='Threshold for light load penalty')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument(
        '--route-time-estimation',
        type=str,
        choices=['Legacy', 'Clarke-Wright', 'BHH', 'CA', 'VRPSolver'],
        help='Method to estimate route times (Legacy, Clarke-Wright, BHH, CA, VRPSolver)'
    )
    
    return parser

def get_parameter_overrides(args) -> Dict[str, Any]:
    """Extract parameter overrides from command line arguments"""
    # Convert args to dictionary, excluding None values
    overrides = {k: v for k, v in vars(args).items() if v is not None}
    
    # Remove non-parameter arguments
    for key in ['config', 'verbose', 'help_params']:
        overrides.pop(key, None)
        
    # Convert dashed args to underscores
    overrides = {k.replace('-', '_'): v for k, v in overrides.items()}
    
    return overrides

def load_parameters(args) -> Parameters:
    """Load parameters with optional command line overrides"""
    # Load base parameters
    if args.config:
        params = Parameters.from_yaml(args.config)
    else:
        params = Parameters.from_yaml()
    
    # Get overrides from command line
    overrides = get_parameter_overrides(args)
    
    # Handle nested parameters
    if 'route_time_estimation' in overrides:
        if not isinstance(params.clustering, dict):
            params.clustering = {}
        params.clustering['route_time_estimation'] = overrides.pop('route_time_estimation')
    
    # Create new Parameters instance with remaining overrides
    if overrides:
        data = params.__dict__.copy()
        data.update(overrides)
        params = Parameters(**data)
    
    return params