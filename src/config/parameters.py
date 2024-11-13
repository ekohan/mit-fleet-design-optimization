from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import yaml

@dataclass
class Parameters:
    """Configuration parameters for the optimization"""
    vehicles: Dict
    variable_cost_per_hour: float
    avg_speed: float
    max_route_time: float
    service_time: float
    depot: Dict
    goods: List[str]
    clustering: Dict
    demand_file: str
    light_load_penalty: float
    light_load_threshold: float
    format: str

    @classmethod
    def from_yaml(cls, path: Path | str = None) -> 'Parameters':
        """Load parameters from YAML file"""
        if path is None:
            path = Path(__file__).parent / 'default_config.yaml'
        
        with open(path) as f:
            data = yaml.safe_load(f)
            return cls(**data) 