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
    compartment_setup_cost: float
    format: str
    post_optimization: bool = True  # Default to True

    @classmethod
    def from_yaml(cls, path: Path | str = None) -> 'Parameters':
        """Load parameters from YAML file"""
        if path is None:
            path = Path(__file__).parent / 'default_config.yaml'
        
        with open(path) as f:
            data = yaml.safe_load(f)
            return cls(**data) 

    def __post_init__(self):
        """Validate parameters after initialization"""
        # Validate clustering weights sum to 1
        geo_weight = self.clustering.get('geo_weight', 0.7)
        demand_weight = self.clustering.get('demand_weight', 0.3)
        
        if abs(geo_weight + demand_weight - 1.0) > 1e-6:
            raise ValueError(
                f"Clustering weights must sum to 1.0. Got: "
                f"geo_weight={geo_weight}, demand_weight={demand_weight}"
            ) 