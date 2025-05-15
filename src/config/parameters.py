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
    expected_vehicles: int = -1
    small_cluster_size: int = 7
    nearest_merge_candidates: int = 10
    max_improvement_iterations: int = 4

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

        if not isinstance(self.small_cluster_size, int) or self.small_cluster_size <= 0:
            raise ValueError(
                f"small_cluster_size must be a positive integer. Got: {self.small_cluster_size}"
            )
        
        if not isinstance(self.nearest_merge_candidates, int) or self.nearest_merge_candidates <= 0:
            raise ValueError(
                f"nearest_merge_candidates must be a positive integer. Got: {self.nearest_merge_candidates}"
            )
        
        if not isinstance(self.max_improvement_iterations, int) or self.max_improvement_iterations < 0:
            raise ValueError(
                f"max_improvement_iterations must be a non-negative integer. Got: {self.max_improvement_iterations}"
            ) 