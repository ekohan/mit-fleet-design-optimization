from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

@dataclass(frozen=True)
class MCVRPInstance:
    """Container for parsed MCVRP instance data."""
    name: str
    source_file: Path
    dimension: int
    capacity: int
    vehicles: int
    depot_id: int
    coords: Dict[int, Tuple[float, float]]
    demands: Dict[int, Tuple[int, int, int]]

    def customers(self) -> List[int]:
        """Return all customer node IDs (excluding the depot)."""
        return [node_id for node_id in self.coords.keys() if node_id != self.depot_id] 