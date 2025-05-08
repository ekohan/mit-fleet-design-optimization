from pathlib import Path
from collections import namedtuple

from .benchmark_types import BenchmarkType
from .cvrp_to_fsm import convert_cvrp_to_fsm, CVRPBenchmarkType
from .mcvrp_parser import parse_mcvrp
from .mcvrp_to_fsm import convert_mcvrp_to_fsm

# Entry tuple for dispatch registry: contains instance directory, parser, converter
_Entry = namedtuple('Entry', ['dir', 'parser', 'converter'])

# Registry mapping benchmark types to their instance directory, parser, and converter
_registry: dict[BenchmarkType, _Entry] = {
    BenchmarkType.SINGLE_COMPARTMENT: _Entry(
        dir=BenchmarkType.SINGLE_COMPARTMENT.default_dir(),
        parser=None,
        converter=lambda stem: convert_cvrp_to_fsm(stem, CVRPBenchmarkType.NORMAL)
    ),
    BenchmarkType.MULTI_COMPARTMENT: _Entry(
        dir=BenchmarkType.MULTI_COMPARTMENT.default_dir(),
        parser=None,
        converter=lambda stem: convert_cvrp_to_fsm(stem, CVRPBenchmarkType.SPLIT)
    ),
    BenchmarkType.MCVRP_3C: _Entry(
        dir=BenchmarkType.MCVRP_3C.default_dir(),
        parser=parse_mcvrp,
        converter=lambda path: convert_mcvrp_to_fsm(path)
    ),
}

def resolve(kind: BenchmarkType, instance_stem: str):
    """Resolve an instance to (customers_df, parameters)."""
    if kind not in _registry:
        raise ValueError(f"Unsupported benchmark type: {kind}")
    entry = _registry[kind]
    inst_dir = entry.dir
    # Determine file name and existence
    if kind == BenchmarkType.MCVRP_3C:
        file_path = inst_dir / f"{instance_stem}.dat"
        if not file_path.exists():
            available = list_instances(kind)
            raise FileNotFoundError(
                f"MCVRP instance '{instance_stem}' not found. Available: {available}"
            )
        return entry.converter(file_path)
    else:
        file_path = inst_dir / f"{instance_stem}.vrp"
        if not file_path.exists():
            available = list_instances(kind)
            raise FileNotFoundError(
                f"CVRP instance '{instance_stem}' not found. Available: {available}"
            )
        return entry.converter(instance_stem)

def list_instances(kind: BenchmarkType) -> list[str]:
    """List available instance stems for given benchmark type."""
    if kind not in _registry:
        raise ValueError(f"Unsupported benchmark type: {kind}")
    inst_dir = _registry[kind].dir
    # Select file extension based on type
    pattern = "*.dat" if kind == BenchmarkType.MCVRP_3C else "*.vrp"
    return sorted([p.stem for p in inst_dir.glob(pattern) if p.is_file()]) 