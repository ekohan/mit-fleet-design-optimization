#!/usr/bin/env python3

import subprocess
from pathlib import Path
import logging
import multiprocessing
from functools import partial

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_all_instances():
    instance_dir = Path(__file__).parent / 'src' / 'benchmarking' / 'cvrp_instances'
    return [f.stem for f in instance_dir.glob('*.vrp')]

def get_existing_results(results_dir: Path) -> set:
    """Load all existing benchmark filenames from results CSV into a set."""
    csv_path = results_dir / 'benchmark_results.csv'
    existing_results = set()
    
    if not csv_path.exists():
        return existing_results
        
    try:
        with open(csv_path, 'r') as f:
            # Skip header
            next(f)
            for line in f:
                # First column contains the filename
                filename = line.split(',')[0].strip('"')
                existing_results.add(filename)
        return existing_results
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error reading benchmark results: {e}")
        return existing_results

def result_exists(instance: str, btype: str, existing_results: set) -> bool:
    """Check if result exists in the set of completed benchmarks."""
    if isinstance(instance, (list, tuple)):
        filename = f"cvrp_{instance}_{btype}"
    else:
        filename = f"cvrp_['{instance}']_{btype}"
    return filename in existing_results

def run_benchmark(instance, btype, next_instance=None):
    """Run a single benchmark with given parameters."""
    logger = logging.getLogger(__name__)
    
    if btype == 'combined':
        cmd = [
            'python',
            'src/benchmarking/cvrp_to_fsm.py',
            '--instance', instance, next_instance,
            '--benchmark-type', 'combined',
            '--format', 'json'
        ]
    else:
        cmd = [
            'python',
            'src/benchmarking/cvrp_to_fsm.py',
            '--instance', instance,
            '--benchmark-type', btype,
            '--format', 'json'
        ]
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error processing benchmark: {e}")
        return False

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    instances = get_all_instances()
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    existing_results = get_existing_results(results_dir)
    logger.info(f"Found {len(existing_results)} existing benchmark results")
    
    # Create list of all tasks to run
    tasks = []
    for instance in instances:
        for btype in ['normal', 'split', 'scaled', 'spatial']:
            if not result_exists(instance, btype, existing_results):
                tasks.append((instance, btype, None))
    
    if not tasks:
        logger.info("All benchmarks have already been run. No work needed.")
        return
    
    logger.info(f"Found {len(tasks)} benchmarks to run")
    
    # Run tasks in parallel
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.starmap(run_benchmark, tasks)
    
    successful = sum(1 for r in results if r)
    logger.info(f"Completed {successful}/{len(tasks)} benchmarks successfully")

if __name__ == "__main__":
    main()
