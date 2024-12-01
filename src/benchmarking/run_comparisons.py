import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))


from src.benchmarking.comparison import run_comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--instances-dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='src/config/default_config.yaml')
    parser.add_argument('--time-limit', type=int, default=300)
    parser.add_argument('--output-dir', type=str, default='results/benchmarks')
    args = parser.parse_args()
    
    # Find all .vrp files
    instance_files = list(Path(args.instances_dir).glob('*.vrp'))
    results = []
    
    # Run comparisons
    for instance_path in tqdm(instance_files):
        try:
            result = run_comparison(
                instance_path,
                args.config,
                args.time_limit
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {instance_path}: {e}")
    
    # Convert results to DataFrame
    results = [r for r in results if r is not None]  # Filter out None results
    if not results:
        print("No valid results to process")
        return
        
    results_df = pd.DataFrame([r.to_dict() for r in results])
    print(f"Processed {len(results)} results successfully")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    results_df.to_csv(output_dir / 'comparison_results.csv', index=False)
    
    # Generate visualizations
    create_comparison_plots(results_df, output_dir)

def create_comparison_plots(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization plots for the comparison results"""
    # Debug information
    print(f"DataFrame shape: {results_df.shape}")
    print("Available columns:", results_df.columns.tolist())
    
    # Check if DataFrame is empty
    if results_df.empty:
        print("No results to plot - all comparisons failed")
        return
    
    # Vehicle comparison plot
    plt.figure(figsize=(10, 6))
    try:
        sns.scatterplot(
            data=results_df,
            x='CVRP Vehicles',
            y='MCV Vehicles'
        )
        plt.plot([0, results_df['CVRP Vehicles'].max()], 
                 [0, results_df['CVRP Vehicles'].max()],
                 'r--', alpha=0.5)
        plt.title('CVRP vs MCV Vehicle Count Comparison')
        plt.savefig(output_dir / 'vehicle_comparison.png')
    except Exception as e:
        print(f"Error creating plot: {e}")
        print("This might indicate missing or incorrectly named columns in the results")
    finally:
        plt.close()
    
    # More plots can be added here...

if __name__ == '__main__':
    main() 