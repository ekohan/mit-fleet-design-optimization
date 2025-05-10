# Fleet‑Size‑and‑Mix Optimizer (FSM‑Opt) for Multi-Compartment Vehicle Routing

Fast, reproducible tooling for **multi‑compartment vehicle fleet design** in urban food distribution.  This repository provides a comprehensive Python implementation supporting the research paper on [Designing Multi-Compartment Vehicle Fleets for Last-Mile Food Distribution Systems](https://arxiv.org/search/?query=fleet+size+mix&searchtype=all&source=header).

*Written for transparent research, hardened for production use.*

---

## Why FSM‑Opt?

* **Scales** — 1 000 + customers solved in seconds via a cluster‑first / MILP‑second approach.  
* **Extensible** — modular parsers, clustering engines, and solver back‑ends.  
* **Reproducible** — every experiment in the forthcoming journal article reruns with a single script.  

---

## Installation

```bash
git clone <repo> && cd fsm‑opt
./init.sh            # creates `mit-fleet-env`, installs deps
```

Activate the Python environment:

```bash
source mit-fleet-env/bin/activate
```

---

## Directory Map

```
src/
  clustering.py            # k‑means, sweep, grid‑grow...
  fsm_optimizer.py         # MILP fleet‑selection core
  benchmarking/
    vrp_to_fsm.py          # unified CLI (CVRP & MCVRP → FSM)
    parsers/               # VRPLIB, Henke .dat, geo helpers
    run_all_mcvrp.py       # batch reproduce Henke results
tests/                     # >150 unit / component / E2E tests
docs/                      # algorithm notes, design drivers
results/                   # auto‑generated outputs
```

---

## Core Workflow

```mermaid
graph LR
    A[Raw VRP instance] -->|convert| B[clusters.csv]
    B -->|MILP| C[fleet.json + xlsx]
```

---

## Running the Optimization

Execute the fleet optimization pipeline:
```bash
python src/main.py
```

### Command Line Options
The optimization can be customized using command line arguments:

```bash
# Show detailed parameter help
python src/main.py --help-params TODO: check params

# Use custom configuration
python src/main.py --config my_config.yaml

# Override specific parameters
python src/main.py --avg-speed 45 --max-route-time 12 --service-time 15

# Change clustering method and distance metric
python src/main.py --clustering-method agglomerative --clustering-distance composite --geo-weight 0.5 --demand-weight 0.5

# Combine all clustering methods at once
python src/main.py --clustering-method combine --verbose
```

#### Core Parameters
- `--avg-speed`: Average vehicle speed in km/h
- `--max-route-time`: Maximum route time in hours
- `--service-time`: Service time per customer in minutes
- `--route-time-estimation`: Method to estimate route times (BHH, TSP, Legacy)
- `--light-load-penalty`: Penalty cost for light loads (0 to disable)
- `--light-load-threshold`: Threshold for light load penalty (0.0 to 1.0)
- `--compartment-setup-cost`: Cost per additional compartment beyond the first one

#### Clustering Options
- `--clustering-method`: Algorithm choice (minibatch_kmeans, kmedoids, agglomerative, combine)
- `--clustering-distance`: Distance metric (euclidean, composite)
- `--geo-weight`: Weight for geographical distance (0.0 to 1.0)
- `--demand-weight`: Weight for demand distance (0.0 to 1.0)

## Clustering Methods

The system supports several clustering methods to group customers efficiently:

### Basic Methods
- `minibatch_kmeans`: Fast K-means clustering based on geographical coordinates
- `kmedoids`: K-medoids clustering that's more robust to outliers
- `agglomerative`: Hierarchical clustering that can be customized to consider both geographical distance and demand similarity

### Combine Method
The `combine` method is a comprehensive approach that:
1. Runs multiple clustering algorithms:
   - MiniBatch K-means (geographical coordinates)
   - K-medoids (geographical coordinates)
   - Agglomerative clustering (with various weights for geographical distance and demand similarity)
2. Evaluates each cluster's feasibility based on:
   - Vehicle capacity constraints
   - Maximum route time
   - Product type compatibility
3. Selects the best clusters across all methods to create a final solution

This will generate and combine multiple clustering solutions to produce more robust results.

#### Input/Output
- `--demand-file`: Name of the demand file to use (must be in data directory)
- `--config`: Path to custom config file
- `--verbose`: Enable verbose output

## Route Time Estimation Methods

The system supports multiple methods for estimating route times:

1. **BHH**: Beardwood-Halton-Hammersley theorem approximation (Default)
2. **TSP**: Detailed VRP solver-based estimation
3. **Quick & Dirty**: TODO, decide if relevant 

## Results and Visualization

Outputs include detailed Excel and JSON summaries with metrics:

* Fleet size and cost breakdowns
* Vehicle utilization rates
* Computational performance (execution time, scalability)

Interactive maps visualize solution structures:

* Cluster assignments
* Vehicle routes and compartment utilization
* Geographic demand distribution

## Solver Backend

Default solver: Gurobi (fallback to CBC if unavailable). Set `FSM_SOLVER` to specify (`gurobi`, `cbc`, or `auto`).

---

## Benchmarking

### 1 · VRP → FSM Pipeline

Convert **any** supported VRP benchmark and immediately solve the corresponding FSM instance.

The benchmarking pipeline:

1. **VRP to FSM Conversion:** Adapts single-compartment (CVRP) and multi-compartment (MCVRP) benchmark datasets to the FSM model.
2. **Optimization Execution:** Solves adapted FSM instances using MILP and clustering heuristics.
3. **Performance Analysis:** Evaluates computational efficiency, scalability, and solution quality.

TODO: define CLI interface, module or independent script?

#### VRP to FSM Conversion

Converts standard VRP instances for fair comparison:

* **MCVRP Instances (Henke et al.):** Direct mapping without additional scaling.
* **CVRP Instances (Uchoa et al.):** Adapted through several strategies:

  * **Split:** Demand split across Dry, Chilled, Frozen.
  * **Scaled:** Demands and capacities scaled proportionally.
  * **Combined:** Multiple instances merged, each representing different product categories.
  * **Spatial Differentiation:** Geographically varied product distribution.

```bash
# Henke 10‑customer MCVRP
python -m benchmarking.vrp_to_fsm \
       --vrp-type mcvrp --instance 10_3_3_3_01

# Uchoa CVRP adapted via "split" strategy
python -m benchmarking.vrp_to_fsm \
       --vrp-type cvrp --instance X-n106-k14 \
       --benchmark-type split --num-goods 3
```

Supported datasets
: *Henke 15* MCVRP (bundled) · *Uchoa 17* CVRP (auto‑download via PyVRP).

### 2 · Reference Bounds (sanity checks)

`run_benchmark.py` produces two fast baselines:

* **Upper Bound (Single-Compartment):** Solves each product type separately, establishing performance limits.
* **Lower Bound (Multi-Compartment):** Aggregates all demands, providing theoretical optimal efficiency.

Helpful for heuristic health‑checks; not required for routine usage.

---

## Reproducing Paper Results

```bash
# All 153 Henke instances
python -m benchmarking.run_all_mcvrp

# Selected Uchoa adaptations
python -m benchmarking.vrp_to_fsm \
       --vrp-type cvrp --instance X-n101-k25 \
       --benchmark-type split
```

Artifacts appear in `results/` as
`vrp_<type>_<instance>_<variant>.{json,xlsx}`.

TODO: actually add a single tool to do this

---

## Contribution Guidelines

We welcome community contributions:

* Fork the repository
* Create a feature branch (`git checkout -b feature-name`)
* Submit pull requests to the main branch

Ensure all tests pass and follow established coding standards and documentation practices.

## Citation

If you use this work, please cite:

```latex
@article{YourPaper,
  author = {Eric Kohan},
  title = {Fleet Size and Mix Optimization for Multi-Compartment Vehicles},
  journal = {Top Journal},
  year = {2025},
  volume = {},
  number = {},
  pages = {},
  doi = {}
}
```

## License

This project is released under the MIT License.