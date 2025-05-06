# tests/core/test_fsm_scenarios.py

import pandas as pd
import pytest
from src.fsm_optimizer import solve_fsm_problem
from src.config.parameters import Parameters

# ─────────────────────────────────────────────────────────────────────────────
# Scenario definitions: (name, clusters, configs, params_updates, expected)
# ─────────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    (
        "S0_happy",
        # clusters
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 1, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        # configs
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            }
        ],
        # parameter overrides
        {"variable_cost_per_hour": 1},
        # expected results
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 6,
        },
    ),
    (
        "S1_size_filter",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 8, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            },
            {
                "Config_ID": 2,
                "Vehicle_Type": "B",
                "Capacity": 2,
                "Fixed_Cost": 2,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            },
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 6,
        },
    ),
    (
        "S2_cheapest",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 3, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            },
            {
                "Config_ID": 2,
                "Vehicle_Type": "B",
                "Capacity": 4,
                "Fixed_Cost": 4,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            },
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": set(),
            "vehicles_used": {"B": 1},
            "total_fixed_cost": 4,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 5,
        },
    ),
    (
        "S3_light_load",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 3, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            }
        ],
        {
            "variable_cost_per_hour": 1,
            "light_load_penalty": 100,
            "light_load_threshold": 0.5,
        },
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 100,
            "total_compartment_penalties": 0,
            "total_cost": 106,
        },
    ),
    (
        "S4_compartment_penalty",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 1, "Chilled": 1, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 10,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 1,
                "Frozen": 0,
            }
        ],
        {
            "variable_cost_per_hour": 1,
            "compartment_setup_cost": 50,
        },
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1},
            "total_fixed_cost": 5,
            "total_variable_cost": 1,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 50,
            "total_cost": 56,
        },
    ),
    (
        "S5_infeasible",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 10, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            }
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 5,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            }
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": {"C1"},
            "vehicles_used": {},
            "total_fixed_cost": 0,
            "total_variable_cost": 0,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 0,
        },
    ),
    (
        "S6_two_clusters",
        [
            {
                "Cluster_ID": 1,
                "Customers": ["C1"],
                "Total_Demand": {"Dry": 2, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            },
            {
                "Cluster_ID": 2,
                "Customers": ["C2"],
                "Total_Demand": {"Dry": 9, "Chilled": 0, "Frozen": 0},
                "Route_Time": 1.0,
            },
        ],
        [
            {
                "Config_ID": 1,
                "Vehicle_Type": "A",
                "Capacity": 5,
                "Fixed_Cost": 5,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            },
            {
                "Config_ID": 2,
                "Vehicle_Type": "B",
                "Capacity": 10,
                "Fixed_Cost": 8,
                "Dry": 1,
                "Chilled": 0,
                "Frozen": 0,
            },
        ],
        {"variable_cost_per_hour": 1},
        {
            "missing_customers": set(),
            "vehicles_used": {"A": 1, "B": 1},
            "total_fixed_cost": 13,
            "total_variable_cost": 2,
            "total_light_load_penalties": 0,
            "total_compartment_penalties": 0,
            "total_cost": 15,
        },
    ),
]

@pytest.mark.parametrize(["name", "clusters", "configs", "upd", "exp"], SCENARIOS)
def test_fsm_scenarios(name, clusters, configs, upd, exp):
    # Build DataFrames
    clusters_df = pd.DataFrame(clusters)
    config_df   = pd.DataFrame(configs)
    # Customers=index of clusters
    customers_df = pd.DataFrame([
        {
            "Customer_ID": c,
            "Dry_Demand": row["Total_Demand"].get("Dry", 0),
            "Chilled_Demand": row["Total_Demand"].get("Chilled", 0),
            "Frozen_Demand": row["Total_Demand"].get("Frozen", 0),
        }
        for row in clusters for c in row["Customers"]
    ])
    # Load & override params
    params = Parameters.from_yaml()
    for k,v in upd.items():
        setattr(params, k, v)
    # Solve or validate infeasible
    from src import fsm_optimizer
    if exp["missing_customers"]:
        # Infeasible: model should inject NoVehicle and warn
        model, y_vars, x_vars, c_vk = fsm_optimizer._create_model(
            clusters_df, config_df, params
        )
        selected = fsm_optimizer._extract_solution(clusters_df, y_vars, x_vars)
        missing = fsm_optimizer._validate_solution(
            selected, customers_df, config_df
        )
        assert missing == exp["missing_customers"]
    else:
        result = solve_fsm_problem(
            clusters_df, config_df, customers_df, params, verbose=False
        )
        # Compare expected
        assert result["missing_customers"] == exp["missing_customers"]
        assert result["vehicles_used"]       == exp["vehicles_used"]
        assert result["total_fixed_cost"]    == exp["total_fixed_cost"]
        assert pytest.approx(result["total_variable_cost"]) == exp["total_variable_cost"]
        assert result["total_light_load_penalties"]  == exp["total_light_load_penalties"]
        assert result["total_compartment_penalties"] == exp["total_compartment_penalties"]
        assert result["total_cost"]          == exp["total_cost"]