import pandas as pd
import pytest

import fleetmix.post_optimization as post_opt
import fleetmix.fsm_optimizer as fsm
from fleetmix.config.parameters import Parameters

# Helper to create a minimal clusters DataFrame with goods columns
def make_cluster_df(cluster_id):
    return pd.DataFrame([{  
        'Cluster_ID': cluster_id,
        'Config_ID': 1,
        # include all goods by default
        **{g: 1 for g in Parameters.from_yaml().goods}
    }])


def test_no_merges(monkeypatch):
    """Scenario B: no feasible merges => exit immediately without solving"""
    calls = {'gen': 0, 'solve': 0}

    def fake_gen(selected_clusters, configurations_df, customers_df, params):
        calls['gen'] += 1
        return pd.DataFrame()  # always empty

    def fake_solve(combined, configurations_df, customers_df, params):
        calls['solve'] += 1
        return {'selected_clusters': make_cluster_df('m'), 'total_cost': 50}

    monkeypatch.setattr(post_opt, 'generate_post_optimization_merges', fake_gen)
    monkeypatch.setattr(fsm, 'solve_fsm_problem', fake_solve)

    initial_clusters = make_cluster_df('c')
    initial_solution = {'selected_clusters': initial_clusters, 'total_cost': 100}
    params = Parameters.from_yaml()

    result = post_opt.improve_solution(initial_solution, pd.DataFrame(), pd.DataFrame(), params)
    assert result is initial_solution
    assert calls['gen'] == 1
    assert calls['solve'] == 0


def test_single_merge_then_no_more(monkeypatch):
    """Scenario A: one merge lowers cost, next yields none"""
    calls = {'gen': 0, 'solve': 0}

    def fake_gen(selected_clusters, configurations_df, customers_df, params):
        calls['gen'] += 1
        if calls['gen'] == 1:
            return make_cluster_df('m1')
        return pd.DataFrame()

    def fake_solve(combined, configurations_df, customers_df, params):
        calls['solve'] += 1
        return {'selected_clusters': make_cluster_df('m1'), 'total_cost': 90}

    monkeypatch.setattr(post_opt, 'generate_post_optimization_merges', fake_gen)
    monkeypatch.setattr(fsm, 'solve_fsm_problem', fake_solve)

    initial_clusters = make_cluster_df('c1')
    initial_solution = {'selected_clusters': initial_clusters, 'total_cost': 100}
    params = Parameters.from_yaml()

    result = post_opt.improve_solution(initial_solution, pd.DataFrame(), pd.DataFrame(), params)
    assert result['total_cost'] == 90
    assert calls['gen'] == 2
    assert calls['solve'] == 1


def test_iteration_cap(monkeypatch):
    """Scenario C: always merge & improve => stops at iteration cap"""
    calls = {'gen': 0, 'solve': 0}

    def fake_gen(selected_clusters, configurations_df, customers_df, params):
        calls['gen'] += 1
        # always return a dummy merge
        return make_cluster_df(f'g{calls["gen"]}')

    def fake_solve(combined, configurations_df, customers_df, params):
        calls['solve'] += 1
        # decreasing cost each call
        cost = 100 - calls['solve']
        # ensure selected_clusters changes
        return {'selected_clusters': make_cluster_df(f'g{calls["solve"]}'), 'total_cost': cost}

    monkeypatch.setattr(post_opt, 'generate_post_optimization_merges', fake_gen)
    monkeypatch.setattr(fsm, 'solve_fsm_problem', fake_solve)

    initial_clusters = make_cluster_df('c0')
    initial_solution = {'selected_clusters': initial_clusters, 'total_cost': 100}
    params = Parameters.from_yaml()
    params.max_improvement_iterations = 3

    result = post_opt.improve_solution(initial_solution, pd.DataFrame(), pd.DataFrame(), params)
    assert calls['solve'] == 3
    assert result['total_cost'] == 100 - 3 