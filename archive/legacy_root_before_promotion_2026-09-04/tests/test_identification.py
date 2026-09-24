from topoident.ieee33 import simulate_meter_data
from topoident.terminal_pmu import run_terminal_pmu_experiment
from topoident.zhang2020 import basic_identification


def test_basic_identification_keeps_most_true_edges():
    data = simulate_meter_data(samples=80, p_noise=0.002, q_noise=0.002, v_noise=0.0)
    estimate = basic_identification(data.p, data.q, data.v, max_edges=80)
    truth = {tuple(sorted((branch.u, branch.v))) for branch in data.branches}
    assert len(set(estimate.edges) & truth) >= 30


def test_optimized_pmu_improves_worst_pair_separation_and_accuracy():
    result = run_terminal_pmu_experiment(design_samples=8, trials=2)
    assert result.separation_by_budget[2] > result.separation_by_budget[0]
    assert result.accuracy_by_scheme["optimized_2_pmu"] >= 0.99
    assert (
        result.accuracy_by_scheme["optimized_2_pmu"] > result.accuracy_by_scheme["smart_meter_only"]
    )
