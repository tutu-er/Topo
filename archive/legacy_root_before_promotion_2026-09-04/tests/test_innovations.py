from topoident.innovation_active import run_active_probe_design
from topoident.innovation_bayesian import run_bayesian_hidden_loads
from topoident.innovation_hmm import run_dynamic_hmm
from topoident.innovation_potential import run_potential_sparse_recovery
from topoident.innovation_topology_free import run_topology_free_latent_tree


def test_bayesian_marginalization_beats_zero_injection():
    result = run_bayesian_hidden_loads(operations=3, prior_draws=6, test_draws=3)
    assert result["marginalized_bayes_accuracy"] > result["zero_injection_wls_accuracy"]


def test_joint_active_design_improves_separation():
    result = run_active_probe_design(trials=20)
    assert result["separation"]["joint_active_pmu"] > result["separation"]["passive_sm"]


def test_hmm_uses_temporal_structure():
    result = run_dynamic_hmm(horizon=50)
    assert result["viterbi_accuracy"] > result["independent_accuracy"]


def test_potential_projection_returns_a_tree():
    result = run_potential_sparse_recovery(samples=24, iterations=8)
    assert result["projected_edges"] == result["true_edges"]
    assert result["edge_recall"] >= 0.8


def test_topology_free_method_recovers_terminal_splits_without_candidates():
    result = run_topology_free_latent_tree(samples=240, bootstrap_draws=12)
    assert result["uses_candidate_topologies"] is False
    assert result["uses_candidate_edges"] is False
    assert result["split_recall"] >= 0.5
    assert result["inferred_latent_nodes"] > 0
