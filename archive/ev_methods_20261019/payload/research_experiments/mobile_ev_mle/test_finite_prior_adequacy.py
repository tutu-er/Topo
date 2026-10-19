import json
import math

import numpy as np
import pytest
from scipy.stats import chi2, norm

from research_experiments.mobile_ev_mle.finite_prior_adequacy import audit_fixed_mean_loss, audit_saved_eig, fixed_mean_statistic


SCOPE = "pre-H fixed atom means; known conditional root and Gaussian whitening covariance"


def test_dimension_one_threshold_has_independent_normal_identity():
    result = audit_fixed_mean_loss(0.0, 1, alpha=0.01, fixed_mean_scope=SCOPE)
    expected = float(norm.ppf(0.995) ** 2)
    assert result["threshold"] == pytest.approx(expected, rel=2e-14)
    at_threshold = audit_fixed_mean_loss(result["threshold"], 1, alpha=0.01, fixed_mean_scope=SCOPE)
    assert at_threshold["reject"] is False


@pytest.mark.parametrize("dimension", [1, 5, 132])
def test_included_true_fixed_mean_dominates_gaussian_noise_norm(dimension):
    rng = np.random.default_rng(80421 + dimension)
    means = rng.normal(size=(7, dimension))
    for atom in range(len(means)):
        for _ in range(12):
            noise = rng.normal(size=dimension)
            observed = means[atom] + noise
            score = fixed_mean_statistic(observed, means)
            assert score <= float(noise @ noise) + 2e-12
            independently_computed = min(float(np.dot(observed - mean, observed - mean)) for mean in means)
            assert score == pytest.approx(independently_computed, abs=2e-12)


@pytest.mark.parametrize("dimension,loss", [(1, 4.0), (2, 8.0), (5, 15.0), (132, 160.0), (132, 250.0)])
def test_chernoff_bound_dominates_non_underflow_exact_tail(dimension, loss):
    result = audit_fixed_mean_loss(loss, dimension, fixed_mean_scope=SCOPE)
    exact_tail = float(chi2.sf(loss, dimension))
    assert math.log(exact_tail) <= result["log_tail_upper_bound"] + 2e-14
    assert 0.0 < exact_tail <= math.exp(result["log_tail_upper_bound"]) <= 1.0


def test_extreme_loss_has_finite_log_bound_and_explicit_display_underflow():
    result = audit_fixed_mean_loss(25910.706185844334, 132, fixed_mean_scope=SCOPE)
    assert result["reject"] and result["underflow"]
    assert result["sf_display"] == 0.0
    assert math.isfinite(result["log_tail_upper_bound"])
    assert result["log_tail_upper_bound"] < -10000.0
    assert result["topology_correctness_inferred"] is False
    assert result["scope_verified_from_summary"] is False


@pytest.mark.parametrize("loss,dimension,alpha", [(-1.0, 132, .01), (math.inf, 132, .01),
    (0.0, 0, .01), (0.0, 1.5, .01), (0.0, True, .01), (0.0, 132, 0.0), (0.0, 132, True)])
def test_invalid_inputs_raise(loss, dimension, alpha):
    with pytest.raises(ValueError):
        audit_fixed_mean_loss(loss, dimension, alpha=alpha, fixed_mean_scope=SCOPE)


def test_scope_is_required_and_not_inferred_from_summary():
    with pytest.raises(TypeError):
        audit_fixed_mean_loss(1.0, 132)
    with pytest.raises(ValueError):
        audit_fixed_mean_loss(1.0, 132, fixed_mean_scope=" ")


def test_saved_audit_keeps_missing_records_and_ignores_truth_action_and_entropy(tmp_path):
    source = {"truth": "PRIVATE_TRUTH", "future": "PRIVATE_FUTURE", "histories": [
        {"input_identifier": "H1", "best_history_loss": 25910.7, "history_dimension": 132,
         "truth": "PRIVATE_ROW_TRUTH", "topology_entropy": 0.0, "selected_action": "PRIVATE_ACTION"},
        {"input_identifier": "H2", "history_dimension": 132}]}
    summary_path, out = tmp_path / "summary.json", tmp_path / "adequacy.json"
    summary_path.write_text(json.dumps(source), encoding="utf-8")
    result = audit_saved_eig(summary_path, out, fixed_mean_scope=SCOPE, prior_config_hash="provided-hash")
    assert result["history_count"] == 2
    assert result["audited_history_count"] == result["reject_count"] == 1
    assert result["unavailable_history_count"] == 1
    assert result["histories"][1]["reject"] is None
    assert result["provided_prior_config_hash"] == "provided-hash"
    serialized = out.read_text(encoding="utf-8")
    assert "PRIVATE_" not in serialized
    assert result == json.loads(serialized)
