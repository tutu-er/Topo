"""Independent rank, interval and policy-interface checks."""

import inspect
import math

import numpy as np
import pytest
from scipy.stats import binom

try:
    from .adaptive_statistics import (
        ScoreRecord, policy_trial_scores, conformal_upper_quantile,
        calibrate_policy_gates, apply_candidate_gates,
        binomial_exact_interval, history_event_summary,
        paired_binary_effects, paired_continuous_effects, replay_eig_shortlist,
    )
except ImportError:
    from adaptive_statistics import (
        ScoreRecord, policy_trial_scores, conformal_upper_quantile,
        calibrate_policy_gates, apply_candidate_gates,
        binomial_exact_interval, history_event_summary,
        paired_binary_effects, paired_continuous_effects, replay_eig_shortlist,
    )


def complete_record(relative=2.0, absolute=3.0):
    return ScoreRecord(relative, absolute, 1.0, True, True, False, False, 2, 0)


def calibration(records=None, ids=None):
    records = [complete_record()] * 20 if records is None else records
    ids = [f"cal_{i}" for i in range(len(records))] if ids is None else ids
    return calibrate_policy_gates(records, ids, policy_id="policy_sha",
                                  population_id="population_sha")


def apply(losses, cal, **kwargs):
    return apply_candidate_gates(losses, cal, history_id="fresh_eval",
                                 policy_id="policy_sha", population_id="population_sha",
                                 **kwargs)


def test_full_policy_scores_include_every_method_and_round():
    losses = np.array([[[3.0, 5.0], [7.0, 4.0]],
                       [[2.0, 9.0], [5.0, 10.0]]])
    score = policy_trial_scores(losses, 1)
    assert score.relative_gap == 7.0
    assert score.absolute_loss == 10.0
    assert score.minimum_absolute_loss == 5.0
    assert score.fit_count == 8
    assert score.complete and not score.coverage_failure


def test_each_trial_can_have_a_different_candidate_count():
    a = policy_trial_scores([1.0, 3.0], 1)
    b = policy_trial_scores([1.0, 5.0, 3.0], 2)
    assert a.relative_gap == b.relative_gap == 2.0
    assert a.absolute_loss == b.absolute_loss == 3.0
    cal = calibration([a, b])
    assert math.isinf(cal.relative.threshold)
    assert cal.complete_history_count == 2


@pytest.mark.parametrize("losses,success", [([1.0, np.nan], None),
                                            ([1.0, np.inf], None),
                                            ([1.0, -2.0], None),
                                            ([1.0, 2.0], [True, False])])
def test_any_required_failure_maps_calibration_to_infinity(losses, success):
    score = policy_trial_scores(losses, 0, success=success)
    assert score.compute_failure and not score.complete
    assert score.relative_gap == score.absolute_loss == math.inf
    records = [complete_record()] * 19 + [score]
    cal = calibration(records)
    assert math.isinf(cal.relative.threshold) and math.isinf(cal.absolute.threshold)
    assert cal.relative.sample_count == 20
    assert cal.compute_failure_count == 1


def test_missing_truth_is_recorded_and_cannot_be_restored_by_the_gate():
    score = policy_trial_scores([1.0, 2.0], None)
    assert score.coverage_failure and not score.coverage_ok
    assert not score.compute_failure
    cal = calibration([score] * 20)
    result = apply([1.0, 2.0], cal, candidate_names=["model_a", "model_b"])
    assert result["retained_combined"] == ["model_a", "model_b"]
    assert "missing_truth" not in result["retained_combined"]
    assert cal.summary()["missed_truth_repaired"] is False
    assert result["coverage_status"] == "requires_external_audit"


def test_leave_one_out_rank_rejection_count_is_exact_for_distinct_scores():
    values = np.arange(20.0)
    alpha = 0.1
    rejects = 0
    for held_out in range(len(values)):
        quantile = conformal_upper_quantile(np.delete(values, held_out), alpha)
        rejects += values[held_out] > quantile.threshold
    assert rejects == math.floor(alpha * len(values))


def test_small_calibration_sample_has_a_vacuous_gate_and_ties_are_retained():
    assert math.isinf(conformal_upper_quantile([0.0] * 5, 0.05).threshold)
    assert conformal_upper_quantile([], 0.05).rank_tail_bound == 0.0
    quantile = conformal_upper_quantile([0.0] * 99, 0.05)
    assert not (0.0 > quantile.threshold)
    assert quantile.rank_tail_bound <= 0.05


def test_joint_absolute_and_relative_gate_has_the_allocated_rank_budget():
    n = 20
    rejected = 0
    for held_out in range(n):
        records = [complete_record(relative=float(i), absolute=float(50 - i))
                   for i in range(n) if i != held_out]
        cal = calibrate_policy_gates(records, [f"h_{i}" for i in range(n) if i != held_out],
                                     policy_id="p", population_id="pop", alpha_total=0.2)
        truth_loss = 50.0 - held_out
        result = apply_candidate_gates([truth_loss - held_out, truth_loss], cal,
                                       history_id=f"h_{held_out}", policy_id="p",
                                       population_id="pop")
        rejected += result["combined_reject"][1]
    assert rejected == 4
    assert rejected / n <= 0.2


def test_absolute_loss_can_alarm_when_the_relative_winner_has_zero_gap():
    result = apply([4.0, 4.0], calibration())
    assert result["relative_reject"] == [False, False]
    assert result["absolute_reject"] == result["combined_reject"] == [True, True]
    assert result["absolute_alarm"]
    assert result["retained_combined"] == []
    equal = apply([1.0, 3.0], calibration())
    assert equal["combined_reject"] == [False, False]


def test_fit_failure_abstains_for_the_whole_step():
    result = apply([1000.0, 1000.0], calibration(), success=[True, False])
    assert result["compute_failure"]
    assert result["combined_reject"] == result["relative_reject"] == [False, False]
    assert result["retained_combined"] == [0, 1]
    assert not result["absolute_alarm"]
    signature = inspect.signature(apply_candidate_gates)
    assert "true_index" not in signature.parameters


def test_context_and_independent_histories_are_enforced():
    cal = calibration()
    with pytest.raises(ValueError, match="overlaps"):
        apply_candidate_gates([1.0, 2.0], cal, history_id="cal_0",
                              policy_id="policy_sha", population_id="population_sha")
    with pytest.raises(ValueError, match="context"):
        apply_candidate_gates([1.0, 2.0], cal, history_id="new",
                              policy_id="other_policy", population_id="population_sha")
    with pytest.raises(ValueError, match="context"):
        apply_candidate_gates([1.0, 2.0], cal, history_id="new",
                              policy_id="policy_sha", population_id="other_population")
    with pytest.raises(ValueError, match="unique"):
        calibration([complete_record()] * 2, ids=[1, "1"])


def test_binomial_interval_inverts_independent_binomial_tails():
    lower, upper = binomial_exact_interval(3, 20)
    assert binom.sf(2, 20, lower) == pytest.approx(0.025, abs=1e-12)
    assert binom.cdf(3, 20, upper) == pytest.approx(0.025, abs=1e-12)
    assert binomial_exact_interval(0, 20)[1] == pytest.approx(1.0 - 0.025 ** (1.0 / 20))
    assert binomial_exact_interval(20, 20)[0] == pytest.approx(0.025 ** (1.0 / 20))
    assert binomial_exact_interval(0, 0) == (0.0, 1.0)


def test_history_unit_events_reject_noise_pooling_by_duplicate_ids():
    result = history_event_summary(["h1", "h2", "h3"], [True, False, False])
    assert result["history_count"] == 3 and result["count"] == 1
    assert result["unit"] == "independent_history"
    with pytest.raises(ValueError, match="unique"):
        history_event_summary(["same_history", "same_history"], [True, False])


def test_paired_binary_effects_use_discordant_histories():
    a = np.array([True, True, True, False, False, False])
    b = np.array([True, False, False, True, False, False])
    result = paired_binary_effects(range(6), a, b)
    assert (result["improved"], result["worsened"], result["tied"]) == (2, 1, 3)
    assert result["effect"] == pytest.approx(1.0 / 6.0)
    assert result["exact_mcnemar_pvalue"] == 1.0
    assert result["conservative_exact_interval"][0] <= result["effect"]
    assert result["conservative_exact_interval"][1] >= result["effect"]
    all_improved = paired_binary_effects(range(5), [True] * 5, [False] * 5)
    assert all_improved["exact_mcnemar_pvalue"] == pytest.approx(2.0 / 2 ** 5)
    tied = paired_binary_effects(range(5), [False] * 5, [False] * 5)
    assert tied["effect"] == 0.0 and tied["exact_mcnemar_pvalue"] == 1.0


def test_paired_continuous_effects_keep_the_history_pairing():
    result = paired_continuous_effects([1, 2, 3], [101.0, 201.0, 301.0], [100.0, 200.0, 300.0])
    assert result["effect"] == 1.0 and result["paired_standard_error"] == 0.0
    assert not result["exact_confidence_interval"]


def test_dynamic_eig_replay_uses_actual_actions_and_independent_validation():
    from research_experiments.mobile_ev_mle.bayes import posterior_weights
    post = posterior_weights([0.0], [[0.0], [0.0]], [0, 1], [0.5, 0.5])
    calls = []
    def build(action):
        calls.append(action)
        return np.array([[-action], [action]])
    result = replay_eig_shortlist(
        post, {"none": 0.0, "constant": 0.3, "milp": 0.8}, build,
        eligible_names=["constant", "milp"], samples_per_topology=128,
        selection_seed=41, validation_seeds=(42, 43), replay_context_id="physical_bank_sha")
    assert sorted(calls) == [0.0, 0.3, 0.8]
    assert result["selected_action"] == "milp"
    assert len(set(result["mean_hashes"].values())) == 3
    assert result["validation_action_frozen"]
    assert result["fixed_action_validation_comparisons"]["constant"]["eig_gain"] > 0.0
    assert result["fixed_action_validation_comparisons"]["constant"]["independent_integration_repeats"] == 2
    assert not result["evaluation_truth_input"] and not result["realized_future_noise_input"]


def test_dynamic_eig_replay_checks_atom_rows_and_seed_separation():
    class StubPosterior:
        log_weights = np.log([0.5, 0.5])
    with pytest.raises(ValueError, match="independent"):
        replay_eig_shortlist(StubPosterior(), {"a": 1.0}, lambda a: np.zeros((2, 1)),
                             selection_seed=1, validation_seeds=(1, 2), replay_context_id="p")
    with pytest.raises(ValueError, match="posterior atom rows"):
        replay_eig_shortlist(StubPosterior(), {"a": 1.0}, lambda a: np.zeros((3, 1)),
                             replay_context_id="p")


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan])
def test_invalid_risk_levels_are_rejected(alpha):
    with pytest.raises(ValueError):
        conformal_upper_quantile([1.0, 2.0], alpha)
