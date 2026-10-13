"""Hand-computed regression checks of descriptive frozen-result analysis."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.gradient_price_analysis import (
    analyze_trials, compare_plans_exact, signed_actual_margin)
from research_experiments.mobile_ev_mle.gradient_price_analysis import (
    paired_correlation, true_impedance_errors, load_completed_phase, analysis_markdown)


def sample_trial(identifier="1", *, case="star", predicted_gain=1., actual_effect=2.):
    plan = dict(site=0, p=[[.1, 0.], [.1, 0.]], q=[[0., 0.], [0., 0.]],
                sites=[0, 0], soc=[.4, .41, .42])
    fitted = dict(name="truth", success=True, supports=[[0, 1], [0], [1]],
                  r=[.11, .22, .33], x=[.055, .11, .165], diagnostics=dict(seconds=.5))
    baseline = dict(method="constant_fixed", status="complete", true_name="truth", true_covered=True,
        losses=dict(truth=2., wrong=4.), fits=[deepcopy(fitted)], seconds=.7, ac_calls=5)
    design = dict(method="optimized_designed", status="complete", true_name="truth", true_covered=True,
        losses=dict(truth=2., wrong=4.+actual_effect), fits=[deepcopy(fitted)], seconds=.9, ac_calls=7)
    full = dict(success=True, plan=deepcopy(plan), score=10.+predicted_gain, best_seed_score=10.+predicted_gain-1e-12,
        gradient_gain_over_best_seed=1e-12, seconds=3., total_ac_calls=20,
        site_records=[dict(site=0, constant_score=10., steps=[dict(accepted=True, actual_local_score_gain=1e-12)])])
    return dict(history_id=str(identifier), case=case, true_name="truth", true_covered=True, complete=True,
        candidate_names=["truth", "wrong"], policy_id="fixed-policy", source_sha256={"selector.py":"abc"},
        historical_cost=dict(seconds=.4), total_ac_calls=40,
        truth=dict(supports=[], r_pu=[.1, .2, .3], x_pu=[.05, .1, .15]),
        selection=dict(activated=True, design_candidates=["truth", "wrong"], seconds=6., selector_ac_calls=35,
                       searches=dict(full=full, pq_only=dict(success=True, plan=deepcopy(plan)))),
        records=[baseline, design])


def test_signed_actual_margin_keeps_wrong_winner_negative_and_true_winner_positive():
    trial = sample_trial(actual_effect=-3.)
    assert signed_actual_margin(trial, trial["records"][0]) == 2.
    assert signed_actual_margin(trial, trial["records"][1]) == -1.
    result = analyze_trials([trial])
    row = result["histories"][0]
    assert row["actual_design_minus_constant_margin_full_SSE"] == -3.
    assert row["predicted_gain_vs_fixed0_half_SSE"] == 1.
    assert result["all"]["actual_design_minus_constant_margin"]["mean"] == -3.


def test_small_numeric_seed_gain_is_not_a_material_gradient_gain():
    result = analyze_trials([sample_trial()])
    rows = result["all"]
    assert rows["accepted_steps"] == 1
    assert rows["material_accepted_steps"] == 0
    assert rows["material_gradient_gain_over_best_seed_count"] == 0
    assert rows["material_gain_threshold_half_SSE"] == 1e-6


def test_exact_plan_comparison_counts_arrays_separately_without_tolerance():
    trial = sample_trial()
    full = trial["selection"]["searches"]["full"]["plan"]
    pq = deepcopy(full)
    assert compare_plans_exact(full, pq)["all_arrays_equal"]
    pq["p"][0][0] += 1e-12
    compared = compare_plans_exact(full, pq)
    assert not compared["all_arrays_equal"]
    assert not compared["field_equal"]["p"] and compared["field_equal"]["soc"]
    trial["selection"]["searches"]["pq_only"]["plan"] = pq
    result = analyze_trials([trial])["all"]
    assert result["full_pq_plan_comparison_denominator"] == 1
    assert result["full_pq_all_arrays_exact_match_count"] == 0
    assert result["full_pq_each_array_exact_match_count"] == dict(p=0, q=1, sites=1, soc=1)


def test_case_denominators_coverage_and_evaluation_only_reference_coverage():
    covered = sample_trial("1", case="strong")
    excluded = sample_trial("2", case="weak")
    excluded["selection"]["design_candidates"] = ["wrong"]
    missing = sample_trial("3", case="star")
    missing.update(true_name=None, true_covered=False, complete=False)
    for row in missing["records"]:
        row.update(true_name=None, true_covered=False, status="joint_fit_failure")
    result = analyze_trials([covered, excluded, missing])
    assert result["all"]["all_history_count"] == 3
    assert result["all"]["complete_count"] == 2
    assert result["all"]["incomplete_count"] == 1
    assert result["all"]["candidate_coverage_failure_count"] == 1
    assert result["all"]["design_truth_reference_denominator"] == 2
    assert result["all"]["design_truth_reference_covered_count"] == 1
    assert result["by_case"]["weak"]["design_truth_reference_excluded_count"] == 1
    assert result["by_case"]["star"]["methods"]["optimized_designed"]["failed_count"] == 1


def test_paired_correlations_use_history_units_and_constant_inputs_are_na():
    trials = [sample_trial(i, predicted_gain=i, actual_effect=2*i) for i in (1, 2, 3)]
    correlation = analyze_trials(trials)["all"]["predicted_vs_observed_gain_correlation"]
    assert correlation["n"] == 3
    assert correlation["pearson"] == pytest.approx(1.)
    assert correlation["spearman"] == pytest.approx(1.)
    same = paired_correlation([1., 1.], [1., 2.])
    assert same["pearson"] is None and same["spearman"] is None
    assert same["unavailable_reason"] == "constant_prediction"
    assert paired_correlation([1.], [2.])["unavailable_reason"] == "fewer_than_two_paired_histories"
    assert paired_correlation([None, 1.], [2., 2.])["n"] == 1


def test_support_aligned_true_impedance_rmse_and_failed_fit_exclusion():
    trial = sample_trial()
    # Fit support order differs from the true canonical edge order.
    fitted = trial["records"][1]["fits"][0]
    fitted.update(supports=[[1], [0, 1], [0]], r=[.33, .11, .22], x=[.165, .055, .11])
    result = analyze_trials([trial])["all"]["methods"]["optimized_designed"]["impedance"]
    common, leaves = result["by_edge_class"]["common_full"], result["by_edge_class"]["leaf"]
    assert common["r"]["absolute_rmse_pu"] == pytest.approx(.01)
    assert leaves["r"]["absolute_rmse_pu"] == pytest.approx(np.sqrt((.02**2+.03**2)/2))
    assert leaves["r"]["relative_rmse"] == pytest.approx(.1)
    assert leaves["x"]["relative_rmse"] == pytest.approx(.1)
    fitted["success"] = False
    errors, reason = true_impedance_errors(trial, trial["records"][1])
    assert errors is None and reason == "true_fit_failed"
    result = analyze_trials([trial])["all"]["methods"]["optimized_designed"]["impedance"]
    assert result["successful_true_fit_history_count"] == 0
    assert result["by_edge_class"]["leaf"]["r"]["absolute_rmse_pu"] is None


def test_weak_internal_relative_error_uses_the_small_true_parameter():
    trial = sample_trial(case="weak")
    trial["truth"] = dict(supports=[[0, 1]], r_pu=[.1, .0001, .2, .3, .4], x_pu=[.05, .00005, .1, .15, .2])
    # Truth canonical order is full, internal pair01, leaf0, leaf1, leaf2;
    # the fitted arrays below deliberately use a different support order.
    fitted = trial["records"][1]["fits"][0]
    fitted.update(supports=[[0, 1, 2], [0], [1], [2], [0, 1]],
                  r=[.1, .2, .3, .4, .0003], x=[.05, .1, .15, .2, .00015])
    result = analyze_trials([trial])["all"]["methods"]["optimized_designed"]["impedance"]["by_edge_class"]["internal"]
    assert result["r"]["absolute_rmse_pu"] == pytest.approx(.0002)
    assert result["r"]["relative_rmse"] == pytest.approx(2.)
    assert result["x"]["relative_rmse"] == pytest.approx(2.)
    assert result["r"]["true_parameter_min_pu"] == .0001


def test_fit_and_selection_time_are_reported_separately_and_missing_fields_are_forward_compatible():
    trial = sample_trial()
    result = analyze_trials([trial], calibration=dict(relative=dict(threshold=6.75), scope="declared mixture"))
    assert result["all"]["timing"]["selection_seconds"]["mean"] == 6.
    assert result["all"]["timing"]["historical_fit_seconds"]["mean"] == .4
    assert result["all"]["methods"]["optimized_designed"]["joint_fit_seconds"]["mean"] == .5
    assert result["calibration_threshold"] == 6.75
    minimal = analyze_trials([dict(history_id="minimal", records=[])])
    assert minimal["all"]["all_history_count"] == 1
    assert minimal["all"]["completeness_unknown_count"] == 1
    assert minimal["all"]["methods"]["optimized_designed"]["missing_count"] == 1
    assert "NA" in analysis_markdown(minimal)


def test_completed_raw_or_compact_loader_rejects_partial_phase(tmp_path):
    phase = tmp_path/"pilot"; phase.mkdir()
    (phase/"config.json").write_text(json.dumps(dict(count=2)))
    (phase/"summary.json").write_text("{}")
    data = json.dumps(sample_trial("1")).encode()
    (phase/"trial_1.json.gz").write_bytes(gzip.compress(data))
    with pytest.raises(ValueError, match="incomplete"):
        load_completed_phase(tmp_path, "pilot")
    (phase/"trial_2.json").write_text(json.dumps(sample_trial("2")))
    trials, provenance, _, _ = load_completed_phase(tmp_path, "pilot")
    assert len(trials) == 2 and len(provenance["input_sha256"]) == 2
    (phase/"summary.json").unlink()
    with pytest.raises(ValueError, match="completed summary"):
        load_completed_phase(tmp_path, "pilot")
