"""Synthetic scoring/unit smoke against the current portable DR functions.

No quote optimizer, probability integral, device constructor, AC, parameter
fit, real observation file or completed native F1 is touched.  Only the
selection-state fields needed by score_observations are initialized using the
neighbor's own _selection_rule; this is not a complete planning-case test.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np

from . import candidates as own


ROOT = Path(__file__).resolve().parents[2]
NEIGHBOR = Path("C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo/research_experiments/wzzt_dr_selection/dr_pricing")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def scorer(module, *, slots, nodes, scale, weight=1., overrides=None):
    instance = module.PricingProblem.__new__(module.PricingProblem)
    instance.T, instance.N = slots, nodes
    rule = dict(history_count=288, new_count=slots, new_day_weight=weight,
                historical_loss_coarse=8.1*scale/1e6, historical_loss_fine=8.*scale/1e6,
                relative_band=.03, absolute_tolerance=1e-12*scale,
                candidate_ids=["coarse", "fine"], support_counts=[3, 4],
                historically_eligible=[True, True], candidate_count=2)
    rule.update(overrides or {})
    instance._selection_rule(rule)
    return instance


def main():
    started = datetime.now(timezone.utc).isoformat()
    files = [NEIGHBOR/"problem.py", NEIGHBOR/"probability.py", Path(own.__file__), Path(__file__)]
    before_hashes = {str(path): sha(path) for path in files}
    package = types.ModuleType("neighbor_portable_dr_scoring_smoke")
    package.__path__ = [str(NEIGHBOR)]
    sys.modules[package.__name__] = package
    load_module(package.__name__+".probability", NEIGHBOR/"probability.py")
    problem = load_module(package.__name__+".problem", NEIGHBOR/"problem.py")

    slots, nodes, base_kw, scale = 96, 3, 200., 1e6
    coarse = dict(supports=[[0], [1], [2]], r=[.1, .12, .11], x=[.02, .03, .025])
    fine = dict(supports=[[0], [1], [2], [1, 2]], r=[.1, .12, .11, .2], x=[.02, .03, .025, .05])
    original_parameters = own.digest([coarse, fine])
    rc, xc = own.response_matrices(coarse, nodes)
    rf, xf = own.response_matrices(fine, nodes)
    rng = np.random.default_rng(2026111019)
    natural_p = rng.uniform(.01, .05, (slots, nodes))
    natural_q = rng.uniform(.004, .02, (slots, nodes))
    delivered = rng.uniform(-.5, .5, (3, slots))
    resource_node = np.array([0, 2, 2])
    q_over_p = np.array([.3, 0., .5])
    delivered_saved = delivered.copy()
    dp_kw, dq_kvar = problem.aggregate_delivery(delivered, resource_node, q_over_p, nodes)
    incidence = np.eye(nodes)[resource_node].T
    expected_dp = delivered.T @ incidence.T
    expected_dq = (delivered*q_over_p[:, None]).T @ incidence.T
    mapping_error = max(float(np.max(abs(dp_kw-expected_dp))), float(np.max(abs(dq_kvar-expected_dq))))
    assert mapping_error <= 1e-14
    assert np.array_equal(delivered_saved, delivered)
    p, q = natural_p+dp_kw/base_kw, natural_q+dq_kvar/base_kw
    power = np.column_stack((p, q))
    hc, hf = np.column_stack((rc, xc)), np.column_stack((rf, xf))
    direct_predictions = [p@rc.T+q@xc.T, p@rf.T+q@xf.T]
    predictions = [power@hc.T, power@hf.T]
    prediction_error = max(float(np.max(abs(a-b))) for a, b in zip(predictions, direct_predictions))
    assert prediction_error <= 1e-14
    increment_error = float(np.max(abs((direct_predictions[1]-(natural_p@rf.T+natural_q@xf.T))
                                      -(dp_kw/base_kw@rf.T+dq_kvar/base_kw@xf.T))))
    assert increment_error <= 1e-14
    drop = direct_predictions[1] + rng.uniform(-2e-6, 2e-6, (slots, nodes))
    canonical = scorer(problem, slots=slots, nodes=nodes, scale=1.)
    micro = scorer(problem, slots=slots, nodes=nodes, scale=scale)
    canonical_result = canonical.score_observations(power, drop, hc, hf)
    micro_result = micro.score_observations(power, scale*drop, scale*hc, scale*hf)
    residual_c, residual_f = scale*(drop-predictions[0]), scale*(drop-predictions[1])
    residual_result = micro.score_residuals(residual_c, residual_f)
    expected_losses = np.array([np.mean(abs(residual_c)), np.mean(abs(residual_f))])
    expected_combined = .75*np.array([8.1, 8.])+.25*expected_losses
    expected_eligible = expected_combined <= 1.03*expected_combined.min()+1e-6
    expected_selected = min(np.flatnonzero(expected_eligible), key=lambda j: ([3,4][j], expected_combined[j], j))
    assert micro_result["selected"] == ["coarse", "fine"][expected_selected] == canonical_result["selected"]
    assert micro_result["eligible"] == canonical_result["eligible"]
    losses = np.array([micro_result["new_losses"][name] for name in ("coarse", "fine")])
    combined = np.array([micro_result["combined_losses"][name] for name in ("coarse", "fine")])
    score_error = max(float(np.max(abs(losses-expected_losses))), float(np.max(abs(combined-expected_combined))))
    assert score_error <= 1e-9
    unit_error = max(abs(micro_result[key]-scale*canonical_result[key]) for key in ("D", "threshold", "selection_margin"))
    assert unit_error <= 1e-8
    residual_api_error = max(abs(micro_result["new_losses"][name]-residual_result["new_losses"][name])
                             for name in ("coarse", "fine"))
    assert residual_api_error <= 1e-9
    assert micro_result["identity_error"] <= 1e-9
    weighted = scorer(problem, slots=slots, nodes=nodes, scale=scale, weight=2.)
    weighted_result = weighted.score_observations(power, scale*drop, scale*hc, scale*hf)
    assert abs(weighted.new_fraction-.4) <= 1e-15 and abs(micro.new_fraction-.25) <= 1e-15
    assert abs(micro.threshold-(2*weighted.threshold-micro.epsilon)) <= 1e-12

    rejected = {}
    checks = {
        "wrong_observation_shape": lambda: micro.score_observations(power.T, scale*drop, scale*hc, scale*hf),
        "out_of_range_resource_node": lambda: problem.aggregate_delivery(delivered, [0,2,3], q_over_p, nodes),
        "fractional_resource_node": lambda: problem.aggregate_delivery(delivered, [0,2,1.5], q_over_p, nodes),
        "missing_explicit_new_day_weight": lambda: _missing_weight(problem, slots, nodes),
        "more_than_two_candidates": lambda: scorer(problem, slots=slots, nodes=nodes, scale=scale, overrides={"candidate_count":3}),
        "equal_raw_support_counts": lambda: scorer(problem, slots=slots, nodes=nodes, scale=scale, overrides={"support_counts":[4,4]}),
        "historically_ineligible_pair": lambda: scorer(problem, slots=slots, nodes=nodes, scale=scale, overrides={"historical_loss_coarse":9.}),
    }
    for name, action in checks.items():
        try:
            action()
        except ValueError:
            rejected[name] = True
        else:
            raise AssertionError("Current portable package did not reject " + name)
    assert own.digest([coarse, fine]) == original_parameters
    after_hashes = {str(path): sha(path) for path in files}
    assert before_hashes == after_hashes, "Source changed during the portable interface smoke"
    receipt = dict(passed=True, checked_started_utc=started,
                   checked_finished_utc=datetime.now(timezone.utc).isoformat(),
                   target="current dr_pricing.problem aggregate_delivery, _selection_rule, score_observations and score_residuals",
                   slots=slots, observation_nodes=nodes, resources=3, power_base_kw=base_kw, reactive_base_kvar=base_kw,
                   synthetic_only=True, mapping_max_error=mapping_error, response_prediction_max_error=prediction_error,
                   increment_added_once_identity_error=increment_error, score_max_error=score_error,
                   residual_vs_observation_API_max_error=residual_api_error, residual_unit_scaling_max_error=unit_error,
                   score_selection_identity_error=micro_result["identity_error"], selected_micro=micro_result["selected"],
                   selected_canonical=canonical_result["selected"], same_selection_under_declared_unit_conversion=True,
                   explicit_weight1_new_fraction=micro.new_fraction, explicit_weight2_new_fraction=weighted.new_fraction,
                   weight1_threshold=micro.threshold, weight2_threshold=weighted.threshold,
                   rejected=rejected, frozen_response_parameters_unchanged=True, source_hashes=after_hashes,
                   source_unchanged_during_smoke=True, neighbor_modified=False, neighbor_contacted=False,
                   new_fit_calls=0, new_search_calls=0, pricing_optimizers_invoked=0, probability_integrals_evaluated=0,
                   device_models_constructed=0, AC_calls=0, native_F1_or_test_observations_read=False,
                   complete_planning_case_loaded=False, direct_bridge_loader_tested=False,
                   coverage="Synthetic prediction, unit/increment mapping and fixed-pair scoring only; no quote optimization, noise-law validation, device or end-to-end DR recovery guarantee")
    target = ROOT/"docs/day96_t_benchmark_20261110/portable_bridge_verification.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return receipt


def _missing_weight(module, slots, nodes):
    instance = module.PricingProblem.__new__(module.PricingProblem)
    instance.T, instance.N = slots, nodes
    instance._selection_rule(dict(history_count=288, new_count=slots,
                                  historical_loss_coarse=8.1, historical_loss_fine=8.,
                                  relative_band=.03, absolute_tolerance=1e-6,
                                  candidate_ids=["coarse", "fine"], support_counts=[3,4],
                                  historically_eligible=[True, True], candidate_count=2))


if __name__ == "__main__":
    main()
