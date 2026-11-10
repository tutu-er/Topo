"""End-to-end synthetic planning/freezing checks; no private grid or AC data."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit, logit

from research_experiments.wzzt_dr_selection.dr_pricing.bayes_qcqp.core import separation_threshold
from research_experiments.wzzt_dr_selection.dr_pricing.bayes_qcqp.run import plan, sample


def _json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(tmp_path, *, linear=False, initial_prior=0., history=0., eta=.6, beta=.8):
    """One-resource case with analytic directional costs and frozen parameters."""
    folder = tmp_path / "case"
    folder.mkdir()
    config = dict(format_version=1, arrays_file="arrays.npz", case_id="synthetic_test",
        dt_hours=1., decision_slots=1, residual_unit="synthetic", node_labels=[101],
        resource_ids=["101_flexible"],
        information=dict(parameters_frozen=True, actual_future_used=False, truth_used=False),
        selection=dict(history_count=1, new_count=1, new_day_weight=2.,
            historical_loss_coarse=1., historical_loss_fine=1., relative_band=.03,
            absolute_tolerance=1e-6, candidate_ids=["C", "F"], support_counts=[1,2],
            historically_eligible=[True,True], candidate_count=2),
        probability=dict(target=.9, independent_cells=True, shared_candidate_noise=True,
                         scope="conditional_voltage"))
    arrays = dict(baseline_coarse=np.array([[.5 if linear else 0.]]),
        baseline_fine=np.array([[-.5 if linear else 0.]]),
        effect_coarse=np.array([[1. if linear else 4.]]),
        effect_fine=np.array([[1. if linear else 0.]]), sigma=np.ones((1,1)),
        resource_lower_kw=np.array([[-.5]]), resource_upper_kw=np.array([[1.]]),
        resource_cost=np.ones((1,1)), resource_efficiency=np.ones(1),
        resource_energy_lower_kwh=np.array([-np.inf]),
        resource_energy_upper_kwh=np.array([np.inf]),
        resource_reset=np.ones((1,1),bool), resource_terminal=np.zeros((1,1),bool))
    _json(folder/"case.json", config)
    np.savez_compressed(folder/"arrays.npz", **arrays)
    np.savez_compressed(folder/"bayes_arrays.npz", likelihood_sd=np.ones((1,1)),
                        response_relative_sd=np.zeros((1,1)))
    prob_f = float(expit(initial_prior))
    settings = dict(likelihood="fixed_common_diagonal_gaussian", arrays_file="bayes_arrays.npz",
        posterior_threshold=eta, event_probability_target=beta,
        base_prior=[1-prob_f, prob_f], history_rows=int(history!=0.),
        history_log_bayes_factor=history,
        solver=dict(seconds=15., mip_gap=1e-7, threads=1, seed=0))
    if history:
        receipt = folder/"history_evidence.json"
        _json(receipt, dict(history_rows=1, log_bayes_factor=history,
                           method="common_diagonal_gaussian_squared_residuals"))
        settings["history_receipt"] = dict(path=receipt.name, sha256=_sha(receipt))
    _json(folder/"bayes_settings.json", settings)
    return folder, folder/"bayes_settings.json"


@pytest.mark.parametrize("prior,chosen", [(0.,"F"),(-.1,"C")])
def test_expected_either_compares_both_directions_with_their_actual_costs(tmp_path,prior,chosen):
    pytest.importorskip("gurobipy")
    folder, settings = inputs(tmp_path, linear=True, initial_prior=prior)
    out = tmp_path/"plan"
    result = plan(folder,settings,"expected_either",out)
    cut = float(logit(.6))
    expected_costs = dict(F=(cut-prior)**2, C=2*(cut+prior)**2)
    assert result["qualified"] and set(result["branch_results"])=={"F","C"}
    assert result["selected_branch"]==chosen
    assert result["fee"]==pytest.approx(expected_costs[chosen], abs=2e-6)
    assert result["lower_bound"]==pytest.approx(min(expected_costs.values()), abs=2e-6)
    assert result["cost_gap"] < 2e-6
    freeze = json.loads((out/"policy_freeze.json").read_text())
    assert freeze["status"]=="frozen" and freeze["quote"]["sha256"]==_sha(out/"quote.npz")
    assert result["expectation_is_not_probability"]


def test_history_and_prior_count_once_and_zero_fee_has_global_certificate(tmp_path):
    folder,settings = inputs(tmp_path, initial_prior=.3, history=.2)
    out=tmp_path/"zero"
    result=plan(folder,settings,"expected_either",out)
    assert result["initial_log_odds"]==pytest.approx(.5)
    assert result["expected_log_odds"]==pytest.approx(.5)
    assert result["posterior_at_expected_log_odds"]==pytest.approx(expit(.5))
    assert result["qualified"] and result["analytic_zero_cost_optimum"]
    assert result["fee"]==result["lower_bound"]==result["upper_bound"]==0.
    assert set(result["branch_results"])=={"F"}
    assert result["branch_results"]["F"]["solver_status"]=="CERTIFIED_COST_ZERO"
    request=json.loads((out/"request.json").read_text())
    assert request["history_count"]==1
    assert request["historical_log_bayes_factor"]==.2
    assert request["new_day_weight"]==1 and request["legacy_new_day_weight"]==2
    with np.load(out/"quote.npz",allow_pickle=False) as z:
        assert np.max(np.abs(z["mean_u"]))==0.


def test_nominal_pair_event_uses_the_true_separation_threshold(tmp_path):
    pytest.importorskip("gurobipy")
    folder,settings=inputs(tmp_path,eta=.8,beta=.8)
    result=plan(folder,settings,"pair_event_nominal",tmp_path/"pair")
    threshold=separation_threshold(.8,.8)
    assert result["qualified"] and result["selected_branch"]=="both"
    assert result["nominal_separation"]==pytest.approx(threshold, abs=2e-6)
    assert result["fee"]==pytest.approx(threshold/16, abs=2e-6)
    assert min(result["ideal_pair_probability_F"],result["ideal_pair_probability_C"])>=.8-2e-7
    assert "nominal deterministic" in result["scope"]


def test_sampling_is_post_freeze_and_does_not_change_the_frozen_policy(tmp_path):
    pytest.importorskip("gurobipy")
    folder,settings=inputs(tmp_path,eta=.8,beta=.8)
    out=tmp_path/"planned"
    plan(folder,settings,"pair_event_nominal",out)
    quote=out/"quote.npz"
    hashes={name:_sha(out/name) for name in ("quote.npz","policy_freeze.json","request.json")}
    # A numerically identical bare array is not a policy frozen by this flow.
    bare=tmp_path/"not_frozen.npz"
    bare.write_bytes(quote.read_bytes())
    with pytest.raises(ValueError):
        sample(folder,settings,bare,tmp_path/"rejected",count=4096,seed=23)
    sampled=sample(folder,settings,quote,tmp_path/"sampled",count=4096,seed=23)
    assert sampled["count"]==4096 and sampled["no_model_or_quote_changes"]
    assert abs(sampled["mean_error"]) <= 5*sampled["mean_standard_error"]+1e-12
    assert abs(sampled["ideal_F_correct_confidence_successes"]/4096
               -sampled["ideal_F_analytic_probability"])<.035
    assert abs(sampled["ideal_C_correct_confidence_successes"]/4096
               -sampled["ideal_C_analytic_probability"])<.035
    assert {name:_sha(out/name) for name in hashes}==hashes
    original=quote.read_bytes()
    try:
        quote.write_bytes(original+b"changed_after_freeze")
        with pytest.raises(ValueError,match="Frozen quote reference changed"):
            sample(folder,settings,quote,tmp_path/"tampered",count=4096,seed=23)
    finally:
        quote.write_bytes(original)
    assert {name:_sha(out/name) for name in hashes}==hashes
