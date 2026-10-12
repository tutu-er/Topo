"""Focused design checks: feasibility, shared-history refit and failure retention."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from . import design
from .design import (
    EVAction, EVConstraints, FutureForecast, check_action,
    enumerate_actions, expected_observations, select_action,
)
from .likelihood import MeasurementModel, Observations, RadialCandidate, ac_predict


def _constant(name, site, n_slots=8, n_sites=3, energy=.30, dt=.25):
    p = np.zeros((n_slots, n_sites))
    p[:, site] = energy / (n_slots * dt)
    return EVAction(name, p, np.zeros_like(p), np.full(n_slots, site), "constant")


def _fit(loss=0., n=2, t=2, success=True):
    return SimpleNamespace(r=np.full(n+1, .01), x=np.full(n+1, .005),
                           p=np.ones((t,n)), q=np.zeros((t,n)),
                           source_p=np.zeros(t), source_q=np.zeros(t),
                           bias_p=np.zeros(n), bias_q=np.zeros(n), bias_v=np.zeros(n),
                           loss=loss, loglik=-loss/2, success=success, diagnostics={})


def test_library_has_equal_energy_feasible_constant_pulses_and_movement():
    constraints = EVConstraints(travel_energy=.01*(np.ones((3,3))-np.eye(3)))
    actions, rejected = enumerate_actions(8, 3, constraints, max_actions=12)
    assert {"constant", "fixed_pulse", "mobile"} <= {a.family for a in actions}
    assert len(actions) <= 12
    for action in actions:
        check = check_action(action, constraints)
        assert check["feasible"], (action.name, check)
        assert check["energy_pu_hours"] == pytest.approx(.30)
        assert action.metadata(constraints)["energy_kwh"] == pytest.approx(6.0)
        assert np.all(action.q == 0)
        assert np.all(action.p[action.sites == -1] == 0)
    assert any(a.name == "constant_site_0" for a in actions)
    assert all("name" in proposal for proposal in rejected)


def test_access_and_initial_travel_are_enforced_by_library():
    access = np.ones((8, 3), bool)
    access[0:3, 1] = False
    constraints = EVConstraints(availability=access, initial_site=0)
    actions, _ = enumerate_actions(8, 3, constraints)
    for action in actions:
        assert check_action(action, constraints)["feasible"]
        assert not np.any(action.p[~access] > 0)
        first_site = next(site for site in action.sites if site >= 0)
        if first_site != 0:
            assert action.sites[0] == -1


def test_travel_energy_soc_q_and_access_failures_are_retained():
    valid = _constant("valid", 0)
    assert check_action(valid, EVConstraints())["feasible"]
    assert "soc" in check_action(valid, EVConstraints(soc_max=.41))["issues"]
    access = np.ones((8,3),bool)
    access[0,0] = False
    assert "charger_access" in check_action(valid, EVConstraints(availability=access))["issues"]
    q = valid.q.copy()
    q[0,0] = .1
    unsupported = EVAction("independent_q",valid.p,q,valid.sites)
    assert "unsupported_reactive_power" in check_action(unsupported, EVConstraints())["issues"]
    sites = np.r_[np.zeros(4,int),np.ones(4,int)]
    p = np.zeros((8,3))
    p[:4,0],p[4:,1] = .15,.15
    no_elapsed_move = EVAction("instant_move",p,np.zeros_like(p),sites)
    assert "insufficient_travel_time" in check_action(no_elapsed_move, EVConstraints())["issues"]
    travel_p = valid.p.copy()
    travel_sites = valid.sites.copy()
    travel_sites[2] = -1
    charging_move = EVAction("charging_move",travel_p,valid.q,travel_sites)
    assert "charging_during_travel" in check_action(charging_move, EVConstraints())["issues"]


def test_prediction_includes_ev_once_and_same_calibrated_bias():
    candidate = RadialCandidate.from_supports("star", 2)
    fitted = _fit(n=2)
    fitted.p[:] = .2
    model = MeasurementModel(.01,.01,.001,master_p_sigma=.01,master_q_sigma=.01,
                             bias_p=.05,bias_q=.02,bias_v=.003)
    history = Observations(np.full((2,2),.25),np.full((2,2),.02),np.ones((2,2)),
                           master_p=np.ones(2),master_q=np.zeros(2))
    forecast = FutureForecast(np.full((2,2),.2),np.zeros((2,2)),
                              np.zeros(2),np.zeros(2),np.ones(2))
    p = np.zeros((2,2))
    p[:,0] = .1
    action = EVAction("charge",p,np.zeros_like(p),np.zeros(2,int))
    # FitResult's calibrated bias is the authoritative shared meter offset.
    fitted.bias_p[:] = .05
    fitted.bias_q[:] = .02
    fitted.bias_v[:] = .003
    future = expected_observations(candidate,fitted,history,model,forecast,action)
    physical = ac_predict(candidate,fitted.r,fitted.x,forecast.p+p,forecast.q)
    assert np.allclose(future.p[:,0],.35)
    assert np.allclose(future.p[:,1],.25)
    assert np.allclose(future.v,physical["v"]+.003)
    assert np.allclose(future.master_p,physical["master_p"])
    assert future.source_p is None


def test_history_joint_parameter_refit_discriminates_but_future_only_cannot():
    reference = SimpleNamespace(name="a_reference")
    rival = SimpleNamespace(name="b_rival")
    history_p = np.array([[1.,0.],[2.,0.]])
    observed_response = np.array([.0103,.0199])
    history_v = 1 - observed_response[:,None]*np.ones((1,2))
    history = Observations(history_p,np.zeros_like(history_p),history_v)
    model = MeasurementModel(1.,1.,.001)
    historical_feature = history_p.sum(axis=1)
    beta_h = (historical_feature @ observed_response) / (historical_feature @ historical_feature)
    baseline = 2*np.sum((observed_response-beta_h*historical_feature)**2)/model.v_sigma**2
    fitted = _fit(loss=baseline)
    fitted.p[:] = history_p
    forecast = FutureForecast(np.ones((2,2)) * np.array([1.,0.]),
                              np.zeros((2,2)),np.zeros(2),np.zeros(2),np.ones(2))
    actions = [_constant("site_0",0,2,2,.10),_constant("site_1",1,2,2,.10)]
    calls = []
    def predictor(candidate, r, x, p, q, source_p=0., source_q=0., root_v=1.):
        response = .01*(p[:,0]+2*p[:,1])
        return dict(p=p,q=q,v=1-response[:,None]*np.ones((1,2)),
                    master_p=p.sum(axis=1),master_q=q.sum(axis=1),
                    source_p=source_p,source_q=source_q)
    def joint_fit(candidate, joined, fit_model, **kwargs):
        assert np.array_equal(joined.p[:2],history.p)
        assert np.array_equal(joined.v[:2],history.v)
        assert kwargs["initial"] is fitted
        x = joined.p.sum(axis=1)
        y = (1-joined.v).mean(axis=1)
        beta = x @ y / (x @ x)
        loss = 2*np.sum((y-beta*x)**2)/fit_model.v_sigma**2
        calls.append(dict(joined=joined,beta=beta,loss=loss))
        return _fit(loss=loss,t=4)
    with patch.object(design,"ac_predict",predictor), patch.object(design,"fit_candidate",joint_fit):
        result = select_action([reference,rival],history,
                               {"a_reference":fitted,"b_rival":fitted},model,actions,
                               forecast=forecast,voltage_limits=None)
    assert result["selected"] == "site_1"
    assert result["scores"][1]["score"] > result["scores"][0]["score"]
    assert result["scores"][1]["score"] > 0
    # The two equal future slots can be fitted exactly by a future-only beta.
    future = calls[1]["joined"]
    future_x = future.p[2:].sum(axis=1)
    future_y = (1-future.v[2:]).mean(axis=1)
    beta_future = future_x @ future_y / (future_x @ future_x)
    assert np.sum((future_y-beta_future*future_x)**2) < 1e-25
    assert result["incremental_score_bound"].startswith("uncertified")
    assert result["local_inner_optimization"]


def test_failed_refit_and_negative_baseline_never_reward_actions():
    candidate = RadialCandidate.from_supports("a_reference",2)
    rival = RadialCandidate.from_supports("b_rival",2)
    fitted = _fit(loss=10)
    fitted.p[:] = .1
    history = Observations(fitted.p,fitted.q,np.ones((2,2)))
    model = MeasurementModel(.01,.01,.001)
    actions = [_constant("failed",0,2,2,.10),_constant("negative",1,2,2,.10)]
    outcomes = iter([_fit(loss=1e9,success=False),_fit(loss=1.)])
    with patch.object(design,"fit_candidate",lambda *a,**k:next(outcomes)):
        result = select_action([candidate,rival],history,
                               {candidate.name:fitted,rival.name:fitted},model,actions)
    assert result["selected"] is None
    assert result["failed_action_count"] == 2
    assert all(row["score"] is None for row in result["scores"])
    assert result["scores"][1]["competitor_fits"][0]["increment"] == -9
    assert result["scores"][1]["competitor_fits"][0]["negative_increment"]
    assert result["scores"][1]["failures"] == ["historical_baseline_instability:b_rival"]


def test_nominal_voltage_screen_reports_failure_without_inner_fit():
    candidate = RadialCandidate.from_supports("a_reference",2)
    rival = RadialCandidate.from_supports("b_rival",2)
    fitted = _fit()
    fitted.p[:] = .1
    history = Observations(fitted.p,fitted.q,np.ones((2,2)))
    model = MeasurementModel(.01,.01,.001)
    action = _constant("unsafe_nominal",0,2,2,.10)
    physical = ac_predict(candidate,fitted.r,fitted.x,action.p+fitted.p,fitted.q)
    physical["voltage"] = np.full((2,3),.85+0j)
    with patch.object(design,"ac_predict",lambda *a,**k:physical), \
         patch.object(design,"fit_candidate",side_effect=AssertionError("must not fit")):
        result = select_action([candidate,rival],history,
                               {candidate.name:fitted,rival.name:fitted},model,[action])
    assert result["selected"] is None
    assert "nominal_voltage_screen" in result["scores"][0]["failures"][0]
    assert result["grid_safety_interpretation"] == "reference_nominal_forecast_screen_only"


def test_no_competitor_and_historical_failures_remain_visible():
    candidate = RadialCandidate.from_supports("only",2)
    failed = RadialCandidate.from_supports("failed",2)
    history = Observations(np.full((2,2),.1),np.zeros((2,2)),np.ones((2,2)))
    result = select_action([candidate,failed],history,
                           {"only":_fit(),"failed":_fit(success=False)},
                           MeasurementModel(.01,.01,.001),[])
    assert result["status"] == "no_competing_candidate"
    assert result["historical_fit_failures"] == ["failed"]
    assert result["selected"] is None
