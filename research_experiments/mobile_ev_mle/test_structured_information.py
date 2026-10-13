"""Independent matrix, finite-difference and complete-AC information audits."""
from dataclasses import replace
from inspect import signature
import numpy as np
import pytest
from .likelihood import RadialCandidate, Observations, MeasurementModel, ac_predict
from .symmetric_ac_local import mean_and_jacobian
from .structured_information import (FutureForecast, InformationCenter,
    InformationRankError, InformationInitializationError, StructuredInformationEvaluator, project_columns,
    efficient_information, information_from_blocks)


def inputs(t=7, horizon=4):
    star = RadialCandidate.from_supports("star", 3)
    by_r = {s:r for s,r in zip(star.supports, [.005, .025, .03, .035])}
    by_x = {s:x for s,x in zip(star.supports, [.003, .01, .012, .014])}
    time = np.arange(t, dtype=float)
    p = .1+.015*np.sin(time[:,None]*.61+np.arange(3))
    q = .035+.008*np.cos(time[:,None]*.93+.7*np.arange(3))
    sp, sq = .08+.01*np.sin(time*.53), .025+.004*np.cos(time*.43)
    hm = MeasurementModel(p_sigma=.0006, q_sigma=.0004, v_sigma=.00012,
        master_p_sigma=.001, master_q_sigma=.001)
    precision = np.broadcast_to(np.array([1., 9., 9., 1.])[:,None], (horizon, 3))
    fm = MeasurementModel(p_sigma=.0006/np.sqrt(5), q_sigma=.0004/np.sqrt(5),
        v_sigma=.00012/np.sqrt(precision),master_p_sigma=.001/np.sqrt(5),
        master_q_sigma=.001/np.sqrt(5))
    physical = ac_predict(star, list(by_r.values()), list(by_x.values()), p, q, sp, sq, 1.)
    history = Observations(p=p, q=q, v=physical["v"], root_v=1.,
        master_p=physical["master_p"], master_q=physical["master_q"])
    forecast = FutureForecast(np.tile(p.mean(0), (horizon,1)),
        np.tile(q.mean(0), (horizon,1)), np.full(horizon, sp.mean()),
        np.full(horizon, sq.mean()), np.ones(horizon))
    centers=[]
    for pair in ((0,1), (0,2), (1,2)):
        candidate=RadialCandidate.from_supports("clade_"+"".join(map(str,pair)),3,(pair,))
        centers.append(InformationCenter(candidate.name,"anchor0",pair,candidate,
            np.array([by_r.get(s,0.) for s in candidate.supports]),
            np.array([by_x.get(s,0.) for s in candidate.supports]),
            p,q,sp,sq,np.ones(t),forecast))
    return history,hm,fm,centers


@pytest.fixture
def evaluator():
    history,hm,fm,centers=inputs()
    return StructuredInformationEvaluator(history,centers,hm,future_model=fm)


def test_free_projection_matches_independent_lstsq_with_rank_deficiency():
    a=np.array([[1.,2.],[3.,-1.],[1.,.2],[4.,5.]])
    b=np.array([[1.,2.],[0.,0.],[2.,4.],[-1.,-2.]])
    out=project_columns(a,b)
    independent=a-b@np.linalg.lstsq(b,a,rcond=None)[0]
    np.testing.assert_allclose(out["projected"],independent,atol=2e-14)
    assert out["rank"]==1
    assert out["nuisance_orthogonality_error"]<1e-13


def test_empty_nuisance_and_unweighted_score_have_known_values():
    line=np.diag([2.,3.])
    out=efficient_information(line,(0,1),weight=np.eye(2))
    np.testing.assert_array_equal(out["information"],np.diag([4.,9.]))
    assert out["score"]==6.5
    assert out["rank"]==0


def test_explicit_near_cutoff_failure_and_exact_zero_rank_allowed():
    b=np.array([[1.,1.],[0.,.002]])
    with pytest.raises(InformationRankError,match="cutoff"):
        project_columns(np.ones((2,2)),b,rcond=.001)
    out=project_columns(np.eye(2),np.zeros((2,2)))
    assert out["rank"]==0
    np.testing.assert_array_equal(out["projected"],np.eye(2))


def test_two_stage_equals_direct_full_nuisance_and_future_gain_is_psd():
    rng=np.random.default_rng(714)
    jh,wh=rng.normal(size=(15,6)),rng.normal(size=(15,4))
    jn,wn=rng.normal(size=(11,6)),rng.normal(size=(11,3))
    out=information_from_blocks(jh,wh,jn,wn,(1,4))
    target=np.vstack((jh[:,(1,4)],jn[:,(1,4)]))
    other=(0,2,3,5)
    nuisance=np.column_stack((np.vstack((jh[:,other],jn[:,other])),
        np.vstack((wh,np.zeros((len(jn),wh.shape[1])))),
        np.vstack((np.zeros((len(jh),wn.shape[1])),wn))))
    direct=target-nuisance@np.linalg.lstsq(nuisance,target,rcond=None)[0]
    expected=direct.T@direct
    np.testing.assert_allclose(out["joint_information"]["information"],expected,rtol=2e-13,atol=1e-13)
    h=efficient_information(out["history_projection"]["projected"],(1,4))["information"]
    assert np.linalg.eigvalsh(expected-h).min()>-1e-10


def test_weight_scaling_and_signal_coordinate_change_are_consistent():
    rng=np.random.default_rng(117)
    line=rng.normal(size=(19,5))
    weight=np.array([[.7,.1],[.1,.4]])
    out=efficient_information(line,(1,3),weight=weight)
    doubled=efficient_information(line,(1,3),weight=2*weight)
    assert doubled["score"]==pytest.approx(2*out["score"])
    scale=np.diag([.01,.03])
    altered=line.copy();altered[:,(1,3)]=line[:,(1,3)]@scale
    scaled=efficient_information(altered,(1,3),weight=np.linalg.inv(scale)@weight@np.linalg.inv(scale))
    assert scaled["score"]==pytest.approx(out["score"],rel=1e-12)


def test_center_arrays_are_frozen_and_incomplete_anchor_is_rejected():
    history,hm,fm,centers=inputs()
    with pytest.raises(ValueError):centers[0].r[0]=.02
    with pytest.raises(ValueError,match="all three"):
        StructuredInformationEvaluator(history,centers[:2],hm,future_model=fm)
    with pytest.raises(ValueError,match="weight"):
        StructuredInformationEvaluator(history,centers,hm,future_model=fm,weight=np.diag([1.,-1.]))


def test_collapsed_supermodels_preserve_star_meter_means(evaluator):
    expected=np.concatenate([getattr(evaluator.history,name).ravel()/np.broadcast_to(
        np.asarray(getattr(evaluator.history_model,name+"_sigma")),getattr(evaluator.history,name).shape).ravel()
        for name in evaluator.channels])
    for center in evaluator.centers:
        ctx=evaluator._contexts[center.id]
        np.testing.assert_allclose(ctx["history_geometry"]["mean"],expected,rtol=2e-13,atol=2e-10)
        assert center.r[ctx["target"][0]]==0.
        assert center.x[ctx["target"][1]-center.candidate.m]==0.


def test_no_ev_increment_zero_and_complete_audit_fields(evaluator):
    out=evaluator.evaluate(0,np.zeros(4),with_gradient=False)
    assert out["success"] and out["score"]==0.
    assert len(out["targets"])==3 and len(out["pairs"])==1
    assert out["fit_calls"]==0 and out["total_ac_calls"]==3
    for target in out["targets"]:
        np.testing.assert_array_equal(target["information_EV_delta"],np.zeros((2,2)))
        assert target["active_boundaries"]["target_geometry_lower_boundary"]
        assert target["future_gain_min_eigenvalue"]>-1e-5
    assert not out["target_weights_are_topology_probabilities"]
    assert out["full_joint_AC_refit_required_for_final_evaluation"]


@pytest.mark.parametrize("step",[.02,.005])
def test_full_ac_information_price_matches_independent_reprojection_fd(evaluator,step):
    power=np.array([1.,4.,5.,2.]);direction=np.array([-.3,.6,-.1,-.2])
    evaluated=evaluator.evaluate(0,power)
    plus=evaluator.evaluate(0,power+step*direction,with_gradient=False)
    minus=evaluator.evaluate(0,power-step*direction,with_gradient=False)
    assert evaluated["success"] and plus["success"] and minus["success"]
    analytic=np.dot(evaluated["pairs"][0]["gradient_kw"],direction)
    numerical=(plus["score"]-minus["score"])/(2*step)
    assert abs(analytic)>1e-4
    assert analytic==pytest.approx(numerical,rel=5e-5,abs=2e-7)
    for target,gp,gm in zip(evaluated["targets"],plus["targets"],minus["targets"]):
        price=np.dot(target["gradient_kw"],direction)
        finite=(gp["score"]-gm["score"])/(2*step)
        assert price==pytest.approx(finite,rel=8e-5,abs=2e-7)


def test_full_ac_two_stage_information_matches_one_shot_joint_projection(evaluator):
    power=np.array([1.,4.,5.,2.]);center=evaluator.centers[1]
    point=evaluator._point(center,power,1);ctx=evaluator._contexts[center.id]
    h=ctx["history_geometry"];n=point["geometry"]
    th,tn=h["line_jacobian"],n["line_jacobian"]
    target=np.vstack((th[:,ctx["target"]],tn[:,ctx["target"]]))
    nuisance=np.column_stack((np.vstack((th[:,ctx["other"]],tn[:,ctx["other"]])),
        np.vstack((h["state_jacobian"],np.zeros((len(tn),h["state_jacobian"].shape[1])))),
        np.vstack((np.zeros((len(th),n["state_jacobian"].shape[1])),n["state_jacobian"]))))
    residual=target-nuisance@np.linalg.lstsq(nuisance,target,rcond=None)[0]
    np.testing.assert_allclose(point["information"]["information"],residual.T@residual,rtol=2e-9,atol=1e-3)


def test_required_physical_failure_keeps_all_failed_targets(evaluator):
    out=evaluator.evaluate(0,np.full(4,40.))
    assert not out["success"] and out["score"] is None and not out["pairs"]
    assert len(out["failures"])==3
    assert out["fit_calls"]==0


def test_rank_switch_is_explicit_failed_curve_not_reduced_family(evaluator,monkeypatch):
    original=evaluator._point
    def changed(center,power,site):
        result=original(center,power,site)
        if np.asarray(power)[0]>2.:
            result["rank_signature"]=(result["rank_signature"][0],result["rank_signature"][1]+1)
        return result
    monkeypatch.setattr(evaluator,"_point",changed)
    out=evaluator.evaluate(0,[2.,3.,4.,3.])
    assert not out["success"] and not out["pairs"]
    assert len(out["failures"])==3
    assert {f["kind"] for f in out["failures"]}=={"InformationRankError"}


def test_multiple_anchors_minimum_and_prices_use_equal_target_means(evaluator):
    h,hm,fm,cs=inputs()
    others=[replace(c,id=c.id+"_b",anchor_id="anchor1",r=c.r*1.03,x=c.x*.98) for c in cs]
    multi=StructuredInformationEvaluator(h,cs+others,hm,future_model=fm)
    out=multi.evaluate(1,[1.,4.,5.,2.])
    assert out["success"] and len(out["pairs"])==2
    assert out["score"]==min(p["score"] for p in out["pairs"])
    for pair in out["pairs"]:
        targets=[t for t in out["targets"] if t["anchor_id"]==pair["anchor_id"]]
        assert pair["score"]==pytest.approx(np.mean([t["score"] for t in targets]))
        np.testing.assert_allclose(pair["gradient_kw"],np.mean([t["gradient_kw"] for t in targets],axis=0))
        assert pair["pq_only_gradient_kw"]==pair["gradient_kw"]


def test_selector_interface_contains_no_truth_or_future_noise(evaluator):
    assert not {"truth","future_noise","future_observations"}&set(signature(StructuredInformationEvaluator).parameters)
    from .gradient_price_search import optimize_price_curves, SearchSettings
    from .single_site_milp import SingleSiteConstraints
    constraints=SingleSiteConstraints(dt_hours=.25,energy=.15,p_charge_max=.3,
        p_discharge_max=0.,availability=np.ones((4,3),bool),initial_site=0)
    out=optimize_price_curves(evaluator,constraints,sites=[0],
        settings=SearchSettings(max_iterations=1,max_starts=1))
    assert out["success"] and out["fit_calls"]==0
    assert out["total_ac_calls"]>0
    assert not out["global_design_optimality_certified"]

def test_late_initialization_failure_preserves_actual_ac_attempt_denominator(monkeypatch):
    from . import structured_information as module
    history,hm,fm,centers=inputs()
    original=module.mean_and_jacobian
    calls=[]
    def fail_last(*args,**kwargs):
        calls.append(args[0].name)
        if len(calls)==6:
            raise RuntimeError("last target no-EV AC failed")
        return original(*args,**kwargs)
    monkeypatch.setattr(module,"mean_and_jacobian",fail_last)
    with pytest.raises(InformationInitializationError) as failure:
        StructuredInformationEvaluator(history,centers,hm,future_model=fm)
    error=failure.value
    assert len(calls)==error.ac_calls==error.initialization_ac_calls==6
    assert error.failed_center_id==centers[-1].id
    assert error.cause_kind=="RuntimeError"
    assert error.initialization_seconds>0.
    assert error.ac_call_count_available
    record=error.record()
    assert record["initialization_ac_calls"]==6
    assert record["ac_call_count_available"]
