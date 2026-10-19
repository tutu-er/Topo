"""Cone coverage, one-sided kinks, full-AC chain rule and fixed-line contracts."""
from dataclasses import replace

import numpy as np
import pytest

from . import likelihood as lk
from .ev_fixed_train_fit import _FixedInputProblem, fit_fixed_inputs, subset_observations, subset_model
from .ev_fixed_train_review import review_fixed_fit
from .gradient_price_study import fit_record, serial
from .ev_rx_fit import DEFAULT_RATIO_BOUNDS, _RxMap, _RxInputProblem, fit_rx_inputs


BOUNDS = dict(r_bounds=(.00065,.0120375),x_bounds=(.00025,.00155625))


def _case(t=12,seed=401):
    c=lk.RadialCandidate.from_supports("pair01",3,[(0,1)])
    rng=np.random.default_rng(seed)
    r=np.linspace(.002,.004,c.m);x=r*np.linspace(.2,.3,c.m)
    p=rng.uniform(.025,.17,(t,3));q=rng.uniform(.005,.1,(t,3))
    sp=rng.uniform(.02,.12,t);sq=rng.uniform(-.04,.05,t);root=rng.uniform(.996,1.004,t)
    model=lk.MeasurementModel(.0006,.0004,.00012,.001,.001,
        bias_p=np.array([.001,-.0008,.0005]),bias_q=np.array([-.0003,.0006,.0002]),
        bias_v=np.array([.0002,-.0001,.0003]))
    phy=lk.ac_predict(c,r,x,p,q,sp,sq,root)
    obs=lk.Observations(p+model.bias_p,q+model.bias_q,phy["v"]+model.bias_v,
                        root,phy["master_p"],phy["master_q"])
    return c,r,x,sp,sq,obs,model


def test_exact_mapping_covers_original_box_intersect_cone_and_corners():
    m=5;mapping=_RxMap(m,**BOUNDS);rng=np.random.default_rng(40)
    np.testing.assert_array_equal(mapping.lower,np.full(m,BOUNDS["r_bounds"][0]))
    np.testing.assert_allclose(mapping.upper,np.full(m,BOUNDS["r_bounds"][1]),atol=3e-18,rtol=0)
    for _ in range(30):
        r=rng.uniform(mapping.lower,mapping.upper);a=rng.uniform(0,1,m)
        r,x,_,_=mapping.forward(r,a)
        assert mapping.contains(r,x)
        rr,aa=mapping.inverse(r,x);_,xx,_,_=mapping.forward(rr,aa)
        np.testing.assert_allclose(rr,r,atol=1e-17,rtol=0)
        np.testing.assert_allclose(xx,x,atol=1e-17,rtol=0)
    for r in (mapping.lower,mapping.upper):
        for a in (np.zeros(m),np.ones(m)):
            rr,x,_,_=mapping.forward(r,a)
            assert mapping.contains(rr,x)
            _,_,width,_,_=mapping.section(r)
            assert np.max(width)<5e-18


def test_initial_projection_and_empty_or_invalid_prior_domains():
    mapping=_RxMap(2,**BOUNDS)
    r,a=mapping.inverse([1,-1],[1,-1]);_,x,_,_=mapping.forward(r,a)
    assert mapping.contains(r,x)
    for rho in ((0,.4),(.5,.2),(.2,np.inf),(.2,.2)):
        with pytest.raises(ValueError,match="ratio_bounds"):
            _RxMap(2,**BOUNDS,ratio_bounds=rho)
    with pytest.raises(ValueError,match="empty intersection"):
        _RxMap(2,(.001,.002),(.009,.01))


@pytest.mark.parametrize("which",["lower","upper"])
def test_mapping_switch_uses_documented_right_derivative(which):
    mapping=_RxMap(1,**BOUNDS)
    r=(mapping.low_switch if which=="lower" else mapping.high_switch).copy();a=np.array([.37])
    _,x,derivative,_=mapping.forward(r,a);step=1e-8
    right=(mapping.forward(r+step,a)[1]-x)/step
    left=(x-mapping.forward(r-step,a)[1])/step
    np.testing.assert_allclose(derivative,right,atol=1e-11,rtol=1e-8)
    assert np.max(abs(left-right))>.01


def test_full_ac_chainrule_matches_centered_differences_away_from_kinks():
    c,r,x,sp,sq,obs,model=_case(t=2)
    problem=_RxInputProblem(c,obs,model,initial=dict(r=r,x=x,source_p=sp,source_q=sq),**BOUNDS)
    point=problem.start.copy();analytic=problem.jacobian(point);numerical=np.zeros_like(analytic)
    for j in range(len(point)):
        h=2e-5 if c.m<=j<2*c.m else 2e-7
        lo=point.copy();hi=point.copy();lo[j]-=h;hi[j]+=h
        numerical[:,j]=(problem.evaluate(hi)-problem.evaluate(lo))/(2*h)
    np.testing.assert_allclose(analytic,numerical,atol=1e-4,rtol=3e-6)
    assert np.count_nonzero(analytic[5:,2*c.m:2*c.m+2])==0
    pvec=problem.physical_vector(point)
    old=_FixedInputProblem(c,obs,model,initial=dict(r=r,x=x,source_p=sp,source_q=sq),**BOUNDS)
    np.testing.assert_array_equal(problem.physical_jacobian(point),old.jacobian(pvec))


def test_full_ac_kink_jacobian_matches_right_difference():
    c,r,x,sp,sq,obs,model=_case(t=1)
    problem=_RxInputProblem(c,obs,model,initial=dict(r=r,x=x,source_p=sp,source_q=sq),**BOUNDS)
    for switch in (problem.mapping.low_switch,problem.mapping.high_switch):
        point=problem.start.copy();point[0]=switch[0]
        h=1e-8;right=point.copy();right[0]+=h
        numeric=(problem.evaluate(right)-problem.evaluate(point))/h
        analytic=problem.jacobian(point)[:,0]
        np.testing.assert_allclose(analytic,numeric,atol=1e-3,rtol=2e-5)


def test_noiseless_training_returns_one_physical_group_and_common_information():
    c,r,x,sp,sq,obs,model=_case()
    counter={}
    fit=fit_rx_inputs(c,obs,model,initial=dict(r=.9*r,x=.9*x,source_p=sp,source_q=sq),
        multistart=2,max_nfev=250,call_counter=counter,seed=42,**BOUNDS)
    assert fit.success,fit.diagnostics
    assert fit.loss<1e-11
    np.testing.assert_allclose(fit.r,r,atol=2e-7,rtol=0)
    np.testing.assert_allclose(fit.x,x,atol=2e-7,rtol=0)
    np.testing.assert_array_equal(fit.p,obs.p-model.bias_p)
    assert fit.diagnostics["ratio_constraints"]["feasible"]
    assert fit.diagnostics["joint_parameter_count"]==2*c.m+2*obs.t
    assert fit.diagnostics["optimizer_parameter_count"]==2*c.m+2*obs.t
    assert not fit.diagnostics["global_optimality_certified"]
    assert counter["fit_calls"]==1 and counter["solver_calls"]==2
    assert counter["ac_calls"]==fit.diagnostics["ac_evaluations"]
    row=serial(fit_record(c,fit))
    assert review_fixed_fit(c,obs,model,row)["success"]
    # Recompute the profile spectrum at the attained state in PHYSICAL coords.
    old=_FixedInputProblem(c,obs,model,initial=fit,**BOUNDS)
    vector=np.r_[fit.r,fit.x,np.c_[fit.source_p,fit.source_q].ravel()]
    jac=old.jacobian(vector);shared=jac[:,:2*c.m]*old.scale[:2*c.m][None,:]
    source=jac[:,2*c.m:];profile=shared-source@np.linalg.lstsq(source,shared,rcond=1e-10)[0]
    np.testing.assert_allclose(fit.diagnostics["common_physical_scale"],old.scale,atol=0,rtol=0)
    np.testing.assert_allclose(fit.diagnostics["local_projected_shared_singular_values"],
        np.linalg.svd(profile,compute_uv=False),atol=2e-12,rtol=1e-9)
    np.testing.assert_allclose(fit.diagnostics["physical_shared_whitened_jacobian"],jac[:,:2*c.m])
    assert fit.diagnostics["information_coordinate_system"].startswith("physical_R_then_X")


def test_future_fixedlines_delegate_exactly_and_ev_added_once():
    c,r,x,sp,sq,obs,model=_case(t=5)
    future_p=obs.p-model.bias_p;future_p[:,2]+=np.array([.02,-.01,.03,0,.01])
    phy=lk.ac_predict(c,r,x,future_p,obs.q-model.bias_q,sp+.01,sq-.015,obs.root_v)
    ev=replace(obs,p=future_p+model.bias_p,v=phy["v"]+model.bias_v,
               master_p=phy["master_p"],master_q=phy["master_q"])
    kw=dict(initial=dict(r=r,x=x,source_p=sp,source_q=sq),fixed_r=r,fixed_x=x,
            multistart=2,seed=31,**BOUNDS)
    fit=fit_rx_inputs(c,ev,model,**kw);old=fit_fixed_inputs(c,ev,model,**kw)
    assert fit.success and fit.loss<1e-12
    np.testing.assert_array_equal(fit.r,r);np.testing.assert_array_equal(fit.x,x)
    np.testing.assert_array_equal(fit.p,ev.p-model.bias_p)
    np.testing.assert_array_equal(fit.source_p,old.source_p)
    np.testing.assert_array_equal(fit.source_q,old.source_q)
    assert fit.loss==old.loss and fit.loglik==old.loglik
    assert fit.diagnostics["shared_parameter_count"]==0
    assert fit.diagnostics["source_node"]==1
    with pytest.raises(ValueError,match="ratio cone"):
        fit_rx_inputs(c,ev,model,fixed_r=r,fixed_x=np.full(c.m,.0015),**BOUNDS)


def test_failed_numerical_and_iteration_limited_restarts_remain_visible(monkeypatch):
    c,r,x,sp,sq,obs,model=_case(t=3)
    counter={}
    failed=fit_rx_inputs(c,obs,model,multistart=2,max_nfev=1,call_counter=counter,**BOUNDS)
    assert not failed.success and np.isfinite(failed.loss)
    assert all("attained_state" in s and s["status"]==0 for s in failed.diagnostics["restarts"])
    assert counter["fit_calls"]==1 and counter["solver_calls"]==2
    np.testing.assert_array_equal(failed.p,obs.p-model.bias_p)
    def broken(*args,**kwargs):raise RuntimeError("deliberate AC failure")
    monkeypatch.setattr(lk,"ac_predict",broken)
    total=fit_rx_inputs(c,obs,model,multistart=2,**BOUNDS)
    assert not total.success and total.status==-99
    assert total.diagnostics["failure_kind"]=="computation"
    assert len(total.diagnostics["restarts"])==2
    assert all("deliberate AC failure" in s["message"] for s in total.diagnostics["restarts"])


def test_collapsed_cone_contact_is_fixed_without_equal_solver_bounds():
    c,_,_,sp,sq,obs,model=_case(t=3)
    rho=DEFAULT_RATIO_BOUNDS[1];r=np.full(c.m,.002);x=r*rho
    phy=lk.ac_predict(c,r,x,obs.p-model.bias_p,obs.q-model.bias_q,sp,sq,obs.root_v)
    contact=replace(obs,v=phy["v"]+model.bias_v,master_p=phy["master_p"],master_q=phy["master_q"])
    fit=fit_rx_inputs(c,contact,model,r_bounds=(.001,.002),x_bounds=(rho*.002,.0011),
        multistart=1,max_nfev=150)
    assert fit.success and fit.loss<1e-12
    np.testing.assert_array_equal(fit.r,r);np.testing.assert_allclose(fit.x,x,atol=1e-18)
    assert fit.diagnostics["optimizer_shared_parameter_count"]==0
    assert fit.diagnostics["optimizer_parameter_count"]==2*obs.t


def test_input_precision_remains_documentary_and_source_or_bias_contracts_rejected():
    c,r,x,sp,sq,obs,model=_case(t=3)
    kw=dict(fixed_r=r,fixed_x=x,multistart=1,**BOUNDS)
    fit=fit_rx_inputs(c,obs,model,**kw)
    same=fit_rx_inputs(c,obs,replace(model,p_sigma=1,q_sigma=2),**kw)
    assert same.loss==fit.loss and same.loglik==fit.loglik
    with pytest.raises(ValueError,match="supplied together"):
        fit_rx_inputs(c,obs,model,fixed_r=r,**BOUNDS)
    with pytest.raises(ValueError,match="known calibrated"):
        fit_rx_inputs(c,obs,replace(model,fit_bias=True),**BOUNDS)
    with pytest.raises(ValueError,match="separate source telemetry"):
        fit_rx_inputs(c,replace(obs,source_p=sp,source_q=sq),model,**BOUNDS)


def test_scalar_compensation_has_equal_history_but_different_ev_response():
    kappa=.25;r=.006;x=.001;delta_x=.0004;delta_r=-kappa*delta_x
    P=np.array([.08,.12,.16]);Q=kappa*P
    history=r*P+x*Q;other=(r+delta_r)*P+(x+delta_x)*Q
    np.testing.assert_allclose(history,other,atol=1e-18,rtol=0)
    for u in (-.2,.1,.3):
        increment=((r+delta_r)*u-r*u)
        assert increment==pytest.approx(delta_r*u,abs=1e-18)
        assert increment==pytest.approx(-kappa*delta_x*u,abs=1e-18)
    lo,hi=DEFAULT_RATIO_BOUNDS;combination=r+kappa*x
    assert combination/(1+kappa*hi)<=r<=combination/(1+kappa*lo)
