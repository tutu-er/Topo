"""Independent physics, derivative, likelihood and identification checks."""
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import root

sys.path.insert(0, str(Path(__file__).parent))
from likelihood import (
    Calibration, MeasurementModel, Observations, RadialCandidate,
    _response_derivatives, ac_predict, concat_observations, fit_candidate,
)


def case(t=10, seed=12):
    candidate = RadialCandidate.from_supports("pair01", 3, [(0, 1)])
    rng = np.random.default_rng(seed)
    # Independent R and X deliberately violate a common X/R restriction.
    r = np.linspace(.008,.026,candidate.m)
    x = np.array([.006,.009,.012,.004,.018])
    p = rng.uniform(.025,.17,(t,3))
    q = rng.uniform(-.012,.085,(t,3))
    sp,sq = rng.uniform(.015,.08,t),rng.uniform(-.02,.03,t)
    root_v = rng.uniform(.995,1.006,t)
    prediction = ac_predict(candidate,r,x,p,q,sp,sq,root_v)
    obs = Observations(p,q,prediction["v"],root_v,
                       prediction["master_p"],prediction["master_q"])
    model = MeasurementModel(.0006,.0004,.00012,.001,.001)
    return candidate,r,x,sp,sq,obs,model,prediction


def independent_nodal(candidate,r,x,p,q,sp,sq,root_v):
    """Rectangular KCL from edge admittances, with no B/Zbus use."""
    m = candidate.m
    y = np.zeros((m+1,m+1),complex)
    for (u,v),z in zip(candidate.edges,r+1j*x):
        g = 1/z
        y[u,u]+=g;y[v,v]+=g;y[u,v]-=g;y[v,u]-=g
    load = np.zeros(m,complex)
    load[0]=sp+1j*sq
    load[list(candidate.terminal_edges)] += p+1j*q
    def residual(real):
        v = np.r_[root_v,real[:m]+1j*real[m:]]
        equation = (y@v)[1:]+np.conj(load/v[1:])
        return np.r_[equation.real,equation.imag]
    solution = root(residual,np.r_[np.full(m,root_v),np.zeros(m)],tol=1e-11)
    assert np.max(abs(residual(solution.x))) < 3e-10
    voltage = np.r_[root_v,solution.x[:m]+1j*solution.x[m:]]
    master = root_v*np.conj((y@voltage)[0])
    return voltage,master


def test_ac_matches_independent_nodal_and_power_balance():
    c,r,x,sp,sq,obs,model,pred=case(t=5)
    for t in range(obs.t):
        v,master=independent_nodal(c,r,x,obs.p[t],obs.q[t],sp[t],sq[t],obs.root_v[t])
        np.testing.assert_allclose(pred["voltage"][t],v[1:],atol=2e-11,rtol=0)
        np.testing.assert_allclose(pred["master_p"][t]+1j*pred["master_q"][t],
                                   master,atol=3e-11,rtol=0)
    assert pred["power_balance_residual"] < 1e-11


def test_analytic_ac_jacobian_matches_central_differences():
    c,r,x,sp,sq,obs,model,pred=case(t=2)
    derivatives=_response_derivatives(c,pred)
    m,n=c.m,c.n
    packed=np.r_[r,x,obs.p[0],obs.q[0],sp[0],sq[0]]
    def response(z):
        pp=ac_predict(c,z[:m],z[m:2*m],z[2*m:2*m+n],
                      z[2*m+n:2*m+2*n],z[-2],z[-1],obs.root_v[0])
        return np.r_[pp["v"][0],pp["master_p"][0],pp["master_q"][0]]
    numerical=np.empty((n+2,len(packed)))
    h=2e-6
    for j in range(len(packed)):
        zlo,zhi=packed.copy(),packed.copy()
        zlo[j]-=h;zhi[j]+=h
        numerical[:,j]=(response(zhi)-response(zlo))/(2*h)
    analytic=np.vstack([derivatives["v"][0],derivatives["master_p"][0],
                        derivatives["master_q"][0]])
    np.testing.assert_allclose(analytic,numerical,atol=2e-8,rtol=2e-6)


def test_noiseless_full_mle_recovers_free_r_x_and_source():
    c,r,x,sp,sq,obs,model,pred=case(t=12)
    result=fit_candidate(c,obs,model,multistart=2,max_nfev=180,seed=4)
    assert result.success, result.diagnostics
    assert result.loss < 1e-7
    np.testing.assert_allclose(result.r,r,atol=2e-5,rtol=0)
    np.testing.assert_allclose(result.x,x,atol=2e-5,rtol=0)
    np.testing.assert_allclose(result.source_p,sp,atol=2e-7,rtol=0)
    np.testing.assert_allclose(result.source_q,sq,atol=2e-7,rtol=0)
    assert result.diagnostics["local_identifiability_full_column_rank"]
    assert not result.diagnostics["global_optimality_certified"]
    assert result.diagnostics["power_balance_residual"]<1e-10
    normalization=obs.t*(3*np.log(2*np.pi*model.p_sigma**2)
                         +3*np.log(2*np.pi*model.q_sigma**2)
                         +3*np.log(2*np.pi*model.v_sigma**2)
                         +np.log(2*np.pi*model.master_p_sigma**2)
                         +np.log(2*np.pi*model.master_q_sigma**2))
    assert result.loglik==pytest.approx(-.5*(result.loss+normalization))


def test_source_and_bias_cannot_be_freed_without_anchor():
    c,r,x,sp,sq,obs,model,pred=case(t=3)
    terminal=Observations(obs.p,obs.q,obs.v,obs.root_v)
    terminal_model=MeasurementModel(model.p_sigma,model.q_sigma,model.v_sigma)
    with pytest.raises(ValueError,match="source P/Q requires"):
        fit_candidate(c,terminal,terminal_model)
    no_calibration=MeasurementModel(.0006,.0004,.00012,.001,.001,fit_bias=True)
    with pytest.raises(ValueError,match="independent calibration"):
        fit_candidate(c,obs,no_calibration)


def test_known_bias_correction_and_independent_joint_calibration():
    c,r,x,sp,sq,obs,model,pred=case(t=12)
    bp=np.array([.001,-.0008,.0005])
    bq=np.array([-.0003,.0006,.0002])
    bv=np.array([.0002,-.0001,.0003])
    shifted=Observations(obs.p+bp,obs.q+bq,obs.v+bv,obs.root_v,
                         obs.master_p,obs.master_q)
    corrected=MeasurementModel(.0006,.0004,.00012,.001,.001,
                               bias_p=bp,bias_q=bq,bias_v=bv)
    known=fit_candidate(c,shifted,corrected,initial=dict(r=r,x=x),multistart=1)
    assert known.success and known.loss<1e-7
    np.testing.assert_allclose(known.predicted["v"],shifted.v,atol=1e-7)
    calibration=Calibration(np.tile(bp,(3,1)),np.tile(bq,(3,1)),np.tile(bv,(3,1)),
                            .0001,.0001,.00002)
    jointly=MeasurementModel(.0006,.0004,.00012,.001,.001,
                             fit_bias=True,calibration=calibration)
    fitted=fit_candidate(c,shifted,jointly,initial=known,multistart=1)
    assert fitted.success and fitted.loss<1e-7
    np.testing.assert_allclose(fitted.bias_v,bv,atol=1e-8)
    assert fitted.diagnostics["local_identifiability_full_column_rank"]
    assert fitted.diagnostics["bias_policy"]=="independent_calibration_joint"


def test_concat_warm_start_and_failures_retained():
    c,r,x,sp,sq,obs,model,pred=case(t=12)
    first=Observations(obs.p[:6],obs.q[:6],obs.v[:6],obs.root_v[:6],
                       obs.master_p[:6],obs.master_q[:6])
    second=Observations(obs.p[6:],obs.q[6:],obs.v[6:],obs.root_v[6:],
                        obs.master_p[6:],obs.master_q[6:])
    short=fit_candidate(c,first,model,initial=dict(r=r,x=x),multistart=1,max_nfev=160)
    joint=fit_candidate(c,concat_observations(first,second),model,initial=short,
                        multistart=1,max_nfev=160)
    assert joint.p.shape==obs.p.shape
    assert joint.success and joint.loss<1e-7
    failed=fit_candidate(c,obs,model,multistart=1,max_nfev=1)
    assert not failed.success
    assert np.isfinite(failed.loss)
    assert failed.diagnostics["restarts"][0]["status"]==0
    assert failed.diagnostics["failure_kind"]=="computation"
