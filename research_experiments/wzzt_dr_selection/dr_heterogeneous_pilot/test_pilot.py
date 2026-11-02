import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,sys
import numpy as np
import pytest
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).parent))
from devices import Devices
from planner import Probability
from research_experiments.wzzt_dr_selection.dr_heterogeneous_resources.build_allocations import profiles


def dev():
    folder=REPO/'research_experiments/wzzt_dr_selection/dr_heterogeneous_resources'
    cfg=json.loads((folder/'library.json').read_text(encoding='utf8'))
    case=next(c for c in json.loads((folder/'case_allocations.json').read_text(encoding='utf8'))['cases'] if c['case']=='paper15')
    case=dict(case,nodes=case['nodes'][:2])
    return Devices(case,cfg,profiles)


def test_zero_keeps_original_background():
    d=dev();u=d.nominal(np.zeros((d.R,24)))
    assert np.array_equal(u,np.zeros_like(u))
    assert d.fees(u)==0
    assert d.audit(u)['passed']


def test_large_requests_preserve_power_deadline_energy():
    d=dev();rng=np.random.default_rng(19)
    requested=rng.normal(size=(40,d.R,96))*10
    delivered=d.project(requested)
    for u in delivered:assert d.audit(u)['passed']


def test_projection_is_causal():
    d=dev();rng=np.random.default_rng(20);u=rng.normal(size=(d.R,96))*.1
    v=u.copy();v[:,60:]+=10
    np.testing.assert_array_equal(d.project(u)[:,:60],d.project(v)[:,:60])


def test_battery_terminal_uses_lossy_recharge():
    d=dev();raw=np.zeros((d.R,96));raw[d.kind==3,0]=-d.cap[d.kind==3]*.5
    delivered=d.project(raw)
    for r in np.flatnonzero(d.kind==3):
        assert delivered[r].sum()>0
        assert .25*np.sum(np.where(delivered[r]>=0,d.eta[r],1/d.eta[r])*delivered[r])==pytest.approx(0,abs=1e-12)


def test_EV_does_not_pay_back_after_disconnection():
    d=dev();raw=np.zeros((d.R,96));raw[d.kind==2,0]=-1.
    u=d.project(raw)
    np.testing.assert_allclose(u[d.kind==2,32:72],0,atol=1e-12)
    np.testing.assert_allclose(u[d.kind==2,:32].sum(1),0,atol=1e-12)
    assert d.audit(u)['passed']


def test_projection_vjp_regular_point():
    d=dev();rng=np.random.default_rng(29)
    u=rng.normal(size=(d.R,96))*.4;direction=rng.normal(size=u.shape);g=rng.normal(size=u.shape)
    value,cache=d.project(u,cache=True)
    analytic=np.sum(d.vjp(g,cache)*direction)
    eps=1e-7
    numeric=np.sum(g*(d.project(u+eps*direction)-d.project(u-eps*direction)))/(2*eps)
    assert analytic==pytest.approx(numeric,rel=2e-5,abs=2e-5)


def test_nominal_fee_gradient_and_resource_aggregation():
    d=dev();rng=np.random.default_rng(31);x=rng.normal(size=(d.R,24))*.04
    u,cache=d.nominal(x,cache=True);cost,g=d.fees(u,gradient=True)
    gx=d.hourly_vjp(g,cache);v=rng.normal(size=x.shape);eps=1e-7
    numerical=(d.fees(d.nominal(x+eps*v))-d.fees(d.nominal(x-eps*v)))/(2*eps)
    assert np.sum(gx*v)==pytest.approx(numerical,rel=2e-4,abs=1e-5)
    pp,qq=d.aggregate(u)
    assert pp.sum()==pytest.approx(u.sum())
    assert qq.sum()==pytest.approx((u*d.qratio[:,None]).sum())


def test_correlated_draws_reproducible_and_stress_changes_mean():
    d=dev();a=d.draw(62,3);b=d.draw(62,3)
    np.testing.assert_array_equal(a,b)
    assert a.shape==(3,d.R,96)
    assert not np.array_equal(a,d.draw(62,3,stress=True))


def test_conditional_probability_chain_gradient():
    d=dev();rng=np.random.default_rng(44);n=2;m=96*n
    packet=dict(mu_C=rng.normal(size=m),mu_F=rng.normal(size=m)*.8,
        L_C=rng.normal(size=(m,3)),L_F=rng.normal(size=(m,3))*.8,
        effect_C=rng.normal(size=(n,d.R))*.2,effect_F=rng.normal(size=(n,d.R))*.1,threshold=.25)
    prob=Probability(packet,d,4,22,23);x=rng.normal(size=(d.R,24))*.05
    out=prob.evaluate(x);assert not out['invalid']
    v=rng.normal(size=x.shape);eps=1e-6
    numerical=(prob.evaluate(x+eps*v)['probability']-prob.evaluate(x-eps*v)['probability'])/(2*eps)
    analytic=float(out['gradient']@v.ravel())
    assert analytic==pytest.approx(numerical,rel=5e-4,abs=3e-6)


def test_scaled_nominal_stays_feasible_and_matches_fee():
    d=dev();rng=np.random.default_rng(81);u=d.nominal(rng.normal(size=(d.R,24)))
    for a in [.1,.5,.9]:
        assert d.audit(a*u)['passed']
        assert d.fees(a*u)==pytest.approx(a*a*d.fees(u))
