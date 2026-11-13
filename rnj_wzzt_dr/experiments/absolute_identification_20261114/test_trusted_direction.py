"""Independent algebra/domain checks for the historical P/Q restriction."""
from copy import deepcopy
import numpy as np
from ...observations import Observation
from ..customer_flexibility_20261113.model import CustomerCase
from .trusted_direction import PQTrustedCase


def main():
    original=CustomerCase();config=deepcopy(original.config)
    pv=next(r for r in config['resources'] if r['kind']=='pv')
    config['expected_terminal_labels']=[1,2]
    config['resources']=[dict(deepcopy(pv),id=f'pv{i}',node_label=i,q_over_p=0.) for i in (1,2)]
    physical=CustomerCase(config)
    t=np.arange(64.)
    p=.01*np.c_[1+.2*np.sin(t),1+.3*np.cos(.7*t)]
    q=.5*p+1e-5*np.c_[np.sin(1.7*t),np.cos(1.9*t)]
    train=Observation(p,q,np.zeros_like(p),(1,2))
    one=PQTrustedCase(physical,train,base_kw=200.,scale=1.)
    two=PQTrustedCase(physical,train,base_kw=200.,scale=2.)
    zero=np.zeros(one.shape)
    assert one.audit(zero)['passed'] and two.audit(zero)['passed']
    assert abs(two.trust_limit-4*one.trust_limit)<1e-15
    assert np.linalg.eigvalsh(one.trust_Q).min()>=-1e-12
    h=int(np.argmax(physical.upper[:,0]))
    boundary=np.sqrt(one.trust_limit/one.trust_Q[0,0])
    assert boundary*1.1<physical.upper[h,0]
    good=zero.copy();good[h,0]=.9*boundary
    bad=zero.copy();bad[h,0]=1.1*boundary
    assert physical.audit(good)['passed'] and physical.audit(bad)['passed']
    assert one.audit(good)['passed'] and not one.audit(bad)['passed']
    assert two.audit(bad)['passed']
    assert one.fee(good)==physical.fee(good)
    measured=one.shift_pq_pu(good,200.)
    direct=200.**2*np.sum((measured[:,2:]-measured[:,:2]*one.kappa)**2)/one.repeat
    assert abs(direct-one.trust_audit(good)['energy'])<1e-12
    expected=24*200.**2*np.mean((q-p*one.kappa)**2,axis=0).sum()
    assert abs(one.trust_limit-expected)<1e-15
    contraction=min(1.,np.sqrt(one.trust_limit/one.trust_audit(bad)['energy']))*(1-1e-9)
    assert one.audit(contraction*bad)['passed']
    # Fix known points in an independent solver model to check attach(), not
    # just the direct evaluator. There is no physical simulation or case fit.
    import gurobipy as gp
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    try:
        for point,expected_status in ((good,2),(bad,3)):
            with gp.Model(env=env) as model:
                model.Params.OutputFlag=0;model.Params.Threads=1;model.Params.NonConvex=0
                model.Params.FeasibilityTol=1e-9;model.Params.DualReductions=0;model.Params.TimeLimit=2.
                u,fee=one.attach(model);u.LB=point;u.UB=point;model.setObjective(fee);model.optimize()
                assert model.Status==expected_status,(model.Status,expected_status)
    finally:
        env.dispose()
    assert 'planning_pq_trust' not in physical.config
    print('P/Q trust checks passed: direct-vs-quadratic identity, hourly units, nested scales, radial feasibility, frozen physical domain, and attached convex-QCQP feasible/infeasible points.')


if __name__=='__main__':main()
