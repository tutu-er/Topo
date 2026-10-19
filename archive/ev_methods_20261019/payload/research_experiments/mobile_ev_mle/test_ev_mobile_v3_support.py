"""Five small algebra/MILP audits; no AC or parameter fitting."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from . import ev_mobile_v3_projection as proxy
from . import ev_mobile_v3_support as support


def pair(a,b,offset=1.,coefficient=1.,minus=0.):
    return proxy.build_projected_pair(a,b,[offset],[[coefficient]],np.zeros((1,0)),
                                      np.zeros((1,1)),const_minus=minus)


def box(lower=-1.,upper=1.):
    return dict(lower=[lower],upper=[upper],integrality=[0],constraints=[],
                u_map=[[1.]],control_lower=[lower],control_upper=[upper])


def test_all_pairs_signed_eta_and_lower_objective_bound_scope():
    pairs=[pair(a,b,minus=2.) for a in ('a','b','c') for b in ('a','b','c') if a!=b]
    out=support.solve_support_start(pairs,box(),[0.],backend='scipy',solver_seconds=1.)
    assert out['success'] and out['all_pairs_count']==6 and out['n_reference_eta']==3
    assert out['parent_exact_score']==pytest.approx(-1.5)
    assert out['support_score']==pytest.approx(-.5)
    assert out['exact_score']==pytest.approx(0.) and out['inheritance_protected']
    assert out['power_kw'][0,0]==pytest.approx(1.)
    # Its optimum -0.5 is below max Q=0: this dual cannot upper-bound max Q.
    assert out['solver']['support_best_bound']==pytest.approx(-.5)
    assert 'not a quadratic-design upper bound' in out['solver']['dual_scope']
    with pytest.raises(ValueError,match='all K'):
        support.solve_support_start(pairs[:-1],box(),[0.],backend='scipy')


def test_full_tangent_identity_not_svd_approximated_and_parent_equality():
    pairs=[pair(a,b,offset=-.8,coefficient=1.3,minus=2.) for a,b in [('a','b'),('b','a')]]
    parent=np.array([[.2]])
    rows=support._tangents(pairs,parent)
    for p,r in zip(pairs,rows):
        for x in np.linspace(-1.,1.,17):
            ell=r['intercept']+r['gradient']@np.array([x])
            q=proxy.evaluate_pair(p,[[x]])
            assert q-ell==pytest.approx(.5*(1.3*(x-.2))**2,abs=1e-12)
        assert r['intercept']+r['gradient']@parent.ravel()==pytest.approx(proxy.evaluate_pair(p,parent))


def test_nonidentity_u_map_keeps_integer_physical_coordinate_and_warm_start():
    domain=dict(lower=[0.,-1.],upper=[1.,1.],integrality=[1,0],u_map=[[0.,2.]],
        control_lower=[-2.],control_upper=[2.],constraints=[LinearConstraint([[1.,0.]],1.,1.)])
    pairs=[pair(a,b) for a,b in [('a','b'),('b','a')]]
    out=support.solve_support_start(pairs,domain,[1.,0.],backend='scipy',solver_seconds=1.)
    assert out['success'] and out['master_start_audit']['feasible']
    np.testing.assert_allclose(out['operation_x'],[1.,1.])
    np.testing.assert_allclose(out['power_kw'],[[2.]])
    assert out['exact_score']==pytest.approx(4.5)
    assert out['support_score']==pytest.approx(2.5)
    assert out['solver']['start_supplied'] and not out['solver']['start_used']


def test_physical_failure_is_retained_and_parent_is_returned(monkeypatch):
    pairs=[pair(a,b) for a,b in [('a','b'),('b','a')]]
    monkeypatch.setattr(support,'solve_linear_mip',lambda *a,**k:
        SimpleNamespace(x=np.array([1.01,10.,10.]),status=1,message='mock bad physical incumbent',
                        mip_dual_bound=-100.,fun=-10.))
    out=support.solve_support_start(pairs,box(),[.2],backend='scipy')
    assert out['success'] and out['selected_origin']=='parent_operation'
    assert out['exact_score']==pytest.approx(.72) and out['inheritance_protected']
    assert out['failures'][0]['kind']=='support_physical_violation'
    assert not out['solver']['operation_incumbent_audit']['feasible']
    np.testing.assert_array_equal(out['operation_x'],[.2])
    invalid=support.solve_support_start(pairs,box(),[2.],backend='scipy')
    assert not invalid['success'] and invalid['MILP_calls']==0


def test_zero_gradient_keeps_parent_without_claiming_the_quadratic_optimum(monkeypatch):
    pairs=[pair(a,b,offset=0.) for a,b in [('a','b'),('b','a')]]
    monkeypatch.setattr(support,'solve_linear_mip',lambda *a,**k:pytest.fail('zero-gradient MILP should be skipped'))
    out=support.solve_support_start(pairs,box(),[0.],backend='scipy')
    assert out['success'] and out['MILP_calls']==0 and out['exact_score']==0.
    assert 'upper_PWL_exploration_still_needed' in out['reason']
    assert proxy.aggregate_pairs(pairs,[[1.]])['score']==pytest.approx(.5)
