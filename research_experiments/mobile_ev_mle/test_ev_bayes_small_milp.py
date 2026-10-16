"""Independent algebra/grid/operation audits for the frozen BC MILP only."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from . import ev_bayes_small_milp as b
from .ev_mobile_service_domain import MobileServiceConfig,build_mobile_service_domain


def domain(lo,hi,*,integer=None,constraints=()):
    n=len(lo)
    return dict(lower=np.asarray(lo,float),upper=np.asarray(hi,float),u_map=np.eye(n),
        integrality=np.zeros(n,int) if integer is None else np.asarray(integer,int),
        control_lower=np.asarray(lo,float),control_upper=np.asarray(hi,float),constraints=constraints)


def bank(means,cov=None,prior=None,weights=None):
    means=np.asarray(means,float);L,K,d=means.shape
    return dict(means=means,covariances=np.broadcast_to(np.eye(d),(L,K,d,d)).copy() if cov is None else np.asarray(cov,float),
        prior=np.full(K,1/K) if prior is None else np.asarray(prior,float),
        weights=np.full(L,1/L) if weights is None else np.asarray(weights,float))


def test_scalar_covariance_and_nonzero_anchor_match_gaussian_formula():
    model=bank([[[1.],[3.]]],cov=[[[[1.]],[[4.]]]],prior=[.3,.7])
    J=np.asarray([[[[1.]],[[2.]]]]);anchor=np.array([[.5]]);u=np.array([[.8]])
    answer=b.frozen_bc_value(model,J,anchor,u)
    delta=(1.+.3)-(3.+.6);bar=2.5
    expected_B=delta**2/(8*bar)+.5*np.log(bar/2.)
    assert answer['pairs'][0]['B']==pytest.approx(expected_B)
    assert answer['C']==pytest.approx(np.sqrt(.21)*np.exp(-expected_B))


def test_uniform_proxy_upper_error_on_dense_covariance_grid():
    cov=np.asarray([[[[1.,.35],[.35,2.]],[[2.,-.2],[-.2,.8]],[[1.5,.1],[.1,1.2]]]])
    model=bank([[[.2,-.3],[.8,.5],[-.4,.9]]],cov=cov,prior=[.2,.3,.5])
    J=np.asarray([[[[1.,.3],[.2,1.]],[[.2,-.4],[.7,.1]],[[-.3,.6],[.8,-.2]]]])
    anchor=np.array([[.2,-.1]]);master=b.build_bc_master(domain([-1.,-1.],[1.,1.]),model,J,anchor,segments=3,exp_segments=8)
    assert not master['shared_control_square']['used']
    assert master['n_exp_binaries']==0
    for x in product(np.linspace(-1,1,9),repeat=2):
        value=b.evaluate_proxy(master,np.asarray([x]))
        direct=b.frozen_bc_value(model,J,anchor,np.asarray([x]))['C']
        assert value['C']==pytest.approx(direct,abs=2e-13)
        assert -2e-12<=value['actual_proxy_error']<=master['proxy_uniform_error']+2e-12
        for pair in value['pairs']:
            assert pair['B_lower']<=pair['B']+1e-12
            assert pair['B_chord']>=pair['B']-1e-12
            assert pair['B_chord']-pair['B']<=pair['epsilon_B']+1e-12


def test_zero_gradient_start_does_not_remove_quadratic_information():
    model=bank([[[0.],[0.]]]);J=np.asarray([[[[1.]], [[-1.]]]])
    result=b.solve_bc_milp(domain([-1.],[1.]),model,J,np.array([[0.]]),segments=4,seconds=5.,backend='scipy')
    assert result['success'] and result['physical_audit']['feasible']
    assert abs(result['power_kw'][0,0])==pytest.approx(1.,abs=1e-8)
    assert result['frozen_C']==pytest.approx(.5*np.exp(-.5),abs=1e-10)
    assert result['frozen_C']<result['frozen_C_anchor']
    assert result['n_exp_binaries']==0 and result['n_clip_binaries']==1
    assert result['solver_best_bound']<=result['proxy_value']+1e-8
    assert result['proxy_certificate'][0]<=result['frozen_C']<=result['proxy_certificate'][1]+1e-10


def test_two_slot_binary_independent_enumeration_audits_proxy_bound():
    op=domain([0.,0.],[1.,1.],integer=[1,1],constraints=[LinearConstraint([[1.,1.]],1.,1.)])
    model=bank([[[.2],[0.]]]);J=np.asarray([[[[2.,1.]],[[0.,0.]]]])
    anchor=np.zeros((2,1));master=b.build_bc_master(op,model,J,anchor,segments=2,exp_segments=8)
    possible=[np.array([[0.],[1.]]),np.array([[1.],[0.]])]
    values=[b.evaluate_proxy(master,u) for u in possible]
    result=b.solve_bc_milp(op,model,J,anchor,segments=2,exp_segments=8,seconds=5.,backend='scipy')
    assert result['success'] and result['proxy_value']==pytest.approx(min(v['proxy'] for v in values),abs=1e-9)
    assert result['solver_best_bound']==pytest.approx(min(v['proxy'] for v in values),abs=1e-9)
    assert result['frozen_C_optimum_lower']<=min(v['C'] for v in values)+1e-9
    np.testing.assert_allclose(result['power_kw'],[[1.],[0.]],atol=1e-9)


def test_unproved_zero_product_claim_never_drops_real_cross_term():
    op=domain([0.,0.],[1.,1.]);op['zero_product_pairs']=[(0,1)]
    model=bank([[[0.],[0.]]]);J=np.asarray([[[[1.,1.]],[[0.,0.]]]])
    master=b.build_bc_master(op,model,J,np.zeros((1,2)),segments=2)
    assert not master['shared_control_square']['used']
    assert 'zero_product_certificate_failed' in master['shared_control_square']['reason']
    exact=b.evaluate_proxy(master,[[1.,1.]])
    assert exact['pairs'][0]['B']==pytest.approx(.5)
    assert exact['C']==pytest.approx(.5*np.exp(-.5))


def test_full_mobile_operation_and_certified_shared_squares_are_preserved():
    cfg=MobileServiceConfig(n_slots=4,n_sites=2,dt_hours=.5,p_charge_max_kw=2.,p_discharge_max_kw=0.,
        allow_discharge=False,battery_capacity_kwh=10.,initial_soc=.4,terminal_energy_kwh=5.,
        eta_charge=1.,eta_discharge=1.,setup_slots=0,travel_slots=1,travel_energy_kwh=.1,
        terminal_site=1,mobile=True,max_moves=1)
    op=build_mobile_service_domain(cfg);means=np.zeros((1,2,4));J=np.zeros((1,2,4,8))
    for t in range(4):J[0,0,t,2*t:2*t+2]=[1.,2.]
    model=bank(means);result=b.solve_bc_milp(op,model,J,np.zeros((4,2)),segments=2,seconds=5.,backend='scipy')
    assert result['success'] and result['physical_audit']['feasible']
    assert result['shared_control_square']['used']
    assert result['shared_control_square']['zero_product_certificate']['valid']
    assert result['n_square_binaries']<=8*2 and result['n_exp_binaries']==0
    decoded=op.decode(result['operation_x'])
    assert decoded['energy_kwh'][-1]==pytest.approx(5.,abs=1e-7)
    assert sum(a.kind=='travel' for a in decoded['selected_arcs'])==1
    assert result['original_operation_constraints_preserved']


def test_power_trust_region_is_an_actual_constraint():
    model=bank([[[0.],[0.]]]);J=np.asarray([[[[1.]], [[-1.]]]])
    result=b.solve_bc_milp(domain([-1.],[1.]),model,J,[[.1]],power_radius=.2,seconds=5.,backend='scipy')
    assert result['success'] and -.1-1e-9<=result['power_kw'][0,0]<=.3+1e-9
    assert result['operation_audit']['feasible']


def test_no_incumbent_and_unsafe_solver_status_retain_failure(monkeypatch):
    monkeypatch.setattr(b,'solve_linear_mip',lambda *a,**k:SimpleNamespace(x=None,fun=None,status=4,
        message='deliberate failure',mip_dual_bound=-100.,mip_gap=None))
    model=bank([[[0.],[0.]]]);J=np.asarray([[[[1.]], [[-1.]]]])
    result=b.solve_bc_milp(domain([-1.],[1.]),model,J,[[0.]])
    assert not result['success'] and result['failure']=='no_solver_incumbent'
    assert result['operation_x'] is None and result['solver_best_bound'] is None
    assert not result['full_AC_optimality_claim']


def test_log_prior_prevents_direct_probability_product_underflow():
    model=bank(np.zeros((1,3,1)),prior=[1.,1e-200,1e-200]);model['log_prior']=np.log(model['prior'])
    value=b.frozen_bc_value(model,np.zeros((1,3,1,1)),[[0.]],[[0.]])
    rare=next(r for r in value['pairs'] if r['i']==1 and r['j']==2)
    assert rare['weight']>0 and rare['weight']==pytest.approx(1e-200,rel=1e-12,abs=0.)
    assert len(value['pairs'])==3


def test_inconsistent_rare_prior_and_log_prior_is_rejected():
    model=bank(np.zeros((1,3,1)),prior=[1.,1e-200,1e-200])
    model['log_prior']=np.log([1.,1e-13,1e-200])
    with pytest.raises(ValueError,match='disagree'):
        b.frozen_bc_value(model,np.zeros((1,3,1,1)),[[0.]],[[0.]])


def test_constant_model_and_large_square_error_are_finite():
    model=bank([[[0.],[0.]]]);J=np.zeros((1,2,1,1))
    result=b.solve_bc_milp(domain([-1.],[1.]),model,J,[[0.]],seconds=5.,backend='scipy')
    assert result['success'] and result['frozen_C']==pytest.approx(.5)
    huge=J.copy();huge[0,0,0,0]=1e6
    master=b.build_bc_master(domain([-1.],[1.]),model,huge,[[0.]],segments=2)
    for u in (-1.,0.,1.):
        value=b.evaluate_proxy(master,[[u]])
        assert np.isfinite(value['proxy']) and 0<=value['proxy']<=.5+1e-10


def test_invalid_covariance_weights_and_dimensions_are_explicit():
    model=bank([[[0.],[0.]]]);J=np.zeros((1,2,1,1))
    wrong={**model,'weights':np.array([.5])}
    with pytest.raises(ValueError,match='sum to one'):b.build_bc_master(domain([-1.],[1.]),wrong,J,[[0.]])
    wrong={**model,'covariances':np.zeros((1,2,1,1))}
    with pytest.raises(ValueError,match='positive definite'):b.build_bc_master(domain([-1.],[1.]),wrong,J,[[0.]])
    with pytest.raises(ValueError,match='dimensions'):b.build_bc_master(domain([-1.],[1.]),model,np.zeros((1,2,1,2)),[[0.]])


def test_probability_module_independently_agrees_on_multi_scenario_bound():
    from .ev_bayes_small_probability import bhattacharyya_bank
    means=np.asarray([[[.2,-.1],[.5,.7],[-.8,.4]],[[.4,.3],[.8,.2],[-.2,-.6]]])
    cov=np.broadcast_to(np.eye(2),(2,3,2,2)).copy()
    cov[0,0]=[[1.,.4],[.4,1.5]];cov[1,2]=[[2.,-.3],[-.3,.8]]
    model=bank(means,cov=cov,prior=[.2,.3,.5],weights=[.3,.7]);model['log_prior']=np.log(model['prior'])
    value=b.frozen_bc_value(model,np.zeros((2,3,2,1)),[[0.]],[[0.]])
    independent=bhattacharyya_bank(model)
    assert value['C']==pytest.approx(independent['bound'],abs=3e-15)
    assert [r['B'] for r in value['pairs']]==pytest.approx([r['B'] for r in independent['pairs']],abs=3e-15)


def test_gurobi_main_backend_matches_scalar_audit():
    from .ev_original_mip_backend import gurobi_available
    if not gurobi_available()['available']:pytest.skip('Gurobi unavailable')
    model=bank([[[0.],[0.]]]);J=np.asarray([[[[1.]], [[-1.]]]])
    result=b.solve_bc_milp(domain([-1.],[1.]),model,J,[[0.]],seconds=5.,backend='gurobi')
    assert result['success'] and result['backend']=='gurobi'
    assert abs(result['power_kw'][0,0])==pytest.approx(1.,abs=1e-8)
    assert result['frozen_C']==pytest.approx(.5*np.exp(-.5),abs=1e-10)
    assert result['solver_best_bound']<=result['proxy_value']+1e-8
