"""Original-domain polynomial identities and complete-AC multi-site checks."""
from types import SimpleNamespace
import numpy as np
import pytest
from .ev_original_ac_check import (multisite_control_map,product_structure,
    fixed_absolute_coefficients,polynomial_residual,original_model_qp,
    build_multisite_context,mixed_control_matrix,evaluate_multisite_predicted_action,
    guard_multisite_predicted_actions)
from .ev_original_ac_check import make_ac_context,evaluate_ac_witness,recenter_at_absolute
from .ev_ac_moving_cross import parameter_increment,residual as moving_residual
from .symmetric_ac_local import mean_and_jacobian,whiten_observations
from .simulation import generate,observe,model
from .candidates import RadialCandidate


def toy_model():
    # Deliberately nonzero cross-time products. They must not be deleted by
    # claiming physical independent-time structure for arbitrary inputs.
    return dict(c=np.array([1.,-2.]),B=np.array([[1.,2.],[3.,-1.]]),
        A0=np.array([[1.,2.],[2.,0.]]),At=np.array([[[1.,2.],[0.,1.]],[[0.,3.],[-1.,2.]]]),
        reference_kw=np.array([1.,1.]),domain=dict(base_center=np.array([.2,.3]),
            shift=np.array([[1.,.5],[.2,1.]]),physical_lower=np.array([-1.,-1.]),physical_upper=np.array([2.,2.]),
            outer_lower=np.array([-4.,-4.]),outer_upper=np.array([4.,4.])))


def test_fixed_absolute_polynomial_keeps_cross_products_and_sign():
    m=toy_model();xi=np.array([.5,.4]);coef=fixed_absolute_coefficients(m,xi)
    for u in (np.array([0.,0.]),np.array([1.,2.]),np.array([.2,-.7])):
        z=parameter_increment(m,u,xi)
        assert polynomial_residual(coef,u)==pytest.approx(moving_residual(m,u,z),abs=1e-13)
    assert coef['cross_pairs']==[(0,1)]
    assert np.linalg.norm(coef['cross'])>0.
    assert not coef['structure']['cross_time_numerically_zero']


def test_curvature_field_rejected_and_infeasible_witness_not_used():
    m=toy_model()
    with pytest.raises(ValueError):fixed_absolute_coefficients(m,[3.,0.])
    m['control_curvature_difference']=np.zeros((2,2))
    with pytest.raises(ValueError):fixed_absolute_coefficients(m,[.5,.4])


@pytest.fixture(scope='module')
def inputs():
    path=generate(9201,history_count=4,future_count=3,truth_family=((0,1),));history,_=observe(path,slice(0,4))
    meter=model(3);bank=[RadialCandidate.from_supports('clade_01',3,((0,1),)),
        RadialCandidate.from_supports('clade_02',3,((0,2),)),RadialCandidate.from_supports('star',3,())]
    obs=whiten_observations(history,meter);fits={}
    for c in bank:
        r=np.full(c.m,.005);x=np.full(c.m,.003)
        g=mean_and_jacobian(c,r,x,path.p[:4],path.q[:4],path.source_p[:4],path.source_q[:4],history.root_v,meter)
        fits[c.name]=SimpleNamespace(r=r,x=x,p=path.p[:4],q=path.q[:4],source_p=path.source_p[:4],source_q=path.source_q[:4],
            success=True,loss=float(np.sum((g['mean']-obs)**2)))
    ref=np.zeros((3,3));ref[:,0]=3.
    return history,bank,fits,meter,ref


def test_multisite_control_map_adds_each_EV_once():
    U=multisite_control_map(2,3,20.)
    p=np.array([[1.,2.,3.],[4.,5.,6.]])
    state=(U@p.ravel()).reshape(2,8)
    assert state[:,:3]==pytest.approx(p/20.)
    assert np.all(state[:,3:]==0)


def test_multisite_context_all_pairs_and_original_line_domain(inputs):
    history,bank,fits,meter,ref=inputs
    built=build_multisite_context(history,bank,fits,meter,meter,ref,0.,6.,20.,
        line_bounds=dict(r_bounds=(.001,.02),x_bounds=(.001,.01)))
    assert built['success'],built['diagnostics']
    assert len(built['models'])==6 and built['diagnostics']['ac_calls']==24
    for m in built['models']:
        assert m['B'].shape[1]==9 and m['At'].shape[0]==9
        assert moving_residual(m,ref.ravel(),np.zeros(m['A0'].shape[1]))==pytest.approx(m['reference_residual'],abs=1e-11)
        nl=m['line_columns']//2;lo=np.asarray(m['domain']['physical_lower']);hi=np.asarray(m['domain']['physical_upper'])
        assert lo[:nl]==pytest.approx(np.full(nl,.001));assert hi[:nl]==pytest.approx(np.full(nl,.02))
        assert lo[nl:2*nl]==pytest.approx(np.full(nl,.001));assert hi[nl:2*nl]==pytest.approx(np.full(nl,.01))
        assert m['product_structure']['cross_time_numerically_zero']
        assert m['product_structure']['same_time_max_norm']>1e-6


def test_original_and_outer_qp_and_fixed_witness_identity(inputs):
    history,bank,fits,meter,ref=inputs
    built=build_multisite_context(history,bank,fits,meter,meter,ref,0.,6.,20.)
    m=built['models'][0];u=np.array([[6.,0.,0.],[0.,4.,0.],[0.,0.,2.]]).ravel()
    qp=original_model_qp(m,u)
    assert qp['original']['success'] and qp['original']['physical_bound_violation']<1e-9
    assert qp['outer']['primal_upper']<=qp['original']['primal_upper']+1e-7
    xi=qp['original']['physical_parameter'];coef=fixed_absolute_coefficients(m,xi)
    assert polynomial_residual(coef,u)==pytest.approx(qp['original']['residual'],abs=1e-10)
    for utest in (np.zeros(9),np.full(9,6.)):
        assert polynomial_residual(coef,utest)==pytest.approx(moving_residual(m,utest,parameter_increment(m,utest,xi)),abs=2e-10)


def test_multisite_mixed_individual_site_time_FD(inputs):
    history,bank,fits,meter,ref=inputs;c=bank[0];fit=fits[c.name]
    base=dict(p=np.full((3,3),.12)+ref/20.,q=np.full((3,3),.04),source_p=np.zeros(3),source_q=np.zeros(3))
    U=multisite_control_map(3,3,20.);theta=np.r_[fit.r,fit.x]
    mixed=mixed_control_matrix(c,theta,base,np.ones(3),meter,('p','q','v','master_p','master_q'),U)
    k=5;eps=.0007
    plus={key:value.copy() for key,value in base.items()};minus={key:value.copy() for key,value in base.items()}
    plus['p'][1,2]+=eps/20.;minus['p'][1,2]-=eps/20.
    gp=mean_and_jacobian(c,fit.r,fit.x,plus['p'],plus['q'],plus['source_p'],plus['source_q'],np.ones(3),meter)
    gm=mean_and_jacobian(c,fit.r,fit.x,minus['p'],minus['q'],minus['source_p'],minus['source_q'],np.ones(3),meter)
    assert mixed['line'][k]==pytest.approx((gp['line_jacobian']-gm['line_jacobian'])/(2*eps),abs=3e-6,rel=1e-5)
    assert mixed['state'][k]==pytest.approx((gp['state_jacobian']-gm['state_jacobian'])/(2*eps),abs=3e-6,rel=1e-5)


def test_failed_center_is_explicit_and_all_pairs_kept_or_failed(inputs):
    history,bank,fits,meter,ref=inputs
    built=build_multisite_context(history,bank,fits,meter,meter,ref,0.,6.,20.,line_bounds=dict(r_bounds=(.006,.02),x_bounds=(.001,.01)))
    assert not built['success'] and built['models']==[] and 'outside' in built['diagnostics']['failure']


def test_multisite_predicted_mean_and_guard_keep_all_pairs(monkeypatch,inputs):
    history,bank,fits,meter,ref=inputs;bank=bank[:2];calls=[]
    from . import ev_original_ac_check as module
    def fake_fit(candidates,observations,meter,**kwargs):
        calls.append(observations)
        found={}
        for i,c in enumerate(candidates):
            found[c.name]=SimpleNamespace(success=True,loss=10.+i)
        return found,{},dict(fit_calls=2,ac_calls=3,seconds=0.)
    monkeypatch.setattr(module.io,'fit_all_candidates',fake_fit)
    monkeypatch.setattr(module.io,'fit_record',lambda c,f:dict(name=c.name,loss=f.loss))
    out=evaluate_multisite_predicted_action(history,bank,fits,meter,meter,meter,ref)
    assert out['success'] and len(out['pairs'])==2 and out['required_fit_states']==4
    assert out['mean_added_score']==pytest.approx(np.mean([r['added'] for r in out['pairs']]))
    # Future physical forecast removes the known bias once; predicted meters
    # add that same calibrated bias once before joint likelihood evaluation.
    assert np.asarray(calls[0].p[4:])==pytest.approx(np.tile(history.p.mean(axis=0),(3,1))+ref/20.)
    guard=guard_multisite_predicted_actions(history,bank,fits,meter,meter,meter,ref,ref)
    assert guard['cache_hit'] and guard['evaluated_actions']==1 and not guard['accepted']


def test_predicted_action_failure_does_not_drop_required_pair(monkeypatch,inputs):
    history,bank,fits,meter,ref=inputs;bank=bank[:2]
    from . import ev_original_ac_check as module
    def failed(*args,**kwargs):return {},{'bad':'failed'},dict(fit_calls=2,ac_calls=1,seconds=0.)
    monkeypatch.setattr(module.io,'fit_all_candidates',failed)
    out=evaluate_multisite_predicted_action(history,bank,fits,meter,meter,meter,ref)
    assert not out['success'] and len(out['pairs'])==2 and out['mean_added_score'] is None


def test_complete_AC_same_witness_and_recenter_preserve_absolute_box(inputs):
    history,bank,fits,meter,ref=inputs
    built=build_multisite_context(history,bank,fits,meter,meter,ref,0.,6.,20.)
    m=built['models'][0];ctx=make_ac_context(m,history,bank,fits,meter,meter)
    xi=np.asarray(m['domain']['base_center'])+np.asarray(m['domain']['shift'])@ref.ravel()
    checked=evaluate_ac_witness(m,ref.ravel(),xi,ctx)
    assert checked['error_norm']<2e-9 and checked['physical_box_feasible']
    assert checked['complete_AC_voltage_feasible']
    # Move the line and both state blocks to a new independently legal point.
    target=xi.copy();target[0]+=.0002;target[m['line_columns']]+=.00001
    target[-m['future_state_columns']]+=.00002
    new=recenter_at_absolute(m,ref.ravel(),target,ctx)
    assert new['domain']['physical_lower']==pytest.approx(m['domain']['physical_lower'])
    assert new['domain']['physical_upper']==pytest.approx(m['domain']['physical_upper'])
    assert new['new_proxy_requires_new_bounds']
    again=evaluate_ac_witness(new,ref.ravel(),target,ctx)
    assert again['error_norm']<2e-9
    assert new['expansion_history_cost']==pytest.approx(again['complete_AC_H_cost'])


def test_AC_witness_uses_EV_once_at_all_active_sites(inputs):
    history,bank,fits,meter,ref=inputs
    built=build_multisite_context(history,bank,fits,meter,meter,ref,0.,6.,20.)
    m=built['models'][0];ctx=make_ac_context(m,history,bank,fits,meter,meter)
    u=np.array([[1.,2.,3.],[0.,4.,0.],[6.,0.,0.]])
    xi=np.asarray(m['domain']['base_center'])+np.asarray(m['domain']['shift'])@ref.ravel()
    checked=evaluate_ac_witness(m,u.ravel(),xi,ctx)
    assert checked['reference_future_state']['p']==pytest.approx(np.asarray(ctx['reference_background']['p'])+u/20.)
    assert checked['fit_calls']==0 and checked['ac_calls']==3


def test_new_absolute_QP_and_polynomial_match_independent_checks(inputs):
    from .ev_original_domain_milp import (from_moving_model,evaluate_original_pair,
        fixed_absolute_witness,polynomial_residual as master_residual)
    history,bank,fits,meter,ref=inputs
    built=build_multisite_context(history,bank,fits,meter,meter,ref,0.,6.,20.)
    for m in (toy_model(),built['models'][0]):
        m=dict(m);m.setdefault('name','toy')
        u=np.linspace(.1,.8,m['B'].shape[1])
        pair=from_moving_model(m);new=evaluate_original_pair(pair,u)
        old=original_model_qp(m,u)['original']
        assert new['success'] and old['success']
        assert new['primal_upper']==pytest.approx(old['primal_upper'],rel=1e-7,abs=1e-7)
        assert new['tangent_lower_numeric']<=old['primal_upper']+1e-8
        xi=new['physical_parameter'];coef=fixed_absolute_coefficients(m,xi)
        poly=fixed_absolute_witness(pair,xi)
        assert master_residual(poly,u)==pytest.approx(polynomial_residual(coef,u),abs=1e-10)
        assert master_residual(poly,u)==pytest.approx(new['residual'],abs=2e-10)
