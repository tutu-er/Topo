"""Complete profile/gradient/MILP/AC-acceptance component checks."""
from copy import deepcopy
from dataclasses import replace
from inspect import signature
from types import SimpleNamespace
from time import perf_counter

import numpy as np
import pytest

from . import ev_gradient_flow_design as flow
from . import ev_fixed_train_design as fixed
from . import likelihood as lk
from .simulation import model


@pytest.fixture(scope='module')
def context():
    bank=[lk.RadialCandidate.from_supports('star',3,()),
          lk.RadialCandidate.from_supports('cherry01',3,((0,1),))]
    T=3;tm=model(3)
    fm=replace(tm,v_sigma=np.full((T,3),.00012),
        master_p_sigma=np.full(T,.001),master_q_sigma=np.full(T,.001))
    p=np.array([[.11,.12,.13],[.12,.11,.13],[.13,.12,.11],[.115,.125,.12]])
    q=np.full_like(p,.035);sp=np.array([.08,.1,.09,.11]);sq=np.full(4,.03)
    fits={c.name:SimpleNamespace(r=np.full(c.m,.004),x=np.full(c.m,.0025),
        p=p.copy(),q=q.copy(),source_p=sp.copy(),source_q=sq.copy(),success=True,loss=1.)
        for c in bank}
    phy=lk.ac_predict(bank[0],fits['star'].r,fits['star'].x,p,q,sp,sq,np.ones(4))
    tr=lk.Observations(p=p+tm.bias_p,q=q+tm.bias_q,v=phy['v']+tm.bias_v,
        root_v=np.ones(4),master_p=phy['master_p'],master_q=phy['master_q'])
    u=np.zeros((T,3));u[:,0]=4.
    ctx=fixed.build_fixed_contexts(tr,bank,fits,tm,fm,-6.,6.,u)
    assert ctx['success'],ctx['diagnostics']
    return ctx


def _group(gradient):
    return [[dict(a='a',b='b',gradient_per_kw=np.asarray(gradient,float))]]


def test_binary_aggregation_is_average_of_two_directed_source_profiles():
    rows=[dict(a='a',b='b',success=True,half_SSE=2.,gradient_per_kw=np.ones((2,1))),
          dict(a='b',b='a',success=True,half_SSE=6.,gradient_per_kw=np.full((2,1),3.))]
    out=flow._aggregate(rows,'binarymean')
    assert out['score']==4.
    assert out['gradient_per_kw']==pytest.approx(np.full((2,1),2.))
    assert flow._aggregate(rows,'nearest')['score']==4.
    rows[0]['success']=False
    assert not flow._aggregate(rows,'nearest')['success']


def test_pair_restart_seed_is_stable_when_other_topologies_are_removed():
    original=['a__to__b','a__to__c','b__to__a']
    filtered=['a__to__b','b__to__a']
    all_seeds={name:flow.source_pair_seed(23,name) for name in original}
    assert {name:all_seeds[name] for name in filtered}=={
        name:flow.source_pair_seed(23,name) for name in filtered}
    assert len(set(all_seeds.values()))==3


def test_nearest_ties_keep_all_required_directional_branches():
    rows=[dict(a='a',b=b,success=True,half_SSE=1.,gradient_per_kw=np.array([[g]]))
        for b,g in [('b',1.),('c',-1.)]]
    out=flow._aggregate(rows,'nearest')
    assert len(out['active_groups'][0])==2
    assert out['nearest'][0]['active']==['b','c']
    with pytest.raises(ValueError,match='two'):
        flow._aggregate(rows,'binarymean')


@pytest.mark.parametrize('method',['ac','i2'])
def test_complete_profile_keeps_all_pairs_lines_and_envelope_gradient(context,method):
    engine=flow.ProfileEngine(context,fit_options=dict(multistart=1,max_nfev=120,tol=1e-11),seed=23)
    u=context['reference_kw'].copy()
    out=engine.evaluate(u,method)
    assert out['success'] and out['line_parameters_unchanged']
    assert len(out['pairs'])==out['expected_pair_count']==2
    grad=flow._aggregate(out['pairs'],'binarymean')
    eps=.0002;plus=u.copy();minus=u.copy();plus[0,0]+=eps;minus[0,0]-=eps
    high=flow._aggregate(engine.evaluate(plus,method)['pairs'],'binarymean')['score']
    low=flow._aggregate(engine.evaluate(minus,method)['pairs'],'binarymean')['score']
    assert grad['gradient_per_kw'][0,0]==pytest.approx((high-low)/(2*eps),rel=3e-5,abs=1e-7)
    before=engine.counters();again=engine.evaluate(u,method)
    assert again is out
    assert engine.counters()['pair_fit_calls']==before['pair_fit_calls']
    assert engine.counters()['profile_cache_hits']==before['profile_cache_hits']+1
    if method=='i2':assert engine.counters()['response_calls']['ac_value_calls']==0


def test_new_profile_agrees_with_old_complete_AC_acceptance(context):
    options=dict(multistart=1,max_nfev=150,tol=1e-11)
    u=context['reference_kw'].copy();u[1,0]=5.
    new=flow.ProfileEngine(context,fit_options=options,seed=19).evaluate(u)
    old=fixed.evaluate_fixed_action(context,u,fit_options=options,seed=19)
    assert new['success'] and old['success']
    for a,b in zip(new['pairs'],old['pairs']):
        assert a['half_SSE']==pytest.approx(b['added'],rel=2e-8,abs=1e-8)


def test_cache_isolated_ac_i2_and_training_context(context):
    cache={};options=dict(multistart=1,max_nfev=120)
    a=flow.ProfileEngine(context,fit_options=options,cache=cache,seed=1)
    u=context['reference_kw'];a.evaluate(u,'ac');a.evaluate(u,'i2')
    assert len(cache)==2
    b=flow.ProfileEngine(context,fit_options=options,cache=cache,seed=1)
    b.evaluate(u,'ac')
    assert b.counters()['pair_fit_calls']==0 and b.counters()['profile_cache_hits']==1
    altered=deepcopy(context);altered['background']['p'][0,0]+=.001
    c=flow.ProfileEngine(altered,fit_options=options,cache=cache,seed=1)
    assert c.signature!=a.signature
    c.evaluate(u,'ac');assert c.counters()['pair_fit_calls']==2


def test_expired_budget_retains_unattempted_pair_denominator_without_caching(context):
    engine=flow.ProfileEngine(context,fit_options=dict(multistart=1,max_nfev=120))
    u=context['reference_kw']
    out=engine.evaluate(u,'ac',deadline=perf_counter()-1.)
    assert not out['success'] and out['deadline_reached']
    assert len(out['pairs'])==out['expected_pair_count']==2
    assert all(not pair['success'] and not pair['attempted'] for pair in out['pairs'])
    assert engine.counters()['pair_fit_calls']==0
    assert len(engine.cache)==0
    assert engine.evaluate(u,'ac')['success']


def test_changed_line_snapshot_rejected_before_cache_lookup(context):
    ctx=deepcopy(context)
    engine=flow.ProfileEngine(ctx,fit_options=dict(multistart=1,max_nfev=120))
    engine.evaluate(ctx['reference_kw'])
    line=ctx['lines']['star'];line['r'].setflags(write=True);line['r'][0]+=.001
    with pytest.raises(ValueError,match='snapshot changed'):
        engine.evaluate(ctx['reference_kw'])


@pytest.mark.parametrize('mobile',[False,True])
def test_linear_MILP_optimizes_full_operation_domain_with_trust_and_finite_bound(mobile):
    domain=fixed._service(6,3,mobile=mobile,discharge=True)
    u=np.zeros((6,3));u[:,0]=2.
    initial=fixed._realize(domain,u);assert initial['success']
    g=np.arange(18,dtype=float).reshape(6,3)+1.
    result=flow.solve_gradient_step(domain,u,_group(g),trust_kw=2.,
        solver_seconds=4.,backend='scipy',start=initial['x'])
    assert result['success'] and result['operation']['feasible']
    assert np.max(abs(result['power_kw']-u))<=2.+1e-7
    assert result['predicted_directional_gain']>=-1e-7
    assert result['linear_best_bound']>=result['predicted_directional_gain']-1e-7
    assert result['full_AC_global_gap'] is None
    if not mobile:assert result['power_kw'][:,1:]==pytest.approx(np.zeros((6,2)))


def test_tied_opposite_prices_do_not_create_false_directional_improvement():
    domain=fixed._service(6,3,mobile=False,discharge=True)
    u=np.zeros((6,3));u[:,0]=2.
    g=np.zeros_like(u);g[0,0]=1.
    groups=[[dict(gradient_per_kw=g),dict(gradient_per_kw=-g)]]
    result=flow.solve_gradient_step(domain,u,groups,backend='scipy',solver_seconds=4.)
    assert result['success']
    assert result['predicted_directional_gain']==pytest.approx(0.,abs=1e-7)


def _fake_engine(monkeypatch, score, gradient, *, reject_proposals=False,fail_all=False):
    class Engine:
        def __init__(self,*args,**kwargs):self.calls=0
        def evaluate(self,u,method='ac',**kwargs):
            self.calls+=1
            constant=np.count_nonzero(abs(np.asarray(u)[:,0]-2.)>1e-5)==0
            success=not fail_all and (not reject_proposals or constant)
            value=float(score(u))
            rows=[dict(a=a,b=b,success=success,half_SSE=value,
                gradient_per_kw=np.asarray(gradient(u)),source=np.zeros((6,2)))
                for a,b in [('a','b'),('b','a')]]
            return dict(success=success,pairs=rows,method=method)
        def counters(self):return dict(profile_evaluations=self.calls)
    monkeypatch.setattr(flow,'ProfileEngine',Engine)


@pytest.mark.parametrize('zero',[False,True])
def test_selector_preserves_baseline_and_accepts_only_complete_AC_increase(monkeypatch,zero):
    weights=np.arange(1,7)[:,None]
    _fake_engine(monkeypatch,lambda u:np.sum(weights*np.asarray(u)**2),
        lambda u:np.zeros_like(u) if zero else 2*weights*np.asarray(u))
    domain=fixed._service(6,3,mobile=False,discharge=True)
    u=np.zeros((6,3));u[:,0]=2.
    selected,record=flow.select_gradient_controls(dict(success=True,control_shape=(6,3),models=[{},{}]),
        domain,dict(constant=u),aggregation='binarymean',
        options=dict(backend='scipy',max_starts=1,max_iterations=2,total_seconds=20.))
    assert record['success'] and record['inheritance_protected']
    assert record['selected_operation']['feasible']
    assert record['selected_score']>=record['initial_best_score']
    if zero:
        assert selected==pytest.approx(u)
        assert record['iterations'][0]['stop_reason'].startswith('zero_gradient')
    else:
        assert record['selected_score']>record['initial_best_score']
        assert any(row['accepted'] for row in record['iterations'])
    assert record['global_AC_gap'] is None
    assert not record['uses_validation'] and not record['uses_truth']


def test_incomplete_candidate_source_fit_rejects_whole_proposal(monkeypatch):
    weights=np.arange(1,7)[:,None]
    _fake_engine(monkeypatch,lambda u:np.sum(weights*np.asarray(u)**2),
        lambda u:2*weights*np.asarray(u),reject_proposals=True)
    domain=fixed._service(6,3,mobile=False,discharge=True)
    u=np.zeros((6,3));u[:,0]=2.
    selected,record=flow.select_gradient_controls(dict(success=True,control_shape=(6,3),models=[{},{}]),
        domain,dict(constant=u),options=dict(backend='scipy',max_starts=1,max_iterations=1))
    assert selected==pytest.approx(u)
    assert not record['iterations'][0]['accepted']
    assert record['iterations'][0]['rejection_reason']=='required_AC_source_fit_failed'


def test_no_successful_initial_is_kept_as_failure(monkeypatch):
    _fake_engine(monkeypatch,lambda u:1.,lambda u:np.ones_like(u),fail_all=True)
    domain=fixed._service(6,3,mobile=False,discharge=True)
    u=np.zeros((6,3));u[:,0]=2.
    selected,record=flow.select_gradient_controls(dict(success=True,control_shape=(6,3),models=[{},{}]),
        domain,dict(constant=u),options=dict(backend='scipy'))
    assert selected is None and not record['success']
    assert 'no feasible complete' in record['failure']


def test_public_selector_has_no_future_truth_or_validation_argument():
    args=set(signature(flow.select_gradient_controls).parameters)
    assert not args & {'truth','validation','future_noise','background_true'}
    assert args>={'context','domain','initial_powers','method','aggregation','options'}


def test_real_selector_runs_full_profile_operation_and_acceptance(context):
    domain=fixed._service(3,3,mobile=False,discharge=True)
    u=context['reference_kw'].copy()
    selected,record=flow.select_gradient_controls(context,domain,dict(constant=u),
        method='i2',aggregation='binarymean',fit_options=dict(multistart=1,max_nfev=150),
        options=dict(backend='scipy',max_starts=1,max_iterations=1,total_seconds=20.))
    assert record['success'] and record['selected_operation']['feasible']
    assert record['inheritance_protected']
    assert len(record['selected_profile']['pairs'])==2
    assert record['counters']['response_calls']['ac_value_calls']>0
    assert record['counters']['response_calls']['approximate_value_calls']>0
    assert record['seconds']>=record['initialization_seconds']
    assert record['linear_MILP_seconds']>=0.
    assert selected.shape==(3,3)


def test_initial_score_obeys_budget_and_unscored_feasible_fallback_is_separate(context):
    domain=fixed._service(3,3,mobile=False,discharge=True)
    u=context['reference_kw'].copy()
    selected,record=flow.select_gradient_controls(context,domain,dict(constant=u),
        options=dict(backend='scipy',total_seconds=1e-8))
    assert selected is None and not record['success']
    assert record['selected_score'] is None
    assert record['fallback_power']==pytest.approx(u)
    assert record['fallback_operation']['feasible']
    assert record['counters']['pair_fit_calls']==0
    assert record['initial_candidates'][0]['profile']['expected_pair_count']==2
