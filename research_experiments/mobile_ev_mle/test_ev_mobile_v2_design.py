"""Capped-task bounds and reprofiled shared-line AC envelope derivatives."""
from copy import deepcopy
from time import perf_counter

import numpy as np
import pytest

from . import ev_mobile_v2_design as design
from . import ev_systematic_design as scalar
from . import ev_fixed_train_design as fixed
from . import likelihood as lk
from .ev_rx_fit import fit_rx_inputs
from .test_ev_systematic_design import context,domain
from .test_ev_systematic_refit import case,OPTIONS


def test_cap_scale_and_easy_task_saturation():
    assert design.TASK_CAP==pytest.approx(5.411086908190808)
    assert design.capped_value([1,100],5)==3
    assert design.capped_value([2,1000],5)==3.5
    with pytest.raises(ValueError):design.capped_value([1],0)
    with pytest.raises(ValueError):design.capped_value([float('nan')])


@pytest.mark.parametrize('grid',[3,9])
@pytest.mark.parametrize('cap',[.1,.5,20.])
def test_capped_milp_bounds_independent_dense_power_audit(grid,cap):
    ctx=context(4);op=domain();table=scalar.build_systematic_tables(ctx,op,grid_points=grid)
    out=design.solve_capped_milp(ctx,op,tables=table,cap=cap,backend='scipy',solver_seconds=5.)
    assert out['success'] and out['operation']['feasible']
    # Net-zero two-time domain: p=(x,-x), -1<=x<=1. This independent
    # dense audit is a formula test, not the operational action selector.
    exact=max(design.capped_value([.5*np.sum((p.c+p.B@np.array([x,-x]))**2)
        for p in ctx['pairs']],cap) for x in np.linspace(-1,1,101))
    assert out['frozen_objective_lower']<=exact+1e-7
    assert out['frozen_objective_upper']>=exact-1e-7
    assert out['frozen_objective_upper']<=cap+1e-7
    assert out['added_continuous_count']>=12
    assert out['added_binary_count']==(0 if grid==3 else 16)
    assert out['full_AC_global_gap'] is None


def test_capped_tables_reject_different_physical_or_route_domain():
    ctx=context();op=domain();table=scalar.build_systematic_tables(ctx,op,grid_points=3)
    with pytest.raises(ValueError,match='tables differ'):
        design.solve_capped_milp(ctx,domain(T=3),tables=table,backend='scipy')


def test_joint_aggregation_preserves_signed_gaps_and_tied_gradients():
    names=['a','b','c'];pairs=[]
    for a in names:
        for b in names:
            if a!=b:pairs.append(dict(a=a,b=b,half_gap=-1.,success=True,gradient_per_kw=np.ones((2,1))))
    out=design._joint_aggregate(pairs,names)
    assert out['score']==-1
    assert all(len(g)==2 for g in out['active_groups'])
    assert not design._joint_aggregate(pairs[:-1],names)['success']


@pytest.fixture(scope='module')
def physical():
    a,blocks,meters,truth,_,_=case(seed=91,noisy=True)
    tr=blocks[0];tm=meters[0];fm=meters[2]
    # Two future points keep this derivative audit small, with the same
    # per-channel meter geometry as the nonlinear fitting code.
    from .ev_fixed_train_fit import subset_model
    fm=subset_model(fm,blocks[2],slice(0,2));T=2
    b=lk.RadialCandidate.from_supports('cherry',3,[(0,1)])
    fits={c.name:fit_rx_inputs(c,tr,tm,seed=81,**OPTIONS) for c in (a,b)}
    assert all(f.success for f in fits.values())
    p=np.tile(np.mean(tr.p-np.asarray(tm.bias_p),axis=0),(T,1))
    q=np.tile(np.mean(tr.q-np.asarray(tm.bias_q),axis=0),(T,1))
    forecast=dict(p=p,q=q,root=np.ones(T),source_by_candidate={name:
        dict(source_p=np.full(T,np.mean(f.source_p)),source_q=np.full(T,np.mean(f.source_q)))
        for name,f in fits.items()},metadata=dict(training_only=True,source_interface=1,
            uses_validation=False,uses_truth=False,uses_future_noise=False,kind='audit training mean'))
    power=np.array([[.4,0,0],[-.2,0,0]])
    ctx=fixed.build_fixed_contexts(tr,[a,b],fits,tm,fm,-2.,2.,power,
        forecast=forecast,base_kw=20.,source_p_bounds=OPTIONS['source_p_bounds'],
        source_q_bounds=OPTIONS['source_q_bounds'])
    assert ctx['success'],ctx['diagnostics']
    return ctx,power


def test_joint_predictive_profiles_share_lines_and_have_fixed_feasible_witnesses(physical):
    ctx,power=physical;before={k:fixed._line_id(v['r'],v['x']) for k,v in ctx['lines'].items()}
    engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS,seed=43);out=engine.evaluate(power)
    assert out['success'] and out['joint_fit_calls']==4
    assert len(out['references'])==2 and len(out['pairs'])==2
    for reference in out['references']:
        for fit in reference['fits'].values():
            assert fit['loss']<=fit['feasible_witness']['loss']+1e-7
            assert fit['gradient_reliable'] and fit['line_domain_feasible'] and fit['source_box_feasible']
            assert len(fit['state']['r'])==len(ctx['lines'][reference['a']]['r']) or len(fit['state']['r']) in (4,5)
    assert before=={k:fixed._line_id(v['r'],v['x']) for k,v in ctx['lines'].items()}
    assert engine.evaluate(power) is out and engine.cache_hits==1
    assert not out['uses_truth'] and not out['uses_validation'] and not out['uses_future_noise']
    assert out['full_AC_global_gap'] is None


@pytest.mark.parametrize('site',[0,1,2])
def test_joint_envelope_prices_match_full_reprofile_finite_differences(physical,site):
    ctx,power=physical;engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS,seed=43)
    direction=np.zeros_like(power);direction[:,site]=[.7,-.3]
    audit=design.audit_envelope_gradient(engine,power,direction,step=1e-3)
    assert audit['success']
    assert audit['max_scaled_error']<.004,audit
    assert any((count or 0)>0 for row in audit['active_constraint_counts'] for count in row.values())


def test_deadline_does_not_drop_failed_competitors_or_claim_valid_gradient(physical):
    ctx,power=physical;engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS)
    out=engine.evaluate(power,deadline=perf_counter()-1.)
    assert not out['success'] and out['deadline_reached']
    assert len(out['pairs'])==2 and all(not p['success'] for p in out['pairs'])
    assert all(not fit['attempted'] for r in out['references'] for fit in r['fits'].values())


def test_reference_line_mutation_is_rejected_before_cached_score(physical):
    ctx,power=physical;ctx=deepcopy(ctx)
    engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS);engine.evaluate(power)
    line=ctx['lines']['star'];line['r']=line['r'].copy();line['r'][0]+=.001
    with pytest.raises(ValueError,match='snapshot changed'):engine.evaluate(power)


def test_physical_failure_retains_every_model_and_pair(physical,monkeypatch):
    ctx,power=physical;engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS)
    def fail(*args,**kwargs):raise RuntimeError('declared physical diagnostic failure')
    monkeypatch.setattr(fixed,'_geometry',fail)
    out=engine.evaluate(power)
    assert not out['success'] and len(out['pairs'])==2
    assert all(len(row['fits'])==2 for row in out['references'])
    assert all(not p['success'] for p in out['pairs'])


def test_parent_predictive_scores_reuse_cache_and_only_pay_new_work(physical):
    ctx,power=physical;engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS,seed=43)
    zero=np.zeros_like(power);engine.evaluate(zero)
    op=domain(T=2,n=3)
    chosen,row=design.joint_gradient_policy(ctx,op,dict(parent=zero),joint_options=OPTIONS,
        seed=43,backend='scipy',max_starts=1,max_iterations=0,total_seconds=5.,engine=engine)
    assert row['success'] and row['inheritance_protected']
    assert row['counters']['joint_fit_calls']==0 and row['counters']['cache_hits']==1
    assert row['gradient_MILP_calls']==0
    np.testing.assert_array_equal(chosen,zero)
    with pytest.raises(ValueError,match='engine differs'):
        design.joint_gradient_policy(ctx,op,dict(parent=zero),joint_options=OPTIONS,seed=44,engine=engine)


def test_zero_power_step_avoids_repeated_K_squared_fits(physical,monkeypatch):
    ctx,power=physical;engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS,seed=43)
    zero=np.zeros_like(power);engine.evaluate(zero);op=domain(T=2,n=3)
    def same_action(operation,current,*args,**kwargs):
        return dict(success=True,power_kw=current.copy(),x=kwargs['start'],predicted_directional_gain=0.)
    monkeypatch.setattr(design.gradient,'solve_gradient_step',same_action)
    _,row=design.joint_gradient_policy(ctx,op,dict(parent=zero),joint_options=OPTIONS,seed=43,
        backend='scipy',max_starts=1,max_iterations=2,total_seconds=5.,engine=engine)
    assert row['gradient_MILP_calls']==1 and row['counters']['profile_evaluations']==0
    assert row['starts'][0]['iterations'][0]['repeated_AC_score_avoided']


def test_near_duplicate_initial_uses_original_scored_canonical_action(physical):
    ctx,power=physical;zero=np.zeros_like(power);near=zero.copy();near[0,0]=1e-12
    engine=design.JointPredictiveEngine(ctx,fit_options=OPTIONS,seed=43);engine.evaluate(zero)
    chosen,row=design.joint_gradient_policy(ctx,domain(T=2,n=3),dict(first=zero,near=near),
        joint_options=OPTIONS,seed=43,backend='scipy',max_iterations=0,total_seconds=5.,engine=engine)
    np.testing.assert_array_equal(chosen,zero)
    duplicate=row['initial_candidates'][1]
    assert duplicate['near_duplicate'] and duplicate['new_curve_not_scored']
    assert duplicate['canonical_origin']=='first' and row['counters']['profile_evaluations']==0
