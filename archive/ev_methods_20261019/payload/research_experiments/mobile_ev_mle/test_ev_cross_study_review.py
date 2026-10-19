"""Analytic audit tests catch changed costs, bounds, units and pairing."""
from copy import deepcopy
import numpy as np
import pytest
from .ev_cross_study_review import (moving_geometry_audit,replay_qp,operation_audit,
                                   paired_identity_audit)
from . import ev_cross_study_review as review


def scalar_model():
    return dict(name='scalar',a='a',b='b',c=[1.],B=[[.2]],A0=[[1.]],At=[[[.4]]],reference_kw=[1.],
        C_at_reference=[[1.4]],reference_residual=[1.2],original_AC_global_certificate=False,
        domain=dict(base_center=[0.],shift=[[.1]],physical_lower=[-.5],physical_upper=[.5],
            action_lower=[0.],action_upper=[2.],outer_lower=[-.7],outer_upper=[.5]))


def test_moving_box_and_reference_closure_are_replayed():
    assert moving_geometry_audit(scalar_model())['columns']==1
    m=scalar_model();m['domain']['outer_lower']=[-.6]
    with pytest.raises(ValueError):moving_geometry_audit(m)


def test_original_fixed_QP_known_endpoint_cost_and_physical_map():
    m=scalar_model();saved=dict(z=[-.6],residual=[.36],primal_upper=.36**2,
        tangent_lower_numeric=.36**2,success=True,physical_parameter=[-.5],physical_bound_violation=0.)
    out=replay_qp(m,[1.],saved,domain='original')
    assert out['upper']==pytest.approx(.1296)
    assert out['lower']==pytest.approx(.1296,abs=1e-12)
    damaged=deepcopy(saved);damaged['primal_upper']=.3
    with pytest.raises(ValueError):replay_qp(m,[1.],damaged,domain='original')


def test_outer_QP_relaxation_analytic_cost_and_domain():
    m=scalar_model();saved=dict(z=[-.7],residual=[.22],q_primal_upper=.22**2,
        q_tangent_lower=.22**2,success=True)
    out=replay_qp(m,[1.],saved,domain='outer',kind='exchange')
    assert out['upper']==pytest.approx(.0484)
    assert out['upper']<.1296
    with pytest.raises(ValueError):replay_qp(m,[1.],saved,domain='original',kind='exchange')


def test_nonoptimal_inner_witness_uses_box_tangent_not_primal_as_lower():
    m=scalar_model();saved=dict(z=[0.],residual=[1.2],primal_upper=1.44,
        tangent_lower_numeric=0.,success=False)
    out=replay_qp(m,[1.],saved,domain='original')
    assert out['lower']==0. and out['upper']==pytest.approx(1.44)
    damaged=deepcopy(saved);damaged['tangent_lower_numeric']=1.44
    with pytest.raises(ValueError):replay_qp(m,[1.],damaged,domain='original')


def test_equal_energy_curve_and_noEV_diagnostic_are_distinct():
    cfg=dict(horizon=4,base_kw=20.,site=0,power_kw=6.,grid_energy_kwh=3.,eta=.95,terminal_battery_kwh=26.85)
    assert operation_audit([0,6,6,0],cfg)['grid_energy_kwh']==3.
    assert operation_audit([0]*4,cfg,diagnostic=True)['grid_energy_kwh']==0.
    with pytest.raises(ValueError):operation_audit([0]*4,cfg)
    wrong={**cfg,'base_kw':1000.}
    with pytest.raises(ValueError):operation_audit([3]*4,wrong)


def test_same_history_quality_repeats_count_once_and_reject_data_changes():
    first=dict(base_seed=2,quality=dict(kind='uniform'),history={'p':[1]},historical_fits=[],
        historical_failures={},truth={'p':[1]},white_future={'p':[0]},seeds={'p':2},correlation={'rho':.9})
    second=deepcopy(first);second['quality']['kind']='temporal'
    result=paired_identity_audit([first,second])
    assert result['independent_histories']==1 and result['quality_conditions']==2
    second['history']['p']=[2]
    with pytest.raises(ValueError):paired_identity_audit([first,second])


def guard_fixture(candidate_score,*,candidate_success=True,accepted=False):
    baseline=dict(u=[3]*4,score=1.,success=True,fit_calls=16)
    candidate=dict(u=[0,6,6,0],score=candidate_score,success=candidate_success,fit_calls=16)
    chosen=candidate['u'] if accepted else baseline['u']
    trial=dict(rows=[dict(method='constant',u_kw=baseline['u']),dict(method='cross_raw_milp',u_kw=candidate['u']),
                    dict(method='cross_guarded_milp',u_kw=chosen)])
    guard=dict(baseline=baseline,candidate=candidate,cache_hit=False,accepted=accepted,gain_tolerance=.01,
        selected_u=chosen,score_gain=None if not candidate_success else candidate_score-1.,totals=dict(fit_calls=32),reason='fixture')
    return trial,guard


@pytest.mark.parametrize('score,success,accepted',[(1.2,True,True),(1.001,True,False),(None,False,False)])
def test_guard_acceptance_checked_from_replayed_scores(monkeypatch,score,success,accepted):
    def fake_audit(trial,action,*args):
        return dict(success=action['success'],score=action['score'],saved_fit_states=16)
    monkeypatch.setattr(review,'guard_action_audit',fake_audit)
    trial,guard=guard_fixture(score,candidate_success=success,accepted=accepted)
    out=review.guard_audit(trial,guard,None,None,None,None,{})
    assert out['accepted']==accepted and out['unique_fit_states']==32
    broken=deepcopy(guard);broken['accepted']=not accepted
    with pytest.raises(ValueError):review.guard_audit(trial,broken,None,None,None,None,{})


def test_cached_guard_records_keep_unique_cost_and_selected_baseline(monkeypatch):
    def fake_audit(trial,action,*args):
        return dict(success=True,score=action['score'],saved_fit_states=16)
    monkeypatch.setattr(review,'guard_action_audit',fake_audit)
    trial,guard=guard_fixture(1.)
    guard.update(candidate=deepcopy(guard['baseline']),cache_hit=True,score_gain=0.,totals=dict(fit_calls=16))
    trial['rows'][1]['u_kw']=[3]*4
    out=review.guard_audit(trial,guard,None,None,None,None,{})
    assert out['saved_fit_states']==32 and out['unique_fit_states']==16 and not out['accepted']


def test_numerical_bound_inconsistency_retains_QP_replay_and_invalid_interval():
    m=scalar_model();qp=dict(z=[-.7],residual=[.22],q_primal_upper=.0484,q_tangent_lower=.0484,success=True)
    entry=dict(origin='known',u=[1.],qp_results=[qp],pair_score_lower=[.0242],pair_score_upper=[.0242],
               score_lower=.0242,score_upper=.0242)
    saved=dict(required_pairs=['scalar'],full_curve_enumeration=False,original_global_optimum_certified=False,
        offsets={'scalar':0.},score_scale=.5,evaluations=[entry],objective_lower=.0242,objective_upper=.01,
        u=[1.],best_evaluation_index=0,gap=0.,raw_gap=-.0142,status='numerical_bound_inconsistency',
        completed=False,failures=[dict(kind='numerical_bound_inconsistency')],initial_fixed_witness_upper=.7442,iterations=[])
    out=review.exchange_audit([m],saved,{'b':0.})
    assert out['qp_states']==1 and not out['valid_global_proxy_interval'] and out['raw_gap']<0
    broken=deepcopy(saved);broken.update(status='gap_reached',completed=True)
    with pytest.raises(ValueError):review.exchange_audit([m],broken,{'b':0.})


def test_v2_initial_QP_witness_bound_and_planning_gap_are_audited():
    m=scalar_model();qp=dict(z=[-.7],residual=[.22],q_primal_upper=.0484,q_tangent_lower=.0484,success=True)
    entry=dict(origin='initial_feasible_control',u=[1.],qp_results=[qp],pair_score_lower=[.0242],
        pair_score_upper=[.0242],score_lower=.0242,score_upper=.0242)
    saved=dict(required_pairs=['scalar'],full_curve_enumeration=False,original_global_optimum_certified=False,
        offsets={'scalar':0.},score_scale=.5,evaluations=[entry],objective_lower=.0242,objective_upper=.045,
        u=[1.],best_evaluation_index=0,gap=.0208,raw_gap=.0208,status='gap_reached_initial_witness_bound',
        completed=True,failures=[],initial_fixed_witness_upper=.045,midpoint_fixed_witness_upper=.7442,
        initialization='all required QP explanations at a feasible control; no midpoint witnesses',iterations=[],target_gap=.1)
    out=review.exchange_audit([m],saved,{'b':0.})
    assert out['initial_witness_upper_replayed']==pytest.approx(.045,abs=1e-12)
    assert out['valid_global_proxy_interval']
    broken=deepcopy(saved);broken['target_gap']=.01
    with pytest.raises(ValueError):review.exchange_audit([m],broken,{'b':0.})


def test_call_counts_charge_cached_curve_once_and_separate_history():
    none=dict(method='none',u_kw=[0]*4,site=0,validation=dict(feasible=True),cache_hit=False,
              fits=[dict(name='a')],cost=dict(fit_calls=4,ac_calls=10))
    constant=dict(method='constant',u_kw=[3]*4,site=0,validation=dict(feasible=True),cache_hit=False,
                  fits=[dict(name='b')],cost=dict(fit_calls=4,ac_calls=12))
    duplicate=deepcopy(constant);duplicate.update(method='cross_raw_milp',cache_hit=True)
    trial=dict(config=dict(base_kw=20.),historical_cost=dict(fit_calls=4,ac_calls=99),evaluation_ac_calls=22,
        selection_ac_calls=148,selection_forecast_calls=4,rows=[none,constant,duplicate],
        selection=dict(context_ac_calls=48,ac_guard=dict(totals=dict(fit_calls=16,fit_ac_calls=96,forecast_calls=4,ac_calls=100)),
            fixed_solver=dict(n_binary=8),cross_solver=dict(iterations=[],qp_calls=12),curves_audit=dict(curve=[None]*12)))
    out=review.cost_audit(trial)
    assert out['evaluation_executed_fit_calls']==8 and out['evaluation_executed_AC_calls']==22
    assert out['evaluation_action_cache_hits']==1 and out['historical_cost_record']['fit_calls']==4
    assert out['fixed_chord_solver_calls']==1 and out['cross_chord_solver_calls']==0
    assert out['guard_unique_forecast_AC_calls']==4 and out['evaluator_future_truth_forward_attempts']==3
    broken=deepcopy(trial);broken['rows'][2]['cache_hit']=False
    with pytest.raises(ValueError):review.cost_audit(broken)


def test_tight_master_affine_extrema_and_secant_error_are_audited_without_LP():
    m=dict(name='four',a='a',b='b',c=[1.],B=[[1.,-1.,2.,-2.]],A0=[[1.]],
        At=np.zeros((4,1,1)).tolist(),domain=dict(outer_lower=[-1.],outer_upper=[1.]))
    exchange=dict(score_scale=.5,witnesses=[[[0.]]])
    master=dict(interval_audits=[dict(name='four__w0',residual_lower=[-17*np.sqrt(.5)],residual_upper=[19*np.sqrt(.5)])],
        segments=4,coarse_upper=180.5+1e-12,numeric_guard=1e-12,
        secant_error_by_quadratic=[10.125],u=[3.]*4,objective_lower=.5)
    item=dict(master=master,compressions=[dict(pair='four',witness=0,box_numerical_upward_guard=0.)])
    out=review.tight_master_audit([m],exchange,item,{'b':0.})
    assert out['analytic_feasible_affine_extrema_checked'] and out['quadratics']==1
    broken=deepcopy(item);broken['master']['interval_audits'][0]['residual_lower'][0]+=.1
    with pytest.raises(ValueError):review.tight_master_audit([m],exchange,broken,{'b':0.})
