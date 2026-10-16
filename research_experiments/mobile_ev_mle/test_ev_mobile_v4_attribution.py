"""Predeclared offline attribution audits using synthetic saved records only."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from . import ev_mobile_v4_attribution as a
from .information_context import meter_to_record
from .likelihood import MeasurementModel, Observations


def _meta(index=0):
    meter=MeasurementModel(p_sigma=.1,q_sigma=.1,v_sigma=.1,master_p_sigma=.1,master_q_sigma=.1)
    obs=Observations(p=[[.1,.2,.3]],q=[[.03,.06,.09]],v=[[1.,1.,1.]],root_v=[1.],master_p=[.6],master_q=[.18])
    parameters=dict(r=[.01]*4,x=[.003]*4,p=[[.1,.2,.3]],q=[[.03,.06,.09]],sp=[.1],sq=[.03],root=[1.])
    return dict(spec=dict(index=index,bank='n3',seed=23),true_name='star',candidate_names=['star','b','c','d'],
        observed=a.io.observations_record(obs),observed_meter=meter_to_record(meter),future_meter=meter_to_record(meter),
        trained={k:dict(r=[.01]*4,x=[.003]*4,source_p=[.1],source_q=[.03]) for k in ['star','b','c','d']},
        candidate_bank=[dict(name='star',n=3,supports=[])],
        predicted=parameters,true_by_regime={r:deepcopy(parameters) for r in a.REGIMES},
        powers={m:[[0.,0.,0.]] for m in a.METHODS},base_kw=20.,fit_options=dict(r_bounds=[.001,.1]),fit_seed=4023,
        nominal={a.METHODS[0]:dict(Phi=1.,true_ref_m=1.),a.METHODS[1]:dict(Phi=2.,true_ref_m=2.)},
        observed_halfM={r:{a.METHODS[0]:1.,a.METHODS[1]:2.} for r in a.REGIMES},
        selected_MD_highest_Phi_direct_parent='mobile_charge',FD_is_selected_guard_anchor=False)


def _context(meta):
    fm=a.meter_from_record(meta['future_meter']);obs=a._observation(meta['observed'])
    return dict(candidate=None,bank=meta['candidate_names'],D=obs,dm=fm,fm=fm,fits={})


def _outcome(complete=True):
    if not complete:return dict(complete=False,comparisons={},losses={},failures={k:'failed fit' for k in ['star','b','c','d']})
    values=dict(star=2.,b=4.,c=8.,d=12.)
    return dict(complete=True,losses=values,comparisons={k:dict(joint=dict(success=True,loss=v),fixed_source_success=True)
        for k,v in values.items()},failures={})


def _fake_target(meta,calls):
    def make(candidate,parameters,power,meter,**kwargs):
        calls.append((parameters,power))
        return a._observation(meta['observed']),dict(EV_in_total_PQ_once=True,future_white_noise=False)
    return make


def test_complete_phase_gate_requires_all_eight_terminal_files_and_source_identity(tmp_path):
    frozen=dict(policy_id='policy',source_sha256={'science.py':'hash'},
        config=dict(posthoc_diagnostic_evaluation_indices=[0,6]),
        specifications=dict(evaluation=[dict(index=i,seed=100+i,regimes=['regular','shift']) for i in range(8)]))
    a._write(tmp_path/'freeze.json',frozen)
    for spec in frozen['specifications']['evaluation'][:7]:
        a._write(tmp_path/'evaluation'/(str(spec['index']).zfill(3)+'.json.gz'),dict(spec=spec,policy_id='policy',status='complete'))
    with pytest.raises(ValueError,match='not finished'):a.readiness(tmp_path,current_sources=frozen['source_sha256'])
    spec=frozen['specifications']['evaluation'][7]
    a._write(tmp_path/'evaluation'/'007.json.gz',dict(spec=spec,policy_id='policy',status='fatal_failure'))
    _,manifest,selected=a.readiness(tmp_path,current_sources=frozen['source_sha256'])
    assert len(manifest)==8 and set(selected)=={0,6}
    with pytest.raises(ValueError,match='source differs'):a.readiness(tmp_path,current_sources={'science.py':'changed'})
    frozen['config']['posthoc_diagnostic_evaluation_indices']=[0,1]
    a._write(tmp_path/'freeze.json',frozen)
    with pytest.raises(ValueError,match='predeclared'):a.readiness(tmp_path,current_sources=frozen['source_sha256'])


def test_exact_targets_cache_all_64_slots_but_keep_256_candidate_denominator(tmp_path,monkeypatch):
    calls=[];fits=[]
    for index in a.INDICES:
        meta=_meta(index);a._write(tmp_path/f'history_{index:03d}'/'history.json.gz',meta)
        monkeypatch.setattr(a,'make_target',_fake_target(meta,calls))
        def evaluate(*args):fits.append(args);return _outcome()
        for regime,bits,method in product(a.REGIMES,a.STATES,a.METHODS):
            a.evaluate_slot(tmp_path,meta,_context(meta),regime,bits,method,evaluate=evaluate)
    summary=a.summarize(tmp_path)
    assert summary['target_slot_denominator']==64 and summary['completed_target_slots']==64
    assert summary['candidate_slots']==dict(expected=256,attempted=256,success=256,failed=0,not_run=0)
    assert summary['unique_evaluation_calls']==1 and summary['unique_attempted_joint_candidate_posts']==4
    assert summary['cached_slot_reuses']==63 and len(fits)==1 and len(calls)==1
    assert summary['complete_factor_groups']==4
    assert all(e['full_gap_decomposition_error']==0. for e in summary['events'])
    # The fake fit call receives saved D/options/seed, not a selector or truth.
    assert fits[0][-2]==_meta()['fit_options'] and fits[0][-1]==4023


def test_failed_fit_is_cached_and_resume_never_silently_retries(tmp_path,monkeypatch):
    meta=_meta();context=_context(meta);calls=[];fits=[]
    monkeypatch.setattr(a,'make_target',_fake_target(meta,calls))
    def evaluate(*args):fits.append(args);return _outcome(False)
    first=a.evaluate_slot(tmp_path,meta,context,'regular',(0,0,0),a.METHODS[0],evaluate=evaluate)
    again=a.evaluate_slot(tmp_path,meta,context,'regular',(0,0,0),a.METHODS[0],evaluate=evaluate)
    other=a.evaluate_slot(tmp_path,meta,context,'regular',(0,0,0),a.METHODS[1],evaluate=evaluate)
    assert first==again and first['status']=='fit_failure'
    assert other['cache_reused'] and len(fits)==1
    assert first['candidate_slots']==dict(expected=4,attempted=4,success=0,failed=4,not_run=0)


def test_target_generation_failure_preserves_unrun_candidate_denominator(tmp_path,monkeypatch):
    meta=_meta();a._write(tmp_path/'history_000'/'history.json.gz',meta)
    def fail(*args,**kwargs):raise ValueError('invalid physical target')
    monkeypatch.setattr(a,'make_target',fail)
    def should_not_fit(*args):raise AssertionError('fit called despite target failure')
    row=a.evaluate_slot(tmp_path,meta,_context(meta),'regular',(0,0,0),a.METHODS[0],evaluate=should_not_fit)
    assert row['status']=='target_failure'
    summary=a.summarize(tmp_path)
    assert summary['completed_target_slots']==1 and summary['candidate_slots']['expected']==256
    assert summary['candidate_slots']['not_run']==256 and summary['unique_evaluation_calls']==0


def test_shapley_effects_and_full_score_gap_accounting_are_exact():
    meta=_meta();FD,MD=a.METHODS
    meta['nominal']={FD:dict(Phi=1.,true_ref_m=1.),MD:dict(Phi=5.,true_ref_m=7.)}
    meta['observed_halfM']['regular']={FD:0.,MD:26.}
    slots={}
    for bits in a.STATES:
        L,B,S=bits;delta=4.+2*L-3*B+5*S+6*L*B+9*L*B*S
        key=''.join(map(str,bits))
        for method,value in [(FD,10.),(MD,10.+delta)]:
            slots['regular',method,key]=dict(status='complete',halfM=value)
    event=a.event_summary(meta,slots,'regular')
    assert event['success'] and event['delta_000']==4. and event['delta_111']==23.
    assert event['Shapley_delta_halfM']['contributions']==pytest.approx(dict(lines=8.,background=3.,source=8.))
    assert event['metric_aggregation_discrepancy']==2.
    assert event['nominal_vs_same_inference_rule_discrepancy']==-2.
    assert event['observed_minus_111_residual']==3.
    assert event['full_gap_decomposition_error']==0.
    assert event['FD_is_selected_guard_anchor'] is False


def test_incomplete_factor_is_not_replaced_by_zero_or_allocated_to_shapley():
    meta=_meta();slots={}
    for bits,method in product(a.STATES,a.METHODS):
        slots['regular',method,''.join(map(str,bits))]=dict(status='complete',halfM=2.)
    slots['regular',a.METHODS[1],'111']=dict(status='fit_failure',halfM=None)
    event=a.event_summary(meta,slots,'regular')
    assert not event['success']
    assert event['states']['111']['delta_halfM'] is None
    assert 'Shapley_delta_halfM' not in event


def test_target_identity_changes_only_for_relevant_fixed_inputs():
    meta=_meta();parameters=meta['predicted'];power=meta['powers'][a.METHODS[0]]
    original=a.target_identity(meta,parameters,power)
    altered=deepcopy(parameters);altered['r'][0]+=.001
    assert a.target_identity(meta,altered,power)!=original
    same=deepcopy(meta);same['spec']['index']=6
    assert a.target_identity(same,parameters,power)==original
    same['fit_seed']+=1
    assert a.fit_identity(same,meta['observed'])!=a.fit_identity(meta,meta['observed'])


def test_no_source_or_line_parameter_refit_happens_during_restore():
    meta=_meta();meta['candidate_names']=['star']
    context=a.restore(meta)
    assert context['candidate'].name=='star'
    assert context['D'].t==1 and context['D'].n==3
    np.testing.assert_array_equal(context['fits']['star'].r,meta['trained']['star']['r'])


def test_outcome_needs_all_four_successful_candidates_and_keeps_negative_margin():
    outcome=_outcome();outcome['comparisons']['star']['joint']['loss']=5.
    value=a.outcome_values(outcome,'star',['star','b','c','d'])
    assert value['halfM']==-.5 and value['complete']
    outcome['comparisons']['d']['joint']['success']=False
    value=a.outcome_values(outcome,'star',['star','b','c','d'])
    assert not value['complete'] and value['halfM'] is None
    assert value['candidate_slots']['failed']==1
