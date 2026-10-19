"""Compact-location plumbing; test doubles never run a selector or AC fit."""
from copy import deepcopy
import json
import numpy as np
import pytest
from . import ev_compact_location_audit as a
from . import test_ev_original_location_ablation as previous


def fixture():
    t=previous.fixture();physical=t['config']
    physical['device'].update(power_kw=6.,capacity_kwh=60.,throughput_kwh=7.5)
    t.update(phase='development',policy_id='p',source_interface=1,
        config=dict(physical=physical,command_power_kw=6.,development_seeds=[1,2,3,4],evaluation_seeds=list(range(11,23))),
        operations_config=deepcopy(physical))
    oldmethods=list(t['selection']['methods']);newmethods={}
    for old,new in zip(oldmethods,a.METHODS):
        record=t['selection']['methods'][old]
        record['selected_origin']='v2g_fixed';record['guard']['accepted']=False
        record['operation']['cost_units']='worker_minutes'
        newmethods[new]=record
        next(r for r in t['rows'] if r['method']==old)['method']=new
    t['selection']['methods']=newmethods
    t['selection']['constant_operation']=dict(driver_cost=999.,travel_energy_kwh=99.)
    return t


def stationary(t):
    for row in t['rows']:
        row['power_kw']=a.old.remap_to_site0(row['power_kw'])
        t['selection']['methods'][row['method']]['operation']['u_kw']=row['power_kw']


def test_compact_view_keeps_parent_config_and_fixed_source_interface():
    t=fixture();v=a.compact_view(t)
    assert v['config'] is t['config']['physical'] and 'physical' in t['config']
    assert v['config']['history']==1
    t['source_interface']=2
    with pytest.raises(ValueError,match='interface'):a.compact_view(t)


def test_inherited_operation_is_not_replaced_by_constant_when_guard_rejects_new_proposal():
    t=fixture();scope=a.resource_audit(t,a.METHODS[0],t['rows'][0]['power_kw'])
    assert not scope['original_guard_accepted'] and scope['original_operation_feasible']
    assert scope['original_drive_kwh']==.2 and scope['original_worker_cost']==30.
    assert scope['remap_terminal_excess_over_original_target_kwh']==pytest.approx(.2)
    assert scope['terminal_minus_drive_identity_error_kwh']==pytest.approx(0.)
    assert not scope['remap_same_terminal_service_feasible']
    assert not scope['equal_service_comparison']


def test_operation_curve_mismatch_is_preserved_as_failure():
    t=fixture();t['selection']['methods'][a.METHODS[0]]['operation']['u_kw']=np.zeros((2,3))
    scope=a.resource_audit(t,a.METHODS[0],t['rows'][0]['power_kw'])
    assert not scope['original_curve_matches_operation'] and not scope['original_operation_feasible']


def test_stationary_conditions_reuse_original_fits_and_have_zero_paired_difference(monkeypatch):
    t=fixture();stationary(t)
    def forbidden(*args,**kwargs):raise AssertionError('new AC/fit for unchanged curve')
    monkeypatch.setattr(a.old.lk,'ac_predict',forbidden)
    monkeypatch.setattr(a.old.io,'fit_all_candidates',forbidden)
    r=a.process_history(t)
    assert len(r['conditions'])==2 and all(c['status']=='complete' for c in r['conditions'])
    assert all(c['stationary_exact_reuse'] and c['delta_M']==c['delta_noiseless_M']==0. for c in r['conditions'])
    assert r['counts']==dict(forward_AC_calls=0,candidate_fit_calls=0,fit_AC_calls=0,actual_returned_fit_calls=0)
    assert all(len(c['remap']['noisy']['pairwise_half_SSE_differences'])==12 for c in r['conditions'])
    assert not r['statistical_threshold_applied']


def test_new_remap_uses_same_background_noise_one_EV_addition_and_shared_H_N_fit(monkeypatch):
    t=fixture();captures=[];fits=[]
    def forward(candidate,r,x,p,q,sp,sq,root):
        captures.append(np.asarray(p).copy())
        return dict(p=p,q=q,v=np.ones((2,3)),master_p=np.asarray(p).sum(axis=1)+sp+.01,
            master_q=np.asarray(q).sum(axis=1)+sq,root_v=root,ac_equation_residual=0.,power_balance_residual=0.)
    def fit(bank,joint,meter,**kwargs):
        fits.append((joint,kwargs));records={r['name']:deepcopy(r) for r in t['historical_fits']}
        for r in records.values():r['predicted']={k:np.asarray(getattr(joint,k)) for k in ('p','q','v','master_p','master_q')}
        return records,{},dict(fit_calls=4,ac_calls=7,seconds=.001)
    monkeypatch.setattr(a.old.lk,'ac_predict',forward);monkeypatch.setattr(a.old.io,'fit_all_candidates',fit)
    monkeypatch.setattr(a.old.io,'fit_record',lambda candidate,fit:fit)
    result=a.process_history(t)
    assert result['counts']==dict(forward_AC_calls=1,candidate_fit_calls=8,fit_AC_calls=14,actual_returned_fit_calls=8)
    expected=np.asarray(t['truth']['p'])[1:].copy();expected[:,0]+=.1
    assert captures[0]==pytest.approx(expected)
    assert len(fits)==2 and all(joint.t==3 and len(kw['initial'])==4 for joint,kw in fits)
    assert [kw['seed'] for joint,kw in fits]==[902,1302]
    assert all(c['meter_noise_audit']['passed'] for c in result['conditions'])
    assert result['conditions'][1]['remap']['noisy']['cache_hit']
    assert result['new_selector_runs']==0


def test_failed_original_or_missing_action_keeps_two_condition_denominator(monkeypatch):
    t=fixture();stationary(t);t['rows'][0]['power_kw']=None
    result=a.process_history(t)
    assert len(result['conditions'])==2 and result['conditions'][0]['status']=='missing_original_action'
    assert result['conditions'][0]['delta_M'] is None
    t=fixture();stationary(t);t['rows'][0]['fits']=t['rows'][0]['fits'][:3]
    result=a.process_history(t)
    failed=result['conditions'][0]
    assert failed['status']!='complete' and failed['delta_M'] is None
    assert len(failed['original']['noisy']['candidate_costs'])==4
    assert len(failed['original']['noisy']['pairwise_half_SSE_differences'])==12


def test_prepare_manifest_requires_all_parent_histories_and_binds_review_and_sources(tmp_path):
    directory,output=tmp_path/'dev',tmp_path/'audit';directory.mkdir()
    t=fixture();t['source_sha256']={'likelihood.py':a.old.file_digest(a.HERE/'likelihood.py')}
    parent=dict(phase='development',seeds=[1,2,3,4],policy_id='p',config=t['config'],source_sha256=t['source_sha256'])
    a.old.write_json(directory/'freeze.json',parent)
    a.old.write_json(directory/'trial_1.json.gz',t)
    with pytest.raises(ValueError,match='all parent'):a.prepare_manifest(directory,output)
    for seed in (2,3,4):a.old.write_json(directory/f'trial_{seed}.json.gz',dict(t,seed=seed))
    review=dict(success=True,policy_id='p',seeds=[1,2,3,4],trial_sha256={f'trial_{s}.json.gz':a.old.file_digest(directory/f'trial_{s}.json.gz') for s in (1,2,3,4)})
    a.old.write_json(tmp_path/'review.json',review)
    _,freeze,paths=a.prepare_manifest(directory,output,parent_review=tmp_path/'review.json')
    assert freeze['planned_conditions']==8 and len(paths)==4 and a.MATH in freeze['source_sha256']
    assert freeze['parent_review_sha256']==a.old.file_digest(tmp_path/'review.json')
    review['trial_sha256']['trial_1.json.gz']='wrong';a.old.write_json(tmp_path/'bad_review.json',review)
    with pytest.raises(ValueError,match='exact histories'):a.prepare_manifest(directory,tmp_path/'bad',parent_review=tmp_path/'bad_review.json')


def test_summary_keeps_method_specific_H_denominators_and_no_rejection_threshold(tmp_path,monkeypatch):
    t=fixture();stationary(t);result=a.process_history(t)
    a.old.write_json(tmp_path/'history_1.json.gz',result)
    freeze=dict(seeds=[1,2,3,4],phase='development',planned_histories=4,planned_conditions=8,parameter_sha256='param')
    summary=a.save_summary(tmp_path,freeze)
    assert summary['available_histories']==1 and summary['available_conditions']==2 and summary['planned_conditions']==8
    assert all(r['history_denominator']==1 and r['planned_histories']==4 for r in summary['methods'].values())
    assert summary['formal_calibration_histories']==0 and not summary['statistical_threshold_applied']
