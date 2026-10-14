"""Saved-cost examples for baseline choice, order of minima and missing data."""
from copy import deepcopy
import gzip
import json
import pytest
from . import ev_compact_nearest_diagnostic as n


NAMES = ('A','B','C')


def prediction(h=None):
    h = h or dict(A=2.,B=20.,C=0.)
    pairs=[]
    for a in NAMES:
        for b in NAMES:
            if a == b:continue
            for s in (0,1):
                q = h[b]+4.
                if a == 'A' and b == 'B':q=22.+2*s
                if a == 'A' and b == 'C':q=10.+2*s
                qa=h[a]+1.
                pairs.append(dict(name=f'{a}_{b}_{s}',a=a,b=b,ensemble_id=str(s),weight=1/12,
                    success=True,joint_loss=q,self_loss=qa,added=.5*(q-h[b]),
                    joint=.5*(q-h[a]),self_relative=.5*(q-qa)))
    return dict(success=True,pairs=pairs,power_kw=[[1.,0.,0.]]),h


def test_competitor_h_b_and_reference_h_a_choose_different_nearest_models():
    p,h=prediction();result=n.score_prediction(p,h,NAMES)
    assert result['success'] and result['self_success']
    ref=next(r for r in result['reference_minima'] if r['a']=='A')
    assert ref['nearest_added_h_b_candidates']==['B']
    assert ref['nearest_added_h_b']==1.5
    assert ref['nearest_joint_h_a_candidates']==['C']
    assert ref['nearest_joint_h_a']==4.5
    assert result['scores']['phi_add']==pytest.approx((1.5+2+2)/3)
    # Every task retains h_a/h_b and its explicit saved competitor/self cost.
    assert len(result['task_components'])==12
    task=result['task_components'][0]
    assert task['joint_h_a']-task['added_h_b']==pytest.approx(.5*(task['h_b']-task['h_a']))


def test_scenarios_are_averaged_before_minimum_not_after():
    p,h=prediction(dict(A=0.,B=0.,C=0.))
    for r in p['pairs']:
        if r['a']=='A':
            d=0. if (r['b']=='B')==(r['ensemble_id']=='0') else 100.
            r.update(joint_loss=2*d,added=d,joint=d,self_relative=d-.5)
    result=n.score_prediction(p,h,NAMES)
    ref=next(r for r in result['reference_minima'] if r['a']=='A')
    assert ref['nearest_added_h_b']==pytest.approx(50.)
    assert set(ref['nearest_added_h_b_candidates'])=={'B','C'}
    # Taking each scenario's nearest first would give zero here.


def test_missing_difficult_task_does_not_increase_score_by_deletion():
    p,h=prediction();p['pairs'].pop(0)
    result=n.score_prediction(p,h,NAMES)
    assert not result['success'] and result['scores']['phi_add'] is None
    assert any('scenario_law' in e for e in result['errors'])


def test_no_q_aa_is_explicitly_missing_even_if_self_relative_scalar_was_saved():
    p,h=prediction()
    for row in p['pairs']:row.pop('self_loss')
    result=n.score_prediction(p,h,NAMES)
    assert result['success'] and not result['self_success']
    assert result['scores']['phi_H'] is not None and result['scores']['phi_self'] is None
    assert all(r['q_aa'] is None for r in result['task_components'])
    assert len(result['self_errors'])==12


def test_explicit_saved_self_fit_is_usable_without_pair_copy():
    p,h=prediction()
    for row in p['pairs']:row.pop('self_loss')
    p['records']=[dict(a=a,ensemble_id=str(s),fits=[dict(name=a,loss=h[a]+1.,success=True)])
                  for a in NAMES for s in (0,1)]
    result=n.score_prediction(p,h,NAMES)
    assert result['self_success']
    assert all(r['q_aa_source']=='saved_self_fit.loss' for r in result['task_components'])


def test_duplicate_saved_fit_cannot_silently_overwrite_the_same_physical_case():
    p,h=prediction()
    own=dict(name='A',loss=3.,success=True)
    p['records']=[dict(a='A',ensemble_id='0',fits=[own,dict(own,loss=999.)])]
    result=n.score_prediction(p,h,NAMES)
    assert not result['success']
    assert any('duplicate_saved_prediction_fit' in error for error in result['errors'])


@pytest.mark.parametrize('change',['weight','failure','cost_mismatch','duplicate'])
def test_invalid_or_inconsistent_task_sets_remain_unavailable(change):
    p,h=prediction()
    if change=='weight':p['pairs'][0]['weight']+=.01;p['pairs'][2]['weight']-=.01
    if change=='failure':p['pairs'][0]['success']=False
    if change=='cost_mismatch':p['pairs'][0]['added']+=1.
    if change=='duplicate':p['pairs'].append(deepcopy(p['pairs'][0]))
    result=n.score_prediction(p,h,NAMES)
    assert not result['success'] and result['scores']['phi_H'] is None
    assert result['errors']


def test_negative_scores_are_not_clipped_and_cap_applies_before_average():
    p,h=prediction()
    r=p['pairs'][0];r.update(joint_loss=0.,added=-10.,joint=-1.,self_relative=-1.5)
    result=n.score_prediction(p,h,NAMES)
    assert result['success']
    assert result['task_components'][0]['added_h_b']==-10.
    assert result['scores']['cap20_added']<=result['scores']['mean_added']


def test_saved_curve_matching_never_borrows_a_different_action_or_future_result():
    p,h=prediction()
    trial=dict(selection=dict(predicted_AC={},all_prediction_records=[p]))
    assert n.saved_prediction(trial,'quality',[[1.,0.,0.]])[0] is p
    assert n.saved_prediction(trial,'random',[[0.,1.,0.]])[0] is None
    trial['selection']['predicted_AC']['quality']=p
    assert n.saved_prediction(trial,'quality',[[0.,1.,0.]])[1]['reason']=='named_prediction_curve_mismatch'


def test_rank_ties_missing_methods_and_constant_vectors_have_explicit_scope():
    assert n.ranks(dict(a=2.,b=2.,c=0.))==dict(a=1.5,b=1.5,c=3.)
    corr=n.spearman(dict(a=1.,b=2.,c=None),dict(a=2.,b=1.,c=8.))
    assert corr['n_methods']==2 and corr['value']==-1.
    assert n.spearman(dict(a=1.,b=1.),dict(a=2.,b=3.))['reason']=='constant_rank_vector'


def test_anchor_rule_prevents_compensation_between_references_without_truth():
    methods=[dict(method='charge_mobile',self_success=True,actual_M=1.),
             dict(method='v2g_mobile',self_success=True,actual_M=999.)]
    refs=[dict(method=m,a=a,nearest_self_relative=v) for m,vs in
          [('charge_mobile',[2.,2.,2.]),('v2g_mobile',[1.,100.,100.])]
          for a,v in zip(NAMES,vs)]
    checked=n.anchor_diagnostic(methods,refs,NAMES)
    d={r['method']:r for r in checked['decisions']}
    assert d['charge_mobile']['decision']=='accepted'
    assert d['v2g_mobile']['decision']=='rejected' and d['v2g_mobile']['violating_references']==['A']
    methods[0]['actual_M']=1e9;methods[1]['actual_M']=-1e9
    assert n.anchor_diagnostic(methods,refs,NAMES)==checked


def test_anchor_rule_requires_all_reference_predictions_and_preserves_unknown():
    methods=[dict(method='charge_mobile',self_success=True),dict(method='candidate',self_success=True)]
    refs=[dict(method=m,a=a,nearest_self_relative=2.) for m in ('charge_mobile','candidate') for a in NAMES]
    refs=[r for r in refs if not (r['method']=='candidate' and r['a']=='C')]
    checked=n.anchor_diagnostic(methods,refs,NAMES)
    assert checked['decisions'][1]['decision']=='unknown'
    refs=[r for r in refs if not (r['method']=='charge_mobile' and r['a']=='B')]
    checked=n.anchor_diagnostic(methods,refs,NAMES)
    assert all(r['decision']=='unknown' for r in checked['decisions'])


def test_directory_retains_eleven_methods_and_unscored_controls(tmp_path):
    directory,output=tmp_path/'dev',tmp_path/'diagnostic';directory.mkdir()
    (directory/'freeze.json').write_text(json.dumps(dict(phase='development',policy_id='p',seeds=[1])),encoding='utf-8')
    p,h=prediction()
    methods=['none','constant','quality_fixed','quality_mobile','random_mobile',
        'charge_fixed','v2g_fixed','charge_mobile','v2g_mobile','refined_v2g_mobile','capped_v2g_mobile']
    ownfits=[dict(name=a,loss=1. if a=='A' else 2.,success=True) for a in NAMES]
    rows=[dict(method=m,power_kw=[[0.,0.,0.]] if m in ('none','random_mobile') else p['power_kw'],
        status='complete',true_name='A',fits=ownfits,noiseless_AC_refit=dict(all_candidates=True,fits=ownfits)) for m in methods]
    trial=dict(phase='development',seed=1,policy_id='p',proposal=dict(bank=[dict(name=a) for a in NAMES]),
        historical_fits=[dict(name=a,loss=h[a],success=True) for a in NAMES],rows=rows,
        delivery=dict(declared_methods=methods,source_raw_sha256='rawhash'),
        selection=dict(all_prediction_records=[p]))
    raw=json.dumps(trial).encode()
    (directory/'trial_1.slim.json.gz').write_bytes(gzip.compress(raw))
    result=n.analyze_directory(directory,output)
    assert result['method_conditions']==11 and result['prediction_available']==result['self_available']==9
    assert {r['method'] for r in result['unavailable']}=={'none','random_mobile'}
    assert result['new_fits']==result['new_optimizations']==0
    assert result['input_manifest'][0]['source_raw_sha256']=='rawhash'
    trial['phase']='evaluation'
    with pytest.raises(ValueError,match='development'):
        n.analyze_trial(trial)
    (directory/'freeze.json').write_text(json.dumps(dict(phase='calibration')),encoding='utf-8')
    with pytest.raises(ValueError,match='development'):
        n.analyze_directory(directory,output)
