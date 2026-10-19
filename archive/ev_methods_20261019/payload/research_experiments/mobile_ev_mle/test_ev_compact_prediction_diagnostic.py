"""Offline aggregation/forecast arithmetic tests; no simulator or fitter."""
from copy import deepcopy
import pytest
from . import ev_compact_prediction_diagnostic as d
from .test_ev_compact_analysis import trial


def sample():
    t=trial();context=dict(id='merged-id-without-axis-label',weight=1.,axis=0,scenario_id='center',
        alias_records=[dict(axis=0,scenario_id='center',weight=.5),
            dict(axis=1,scenario_id='center',weight=.5,missing_flat_fallback_candidates=['star'])],
        explanation_ids={name:name+'__center' for name in ('star','c01','c02','c12')})
    t['selection']['context']={'context_records':[context]}
    for prediction in t['selection']['predicted_AC'].values():
        for i,pair in enumerate(prediction['pairs']):
            pair.update(name=str(i),ensemble_id=context['id'],added=2.,self_relative=2.,joint=2.)
    return t,context


def test_exact_dedup_aliases_split_weight_without_inventing_independent_samples():
    t,context=sample();prediction=t['selection']['predicted_AC']['charge_mobile']
    rows=d.expand_tasks(prediction,[context]);assert len(rows)==24
    assert sum(row['weight'] for row in rows)==pytest.approx(1.)
    assert d.score_group(rows)['unique_task_count']==12
    for axis in (0,1):
        group=d.score_group([r for r in rows if r['axis']==axis])
        assert group['task_weight']==pytest.approx(.5) and group['mean_added']==2.
    true=[r for r in rows if r['a']=='star']
    assert sum(r['weight'] for r in true)==pytest.approx(.25)
    assert sum(r['flat_center_fallback'] for r in true)==3


def test_global_reference_tradeoff_can_increase_mean_and_decrease_true_reference():
    t,_=sample();prediction=t['selection']['predicted_AC']['v2g_mobile']
    for row in prediction['pairs']:
        row['added']=row['self_relative']=.5 if row['a']=='star' else 10.
    actual=next(r for r in t['rows'] if r['method']=='v2g_mobile')
    for fit in actual['fits']:
        if fit['name']!='star':fit['loss']=1.5
    for fit in actual['noiseless_AC_refit']['fits']:
        if fit['name']!='star':fit['loss']=1.5
    result=d.analyze_trial(t,label='dev');contrast=next(r for r in result['contrast_rows'] if r['method']=='v2g_mobile')
    assert contrast['delta_global_mean']>0 and contrast['delta_true_reference_mean']<0
    assert contrast['delta_actual_M']<0 and contrast['delta_noiseless_M']<0
    assert contrast['mean_actual_reversal'] and contrast['true_reference_mean_also_decreases']
    assert all(row['truth_used'].startswith('offline') for row in result['method_rows'])


def test_mean_competitor_and_hardest_competitor_can_move_in_opposite_directions():
    t,_=sample();prediction=t['selection']['predicted_AC']['v2g_mobile']
    for row in prediction['pairs']:
        row['added']=row['self_relative']=.1 if row['a']=='star' and row['b']=='c01' else 10.
    result=d.analyze_trial(t,label='dev');contrast=next(r for r in result['contrast_rows'] if r['method']=='v2g_mobile')
    assert contrast['delta_true_reference_mean']>0
    assert contrast['delta_true_reference_min_comp_mean_self_relative']==pytest.approx(-1.9)
    assert contrast['hard_competitor_mean_also_decreases']
    method=next(r for r in result['method_rows'] if r['method']=='v2g_mobile')
    assert method['true_reference_hardest_competitor']=='c01'


def test_failed_task_retained_and_group_mean_not_partial_average():
    t,context=sample();prediction=t['selection']['predicted_AC']['charge_mobile']
    prediction['pairs'][0].update(success=False,added=None)
    rows=d.expand_tasks(prediction,[context]);score=d.score_group(rows)
    assert score['failed_or_missing']==2 and score['unique_task_count']==12
    assert score['mean_added'] is None and not score['complete']


def test_saved_forecast_error_is_whitened_and_true_source_only_offline():
    t,context=sample();t['history']['root_v']=[1.]
    t['measurement_models']['fm']={'v_sigma':.001,'p_sigma':.1,'q_sigma':.1}
    t['truth']['source_p']=[.2,.2,.2]
    prediction=t['selection']['predicted_AC']['charge_mobile']
    prediction['records']=[dict(a='star',ensemble_id=context['id'],explanation_id='star__center',success=True,
        forecast_state={'source_p':[.3,.3]},predicted_observations={'v':[[1.002]*3]*2,'p':[[1.1]*3]*2})]
    row=next(r for r in t['rows'] if r['method']=='charge_mobile')
    row['noiseless_AC_refit']['future_mean_observations']={'v':[[1.]*3]*2,'p':[[1.]*3]*2}
    result=d.analyze_trial(t,label='dev');forecast=result['forecast_rows'][0]
    assert forecast['forecast_actual_v_white_RMS']==pytest.approx(2.)
    assert forecast['forecast_actual_p_white_RMS']==pytest.approx(1.)
    assert forecast['forecast_actual_source_p_RMS_kw_kvar']==pytest.approx(2.)


def test_raw_proxy_check_not_attributed_to_inherited_executed_curve():
    t,_=sample();record=t['selection']['methods']['v2g_mobile']
    record['raw']['u']=[0.,2.,0.,0.,2.,0.]
    record['raw']['required_pairs']=['pair'];record['raw']['weights']={'pair':1.}
    t['historical_fits']=[dict(name='c01',loss=2.,success=True)]
    record['raw']['evaluations']=[dict(origin='master_0',mean_score_interval=[3.,3.])]
    record['raw']['best_evaluation_index']=0
    record['raw_prediction']=dict(success=True,pairs=[dict(name='pair',a='star',b='c01',success=True,weight=1.,added=2.,self_relative=2.)])
    record['proxy_AC_checks']=[[dict(name='pair',check={'complete_AC_cost':8.,'max_channel_RMS':.2})]]
    result=d.analyze_trial(t,label='dev');method=next(r for r in result['method_rows'] if r['method']=='v2g_mobile')
    assert method['raw_same_witness_AC_mean']==3. and method['raw_full_AC_refit_mean']==2.
    assert not method['raw_same_curve_as_executed']
    assert method['raw_max_channel_RMS']==.2
