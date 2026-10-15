"""Nested rich inputs, unchanged holdouts, and training-only orchestration."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_fixed_train_training_support as support
from .candidates import RadialCandidate


def test_rich_prefixes_are_exactly_nested_with_unchanged_assets_and_holdouts():
    records=[support.make_training_inputs(990101,0,n) for n in support.LENGTHS]
    for a,b in zip(records,records[1:]):
        n=a['training'].t
        for channel in ('p','q','v','root_v','master_p','master_q'):
            assert np.array_equal(getattr(a['training'],channel),getattr(b['training'],channel)[:n])
            assert np.array_equal(getattr(a['validation'],channel),getattr(b['validation'],channel))
        for key in ('rich96_physical_inputs_sha256','rich96_white_sha256','rich96_observations_sha256',
                    'paired_validation_future_asset_sha256'):
            assert a['generation'][key]==b['generation'][key]
        assert np.array_equal(a['case']['r'],b['case']['r']) and np.array_equal(a['case']['x'],b['case']['x'])
        for channel in support.base.CHANNELS:
            assert np.array_equal(a['future_white'][channel],b['future_white'][channel])
    full=records[-1];truth=full['training_truth']
    assert np.all(abs(truth['p']*20.-2.4)<=.6)
    assert np.all(abs(truth['q']*20.-.8)<=.3)
    assert np.all(abs(truth['source_p']*20.-2.)<=.25)
    assert np.all(abs(truth['source_q']*20.-.7)<=.1)
    assert full['generation']['training_EV_applied'] is False
    for k,seed in full['generation']['rich_seeds'].items():
        assert seed==(full['case']['seeds'][k]+104729)%2**32
    with pytest.raises(ValueError,match='8, 32 or 96'):
        support.condition_config(16)


def test_modifying_unseen_future_does_not_change_rich_training(monkeypatch):
    original=support.base.make_case(990102,support.base.FAMILIES[1])
    changed=deepcopy(original)
    changed['p'][12:]+=.123;changed['q'][12:]-=.02
    changed['source_p'][12:]+=.1;changed['source_q'][12:]-=.05
    for values in changed['white'].values():values[12:]+=10.
    monkeypatch.setattr(support.base,'make_case',lambda *a,**k:original)
    a=support.make_training_inputs(990102,1,32)
    monkeypatch.setattr(support.base,'make_case',lambda *a,**k:changed)
    b=support.make_training_inputs(990102,1,32)
    assert support.identity(support.io.observations_record(a['training']))==support.identity(support.io.observations_record(b['training']))
    assert a['generation']['rich96_observations_sha256']==b['generation']['rich96_observations_sha256']
    assert a['generation']['paired_validation_future_asset_sha256']!=b['generation']['paired_validation_future_asset_sha256']


def test_driver_only_passes_training_to_selector_and_keeps_all_paired_evaluations(monkeypatch):
    bank=[RadialCandidate.from_supports('star',3,()),
        *[RadialCandidate.from_supports('clade_'+''.join(map(str,pair)),3,(pair,))
          for pair in ((0,1),(0,2),(1,2))]]
    trace=[]
    def fit_bank(candidates,obs,meter,*,options,seed,trained=None):
        trace.append(('fit',obs.t,trained is not None))
        fits={};rows=[]
        for i,c in enumerate(candidates):
            r=np.full(c.m,.004) if trained is None else trained[c.name].r.copy()
            x=np.full(c.m,.002) if trained is None else trained[c.name].x.copy()
            fits[c.name]=SimpleNamespace(r=r,x=x,success=True,source_p=np.zeros(obs.t),source_q=np.zeros(obs.t))
            rows.append(dict(candidate=c.name,loss=float(obs.t+i),r=r.tolist(),x=x.tolist(),success=True))
        return fits,dict(fits=rows,all_complete=True,failures={},independent_review=[],
            cost=dict(fit_calls=4,ac_calls=0,solver_calls=0),seconds=0.)
    def selector(training,candidates,trained,tm,fm,**kwargs):
        trace.append(('selector',training.t))
        assert training.t==32 and set(kwargs)=={'seed','fit_options','exchange_options','modes'}
        constant=np.zeros((6,3));constant[:,0]=2.
        choices={name:np.zeros_like(constant) if name=='none' else constant.copy() for name in kwargs['modes']}
        return choices,dict(methods={},constant_operation=dict(feasible=True),
            predicted_fit_calls=0,predicted_ac_calls=0,uses_validation=False)
    monkeypatch.setattr(support,'propose',lambda *a,**k:(bank,dict(uses_truth=False)))
    monkeypatch.setattr(support.study,'fit_bank',fit_bank)
    monkeypatch.setattr(support.study,'PQ_error_scale',lambda *a,**k:dict(first_order_only=True))
    monkeypatch.setattr(support,'select_fixed_controls',selector)
    result=support.run_training_condition(990101,0,32)
    assert result['status']=='complete' and result['source_unchanged']
    assert trace[:3]==[('fit',32,False),('fit',4,True),('selector',32)]
    assert all(row==('fit',6,True) for row in trace[3:])
    assert len(result['methods'])==11 and result['unique_actual_curves']==2
    assert result['statistical_unit']=='one paired asset/history'
    assert result['number_of_training_conditions_is_not_independent_sample_size'] is True
    assert result['statistically_calibrated'] is False
    for method in result['methods']:
        physical=np.asarray(method['physical']['p']);power=np.asarray(method['power_kw'])
        assert physical==pytest.approx(np.asarray(result['truth']['p'])[12:]+power/20.)
        assert method['status']=='complete' and method['M']==pytest.approx(2.)
    trace.clear()
    skipped=support.run_training_condition(990101,0,8,run_selector=False)
    assert skipped['status']=='training_validation_only_complete'
    assert trace==[('fit',8,False),('fit',4,True)]
    assert skipped['selection'] is None and skipped['methods']==[]
