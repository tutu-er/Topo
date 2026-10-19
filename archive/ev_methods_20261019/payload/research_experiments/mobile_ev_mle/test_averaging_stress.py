"""Narrow invariants for the paired correlated-averaging diagnostic."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
from . import averaging_stress as stress
from .information_context import quality_models, meter_to_record
from .simulation import model
from .likelihood import Observations


def meters(kind):
    h=Observations(p=np.full((12,3),.12),q=np.full((12,3),.04),v=np.ones((12,3)),master_p=np.full(12,.46),master_q=np.full(12,.155))
    hm=model(3);nm,jm,q=quality_models(h,hm,kind=kind)
    return hm,nm,q


@pytest.mark.parametrize('kind',['uniform','temporal','spatial'])
def test_actual_generation_changes_only_voltage_variance(kind):
    hm,nm,q=meters(kind);actual=stress.actual_generation_model(hm,nm,q['v_repeat_counts'])
    m=np.asarray(q['v_repeat_counts']);ratio=np.asarray(actual.v_sigma)**2/np.asarray(nm.v_sigma)**2
    np.testing.assert_allclose(ratio,1+.25*(m-1),rtol=1e-14)
    a,n=meter_to_record(actual),meter_to_record(nm)
    assert all(a[k]==n[k] for k in a if k!='v_sigma')
    zero=stress.actual_generation_model(hm,nm,m,rho=0.)
    np.testing.assert_allclose(zero.v_sigma,nm.v_sigma,rtol=1e-14)
    np.testing.assert_array_equal(np.asarray(actual.v_sigma)[m==1],np.asarray(nm.v_sigma)[m==1])


def test_sigma_rejects_wrong_saved_assumption():
    hm,nm,q=meters('uniform')
    with pytest.raises(ValueError,match='nominal averaging'):
        stress.actual_generation_model(hm,replace(nm,v_sigma=.001),q['v_repeat_counts'])


def test_required_failure_retains_all_four_and_truth():
    names=['star','01','02','12'];row={'status':'joint_fit_failure','losses':{'01':0.}}
    actual=stress.decision(row,names,'star');row.update(actual)
    assert actual['retained_candidates']==names and not actual['false_rejection'] and actual['unresolved']
    assert stress.outcome(row)=='failure'


def test_threshold_is_full_sse_inclusive_and_truth_only_evaluation():
    names=['star','01','02','12'];row={'status':'complete','losses':{'star':stress.THRESHOLD,'01':0.,'02':10.,'12':20.}}
    assert stress.decision(row,names,'star')['retained_candidates']==['star','01']
    assert stress.decision(row,names,'02')['retained_candidates']==['star','01']
    row['losses']['star']=np.nextafter(stress.THRESHOLD,np.inf)
    assert stress.decision(row,names,'star')['false_rejection']


@pytest.mark.skipif(not (stress.NOMINAL_ROOT/'evaluation/trial_950016.json.gz').exists(), reason='local nominal first16 evidence required')
def test_all16_nominal_reconstructions_and_detect_truth_tamper():
    freeze=stress.old.read_json(stress.NOMINAL_ROOT/'freeze.json')
    for seed in stress.IDS:
        nominal=stress.old.read_json(stress.NOMINAL_ROOT/'evaluation'/f'trial_{seed}.json.gz')
        path,h,w,check=stress.reconstruct_nominal(nominal,freeze)
        assert check['success'], (seed,check['errors'])
        assert check['nominal_method_count']==11 and check['nominal_array_checks']==66
        assert check['saved_truth_sha256']==check['reconstructed_truth_sha256']
        assert check['history_hash_saved']==check['history_hash_reconstructed']
    bad=deepcopy(nominal);bad['truth']['p'][0][0]+=.001
    check=stress.reconstruct_nominal(bad,freeze)[3]
    assert not check['success'] and 'truth_arrays_or_metadata' in check['errors']


@pytest.mark.skipif(not (stress.NOMINAL_ROOT/'evaluation/trial_950016.json.gz').exists(), reason='local nominal first16 evidence required')
def test_cache_replay_counts_all_candidate_exception_result(monkeypatch):
    nominal=stress.old.read_json(stress.NOMINAL_ROOT/'evaluation/trial_950006.json.gz')
    freeze=stress.old.read_json(stress.NOMINAL_ROOT/'freeze.json')
    def failed(candidates,*args,**kwargs):
        return {},{c.name:'injected_candidate_exception' for c in candidates},dict(fit_calls=len(candidates),ac_calls=4,seconds=0.)
    monkeypatch.setattr(stress.old,'fit_all_candidates',failed)
    alternate=stress.run_trial(nominal,freeze,dict(success=True,errors=[]),input_sha256='injected_test')
    assert not alternate['complete']
    assert len(alternate['records'])==11 and all(r['status']=='joint_fit_failure' for r in alternate['records'])
    assert alternate['actual_joint_fit_calls']==4*alternate['joint_unique_action_count']
    assert alternate['fit_cache_hits']>0
    replay=stress.replay_trial(alternate,nominal)
    assert replay['success'],replay['errors']
    assert replay['unique_action_count']==alternate['joint_unique_action_count']
    assert all(r['unresolved'] and not r['false_rejection'] for r in alternate['records'])


@pytest.mark.skipif(not (stress.OUTPUT/'initial_audit_attempt/raw/trial_950001.json.gz').exists(), reason='local preserved stress acquisition required')
def test_extra_replay_detects_root_tamper():
    from .averaging_stress_checks import audit_one
    nominal=stress.old.read_json(stress.NOMINAL_ROOT/'evaluation/trial_950001.json.gz')
    # Use the preserved initial valid acquisition: the audit implementation changed, science did not.
    record=stress.old.read_json(stress.OUTPUT/'initial_audit_attempt/raw/trial_950001.json.gz')
    freeze=stress.old.read_json(stress.NOMINAL_ROOT/'freeze.json')
    calibration=stress.old.read_json(stress.NOMINAL_ROOT/'calibration/thresholds.json')
    assert audit_one(nominal,record,freeze,calibration)['success']
    bad=deepcopy(record);bad['records'][0]['observations']['root_v'][0]=1.001
    audit=audit_one(nominal,bad,freeze,calibration)
    assert not audit['success'] and any('calibrated_root' in e for e in audit['errors'])
