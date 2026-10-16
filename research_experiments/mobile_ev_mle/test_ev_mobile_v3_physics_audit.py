"""Schema, failure-denominator and tamper checks without AC or optimization."""
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_v3_physics_audit as audit


def _fixture(monkeypatch):
    bank = {name:SimpleNamespace(name=name, n=1, m=1, supports=((0,),)) for name in ('a','b')}
    obs = dict(p=[[1.]], q=[[0.]], v=[[1.]], master_p=[1.], master_q=[0.], root_v=[1.])
    meter = dict(bias_p=0., bias_q=0., bias_v=0., v_sigma=1., master_p_sigma=1., master_q_sigma=1.)
    state = dict(r=[.01], x=[.003], source_p=[0.], source_q=[0.])
    fits = [dict(name=name, candidate=name, supports=[[0]], success=True, loss=0., normalizer=0., loglik=0.,
                 p=[[1.]], q=[[0.]], predicted={k:obs[k] for k in ('p','q','v','master_p','master_q')}, **state)
            for name in bank]
    final = dict(loss=0., normalizer=0., loglik=0., success=True,
                 state={**state, 'source_p':[0.]*3, 'source_q':[0.]*3},
                 blocks={k:dict(loss=0., residual=[[0.,0.,0.]]) for k in ('training','validation','future')})
    evaluation = dict(complete=True, future=obs, losses={'a':0.,'b':0.}, M=0.,
        comparisons={name:dict(joint=dict(post=deepcopy(final), joint_total=0.)) for name in bank})
    power = [[0.]]; key=sha256(np.asarray(power,dtype='<f8').tobytes()).hexdigest()
    record = dict(spec=dict(bank='n3',T=1,index=0), config=dict(base_kw=20.), status='complete', true_name='a',
                  training=obs,training_meter=meter,validation=obs,validation_meter=meter,future_meter=meter,
                  training_result=dict(fits=fits,failures={}),methods={name:dict(power_kw=power,action_key=key,
                  success=True,operation=dict(audit=dict(feasible=True)),evaluation=deepcopy(evaluation))
                  for name in ('constant','highlow')})
    calls=[]
    def replay(candidate,state,blocks,cfg):
        calls.append(candidate.name)
        return dict(loss=0.,normalizer=0.,loglik=0.,line_domain_feasible=True,source_box_feasible=True,
                    physical_voltage_feasible=True,max_AC_equation=0.,max_power_balance=0.,
                    blocks={name:dict(loss=0.,residual=np.zeros((len(o['p']),3))) for name,o,m in blocks})
    monkeypatch.setattr(audit,'_bank',lambda record:bank)
    monkeypatch.setattr(audit.old,'replay_state',replay)
    monkeypatch.setattr(audit.cases,'service_domain',lambda *a,**k:None)
    monkeypatch.setattr(audit,'_operation',lambda *a,**k:dict(success=True))
    return record,calls


def test_duplicate_action_replays_once_but_retains_all_method_candidate_slots(monkeypatch):
    record,calls=_fixture(monkeypatch)
    result=audit.audit_record(record,expected_methods=('constant','highlow'))
    assert result['success']
    assert len(calls)==4  # two training plus two unique-action final states
    assert result['metrics']['joint_method_candidate_slots']==4
    assert result['metrics']['joint_unique_action_candidate_states']==2
    assert result['methods'][1]['action_replay_reused']


def test_failed_fits_are_replayed_and_failures_remain_in_slot_denominators(monkeypatch):
    record,_=_fixture(monkeypatch)
    for row in record['methods'].values():
        row['evaluation']['comparisons']['b']['joint']['post']['success']=False
        row['evaluation']['complete']=False;row['success']=False
    record['methods']['highlow']['planning_failure']='budget'
    result=audit.audit_record(record,expected_methods=('constant','highlow'))
    assert result['success']
    assert result['metrics']['joint_method_candidate_slots']==4
    assert result['metrics']['recorded_joint_fit_failure_slots']==2
    assert result['metrics']['recorded_inference_failure_method_slots']==2
    assert any(r['kind']=='planning_failure' for r in result['recorded_failures'])
    assert any(r['kind']=='fit_failure' for r in result['recorded_failures'])


def test_missing_candidate_and_training_failure_do_not_disappear(monkeypatch):
    record,_=_fixture(monkeypatch)
    record['training_result']['fits'].pop()
    result=audit.audit_record(record,expected_methods=('constant','highlow'))
    assert not result['success'] and not result['complete_physical_coverage']
    assert result['expected']['training_candidate_states']==2
    assert any(r['check']=='training/b' for r in result['missing'])


@pytest.mark.parametrize('tamper',['prediction','action_hash','duplicate','complete'])
def test_tampered_saved_evidence_is_detected(monkeypatch,tamper):
    record,_=_fixture(monkeypatch)
    if tamper=='prediction':record['training_result']['fits'][0]['predicted']['v']=[[1.01]]
    elif tamper=='action_hash':record['methods']['constant']['action_key']='wrong'
    elif tamper=='duplicate':record['methods']['highlow']['evaluation']['losses']['a']=.01
    elif tamper=='complete':
        for row in record['methods'].values():row['evaluation']['complete']=False
    result=audit.audit_record(record,expected_methods=('constant','highlow'))
    assert not result['success'] and result['errors']


def test_fatal_record_is_retained_as_missing_physics(monkeypatch):
    record,_=_fixture(monkeypatch)
    result=audit.audit_record(dict(spec=record['spec'],status='fatal_failure',failure='saved exception'),
                              expected_methods=('constant','highlow'))
    assert not result['success']
    assert result['expected']['method_slots']==2
    assert result['missing'] and result['recorded_failure']=='saved exception'


def test_run_coverage_and_no_overwrite(tmp_path,monkeypatch):
    record,_=_fixture(monkeypatch)
    root=tmp_path/'batch';(root/'development').mkdir(parents=True)
    frozen=dict(policy_id='p',source_sha256={'science.py':'original_sha'},methods=['constant','highlow'],
                specifications=dict(development=[record['spec'],{**record['spec'],'index':1}]))
    audit.old.write(root/'freeze.json',frozen)
    record['policy_id']='p';audit.old.write(root/'development'/'000.json.gz',record)
    output=tmp_path/'audit.json'
    result=audit.run(root,output)
    assert not result['success'] and not result['exact_case_coverage']
    assert result['saved_case_slots']==1 and result['expected_case_slots']==2
    assert result['original_source_sha256']=={'science.py':'original_sha'}
    assert result['optimizer_calls']==result['new_simulation_calls']==0
    with pytest.raises(FileExistsError):audit.run(root,output)
