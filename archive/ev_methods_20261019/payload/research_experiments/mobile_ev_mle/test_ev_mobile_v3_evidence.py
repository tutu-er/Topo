from copy import deepcopy
import base64
import gzip
from hashlib import sha256
import json

import numpy as np
import pytest

from . import ev_mobile_v3_evidence as evidence
from .test_ev_mobile_v3_summary import record, spec


def _record():
    raw=record();raw['policy_id']='p'
    for row in raw['methods'].values():
        row['action_key']=sha256(np.asarray(row['power_kw'],dtype='<f8').tobytes()).hexdigest()
        row['evaluation']['comparisons']=dict(a=dict(joint=dict(post=dict(
            state=dict(r=[.01],x=[.02],source_p=[0.],source_q=[0.]),
            blocks={'future':{'loss':1.,'residual':[[1.]]}},loss=10.,loglik=-5.,
            success=False,optimizer_error='retained failure',optimizer={'restarts':[{'success':False,'state':[1,2]}]}))))
    raw['forecast']={'p':[[1.]],'source_by_candidate':{'a':{'source_p':[0.]}}}
    raw['history_prefits']={'a':{'joint':{'state':{'r':[.01]}}}}
    raw['planning']['projection_fixed']['raw'].update(
        selected_operation_x=[1.,2.],failures=['kept failure'],
        selected_profile=dict(score=4.,pairs=[dict(a='a',b='b',half_gap=4.)],
            references=[dict(a='a',success=True,fits=[dict(success=True,state=[9.]*1000),
                         dict(success=False,optimizer_error='failed solve',state=[1.,2.])])]),
        steps=[dict(master=dict(success=True,best_bound=8.,gap=4.,factor_ranges=[{'coefficient':[1.,2.]}]),
                    support=dict(best_bound=7.))])
    raw['planning']['scalar_fixed']['raw']['grids']=[dict(raw={'best_bound':4.},table=dict(
        success=True,tables=[dict(success=False,optimizer_error='failed table point',state=[3.,4.])],
        background={'array':[4.]*1000},QP_calls=10,nearest_error_upper=.3))]
    return raw


def test_compaction_deduplicates_only_evaluations_and_preserves_failure_evidence():
    raw=_record();compact=evidence.compact_record(raw,raw_sha256='original_file_hash')
    restored=evidence.restore(compact)
    assert len(compact['actions'])==1
    assert restored['training_result']==raw['training_result']
    assert restored['history_prefits']==raw['history_prefits']
    assert restored['forecast']==raw['forecast']
    assert restored['methods']['constant']['evaluation']==raw['methods']['constant']['evaluation']
    assert restored['planning']['projection_fixed']['raw']['selected_operation_x']==[1.,2.]
    assert restored['planning']['projection_fixed']['raw']['steps']==raw['planning']['projection_fixed']['raw']['steps']
    assert 'references' not in restored['planning']['projection_fixed']['raw']['selected_profile']
    assert {x['complete_failed_node']['optimizer_error'] for x in compact['failure_evidence']}=={
        'failed solve','failed table point'}
    check=evidence.validate_record(raw,compact,physical_replay=False,candidate_count=2)
    assert check['success'] and all(r['equal'] for r in check['summary_methods'].values())
    assert not check['restored_is_complete_raw']


def test_digest_tamper_and_conflicting_duplicate_actions_are_rejected():
    raw=_record();compact=evidence.compact_record(raw)
    changed=deepcopy(compact);changed['record']['status']='tampered'
    with pytest.raises(ValueError,match='digest'):evidence.restore(changed)
    raw['methods']['highlow']['evaluation']['losses']['a']=123.
    with pytest.raises(ValueError,match='different saved evaluations'):evidence.compact_record(raw)


def test_incomplete_phase_does_not_create_a_final_pack(tmp_path):
    source=tmp_path/'source';(source/'development').mkdir(parents=True)
    frozen=dict(specifications=dict(development=[spec(),spec(1)],evaluation=[]))
    evidence._write(source/'freeze.json',frozen)
    evidence._write(source/'development'/'000.json.gz',_record())
    destination=tmp_path/'final'
    with pytest.raises(ValueError,match='phase not complete'):
        evidence.pack(source,destination,phases=('development',))
    assert not destination.exists()


def test_complete_pack_preserves_snapshot_context_raw_identity_and_summary(tmp_path):
    source=tmp_path/'source';(source/'development').mkdir(parents=True)
    raw=_record();code=b'known frozen scientific bytes\n';source_hash={'code.py':sha256(code).hexdigest()}
    frozen=dict(policy_id='p',methods=list(raw['methods']),config=dict(candidate_count=2),
        specifications=dict(development=[spec()],evaluation=[]),source_sha256=source_hash,environment={'python':'test'})
    evidence._write(source/'freeze.json',frozen)
    evidence._write(source/'source_snapshot.json.gz',dict(source_sha256=source_hash,
        source_base64={'code.py':base64.b64encode(code).decode()}))
    evidence._write(source/'development'/'000.json.gz',raw)
    (source/'worker.log').write_text('unchanged log',encoding='utf8')
    destination=tmp_path/'packed';manifest=evidence.pack(source,destination,phases=('development',))
    assert manifest['status']=='complete' and manifest['case_count']==1
    assert (destination/'worker.log').read_text(encoding='utf8')=='unchanged log'
    assert json.loads((destination/'environment.json').read_text())=={'python':'test'}
    compact=evidence._read(destination/'development'/'000.json.gz')
    assert compact['source_record']['raw_sha256']==sha256((source/'development'/'000.json.gz').read_bytes()).hexdigest()
    assert evidence.validate_record(raw,compact,physical_replay=False,candidate_count=2)['success']
    with pytest.raises(FileExistsError):evidence.pack(source,destination,phases=('development',))


def test_fatal_and_missing_method_records_remain_restorable():
    raw=dict(spec=spec(),policy_id='p',status='fatal_failure',failure='unavailable final fit')
    compact=evidence.compact_record(raw)
    restored=evidence.restore(compact)
    assert restored['failure']==raw['failure'] and restored['status']=='fatal_failure'
    assert evidence.validate_record(raw,compact,physical_replay=False)['success']
