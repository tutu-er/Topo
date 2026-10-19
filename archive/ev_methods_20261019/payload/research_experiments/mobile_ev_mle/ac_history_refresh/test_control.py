"""Artificial records only: B freeze, calibration and launch safety."""
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pytest

from research_experiments.mobile_ev_mle.ac_history_refresh import control as c


@pytest.fixture
def version(tmp_path):
    module=tmp_path/'module';module.mkdir();B=module/'ac_history_refresh';B.mkdir()
    for name in ('a.py','b.py'):(module/name).write_text('original '+name)
    for name in ('control.py','test_control.py','math.md','protocol.md'):(B/name).write_text('B '+name)
    A=tmp_path/'A';A.mkdir();specs={phase:[dict(phase=phase,index=i,seed=100+i) for i in range(32)] for phase in ('calibration','evaluation')}
    c.atomic_json(A/'freeze.json',dict(policy_id='A-independent-policy',source_sha256={name:c.file_hash(module/name) for name in ('a.py','b.py')},specifications=specs))
    root=tmp_path/'B';v=c.create_freeze(root,A,dict(workers=1),module_root=module,b_root=B,work_root=tmp_path,expected_A_count=2)
    return root,A,v,module,B


def install_result(root,A,v,phase,index,record):
    path=c.result_path(root,phase,index);source=A/phase/path.name
    source.parent.mkdir(exist_ok=True);source.write_bytes(b'original A saved input '+str(index).encode())
    c.io.save_record(path,record)
    c.atomic_json(c.meta_path(root,phase,index),dict(policy_id=v['policy_id'],spec=v['specifications'][phase][index],
        result_sha256=c.file_hash(path),source_A_sha256=c.file_hash(source),science_source_unchanged=True))


def artificial(index,phase='calibration',failure=False):
    spec=dict(phase=phase,index=index,seed=100+index)
    if failure:return dict(spec=spec,status='process_failure',failures={'worker':'retained'},n_independent_histories=1)
    bank=dict(means=[[[0.],[2.]]],covariances=[[[[2.]],[[2.]]]],log_prior=np.log([.5,.5]).tolist())
    action=dict(propagation=dict(status='ok',banks={'G11':bank},rows=[dict(status='ok',mean_MCSE_whitened_max=0.,covariance_MCSE_relative_fro=0.)]),
        regimes={regime:dict(values=[[-3.],[3.]],clean=[0.]) for regime in ('regular','shift')})
    return dict(spec=spec,status='complete',candidate_names=['a','b'],true_name='a',true_index=0,
        plans={label:dict(action_key='same') for label in ('constant','SQMD','BCMD')},actions={'same':action},config={})


def test_freeze_snapshot_and_source_manifest_are_complete_and_immutable(version):
    root,A,v,module,B=version
    assert v['A_source_count']==2 and v['B_source_count']==4
    snapshot=json.loads(gzip.decompress((root/'source_snapshot.json.gz').read_bytes()))
    assert set(snapshot)==set(v['source_sha256'])
    assert c.frozen(root)['policy_id']==v['policy_id']
    with pytest.raises(FileExistsError):c.create_freeze(root,A)
    (B/'control.py').write_text('changed')
    with pytest.raises(ValueError,match='source changed'):c.frozen(root)


def test_missing_B_freeze_rejected_before_any_A_read(tmp_path):
    with pytest.raises(ValueError,match='freeze required'):c.worker(tmp_path,'evaluation',0)


def test_evaluation_rejected_before_calibration_or_A_evaluation_read(version,monkeypatch):
    root,A,v,_,_=version;called=[];original=c.file_hash
    def hash_guard(path):
        if Path(path).parent.name=='evaluation':raise AssertionError('evaluation file opened too early')
        return original(path)
    monkeypatch.setattr(c,'file_hash',hash_guard)
    with pytest.raises(ValueError,match='calibration required'):
        c.worker(root,'evaluation',0,replay_function=lambda *a,**kw:called.append(True))
    assert not called


def test_B_NLL_rule_selects_independent_multiplier_and_binds_all_32(version):
    root,A,v,_,_=version
    c.atomic_json(A/'calibration.json',dict(G11_multiplier=32,forbidden_old_A_calibration=True))
    for i in range(32):install_result(root,A,v,'calibration',i,artificial(i))
    result=c.calibrate_B(root)
    assert result['histories']==32 and result['available_histories']==32
    assert result['G11_multiplier']==8
    assert not result['old_A_calibration_used']
    assert 'SSE_thresholds' not in result
    binding=c.bind_evaluation(root)
    assert binding['calibration_sha256']==c.file_hash(root/'calibration.json')
    changed=deepcopy(result);changed['G11_multiplier']=16;c.atomic_json(root/'calibration.json',changed)
    with pytest.raises(ValueError,match='after evaluation binding'):c.bind_evaluation(root)


def test_all_B_calibration_failures_preserved_and_evaluation_blocked(version):
    root,A,v,_,_=version
    for i in range(32):install_result(root,A,v,'calibration',i,artificial(i,failure=True))
    result=c.calibrate_B(root)
    assert result['histories']==32 and result['available_histories']==0
    assert result['G11_multiplier'] is None
    with pytest.raises(ValueError,match='calibration failed'):c.bind_evaluation(root)


def test_worker_uses_replay_file_contract_and_keeps_A_bytes(version):
    root,A,v,_,_=version;source=A/'calibration/history_000.json.gz';source.parent.mkdir();source.write_bytes(b'fixed saved A')
    before=source.read_bytes();seen=[]
    def fake(source,output,**kwargs):
        seen.append(kwargs);r=artificial(0);c.io.save_record(output,r);return r
    result=c.worker(root,'calibration',0,replay_function=fake)
    assert result['spec']['index']==0 and source.read_bytes()==before
    assert seen[0]['variance_multiplier']==1. and seen[0]['draws']==256
    assert c.io.load_record(c.meta_path(root,'calibration',0))['calibration_binding'] is None
    assert c.validate_result(root,v,'calibration',v['specifications']['calibration'][0])[0]['spec']['index']==0


def test_duplicate_worker_claim_and_atomic_permission_retry(version,monkeypatch):
    root,_,_,_,_=version;claim=root/'claim.json'
    c.atomic_json(claim,dict(status='running',pid=os.getpid()))
    with pytest.raises(RuntimeError,match='already active'):c._claim(claim,dict(status='running',pid=123))
    original=Path.replace;attempts=[]
    def locked(path,target):
        if Path(target).name=='retry.json' and len(attempts)<2:
            attempts.append(True);raise PermissionError('viewer lock')
        return original(path,target)
    monkeypatch.setattr(Path,'replace',locked);monkeypatch.setattr(c.time,'sleep',lambda _:None)
    c.atomic_json(root/'retry.json',dict(value=1))
    assert len(attempts)==2 and c.io.load_record(root/'retry.json')['value']==1


def test_failed_replay_preserves_process_failure_without_regeneration(version):
    root,A,v,_,_=version;source=A/'calibration/history_000.json.gz';source.parent.mkdir();source.write_bytes(b'original A')
    def fail(*args,**kwargs):raise RuntimeError('test failure')
    row=c.worker(root,'calibration',0,replay_function=fail)
    assert row['status']=='process_failure' and row['keep_all_on_failure']
    assert source.read_bytes()==b'original A'


def test_controller_stage_with_mock_workers_keeps_all_32_and_no_child_launch(version,monkeypatch):
    root,A,v,_,_=version
    for i in range(32):
        path=A/'calibration'/('history_%03d.json.gz'%i);path.parent.mkdir(exist_ok=True);path.write_bytes(b'unchanged A '+str(i).encode())
    launches=[]
    def popen(command,**kwargs):
        index=int(command[command.index('--index')+1]);launches.append(index)
        def fake(source,output,**options):
            row=artificial(index);c.io.save_record(output,row);return row
        c.worker(root,'calibration',index,replay_function=fake)
        class Completed:
            pid=900000+index
            returncode=0
            def poll(self):return 0
        return Completed()
    monkeypatch.setattr(c.subprocess,'Popen',popen)
    monkeypatch.setattr(c.time,'sleep',lambda _:None)
    result=c.run_phase(root,'calibration')
    assert len(launches)==32 and sorted(launches)==list(range(32))
    assert result['histories']==32 and result['G11_multiplier']==8
    assert c.io.load_record(root/'controller.json')['status']=='calibration_complete'
    assert not (root/'controller_lock.json').exists()


def test_still_writing_A_input_is_not_treated_as_scientific_failure(version):
    root,A,v,_,_=version;source=A/'calibration/history_000.json.gz'
    source.parent.mkdir();source.write_bytes(b'partial gzip written by active A worker')
    c.atomic_json(A/'launch_claims/calibration_000.json',dict(status='running',pid=123))
    with pytest.raises(RuntimeError,match='not finished saving'):
        c.worker(root,'calibration',0,replay_function=lambda *a,**kw:None)
    assert not c.result_path(root,'calibration',0).exists()
