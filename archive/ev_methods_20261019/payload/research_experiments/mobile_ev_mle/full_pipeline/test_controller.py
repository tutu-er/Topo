"""Controller tests use durable dummy cases, never scientific case generation."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import json
import os

import pytest

from research_experiments.mobile_ev_mle.full_pipeline import controller as c


class InlineExecutor:
    """Exercise durable task handling without launching any actual process."""
    submits=0
    options=[]
    def __init__(self,**kwargs):
        self.options.append(kwargs)
    def submit(self,fn,*args):
        type(self).submits+=1
        future=Future()
        try:future.set_result(fn(*args))
        except BaseException as exc:future.set_exception(exc)
        return future
    def shutdown(self,**kwargs):pass


@pytest.fixture
def dummy(tmp_path,monkeypatch):
    InlineExecutor.submits=0;InlineExecutor.options=[]
    cfg={'fixed':'one-policy'}
    specs=lambda prepared,phase:[dict(phase=phase,cluster_index=i,n=6,source_mode='none') for i in range(2)]
    def path_for(output,spec):return Path(output)/spec['phase']/(c.task_name(spec)+'.json')
    def prepare(output):c._write(Path(output)/'prepared.json.gz',dict(forecast_config={'method':'persistence'}))
    def freeze(output,config,prepared):
        value=dict(identity='frozen-test',config=config,prepared_sha256='prepared',source_sha256={'solver.py':'one'})
        c._write(Path(output)/'freeze.json',value);return value
    calls=[]
    def run(spec,output,prepared,freeze_policy=True):
        calls.append((spec,freeze_policy));print('dummy_case',c.task_name(spec))
        if spec['cluster_index']==1:raise RuntimeError('retained test worker failure')
        identity='frozen-test' if freeze_policy else 'development-unfrozen'
        c._write(path_for(output,spec),dict(spec=spec,freeze_identity=identity,source_unchanged=True,
            status='complete',seconds=.1))
        return dict(spec=spec,path=str(path_for(output,spec)),status='complete',seconds=.1)
    study=SimpleNamespace(CONFIG=cfg,prepare=prepare,read=c._read,freeze=freeze,serial=lambda x:x,
        source_hashes=lambda:{'solver.py':'one'},digest=lambda path:'prepared',specifications=specs,
        path_for=path_for,run=run)
    monkeypatch.setattr(c,'_study',lambda:study)
    return tmp_path,study,calls


def test_read_only_pid_probe_and_lock_reject_live_owner(tmp_path):
    assert c.process_alive(os.getpid())
    assert not c.process_alive(-1)
    with c.controller_lock(tmp_path):
        with pytest.raises(RuntimeError,match='already running'):
            with c.controller_lock(tmp_path):pass
    assert not (tmp_path/'controller.lock.json').exists()


def test_stale_owner_is_preserved_but_live_orphan_worker_blocks(tmp_path):
    c._write(tmp_path/'controller.lock.json',dict(pid=99999999,token='dead'))
    c._write(tmp_path/'controller_workers'/'worker_live.json',dict(pid=os.getpid()))
    with pytest.raises(RuntimeError,match='workers remain active'):
        with c.controller_lock(tmp_path):pass
    (tmp_path/'controller_workers'/'worker_live.json').unlink()
    with c.controller_lock(tmp_path):pass
    assert len(list(tmp_path.glob('controller.lock.stale.*.json')))==1


def test_phase_counts_failure_denominator_logs_and_no_rerun(dummy):
    output,study,calls=dummy
    result=c.run_controller(output,phases=['calibration','evaluation'],workers=3,executor_factory=InlineExecutor)
    assert result['status']=='statistical_complete'
    assert result['counts']['total']==result['counts']['computed']==4
    assert result['counts']['remaining']==0
    for phase,row in result['counts']['phases'].items():
        assert row['complete']==row['computed_with_failure']==1
        assert row['statuses']=={'complete':1,'controller_worker_exception':1}
    assert len(calls)==InlineExecutor.submits==4
    assert len(list((output/'logs').glob('*.stdout.log')))==4
    assert len(list((output/'logs').glob('*.stderr.log')))==4
    failure=c._read(next((output/'controller_failures').glob('*.json')))
    assert 'retained test worker failure' in failure['exception']['traceback']
    result2=c.run_controller(output,phases=['calibration','evaluation'],workers=3,executor_factory=InlineExecutor)
    assert InlineExecutor.submits==4
    assert all(row['resumed'] for row in result2['finished'].values())
    assert not (output/'controller.lock.json').exists()
    assert InlineExecutor.options[0]['mp_context'].get_start_method()=='spawn'


def test_matching_failed_raw_is_completed_and_not_retried(dummy):
    output,study,calls=dummy
    study.prepare(output)
    policy=c._policy(output,output/'prepared.json.gz',False)[1]
    spec=study.specifications({},'development')[0]
    c._write(study.path_for(output,spec),dict(spec=spec,freeze_identity=policy['identity'],
        source_unchanged=True,status='failed',seconds=2.))
    result=c.run_controller(output,phases=['development'],workers=2,executor_factory=InlineExecutor)
    assert len(calls)==1
    assert result['counts']['computed']==2
    assert result['counts']['phases']['development']['computed_with_failure']==2
    assert result['finished'][c.task_name(spec)]['resumed']


@pytest.mark.parametrize('field,value',[('freeze_identity','wrong'),('source_unchanged',False),('spec',{})])
def test_wrong_raw_identity_is_preserved_and_blocks_resume(dummy,field,value):
    output,study,calls=dummy
    study.prepare(output);policy=c._policy(output,output/'prepared.json.gz',False)[1]
    spec=study.specifications({},'development')[0]
    raw=dict(spec=spec,freeze_identity=policy['identity'],source_unchanged=True,status='failed')
    raw[field]=value;c._write(study.path_for(output,spec),raw)
    before=study.path_for(output,spec).read_bytes()
    with pytest.raises(RuntimeError,match='does not match'):
        c.run_controller(output,phases=['development'],workers=2,executor_factory=InlineExecutor)
    assert study.path_for(output,spec).read_bytes()==before
    assert not calls
    assert c._read(output/'controller.json')['status']=='controller_failed'


def test_unfrozen_is_development_only_and_policy_hash_protects_resume(dummy):
    output,study,calls=dummy
    with pytest.raises(ValueError,match='only development'):
        c.run_controller(output,phases=['evaluation'],development_unfrozen=True,executor_factory=InlineExecutor)
    result=c.run_controller(output,phases=['development'],development_unfrozen=True,executor_factory=InlineExecutor)
    assert result['freeze_identity']=='development-unfrozen'
    assert all(not freeze for _,freeze in calls)
    study.source_hashes=lambda:{'solver.py':'changed'}
    with pytest.raises(RuntimeError,match='policy differs'):
        c.run_controller(output,phases=['development'],development_unfrozen=True,executor_factory=InlineExecutor)


def test_actual_spawn_initializer_threads_and_pid_receipt_without_scientific_task(tmp_path):
    c._pin_execution_path()
    with c.ProcessPoolExecutor(max_workers=1,mp_context=c.multiprocessing.get_context('spawn'),
            initializer=c._worker_init,initargs=(str(tmp_path),'dummy-spawn')) as executor:
        assert executor.submit(c._json_digest,{'dummy':1}).result(timeout=30)==c._json_digest({'dummy':1})
        workers=c.live_workers(tmp_path)
        assert len(workers)==1 and workers[0]['controller_token']=='dummy-spawn'
        assert all(value=='1' for value in workers[0]['threads'].values())
    assert not c.live_workers(tmp_path)
