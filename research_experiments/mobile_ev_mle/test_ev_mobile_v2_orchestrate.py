"""Dry controllers: gating and restart behavior without AC or subprocess work."""
from pathlib import Path
from hashlib import sha256
from types import SimpleNamespace
import pytest

from . import ev_mobile_v2_orchestrate as control
from . import ev_systematic_analyze as io


def part(root,bank,phase,p,*,count=None):
    width=48 if phase=='calibration' else 24;count=width if count is None else count
    directory=control.part_directory(root,bank,phase,p);directory.mkdir(parents=True,exist_ok=True)
    io._write(directory/'freeze.json',dict(policy_id=control.POLICY))
    first=control.SEEDS[bank,phase]
    for i in range(width*p,width*p+count):
        (directory/f'case_{i:03d}_{first+i}.json.gz').write_bytes(b'existing immutable test record')
    io._write(directory/'checkpoint.json',dict(bank=bank,phase=phase,completed=count,
        planned=width,last_index=width*p+count-1))
    return directory


def thresholds(root,bank):
    path=Path(root)/'outputs'/f'mobile_v2_formal_{bank}_analysis'/'reliability_v2_thresholds.json'
    io._write(path,dict(calibration_count=96,identity=dict(policy_id=control.POLICY),rules={key:{} for key in control.rel.RULES},
        alpha=.1,gamma_AIC=2.,methods=list(control.EXPECTED_METHODS),
        tool_sha256=sha256(Path(control.rel.__file__).read_bytes()).hexdigest()))
    return path


def test_calibration_gate_requires_all_four_complete_chunks_before_any_merge(tmp_path,monkeypatch):
    for bank in control.BANKS:
        for p in (0,1):part(tmp_path,bank,'calibration',p,count=47 if (bank,p)==('n4',1) else 48)
    controller=control.Controller(tmp_path)
    called=[]
    monkeypatch.setattr(controller,'prepare_thresholds',lambda:called.append('calibrate'))
    assert not controller.advance() and called==[]
    assert controller.state['progress']['counts']['n4_1']==47
    part(tmp_path,'n4','calibration',1,count=48)
    monkeypatch.setattr(controller,'prepare_thresholds',lambda:(_ for _ in ()).throw(RuntimeError('gated preparation reached')))
    with pytest.raises(RuntimeError,match='gated preparation reached'):controller.advance()


def test_both_thresholds_gate_evaluation_and_policy_is_checked(tmp_path):
    thresholds(tmp_path,'n3')
    with pytest.raises(ValueError,match='both immutable threshold'):control.require_thresholds(tmp_path)
    path=thresholds(tmp_path,'n4')
    assert set(control.require_thresholds(tmp_path))=={'n3','n4'}
    value=io._read(path);value['identity']['policy_id']='other';io._write(path,value)
    with pytest.raises(ValueError,match='identity/count/rules differ'):control.require_thresholds(tmp_path)


def test_resume_ranges_skip_existing_records_and_reject_wrong_seed(tmp_path):
    directory=part(tmp_path,'n3','evaluation',0,count=5)
    before={p.name:p.read_bytes() for p in directory.glob('case_*')}
    assert control.missing_ranges(directory,'n3',0)==[(5,24)]
    (directory/f'case_007_{control.SEEDS["n3","evaluation"]+7}.json.gz').write_bytes(b'existing')
    assert control.missing_ranges(directory,'n3',0)==[(5,7),(8,24)]
    assert all((directory/key).read_bytes()==value for key,value in before.items())
    (directory/'case_009_999.json.gz').write_bytes(b'wrong seed')
    with pytest.raises(ValueError,match='wrong-seed'):control.missing_ranges(directory,'n3',0)


def test_full_records_can_resume_after_last_cli_checkpoint_was_not_saved(tmp_path):
    directory=part(tmp_path,'n3','evaluation',1)
    ck=io._read(directory/'checkpoint.json');ck['last_index']=46;io._write(directory/'checkpoint.json',ck)
    assert not control.part_progress(directory,'n3','evaluation',1)['complete']
    io._write(directory/'orchestration_complete.json',dict(policy_id=control.POLICY,phase='evaluation',
        bank='n3',start=24,stop=48,completed=24))
    assert control.part_progress(directory,'n3','evaluation',1)['complete']


def test_seed_overlap_is_rejected_before_statistical_outputs():
    control.require_disjoint(dict(records=[dict(seed=1)]),dict(records=[dict(seed=2)]))
    with pytest.raises(ValueError,match='seed overlap'):
        control.require_disjoint(dict(records=[dict(seed=1)]),dict(records=[dict(seed=1)]))


def test_failed_child_preserves_cases_and_stops_false_completion(tmp_path):
    controller=control.Controller(tmp_path);directory=part(tmp_path,'n3','evaluation',0,count=1)
    original=next(directory.glob('case_*')).read_bytes()
    controller.processes[0]=SimpleNamespace(poll=lambda:2,returncode=2)
    with pytest.raises(RuntimeError,match='existing cases retained'):controller.workers_finished()
    assert next(directory.glob('case_*')).read_bytes()==original


def test_hidden_workers_are_launched_only_with_two_saved_thresholds(tmp_path,monkeypatch):
    controller=control.Controller(tmp_path);calls=[]
    for bank in control.BANKS:thresholds(tmp_path,bank)
    directory=control.part_directory(tmp_path,'n3','calibration',0);directory.mkdir(parents=True)
    io._write(directory/'freeze.json',dict(source_sha256={}))
    monkeypatch.setattr(control,'check_current_sources',lambda f:None)
    monkeypatch.setattr(control,'pid_alive',lambda pid:False)
    def launch(args,**kwargs):
        calls.append((args,kwargs));return SimpleNamespace(pid=700+len(calls))
    monkeypatch.setattr(control.subprocess,'Popen',launch)
    controller.launch_evaluation()
    assert len(calls)==2
    assert [args[args.index('--worker')+1] for args,_ in calls]==['0','1']
    assert all(kwargs['env']['OPENBLAS_NUM_THREADS']=='1' and kwargs['env']['OMP_NUM_THREADS']=='1'
        for _,kwargs in calls)
    assert all(kwargs['creationflags']==getattr(control.subprocess,'CREATE_NO_WINDOW',0) for _,kwargs in calls)


def test_completed_controller_restart_does_not_reopen_data_or_launch_jobs(tmp_path,monkeypatch):
    path=tmp_path/'outputs'/'mobile_v2_orchestration_checkpoint.json'
    io._write(path,dict(policy_id=control.POLICY,status='statistical_complete',controller_pid=None,steps={},workers={},events=[]))
    controller=control.Controller(tmp_path)
    monkeypatch.setattr(controller,'advance',lambda:(_ for _ in ()).throw(AssertionError('must not restart')))
    controller.run()
    assert controller.state['status']=='statistical_complete'


def test_transient_direct_checkpoint_write_is_retried_not_treated_as_complete(tmp_path):
    directory=part(tmp_path,'n3','calibration',0)
    (directory/'checkpoint.json').write_text('{',encoding='utf-8')
    progress=control.part_progress(directory,'n3','calibration',0)
    assert progress['count']==48 and not progress['complete'] and progress['read_error']=='JSONDecodeError'
    io._write(directory/'checkpoint.json',dict(bank='n3',phase='calibration',completed=48,planned=48,last_index=47))
    assert control.part_progress(directory,'n3','calibration',0)['complete']


def test_worker_tail_failure_after_all_records_does_not_force_overwriting_cases(tmp_path,monkeypatch):
    controller=control.Controller(tmp_path)
    for bank in control.BANKS:
        for p in (0,1):part(tmp_path,bank,'evaluation',p)
    controller.processes[0]=SimpleNamespace(poll=lambda:2,returncode=2)
    io._write(tmp_path/'outputs'/'mobile_v2_evaluation_worker0.json',dict(status='failed'))
    monkeypatch.setattr(control,'pid_alive',lambda pid:False)
    assert controller.workers_finished()
    assert controller.state['events'][0]['records_complete'] is True
    assert control.missing_ranges(control.part_directory(tmp_path,'n3','evaluation',0),'n3',0)==[]


def test_changed_calibration_rule_is_rejected_even_if_policy_matches(tmp_path):
    for bank in control.BANKS:thresholds(tmp_path,bank)
    path=thresholds(tmp_path,'n4');value=io._read(path);value['alpha']=.2;io._write(path,value)
    with pytest.raises(ValueError,match='identity/count/rules differ'):control.require_thresholds(tmp_path)
