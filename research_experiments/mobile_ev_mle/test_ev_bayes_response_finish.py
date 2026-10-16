"""IO-only smoke, completeness and restart checks for frozen postprocessing."""
import json
from pathlib import Path

import pytest

from . import ev_bayes_response_finish as f


def frozen(tmp_path,*,status='statistical_complete',specs=None):
    root=tmp_path/'raw';root.mkdir()
    specs={p:[] for p in f.PHASES} if specs is None else specs
    f.atomic_json(root/'freeze.json',dict(policy_id='unit_policy',specifications=specs,source_sha256={}))
    f.atomic_json(root/'controller.json',dict(policy_id='unit_policy',status=status,workers=[]))
    f.atomic_json(root/'source_snapshot.json.gz',dict(source_base64={}))
    for phase in f.PHASES:
        (root/phase).mkdir()
        f.atomic_json(root/('audit_'+phase+'.json'),dict(policy_id='unit_policy',success=True,
            coverage={phase:dict(exact_coverage=True)},records=[]))
    return root


def test_readiness_blocks_pending_without_creating_outputs(tmp_path):
    root=frozen(tmp_path,status='running');out=tmp_path/'finish'
    result=f.readiness(root)
    assert not result['ready'] and 'controller status is running' in result['reasons']
    with pytest.raises(RuntimeError,match='statistical_complete'):f.run(root,out)
    assert not out.exists()


def test_readiness_requires_exact_raw_coverage_and_all_existing_audits(tmp_path):
    specs={p:[] for p in f.PHASES};specs['evaluation']=[dict(index=0)]
    root=frozen(tmp_path,specs=specs)
    assert not f.readiness(root)['ready']
    (root/'evaluation'/'history_000.json.gz').write_bytes(b'raw-content-hash-only')
    assert f.readiness(root)['ready']
    (root/'audit_calibration.json').unlink()
    assert not f.readiness(root)['ready']


def test_no_overwrite_atomic_products_and_hash_validation(tmp_path):
    path=tmp_path/'a.json';f.atomic_json(path,dict(a=1));original=path.read_bytes()
    with pytest.raises(FileExistsError):f.atomic_json(path,dict(a=2))
    assert path.read_bytes()==original
    meta=f.file_hash(path);f._verify_artifacts([meta])
    path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):f._verify_artifacts([meta])


def test_junit_counts_and_single_thread_test_command(tmp_path):
    xml=tmp_path/'junit.xml'
    xml.write_text('<testsuites><testsuite tests="10" failures="1" errors="0" skipped="2" time="1.2"/><testsuite tests="4" failures="0" errors="1" skipped="0" time=".3"/></testsuites>')
    assert f.junit_summary(xml)==dict(tests=14,failures=1,errors=1,skipped=2,passed=10,seconds=1.5)
    command=f._command(dict(kind='tests'),tmp_path,tmp_path,1000)
    assert command[:3]==[f.sys.executable,'-m','pytest']
    assert '--junitxml' in command and 'test_ev_bayes_response_finish.py' in ' '.join(command)


def test_inventory_preserves_failures_and_does_not_double_add_cache_cost():
    record=dict(policy_id='p',spec=dict(index=2),status='inference_failure',seconds=5.,
        failures=dict(joint_inference='integration precision not met'),
        planning_seconds=dict(square=1.,fixed_bc=2.),objective_costs=dict(fixed=dict(seconds=.7,cache_calls=3)),
        planning_logs=dict(fixed_bc=dict(fixed_charge=dict(events=[dict(accepted=False,milp=dict(success=False,
            failure='no_solver_incumbent',stop_reason='TIME_LIMIT'),full_AC_failure=None)]))),
        methods=dict(none=dict(),constant=dict(operation=dict(feasible=True))),actions={})
    row=f._inventory_record(record,'evaluation',dict(path='p',sha256='x'),'p')
    assert row['status']=='inference_failure' and row['proposals'][0]['failure']=='no_solver_incumbent'
    assert row['objective_cache_seconds_already_in_family_seconds']
    assert row['planning_seconds']==record['planning_seconds']
    assert row['operation_failures']==[]


def test_resume_reuses_finished_steps_and_preserves_failed_attempt(tmp_path,monkeypatch):
    root=frozen(tmp_path);output=tmp_path/'finish';seal=dict(raw='fixed')
    monkeypatch.setattr(f,'input_seal',lambda path:seal)
    monkeypatch.setattr(f,'regression_files',lambda:[])
    calls=[];fail=[True]
    def executor(step,root,attempt,work,inputs,bootstrap,on_child):
        calls.append(step['name']);result=attempt/'result';result.mkdir()
        (result/'value.txt').write_text(step['name'])
        if step['name']=='analysis_calibration' and fail[0]:
            fail[0]=False;raise RuntimeError('deliberate interruption')
    def finalize(root,out,state,inputs):
        path=out/'final.json'
        if not path.exists():f.atomic_json(path,dict(done=True))
        return dict(manifest=f.file_hash(path))
    monkeypatch.setattr(f,'finalize',finalize)
    with pytest.raises(RuntimeError,match='deliberate'):f.run(root,output,executor=executor)
    state=f.read(output/'checkpoint.json')
    assert state['steps']['inventory']['status']=='complete'
    failed=Path(state['steps']['analysis_calibration']['attempts'][0]['directory'])
    assert (failed/'result'/'value.txt').exists()
    state=f.run(root,output,executor=executor)
    assert state['status']=='complete'
    assert calls.count('inventory')==1 and calls.count('analysis_calibration')==2
    assert len(state['steps']['analysis_calibration']['attempts'])==2
    assert (failed/'result'/'value.txt').exists()
    before=list(calls);f.run(root,output,executor=executor);assert calls==before
    # Even completed resumes verify every referenced artifact.
    (failed/'result'/'value.txt').write_text('tampered')
    with pytest.raises(ValueError,match='changed'):f.run(root,output,executor=executor)


def test_seal_change_blocks_mixed_version_resume_without_touching_products(tmp_path,monkeypatch):
    root=frozen(tmp_path);out=tmp_path/'finish';seal=dict(version=1)
    monkeypatch.setattr(f,'input_seal',lambda path:dict(seal))
    monkeypatch.setattr(f,'regression_files',lambda:[])
    def stop(*args):raise RuntimeError('stop')
    with pytest.raises(RuntimeError,match='stop'):f.run(root,out,executor=stop)
    previous=(out/'checkpoint.json').read_bytes();seal['version']=2
    with pytest.raises(ValueError,match='seal changed'):f.run(root,out,executor=stop)
    assert (out/'checkpoint.json').read_bytes()==previous


def test_live_finish_or_child_prevents_duplicate_work(tmp_path,monkeypatch):
    root=frozen(tmp_path);out=tmp_path/'finish';out.mkdir()
    f.atomic_json(out/'finish.lock',dict(pid=456))
    monkeypatch.setattr(f,'process_alive',lambda pid:True)
    with pytest.raises(RuntimeError,match='already active'):f.run(root,out)
    assert (out/'finish.lock').exists()


def test_protected_raw_output_directory_is_rejected(tmp_path):
    root=frozen(tmp_path)
    with pytest.raises(ValueError,match='raw phase'):f.run(root,root/'evaluation'/'nested')


def test_step_plan_runs_only_prespecified_postprocessing():
    steps=f.step_plan()
    assert [s['name'] for s in steps]==['inventory','analysis_calibration','analysis_evaluation',
        'diagnostics_calibration','diagnostics_evaluation','history_support_evaluation','regression']
    commands=[f._command(s,Path('root'),Path('out'),1000) for s in steps if s['kind'] in ('analysis','diagnostics')]
    assert all('ev_bayes_response_study' not in ' '.join(c) for c in commands)
    assert all('--all-factorial' not in c for c in commands)


def test_finalization_is_resumable_after_manifest_written(tmp_path,monkeypatch):
    root=frozen(tmp_path);out=tmp_path/'finish';out.mkdir();state=dict(options={},steps={})
    by_phase={p:dict(complete_histories=0,records=0,solver_failed=0,proposal_slots=0,accepted=0,full_AC_failed=0) for p in f.PHASES}
    values=dict(inventory=dict(source='inventory.json',value=dict(by_phase=by_phase,
        phase_audits={p:dict(success=True) for p in f.PHASES},runtime_scope='shared workflow')),
        regression=dict(source='tests_summary.json',value=dict(passed=4,tests=4,errors=0,failures=0,skipped=0)),
        history_support_evaluation=dict(source='history_support.json',value=dict(coverage=dict(exact=True),records=[],source_fit_slots_expected=0)))
    for p in ('calibration','evaluation'):
        values['analysis_'+p]=dict(source='summary.json',value=dict(coverage=dict(exact=True)))
        values['diagnostics_'+p]=dict(source='diagnostics.json',value=dict(coverage=dict(exact=True),records=[]))
    for name,item in values.items():
        attempt=out/name;result=attempt/'result';result.mkdir(parents=True);f.atomic_json(result/item['source'],item['value'])
        state['steps'][name]=dict(attempts=[dict(directory=str(attempt),artifacts=f._artifacts(attempt))])
    a=f.finalize(root,out,state,{})
    old=(out/'evidence_manifest.json').read_bytes()
    b=f.finalize(root,out,state,{})
    assert a==b and (out/'evidence_manifest.json').read_bytes()==old
    manifest=f.read(out/'evidence_manifest.json')
    assert manifest['requires_human_result_review'] and not manifest['method_superiority_claim_generated']
