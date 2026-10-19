"""Resumable postprocessing of the completed, frozen response-uncertainty batch.

This CLI never starts histories or changes science/configuration.  ``--check``
only reads readiness.  Actual work requires controller ``statistical_complete``
and all three successful phase audit files.  Each attempt has fresh products;
completed products are hash-verified before reuse.  A final manifest and a brief
review sheet organize evidence; scientific superiority remains a human review.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import redirect_stdout,redirect_stderr
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
import xml.etree.ElementTree as ET


HERE=Path(__file__).resolve().parent
PHASES=('development','calibration','evaluation')
NUMERIC_THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')
SCHEMA='ev_bayes_response_finish_v1'


def utc():return datetime.now(timezone.utc).isoformat()


def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        import gzip
        with gzip.open(path,'rt',encoding='utf-8') as handle:return json.load(handle)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    if hasattr(value,'tolist'):return value.tolist()
    if hasattr(value,'item'):return value.item()
    return value


def _json(value):return json.dumps(_plain(value),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n'


def atomic_json(path,value,*,replace=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:raise FileExistsError(path)
    temporary=path.with_name(path.name+'.%d.tmp'%os.getpid())
    with temporary.open('x',encoding='utf-8') as handle:handle.write(_json(value))
    if path.exists() and not replace:
        temporary.unlink();raise FileExistsError(path)
    temporary.replace(path)


def file_hash(path):
    path=Path(path);digest=sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):digest.update(chunk)
    return dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=digest.hexdigest())


def _identity(value):return sha256(json.dumps(_plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def process_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes=(wintypes.DWORD,wintypes.BOOL,wintypes.DWORD);kernel.OpenProcess.restype=wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD));kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        code=wintypes.DWORD()
        try:return bool(kernel.GetExitCodeProcess(handle,ctypes.byref(code))) and code.value==259
        finally:kernel.CloseHandle(handle)
    try:os.kill(int(pid),0);return True
    except ProcessLookupError:return False


def readiness(directory):
    """No mutation, raw decompression, source execution, fits or scientific calls."""
    root=Path(directory).resolve();reasons=[];phases={}
    try:frozen=read(root/'freeze.json');controller=read(root/'controller.json')
    except (OSError,ValueError) as exc:return dict(ready=False,input_root=str(root),reasons=[str(exc)])
    if controller.get('status')!='statistical_complete':reasons.append('controller status is '+str(controller.get('status')))
    if controller.get('policy_id')!=frozen.get('policy_id'):reasons.append('controller/freeze policy mismatch')
    if any(w.get('exit_code') is None for w in controller.get('workers',[])):reasons.append('controller still lists unfinished workers')
    for phase in PHASES:
        specs=frozen.get('specifications',{}).get(phase)
        if specs is None:reasons.append('frozen specification missing '+phase);continue
        expected={'history_%03d.json.gz'%s['index'] for s in specs}
        present={p.name for p in (root/phase).glob('history_*.json.gz')}
        audit_path=root/('audit_'+phase+'.json');audit_ok=False
        if audit_path.exists():
            try:
                checked=read(audit_path)
                audit_ok=bool(checked.get('success')) and checked.get('policy_id')==frozen.get('policy_id')
                phase_coverage=checked.get('coverage',{}).get(phase,{})
                audit_ok &= bool(phase_coverage.get('exact_coverage'))
            except (OSError,ValueError):pass
        phases[phase]=dict(expected=len(expected),present=len(present),missing=sorted(expected-present),
                           unexpected=sorted(present-expected),existing_audit_success=bool(audit_ok))
        if expected!=present:reasons.append('raw coverage incomplete or unexpected for '+phase)
        if not audit_ok:reasons.append('successful phase audit missing for '+phase)
    return dict(ready=not reasons,input_root=str(root),controller_status=controller.get('status'),
                policy_id=frozen.get('policy_id'),phases=phases,reasons=reasons)


def regression_files():
    names={p.name for pattern in ('test_ev_bayes_response_*.py','test_ev_bayes_small_*.py') for p in HERE.glob(pattern)}
    names.update(('test_ev_mobile_v4_domain.py','test_ev_rx_precision.py'))
    missing=[name for name in names if not (HERE/name).exists()]
    if missing:raise ValueError('required regression test missing: '+str(missing))
    return [HERE/name for name in sorted(names)]


def step_plan():
    return [dict(name='inventory',kind='inventory'),
            *[dict(name='analysis_'+p,kind='analysis',phase=p) for p in ('calibration','evaluation')],
            *[dict(name='diagnostics_'+p,kind='diagnostics',phase=p) for p in ('calibration','evaluation')],
            dict(name='history_support_evaluation',kind='history_support',phase='evaluation'),
            dict(name='regression',kind='tests')]


def runtime_evidence_files(directory):
    """Operational evidence is sealed separately from the 140 science files."""
    runtime=Path(directory)/'runtime_recovery_v1'
    if not runtime.exists():return []
    result=[]
    for path in sorted(runtime.rglob('*')):
        relative=path.relative_to(runtime)
        if any(part.startswith(('test_tmp','pytest_tmp')) or part=='__pycache__' for part in relative.parts):continue
        if not path.is_file():continue
        if not path.resolve().is_relative_to(runtime.resolve()):raise ValueError('runtime evidence escapes its directory')
        result.append(path)
    return result


def runtime_source_files():
    names=('ev_bayes_response_controller.py','ev_bayes_response_runtime_entry.py',
        'ev_bayes_response_recovery_controller.py','test_ev_bayes_response_runtime_entry.py',
        'test_ev_bayes_response_recovery_controller.py')
    return [HERE/name for name in names if (HERE/name).exists()]


def input_seal(directory):
    root=Path(directory);frozen=read(root/'freeze.json')
    paths=[root/name for name in ('freeze.json','source_snapshot.json.gz','controller.json')]
    paths.extend(root/('audit_'+p+'.json') for p in PHASES)
    paths.extend(root/p/('history_%03d.json.gz'%s['index']) for p in PHASES for s in frozen['specifications'][p])
    runtime_files=runtime_evidence_files(root)
    paths.extend(runtime_files)
    dependencies=[HERE/('ev_bayes_response_'+name+'.py') for name in ('finish','analysis','diagnostics','history_support','audit')]
    science=[HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name for name in frozen.get('source_sha256',{})]
    return dict(input_files=[file_hash(p) for p in paths],postprocess_sources=[file_hash(p) for p in dependencies],
                scientific_sources_current=[file_hash(p) for p in science],
                non_scientific_runtime=dict(evidence_files=[file_hash(p) for p in runtime_files],
                    sources=[file_hash(p) for p in runtime_source_files()],test_temporary_directories_excluded=True),
                regression_tests=[file_hash(p) for p in regression_files()])


def _utc_time(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')) if value else None


def runtime_cost_summary(directory):
    """Report overlapping wall-time scopes; never add them to record.seconds."""
    root=Path(directory);runtime=root/'runtime_recovery_v1';controller=read(root/'controller.json')
    events_path=runtime/'attempts.jsonl'
    events=[json.loads(line) for line in events_path.read_text(encoding='utf-8-sig').splitlines() if line.strip()] if events_path.exists() else []
    incidents=[]
    for path in sorted(runtime.glob('native_failure_*.json')):
        row=read(path);timing=row.get('original_worker_timing',{})
        incidents.append(dict(file=str(path.resolve()),pid=row.get('worker_pid'),
            exit_code=row.get('exit_code_decimal'),first_unsaved_history=row.get('first_unsaved_history'),
            whole_worker_wall_seconds=timing.get('wall_seconds_approx'),
            started_utc=timing.get('started_utc'),fault_utc=timing.get('fault_event_utc'),
            scope='whole original worker, including already saved histories and unfinished work',
            additional_lost_seconds=None,additional_loss_unknown=True,additive_to_record_seconds=False))
    starts={r['attempt_id']:r for r in events if r.get('event')=='attempt_started'}
    finished=[r for r in events if r.get('event')=='attempt_finished']
    completed_ids={r.get('attempt_id') for r in finished}
    attempts=[]
    for row in finished:
        start=starts.get(row.get('attempt_id'),{})
        elapsed=row.get('elapsed_seconds')
        scope=row.get('elapsed_scope','unknown')
        attempts.append(dict(attempt_id=row.get('attempt_id'),phase=row.get('phase'),index=row.get('index'),pid=row.get('pid'),
            attempt=start.get('attempt'),exit_code=row.get('exit_code'),elapsed_seconds=elapsed,
            elapsed_scope=scope,elapsed_unknown=elapsed is None,
            includes_record_seconds=True,additive_to_record_seconds=False,
            final_record_saved=any(v is not None for v in row.get('final_records',{}).values())))
    adopted=[dict(pid=r.get('pid'),phase=r.get('phase'),elapsed_seconds=r.get('elapsed_seconds'),
        elapsed_unknown=r.get('elapsed_seconds') is None,elapsed_scope='since adoption only',
        includes_record_seconds=True,additive_to_record_seconds=False,
        original_pre_adoption_elapsed_seconds=None) for r in events if r.get('event')=='adopted_worker_finished']
    controller_start=_utc_time(controller.get('started'));controller_end=_utc_time(controller.get('updated'))
    controller_span=(controller_end-controller_start).total_seconds() if controller_start and controller_end else None
    recovery_starts=[r for r in events if r.get('event')=='recovery_started']
    recovery_ends=[r for r in events if r.get('event')=='statistical_complete']
    recovery_span=(_utc_time(recovery_ends[-1]['time'])-_utc_time(recovery_starts[0]['time'])).total_seconds() if recovery_starts and recovery_ends else None
    full_attempts=[r for r in attempts if r['elapsed_scope']=='full worker wall time']
    adopted_attempts=[r for r in attempts if r['elapsed_scope']=='since adoption']
    measured=lambda rows:sum(r['elapsed_seconds'] for r in rows if r['elapsed_seconds'] is not None)
    known_native=lambda code:code is not None and (int(code)&0xFFFFFFFF)>=0x80000000
    native_events=[r for r in events if r.get('event') in ('original_native_failure','attempt_finished') and known_native(r.get('exit_code'))]
    return dict(schema=SCHEMA+'_runtime_costs',recovery_present=runtime.exists(),
        original_controller_calendar_span_seconds=controller_span,
        original_controller_span_scope='original started to final updated, including development, waits, audits, interruptions and recovery gaps',
        recovery_calendar_span_seconds=recovery_span,
        original_worker_incidents=incidents,new_attempts=attempts,adopted_worker_segments=adopted,
        new_attempts_started=len(starts),new_attempts_finished=len(finished),
        attempts_without_finish_event=sorted(set(starts)-completed_ids),
        full_new_worker_wall_seconds_measured=measured(full_attempts),
        full_new_worker_wall_attempts_measured=sum(r['elapsed_seconds'] is not None for r in full_attempts),
        new_attempt_elapsed_unknown=sum(r['elapsed_unknown'] for r in attempts)+len(set(starts)-completed_ids),
        adopted_since_adoption_wall_seconds_measured=measured(adopted)+measured(adopted_attempts),
        adopted_elapsed_unknown=sum(r['elapsed_unknown'] for r in adopted)+sum(r['elapsed_unknown'] for r in adopted_attempts),
        native_failure_events=len(native_events),native_failures=[dict(pid=r.get('pid'),phase=r.get('phase'),index=r.get('index'),exit_code=r.get('exit_code')) for r in native_events],
        elapsed_categories_are_overlapping=True,grand_total_seconds=None,
        additional_lost_work_seconds=None,additional_lost_work_not_identifiable=True,
        aggregation_rule='report record.seconds, worker wall, adopted segments and calendar spans separately; no sum across scopes',
        scientific_failures_remain_in_raw_records=True)


def copy_runtime_evidence(directory,output,seal):
    root,output=Path(directory),Path(output);runtime=root/'runtime_recovery_v1'
    evidence=seal.get('non_scientific_runtime',{});copied=[]
    for kind,key in (('runtime_evidence','evidence_files'),('runtime_sources','sources')):
        for meta in evidence.get(key,[]):
            source=Path(meta['path'])
            if file_hash(source)!=meta:raise ValueError('sealed runtime evidence changed '+str(source))
            relative=source.relative_to(runtime.resolve()) if kind=='runtime_evidence' else Path(source.name)
            target=output/kind/relative
            if target.name.endswith('.tmp'):target=target.with_name(target.name+'.preserved')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():raise FileExistsError(target)
            shutil.copyfile(source,target)
            copied.append(dict(kind=kind,input=meta,copied=file_hash(target)))
    return copied


def runtime_review_lines(costs):
    if not costs.get('recovery_present'):return []
    lines=['恢复费用分列单历史新进程、接管后片段和控制器日历跨度；这些时间有重叠，不与record.seconds相加。']
    for row in costs.get('original_worker_incidents',[]):
        elapsed=row.get('whole_worker_wall_seconds')
        value='未知' if elapsed is None else '约%.3f秒'%elapsed
        lines.append('原故障worker %s的整段运行时间%s，包含已保存历史；额外损失耗时未知。'%(row.get('pid'),value))
    lines.append('新单历史进程已记录完整运行时间合计%.3f秒；接管后片段合计%.3f秒。缺失新attempt耗时%d项、缺失接管片段耗时%d项。'%(
        costs.get('full_new_worker_wall_seconds_measured',0.),costs.get('adopted_since_adoption_wall_seconds_measured',0.),
        costs.get('new_attempt_elapsed_unknown',0),costs.get('adopted_elapsed_unknown',0)))
    span=costs.get('original_controller_calendar_span_seconds')
    lines.append('原控制器起点至最终完成的日历跨度%s，包含等待、审计、中断及恢复间隔。'%('未知' if span is None else '为%.3f秒'%span))
    return lines


def _inventory_record(record,phase,path_meta,policy_id):
    if record.get('policy_id')!=policy_id:raise ValueError('raw policy mismatch '+path_meta['path'])
    proposals=[]
    for family,domains in record.get('planning_logs',{}).items():
        for domain,row in domains.items():
            for event in row.get('events',[]):
                mip=event.get('milp',{})
                proposals.append(dict(family=family,domain=domain,accepted=bool(event.get('accepted')),
                    solver_success=bool(mip.get('success')),failure=mip.get('failure'),stop_reason=mip.get('stop_reason'),
                    full_AC_failure=event.get('full_AC_failure'),solver_seconds=mip.get('solver_seconds'),
                    wall_seconds=mip.get('seconds'),start_supplied=mip.get('start_supplied'),
                    start_provided_count=mip.get('start_provided_count'),start_used=mip.get('start_used')))
    integrals=[]
    for model,pred in record.get('prediction',{}).items():
        for candidate,row in pred.get('candidate_updates',{}).items():
            for stage in ('T','TV'):
                bounded=row.get(stage+'_line_integration')
                if bounded:
                    d=bounded.get('diagnostics',{})
                    integrals.append(dict(model=model,candidate=candidate,stage=stage,status=bounded.get('status'),
                        log_mass=bounded.get('log_mass'),points=d.get('points'),ess=d.get('ess'),relative_se=d.get('relative_se')))
    physical=[];prediction_failures=[]
    for key,action in record.get('actions',{}).items():
        prediction_failures.extend(dict(action_key=key,model=m,reason=v) for m,v in action.get('prediction_failures',{}).items())
        for regime,row in action.get('actual',{}).items():
            physical.append(dict(action_key=key,regime=regime,ac_equation_residual=row.get('ac_equation_residual'),
                power_balance_residual=row.get('power_balance_residual'),v_min=row.get('v_min'),v_max=row.get('v_max')))
    return dict(phase=phase,spec=record.get('spec'),status=record.get('status'),input=path_meta,
        policy_id=policy_id,wall_seconds=record.get('seconds'),failures=record.get('failures',{}),failure=record.get('failure'),
        training_complete=record.get('training_result',{}).get('all_complete'),
        training_seconds=record.get('training_result',{}).get('seconds'),
        prediction_seconds={m:r.get('seconds') for m,r in record.get('prediction',{}).items()},
        planning_seconds=record.get('planning_seconds',{}),objective_costs=record.get('objective_costs',{}),
        objective_cache_seconds_already_in_family_seconds=True,cold_standalone_cost_not_measured=True,
        candidate_miss=record.get('candidate_miss'),candidate_slots=len(record.get('candidate_names',[])),
        method_slots=len(record.get('methods',{})),unique_action_slots=len(record.get('actions',{})),
        operation_failures=[m for m,r in record.get('methods',{}).items() if m!='none' and not r.get('operation',{}).get('feasible',False)],
        prediction_failures=prediction_failures,proposals=proposals,bounded_integrals=integrals,
        actual_physical_actions=physical,conditional_noise_repeats=record.get('config',{}).get('actual_noise_repeats'))


def inventory(directory,output,work,seal):
    root,output,work=Path(directory),Path(output),Path(work);frozen=read(root/'freeze.json')
    snapshot=read(root/'source_snapshot.json.gz');source=[]
    for name,expected in frozen['source_sha256'].items():
        data=base64.b64decode(snapshot['source_base64'][name],validate=True)
        if sha256(data).hexdigest()!=expected:raise ValueError('snapshot source hash mismatch '+name)
        path=HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name
        current=file_hash(path)
        if current['sha256']!=expected:raise ValueError('current frozen science hash mismatch '+name)
        source.append(dict(name=name,**current))
    input_map={r['path']:r for r in seal['input_files']};rows=[];audits={}
    for phase in PHASES:
        checked=read(root/('audit_'+phase+'.json'));audits[phase]=dict(success=checked['success'],
            scientific_complete_count=checked.get('scientific_complete_count'),counts=checked.get('counts'),
            maxima=checked.get('maxima'),error_count=checked.get('error_count'))
        saved_by_name={Path(r['input_file']).name:r for r in checked['records']}
        for spec in frozen['specifications'][phase]:
            path=root/phase/('history_%03d.json.gz'%spec['index']);meta=input_map[str(path.resolve())]
            old=saved_by_name.get(path.name)
            if old is None or old.get('input_sha256')!=meta['sha256']:raise ValueError('phase audit/raw hash mismatch '+str(path))
            cache=work/('inventory_'+phase+'_%03d.json'%spec['index'])
            if cache.exists():
                row=read(cache)
                if row.get('input')!=meta or row.get('policy_id')!=frozen['policy_id']:raise ValueError('immutable inventory cache differs '+str(cache))
            else:
                raw=read(path)
                if raw.get('spec')!=spec:raise ValueError('raw/freeze specification mismatch '+str(path))
                row=_inventory_record(raw,phase,meta,frozen['policy_id']);del raw;atomic_json(cache,row)
            rows.append(row)
            print('FINISH_INVENTORY',phase,spec['index'],row['status'],flush=True)
    by_phase={}
    for phase in PHASES:
        subset=[r for r in rows if r['phase']==phase];proposal=[p for r in subset for p in r['proposals']]
        by_phase[phase]=dict(expected_histories=len(frozen['specifications'][phase]),records=len(subset),
            status_counts=dict(Counter(r['status'] for r in subset)),complete_histories=sum(r['status']=='complete' for r in subset),
            wall_seconds_total=sum(r['wall_seconds'] or 0. for r in subset),wall_seconds_measured=sum(r['wall_seconds'] is not None for r in subset),
            proposal_slots=len(proposal),solver_failed=sum(not p['solver_success'] for p in proposal),
            full_AC_failed=sum(bool(p['full_AC_failure']) for p in proposal),accepted=sum(p['accepted'] for p in proposal),
            operation_failure_slots=sum(len(r['operation_failures']) for r in subset),
            prediction_failure_slots=sum(len(r['prediction_failures']) for r in subset),
            source_inference_failures=sum('joint_inference' in r['failures'] for r in subset),
            solver_stop_reasons=dict(Counter(str(p['stop_reason']) for p in proposal)),
            family_observed_seconds={f:sum(float(r['planning_seconds'].get(f,0.)) for r in subset)
                for f in sorted({f for r in subset for f in r['planning_seconds']})})
    runtime_copies=copy_runtime_evidence(root,output,seal)
    runtime_costs=runtime_cost_summary(root)
    atomic_json(output/'runtime_costs.json',runtime_costs)
    result=dict(schema=SCHEMA+'_inventory',policy_id=frozen['policy_id'],source_count=len(source),frozen_sources=source,
        phase_audits=audits,by_phase=by_phase,records=rows,
        runtime_recovery=dict(costs=runtime_costs,copied_evidence=runtime_copies,
            scientific_source_freeze_unchanged=True,operational_sources_are_not_scientific_policy=True),
        runtime_scope='observed shared-cache workflow; objective time is contained in family time; worker wall and calendar spans overlap record.seconds and are reported separately; no cold-standalone claim',
        failures_retained=True,no_new_scientific_histories=True)
    atomic_json(output/'inventory.json',result)


def history_support(directory,output,work):
    from . import ev_bayes_response_history_support as support
    root,output,work=Path(directory),Path(output),Path(work);frozen=read(root/'freeze.json')
    expected=[s for s in frozen['specifications']['evaluation'] if s.get('legacy_audit',False)]
    records=[];inputs=[]
    for spec in expected:
        path=root/'evaluation'/('history_%03d.json.gz'%spec['index']);meta=file_hash(path)
        cache=work/('history_support_evaluation_%03d.json'%spec['index'])
        if cache.exists():
            saved=read(cache)
            if saved['input']!=meta or saved['policy_id']!=frozen['policy_id']:raise ValueError('immutable history-support cache differs')
            result=saved['record']
        else:
            raw=read(path)
            if raw.get('policy_id')!=frozen['policy_id'] or raw.get('spec')!=spec:raise ValueError('history support raw identity mismatch')
            result=support.support_record(raw,multistart=2,max_nfev=100);del raw
            atomic_json(cache,dict(policy_id=frozen['policy_id'],input=meta,record=result))
        records.append(result);inputs.append(dict(file=meta['path'],sha256=meta['sha256']))
        print('FINISH_HISTORY_SUPPORT',spec['index'],result['success'],flush=True)
    result=dict(schema='ev_bayes_response_fixed_center_history_support_v1',phase='evaluation',policy_id=frozen['policy_id'],
        coverage=support.coverage(expected,[r['spec'] for r in records]),records=records,input_files=inputs,
        tool_sha256=file_hash(Path(support.__file__))['sha256'],source_freeze_sha256=file_hash(root/'freeze.json')['sha256'],
        source_fit_slots_expected=sum(r['denominators']['line_center_fit_slots_expected'] for r in records),future_or_truth_used=False,
        per_history_cache=True,multistart=2,max_nfev=100)
    atomic_json(output/'history_support.json',result)
    with (output/'history_support.md').open('x',encoding='utf-8') as handle:handle.write(support.markdown(result))


def junit_summary(path):
    root=ET.parse(path).getroot();suites=[root] if root.tag=='testsuite' else list(root.findall('testsuite'))
    totals={k:sum(int(s.get(k,'0')) for s in suites) for k in ('tests','errors','failures','skipped')}
    totals['seconds']=sum(float(s.get('time','0')) for s in suites)
    totals['passed']=totals['tests']-totals['errors']-totals['failures']-totals['skipped']
    return totals


def _command(step,directory,output,bootstrap):
    prefix=[sys.executable,'-m','research_experiments.mobile_ev_mle.']
    if step['kind'] in ('analysis','diagnostics'):
        command=[prefix[0],prefix[1],prefix[2]+'ev_bayes_response_'+step['kind'],
                 '--input',str(directory),'--output',str(output),'--phase',step['phase']]
        if step['kind']=='analysis':command+=['--bootstrap',str(bootstrap)]
        return command
    if step['kind']=='tests':
        return [sys.executable,'-m','pytest','-q',*[str(p) for p in regression_files()],
                '--junitxml',str(output/'final_tests.xml'),'--basetemp',str(output/'pytest_tmp')]
    return None


def _artifacts(directory):
    return [file_hash(p) for p in sorted(Path(directory).rglob('*')) if p.is_file() and not p.name.endswith('.tmp')]


def _verify_artifacts(artifacts):
    for meta in artifacts:
        if file_hash(meta['path'])!=meta:raise ValueError('completed immutable product changed '+meta['path'])


def execute_step(step,directory,attempt,work,seal,bootstrap,on_child):
    result=attempt/'result';result.mkdir();log=attempt/'stdout.log'
    command=_command(step,directory,result,bootstrap)
    if command is not None:
        env=os.environ.copy();env.update({name:'1' for name in NUMERIC_THREADS});env['PYTHONIOENCODING']='utf-8'
        with log.open('x',encoding='utf-8') as handle:
            handle.write('COMMAND '+json.dumps(command,ensure_ascii=False)+'\n');handle.flush()
            process=subprocess.Popen(command,cwd=HERE.parents[1],env=env,stdout=handle,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));on_child(process.pid)
            try:code=process.wait()
            except BaseException:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                raise
        if code:raise RuntimeError(step['name']+' subprocess exited '+str(code)+'; see '+str(log))
        if step['kind']=='tests':
            tests=junit_summary(result/'final_tests.xml')
            if tests['errors'] or tests['failures']:raise RuntimeError('regression XML has failures')
            atomic_json(result/'tests_summary.json',dict(**tests,command=command,single_numeric_thread=True))
    else:
        with log.open('x',encoding='utf-8') as handle,redirect_stdout(handle),redirect_stderr(handle):
            handle.write('IN_PROCESS '+step['kind']+'\n');handle.flush()
            if step['kind']=='inventory':inventory(directory,result,work,seal)
            elif step['kind']=='history_support':history_support(directory,result,work)
            else:raise ValueError('unknown finish step '+step['kind'])


def _completed_result(state,name,filename):
    step=state['steps'][name]
    return Path(step['attempts'][-1]['directory'])/'result'/filename


def finalize(directory,output,state,seal):
    root,output=Path(directory),Path(output);frozen=read(root/'freeze.json')
    inventory_result=read(_completed_result(state,'inventory','inventory.json'))
    tests=read(_completed_result(state,'regression','tests_summary.json'))
    analyses={p:read(_completed_result(state,'analysis_'+p,'summary.json')) for p in ('calibration','evaluation')}
    diagnostics={p:read(_completed_result(state,'diagnostics_'+p,'diagnostics.json')) for p in ('calibration','evaluation')}
    support=read(_completed_result(state,'history_support_evaluation','history_support.json'))
    gates=dict(controller_statistical_complete=readiness(root)['ready'],
        completed_phase_audits=all(v['success'] for v in inventory_result['phase_audits'].values()),
        analysis_coverage=all(v['coverage']['exact'] for v in analyses.values()),
        diagnostics_coverage=all(v['coverage']['exact'] for v in diagnostics.values()),
        history_support_coverage=bool(support['coverage']['exact']),regression_passed=not tests['errors'] and not tests['failures'])
    manifest=dict(schema=SCHEMA+'_evidence',policy_id=frozen['policy_id'],input_root=str(root.resolve()),created=utc(),
        input_seal=seal,postprocess_options=state['options'],gates=gates,
        artifacts=[meta for step in state['steps'].values() for attempt in step['attempts'] for meta in attempt.get('artifacts',[])],
        phase_counts=inventory_result['by_phase'],tests=tests,
        runtime_recovery=inventory_result.get('runtime_recovery',{}),
        history_support=dict(histories=len(support['records']),source_fit_slots_expected=support['source_fit_slots_expected'],
            successful_histories=sum(bool(r.get('success')) for r in support['records']),
            fit_slots_failed=sum(r['denominators']['fit_slots_failed'] for r in support['records'])),
        diagnostic_failures={p:sum(len(r.get('failures',[])) for r in d['records']) for p,d in diagnostics.items()},
        requires_human_result_review=True,method_superiority_claim_generated=False,
        scientific_scope='frozen histories and saved actions only; history-support source fits were prespecified',
        cost_scope=inventory_result['runtime_scope'])
    if not all(gates.values()):raise ValueError('postprocessing completeness gate failed: '+str(gates))
    manifest_path=output/'evidence_manifest.json'
    if manifest_path.exists():
        previous=read(manifest_path)
        if {k:v for k,v in previous.items() if k!='created'}!={k:v for k,v in manifest.items() if k!='created'}:
            raise ValueError('existing immutable final manifest differs')
        manifest=previous
    else:atomic_json(manifest_path,manifest)
    lines=['# 冻结实验后处理核对单','',f"Policy `{frozen['policy_id']}`。后处理步骤和完整性检查已完成，结果等待人工核对。",'',
           '|阶段|完整历史/记录|求解失败/提案|接受提案|AC提案失败|','|---|---:|---:|---:|---:|']
    for p,row in inventory_result['by_phase'].items():
        lines.append(f"|{p}|{row['complete_histories']}/{row['records']}|{row['solver_failed']}/{row['proposal_slots']}|{row['accepted']}|{row['full_AC_failed']}|")
    lines+=['',f"回归测试：{tests['passed']}通过，{tests['skipped']}跳过，{tests['failures']}失败，{tests['errors']}错误。",'',
        f"历史支持：{len(support['records'])}份预定历史，{support['source_fit_slots_expected']}个拟合槽；失败{manifest['history_support']['fit_slots_failed']}槽。",'',
        '费用按保存的共享缓存流程汇总；目标缓存时间已包含在各family时间中，二者不相加，也未测量冷启动独立费用。',
        *runtime_review_lines(inventory_result.get('runtime_recovery',{}).get('costs',{})),
        '计量噪声重复保留在所属历史内。所有推断、规划、物理或支持拟合失败均留在相应分母。','',
        '请核对校准/评价分离、同域平方对照、父计划排序与MILP新增收益、概率校准及移动贡献；本脚本不自动判定方法优越性。','',
        f"证据清单SHA256：`{file_hash(manifest_path)['sha256']}`。"]
    review=output/'review_summary.md';text='\n'.join(lines)+'\n'
    if review.exists():
        if review.read_text(encoding='utf-8')!=text:raise ValueError('existing immutable review summary differs')
    else:
        with review.open('x',encoding='utf-8') as handle:handle.write(text)
    return dict(manifest=file_hash(manifest_path),review=file_hash(output/'review_summary.md'))


def run(directory,output,*,bootstrap=1000,executor=execute_step):
    root,output=Path(directory).resolve(),Path(output).resolve()
    ready=readiness(root)
    if not ready['ready']:raise RuntimeError('postprocessing blocked until statistical_complete: '+str(ready['reasons']))
    if bootstrap<100:raise ValueError('bootstrap must be >=100')
    if output==root or any(output==root/p or root/p in output.parents for p in PHASES):
        raise ValueError('postprocessing output must not be a raw phase directory')
    output.mkdir(parents=True,exist_ok=True);lock=output/'finish.lock';checkpoint=output/'checkpoint.json'
    if lock.exists():
        previous=read(lock)
        if process_alive(previous['pid']):raise RuntimeError('finish already active: '+str(previous['pid']))
        lock.rename(output/('stale_finish_lock_'+str(os.stat(lock).st_mtime_ns)+'.json'))
    atomic_json(lock,dict(pid=os.getpid(),started=utc(),input_root=str(root)))
    state=None
    try:
        seal=input_seal(root);options=dict(bootstrap=int(bootstrap),history_support_multistart=2,history_support_max_nfev=100,
            diagnostics_all_factorial=False,regression_files=[str(p) for p in regression_files()])
        identity=_identity(dict(seal=seal,options=options))
        if checkpoint.exists():
            state=read(checkpoint)
            if state['input_identity']!=identity:raise ValueError('input/tool/test seal changed; retain this output and use a fresh directory')
            if state.get('status')=='complete':
                for entry in state['steps'].values():
                    for attempt in entry['attempts']:_verify_artifacts(attempt.get('artifacts',[]))
                _verify_artifacts(state['final_products'].values());return state
        else:
            if any(p.name not in ('finish.lock',) and not p.name.startswith('stale_finish_lock_') for p in output.iterdir()):
                raise FileExistsError('fresh output or existing finish checkpoint required')
            state=dict(schema=SCHEMA,policy_id=ready['policy_id'],input_root=str(root),input_identity=identity,
                options=options,created=utc(),status='running',steps={})
        work=output/'work';work.mkdir(exist_ok=True)
        for number,step in enumerate(step_plan()):
            entry=state['steps'].setdefault(step['name'],dict(step=step,status='pending',attempts=[]))
            if entry['status']=='complete':
                _verify_artifacts(entry['attempts'][-1]['artifacts']);continue
            if entry['attempts']:
                old=entry['attempts'][-1]
                if old.get('child_pid') and process_alive(old['child_pid']):raise RuntimeError('prior postprocessing child still alive: '+str(old['child_pid']))
            attempt=output/'steps'/('%02d_'%number+step['name'])/('attempt_%03d'%(len(entry['attempts'])+1))
            attempt.mkdir(parents=True)
            current=dict(directory=str(attempt),started=utc(),status='running',child_pid=None)
            entry['attempts'].append(current);entry['status']='running';state.update(status='running',current_step=step['name'],updated=utc())
            atomic_json(checkpoint,state,replace=True);began=perf_counter()
            def child(pid):
                current['child_pid']=pid;atomic_json(checkpoint,state,replace=True)
            try:
                executor(step,root,attempt,work,seal,bootstrap,child)
                current.update(status='complete',finished=utc(),seconds=perf_counter()-began,child_pid=None,artifacts=_artifacts(attempt))
                entry['status']='complete';atomic_json(checkpoint,state,replace=True)
            except BaseException as exc:
                current.update(status='failed',finished=utc(),seconds=perf_counter()-began,failure=type(exc).__name__+': '+str(exc),
                    traceback=traceback.format_exc(),artifacts=_artifacts(attempt))
                entry['status']='failed';state.update(status='postprocess_failure',updated=utc());atomic_json(checkpoint,state,replace=True);raise
        state['final_products']=finalize(root,output,state,seal)
        state.update(status='complete',updated=utc(),current_step=None,requires_human_result_review=True)
        atomic_json(checkpoint,state,replace=True);return state
    finally:
        if lock.exists():lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True);parser.add_argument('--output')
    parser.add_argument('--check',action='store_true');parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    if args.check:
        print(_json(readiness(args.input)),end='');return
    if not args.output:parser.error('--output is required unless --check')
    result=run(args.input,args.output,bootstrap=args.bootstrap)
    print('RESPONSE_FINISH',result['status'],result['policy_id'],flush=True)


if __name__=='__main__':main()
