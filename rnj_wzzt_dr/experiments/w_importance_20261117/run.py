"""Freeze and run the paired W-importance study after the parent batch exits."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
import traceback

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    __package__='rnj_wzzt_dr.experiments.w_importance_20261117'

from ... import io, observations, history
from ...identification import algorithm
from ...progress import Progress

PACKAGE=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent
VARIANTS=('full','pool_only','skip')
PANELS=('long','grid')


def utc():return datetime.now(timezone.utc).isoformat()


def write(path,value):
    for attempt in range(9):
        try:
            io.write(path,value);return
        except PermissionError:
            if attempt==8:raise
            sleep(.01*2**attempt)


def checked(ref):
    path=Path(ref['path']).resolve()
    if io.ref(path)['sha256']!=ref['sha256']:raise ValueError('Changed sealed source: '+str(path))
    return path


def snapshot(out):
    target=Path(out)/'source_snapshot'/'rnj_wzzt_dr'
    for path in sorted(PACKAGE.rglob('*')):
        if not path.is_file():continue
        rel=path.relative_to(PACKAGE)
        if set(rel.parts)&{'__pycache__','.pytest_cache','outputs'}:continue
        if rel.parts[0]=='experiments' and rel.parts[1] not in (STUDY.name,'wzzt_long_grid_20261117'):continue
        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    return target,[io.ref(p) for p in sorted(target.rglob('*')) if p.is_file()]


def task_specs(entries):
    tasks=[]
    for panel in PANELS:
        chosen=[e for e in entries if panel=='grid' or e['long_selected']]
        for index,entry in enumerate(sorted(chosen,key=lambda e:(e['N'],e['case'],e['pq_sd'],e['repeat'],e['id']))):
            order=VARIANTS[index%3:]+VARIANTS[:index%3]
            for variant in order:
                total,step,reserve,ecap=(14400.,600.,600.,6900.) if panel=='long' else (600.,10.,60.,270.)
                cfg=algorithm.SearchConfig(loss='mse',w_mode=variant,e_seconds=ecap,
                    total_seconds=total,per_mip_seconds=step,output_reserve_seconds=reserve,qp_seconds=300.)
                cfg.validate();ident=f'{panel}__{entry["id"]}__{variant}'
                tasks.append(dict(task_id=ident,panel=panel,input_id=entry['id'],method='101',loss='mse',
                    variant=variant,config=asdict(cfg),result_dir='tasks/'+ident))
    if len(tasks)!=528 or len({t['task_id'] for t in tasks})!=528:raise ValueError('Changed planned denominator')
    return tasks


def prepare(parent,out):
    parent,out=Path(parent).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Keep existing run: '+str(out))
    previous=io.read(parent/'protocol.json')
    entries=[e for e in io.read(checked(previous['inputs_ref'])) if e['panel']=='primary']
    if len(entries)!=160 or sum(e['long_selected'] for e in entries)!=16:raise ValueError('Parent input qualification changed')
    for entry in entries:
        if entry['status']!='complete':raise ValueError('Unavailable frozen input: '+entry['id'])
        for ref in entry['files'].values():checked(ref)
    tasks=task_specs(entries);out.mkdir(parents=True)
    source_root,sources=snapshot(out)
    write(out/'inputs.json',entries);write(out/'tasks.json',tasks)
    protocol=dict(created_utc=utc(),parent_run=str(parent),parent_protocol=io.ref(parent/'protocol.json'),
        inputs_ref=io.ref(out/'inputs.json'),tasks_ref=io.ref(out/'tasks.json'),sources=sources,
        source_root=str(source_root),private_driver=str(source_root/'experiments'/STUDY.name/'run.py'),
        planned_inputs=160,planned_tasks=528,panels=list(PANELS),variants=list(VARIANTS),workers=3,
        dispatch_wall_seconds=96*3600.,environment=io.environment(),
        protocol_document=io.ref(STUDY/'PROTOCOL.md'),fresh_full_reference=True,
        fixed_E_allowance=True,truth_used_in_fitting=False,default_changed=False,
        repeated_input_scope='long16 is a subset of grid160; not 176 independent inputs')
    write(out/'protocol.json',protocol)
    return protocol


class ResearchProgress(Progress):
    def _write(self,path,value):write(path,value)

    def __call__(self,event):
        if event.get('event') is not None:
            detail=event['event'];number=len(list((self.directory/'stages').glob('*.json')))
            write(self.directory/'stages'/f'{number:04d}_{detail["stage"]}.json',dict(
                stage=event['stage'],event=detail,point=event.get('point'),time_utc=utc()))
        super().__call__(event)


def execute(out,task_id):
    out=Path(out).resolve();protocol=io.read(out/'protocol.json')
    tasks=io.read(checked(protocol['tasks_ref']));task=next(t for t in tasks if t['task_id']==task_id)
    entry=next(e for e in io.read(checked(protocol['inputs_ref'])) if e['id']==task['input_id'])
    directory=out/task['result_dir'];directory.mkdir(parents=True,exist_ok=False)
    started=perf_counter();write(directory/'started.json',dict(time_utc=utc(),pid=os.getpid(),task_id=task_id))
    progress=ResearchProgress(directory,method='101',loss='mse',mode='off',every=10.)
    result=dict(task_id=task_id,input_id=task['input_id'],panel=task['panel'],variant=task['variant'],
        method='101',loss='mse',source_protocol=io.ref(out/'protocol.json'),config=task['config'],
        truth_used=False,new_day_loaded=False)
    try:
        if not Path(algorithm.__file__).resolve().is_relative_to(Path(protocol['source_root'])):
            raise RuntimeError('Worker imported mutable source')
        train=observations.load(checked(entry['files']['train.npz']))
        validation=observations.load(checked(entry['files']['validation.npz']))
        if list(train.labels)!=entry['labels'] or list(validation.labels)!=entry['labels']:
            raise ValueError('Node order changed')
        progress(dict(stage='load_train_validation'))
        native=algorithm.fit(train,validation,method='101',config=algorithm.SearchConfig(**task['config']),heartbeat=progress)
        native['handoff']=dict(schema='rnj_wzzt_dr_v2',training_sha256=observations.fingerprint(train),
            validation_sha256=observations.fingerprint(validation),parameter_rule='Same MSE fit and strict-MSE validation')
        history.freeze_history(native,validation);write(directory/'native.json',native);progress.finish(native)
        result.update(status='complete',native=io.ref(directory/'native.json'),
            input_refs={k:entry['files'][k] for k in ('train.npz','validation.npz')},
            input_fingerprints=dict(train=observations.fingerprint(train),validation=observations.fingerprint(validation)),
            training_reference_loss=native['reference_loss'],reference_mse=native['reference_mse'],
            method_seconds=native['total_seconds'],total_cap_exceeded=native['total_cap_exceeded'])
    except Exception as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
            failure_details={k:getattr(exc,k) for k in ('receipt','qp_records','last_point','native_records') if hasattr(exc,k)})
        try:progress(dict(stage='failed',error=result['error']))
        except Exception as logging_error:result['failure_logging_error']=repr(logging_error)
    result['seconds']=perf_counter()-started;write(directory/'result.json',result)
    return result


def status(out):
    out=Path(out);protocol=io.read(out/'protocol.json');tasks=io.read(checked(protocol['tasks_ref']))
    counts={p:dict(planned=0,complete=0,failed=0,interrupted=0,budget_unrun=0,running=0,pending=0) for p in PANELS}
    running=[]
    for task in tasks:
        directory=out/task['result_dir'];group=counts[task['panel']];group['planned']+=1
        if (directory/'result.json').exists():
            state=io.read(directory/'result.json')['status'];group[state]+=1
        elif (directory/'started.json').exists():
            group['running']+=1;running.append(dict(task_id=task['task_id'],variant=task['variant']))
        else:group['pending']+=1
    return dict(time_utc=utc(),panels=counts,running=running,stop_requested=(out/'STOP').exists())


def parent_ready(parent):
    parent=Path(parent);execution=parent/'execution.json'
    if (parent/'running.lock').exists():return False,'parent_controller_active'
    if not execution.exists():return False,'parent_not_terminal'
    if io.read(execution).get('status')!='all_terminal':return False,'parent_has_pending'
    return True,'parent_all_terminal'


def validate_reuse(out,task):
    path=Path(out)/task['result_dir']/'result.json';result=io.read(path)
    if result['task_id']!=task['task_id'] or result['source_protocol']!=io.ref(Path(out)/'protocol.json'):
        raise ValueError('Changed result identity: '+str(path))
    if result['status']=='complete':
        checked(result['native'])
        for ref in result['input_refs'].values():checked(ref)


def report(out,allow_partial):
    try:
        from . import evaluate
        receipt=evaluate.study(out,allow_partial=allow_partial)
        write(Path(out)/'latest_evaluation.json',receipt)
        return receipt
    except Exception as exc:
        receipt=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),time_utc=utc())
        write(Path(out)/'report_failed.json',receipt);return receipt


def run(out):
    out=Path(out).resolve();protocol=io.read(out/'protocol.json')
    if Path(__file__).resolve()!=Path(protocol['private_driver']).resolve():
        return subprocess.call([sys.executable,'-B',protocol['private_driver'],'run','--out',str(out)])
    for ref in protocol['sources']:checked(ref)
    checked(protocol['parent_protocol']);tasks=io.read(checked(protocol['tasks_ref']));checked(protocol['inputs_ref'])
    lock=out/'running.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    active={}
    try:
        while True:
            ready,reason=parent_ready(protocol['parent_run'])
            write(out/'controller.json',dict(stage='ready' if ready else 'waiting_parent',reason=reason,pid=os.getpid(),time_utc=utc()))
            if ready:break
            if (out/'STOP').exists():return 0
            sleep(20.)
        schedule=out/'schedule.json'
        if not schedule.exists():write(schedule,dict(started_utc=utc(),pid=os.getpid()))
        started=datetime.fromisoformat(io.read(schedule)['started_utc']).timestamp()
        stop=False
        for panel in PANELS:
            pending=[]
            for task in (t for t in tasks if t['panel']==panel):
                directory=out/task['result_dir']
                if (directory/'result.json').exists():validate_reuse(out,task)
                elif directory.exists():write(directory/'result.json',dict(task_id=task['task_id'],status='interrupted',
                    error='Existing partial attempt retained; no automatic budget reset',source_protocol=io.ref(out/'protocol.json')))
                else:pending.append(task)
            queue=iter(pending);active.clear();exhausted=False;last_status=0.
            while active or not exhausted:
                stop=stop or (out/'STOP').exists() or datetime.now(timezone.utc).timestamp()-started>=protocol['dispatch_wall_seconds']
                while not stop and len(active)<protocol['workers'] and not exhausted:
                    try:task=next(queue)
                    except StopIteration:exhausted=True;break
                    logs=out/'worker_logs';logs.mkdir(exist_ok=True)
                    stdout=(logs/(task['task_id']+'.stdout')).open('wb')
                    stderr=(logs/(task['task_id']+'.stderr')).open('wb')
                    process=subprocess.Popen([sys.executable,'-B',protocol['private_driver'],'worker','--out',str(out),'--task-id',task['task_id']],
                        stdout=stdout,stderr=stderr,cwd=str(out),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    active[process.pid]=(process,task,stdout,stderr)
                if stop:exhausted=True
                for pid,(process,task,stdout,stderr) in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    stdout.close();stderr.close();del active[pid]
                    path=out/task['result_dir']/'result.json'
                    if not path.exists():write(path,dict(task_id=task['task_id'],status='failed',
                        error='Worker exited without final receipt',exit_code=code,
                        source_protocol=io.ref(out/'protocol.json'),stderr=io.ref(out/'worker_logs'/(task['task_id']+'.stderr'))))
                    result=io.read(path)
                    print(json.dumps(dict(task_id=task['task_id'],status=result['status'],seconds=result.get('seconds')),ensure_ascii=False),flush=True)
                if perf_counter()-last_status>=10.:
                    write(out/'status.json',status(out));last_status=perf_counter()
                    write(out/'controller.json',dict(stage='running',panel=panel,pid=os.getpid(),time_utc=utc(),
                        active=[dict(pid=p,task_id=row[1]['task_id']) for p,row in active.items()]))
                if active:sleep(1.)
            write(out/(panel+'_terminal.json'),dict(time_utc=utc(),details=status(out)['panels'][panel]))
            report(out,True)
            if stop:break
        if datetime.now(timezone.utc).timestamp()-started>=protocol['dispatch_wall_seconds']:
            for task in tasks:
                if not (out/task['result_dir']).exists():write(out/task['result_dir']/'result.json',dict(task_id=task['task_id'],
                    status='budget_unrun',error='Global dispatch allowance exhausted',source_protocol=io.ref(out/'protocol.json')))
        final=status(out);write(out/'status.json',final)
        terminal=sum(row[k] for row in final['panels'].values() for k in ('complete','failed','interrupted','budget_unrun'))
        evaluation=report(out,terminal!=len(tasks))
        passed=terminal==len(tasks) and evaluation.get('status')=='passed'
        write(out/'execution.json',dict(status='complete' if passed else 'terminal_with_audit_failure' if terminal==len(tasks) else 'pending',
            terminal=terminal,planned=len(tasks),evaluation=evaluation,details=final))
        return 0 if passed else 1
    finally:
        live=[dict(pid=p,task_id=row[1]['task_id']) for p,row in active.items() if row[0].poll() is None]
        if live:
            write(out/'controller_interrupted.json',dict(time_utc=utc(),active=live,
                reason='Controller exited while workers remain; lock retained to prevent duplicate fits'))
        else:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--parent',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    for name in ('run','status','report','worker'):
        q=sub.add_parser(name);q.add_argument('--out',type=Path,required=True)
        if name=='worker':q.add_argument('--task-id',required=True)
    args=parser.parse_args()
    if args.action=='prepare':
        protocol=prepare(args.parent,args.out);print({k:protocol[k] for k in ('planned_inputs','planned_tasks','private_driver')})
    elif args.action=='run':raise SystemExit(run(args.out))
    elif args.action=='status':print(json.dumps(status(args.out),ensure_ascii=False,indent=2))
    elif args.action=='worker':execute(args.out,args.task_id)
    else:print(report(args.out,True))


if __name__=='__main__':main()
