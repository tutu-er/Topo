"""Prepare, run, inspect and evaluate the locked long-budget study."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

# Direct execution of the private snapshot uses only that snapshot's package.
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
    from rnj_wzzt_dr.experiments.wzzt_long_grid_20261117.run import main
    if __name__=='__main__':
        main();raise SystemExit

from ... import io, observations, history
from ...identification import algorithm
from ...progress import Progress

PACKAGE=Path(__file__).resolve().parents[2]
STUDY=Path(__file__).resolve().parent
PANELS=('long','grid','ablation','root_noise','sampling_control')


def utc():return datetime.now(timezone.utc).isoformat()


def checked(ref):
    path=Path(ref['path'])
    if io.ref(path)['sha256']!=ref['sha256']:raise ValueError('Changed sealed file: '+str(path))
    return path


def reuse_receipt(out, task):
    """Completed attempts are immutable; a corrupt receipt is not a retry."""
    path=Path(out)/task['result_dir']/'result.json'
    result=io.read(path)
    if result.get('task_id')!=task['task_id']:
        raise ValueError('Task receipt identity changed: '+str(path))
    if result.get('source_protocol')!=io.ref(Path(out)/'protocol.json'):
        raise ValueError('Task protocol identity changed: '+str(path))
    if result['status']=='complete':
        checked(result['native'])
        for ref in result['input_refs'].values():checked(ref)
    return result


def report(out, *, allow_partial=True):
    # An analysis error must not discard or interrupt independent solver jobs.
    try:
        from . import evaluate
        value=evaluate.study(out,allow_partial=allow_partial)
        return dict(status='complete',output=value)
    except Exception as exc:
        value=dict(status='failed',error=repr(exc),traceback=traceback.format_exc(),time_utc=utc())
        io.write(Path(out)/'report_failed.json',value)
        return value


def snapshot(out):
    target=Path(out)/'source_snapshot'/'rnj_wzzt_dr'
    for source in sorted(PACKAGE.rglob('*')):
        if not source.is_file():continue
        rel=source.relative_to(PACKAGE)
        if set(rel.parts)&{'__pycache__','.pytest_cache','outputs'}:continue
        if rel.parts[0]=='experiments' and not source.is_relative_to(STUDY):continue
        dest=target/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    return target,[io.ref(p) for p in sorted(target.rglob('*')) if p.is_file()]


def task_specs(entries):
    specs=[]
    def add(panel,entry,method,loss):
        baseline=method.startswith('RNJ')
        if baseline:total,step,reserve=(14400.,600.,600.) if panel=='long' else (600.,10.,60.)
        elif panel=='long':total,step,reserve=14400.,600.,600.
        elif method=='000':total,step,reserve=3600.,300.,300.
        else:total,step,reserve=600.,10.,60.
        config=asdict(algorithm.SearchConfig(loss=loss,total_seconds=total,per_mip_seconds=step,
            output_reserve_seconds=reserve,qp_seconds=300.))
        ident=f'{panel}__{entry["id"]}__{method}__{loss}'
        specs.append(dict(task_id=ident,panel=panel,input_id=entry['id'],method=method,loss=loss,
            config=config,result_dir='tasks/'+ident))
    primary=[e for e in entries if e['panel']=='primary']
    for panel in PANELS:
        if panel=='long':chosen=[e for e in primary if e['long_selected']]
        elif panel=='grid':chosen=primary
        elif panel=='ablation':chosen=[e for e in primary if e['N'] in (96,768) and e['pq_sd']==.005 and e['repeat']==0]
        elif panel=='root_noise':chosen=[e for e in entries if e['panel']=='root_noise']
        else:chosen=[e for e in entries if e['panel']=='sampling']
        for index,e in enumerate(sorted(chosen,key=lambda e:(e['N'],e['case'],e['pq_sd'],e['repeat'],e['id']))):
            methods=('100','000') if panel=='ablation' and e['N']==96 else ('100',) if panel=='ablation' else ('RNJ1','RNJ15','101')
            losses=('mse','mae') if index%2==0 else ('mae','mse')
            for method in methods:
                for loss in losses:add(panel,e,method,loss)
    if len(specs)!=1272 or len({s['task_id'] for s in specs})!=1272:
        raise ValueError('Changed planned study denominator')
    return specs


def prepare(data,out):
    data,out=Path(data).resolve(),Path(out).resolve()
    if out.exists():raise FileExistsError('Keep prior run: '+str(out))
    completed=io.read(data/'complete.json')
    if completed['planned_inputs']!=192 or not completed['source_unchanged'] or not completed['inputs_unchanged']:
        raise ValueError('Data preparation is not sealed and source-consistent')
    entries=io.read(checked(completed['attempts']))
    if len(entries)!=192:raise ValueError('Missing input attempt denominator')
    specs=task_specs(entries);out.mkdir(parents=True)
    source_root,refs=snapshot(out)
    io.write(out/'inputs.json',entries);io.write(out/'tasks.json',specs)
    protocol=dict(created_utc=utc(),data_complete=io.ref(data/'complete.json'),data_protocol=completed['protocol'],
        inputs_ref=io.ref(out/'inputs.json'),tasks_ref=io.ref(out/'tasks.json'),sources=refs,
        source_root=str(source_root),private_driver=str(source_root/'experiments'/STUDY.name/'run.py'),
        planned_inputs=192,planned_tasks=1272,panels=list(PANELS),workers=3,
        dispatch_wall_seconds=96*3600.,environment=io.environment(),
        protocol_document=io.ref(STUDY/'PROTOCOL.md'),
        execution_addendum=io.ref(STUDY/'EXECUTION_ADDENDUM.md'),
        selection='strict normalized validation MSE; separate normalized MAE cross-view',
        truth_used_in_fitting=False,default_loss_changed=False)
    io.write(out/'protocol.json',protocol)
    return protocol


def direct(train,validation,common,cfg,method):
    started=perf_counter();deadline=started+cfg.total_seconds-common['seconds']
    fitter=algorithm.QPFitter(train,reference_mse=common['reference_mse'],
        reference_loss=common['reference_loss'],loss=cfg.loss,bound=cfg.response_bound,qp_seconds=cfg.qp_seconds)
    reference=dict(common['reference_point'],objective=common['reference_point'][cfg.loss]/common['reference_loss'],
                   reference_mse=common['reference_mse'],reference_loss=common['reference_loss'])
    fitter.cache[algorithm.trees.canonical(reference['supports'])]=reference
    families=[common['guidance']['reference_family']] if method=='RNJ1' else common['guidance']['pool_families']
    unique=list(dict.fromkeys(algorithm.trees.canonical(f) for f in families))
    path=[fitter.fit(algorithm.trees.basis(train.n))];origins=['basis']
    for i,family in enumerate(unique):
        if perf_counter()>=deadline-cfg.output_reserve_seconds:raise TimeoutError('Direct baseline fitting budget')
        p,_=algorithm._clean(fitter,fitter.fit(family),cfg);path.append(p);origins.append('RNJ_'+str(i))
    raw=[*path,fitter.fit([(i,) for i in range(train.n)])];origins.append('singleton_protection')
    out,contraction=algorithm.output_floor.contract_bank(raw,fitter,cfg.internal_floor,
        algorithm.output_floor.calibrate(common['guidance'],train),deadline=deadline)
    selected,mae,bank=algorithm._output(out,train,validation,common['reference_mse'],cfg)
    result=dict(method=method,config=asdict(cfg),selected=selected,selected_mse_view=selected,
        selected_mae_view=mae,selection_metric='validation_mse',frozen_candidates=bank,path=path,events=[],
        QP=fitter.records,MIQP=[],reference_mse=common['reference_mse'],reference_loss=common['reference_loss'],
        training_loss=cfg.loss,common_seconds=common['seconds'],fresh_seconds=perf_counter()-started,
        candidate_origins=origins,output_contraction=contraction,uncontracted_output_path=raw,
        complete_RNJ_output_fallback=False,truth_used=False,test_used=False,native_bound_conflicts=0,
        raw_RNJ_family_count=len(families),unique_RNJ_family_count=len(unique),stop_reason='direct_RNJ_baseline',
        guarantee_scope='Direct fitted RNJ candidates; no structural integer search')
    result['total_seconds']=result['common_seconds']+result['fresh_seconds']
    result['total_cap_exceeded']=result['total_seconds']>cfg.total_seconds
    result['competition']=algorithm.competition(result,validation)
    return result


def execute(out,task,entry):
    out=Path(out);directory=out/task['result_dir']
    if directory.exists():raise FileExistsError('Existing attempt retained: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json',dict(time_utc=utc(),pid=os.getpid(),task_id=task['task_id']))
    started=perf_counter();protocol=io.read(out/'protocol.json')
    progress=Progress(directory,method=task['method'],loss=task['loss'],mode='off',every=10.)
    result=dict(task_id=task['task_id'],method=task['method'],loss=task['loss'],input_id=task['input_id'],
        panel=task['panel'],source_protocol=io.ref(out/'protocol.json'),config=task['config'],truth_used=False,new_day_loaded=False)
    try:
        if not Path(algorithm.__file__).resolve().is_relative_to(Path(protocol['source_root']).resolve()):
            raise RuntimeError('Worker did not import the immutable runtime')
        if entry['status']!='complete':raise ValueError('Data unavailable: '+entry.get('reason','preparation failed'))
        cfg=algorithm.SearchConfig(**task['config']);cfg.validate()
        progress(dict(stage='load_train_validation'))
        train=observations.load(checked(entry['files']['train.npz']))
        validation=observations.load(checked(entry['files']['validation.npz']))
        if list(train.labels)!=entry['labels'] or list(validation.labels)!=entry['labels']:raise ValueError('Node order changed')
        if task['method'].startswith('RNJ'):
            common=algorithm.prepare_common(train,cfg)
            native=direct(train,validation,common,cfg,task['method'])
        else:
            native=algorithm.fit(train,validation,method=task['method'],config=cfg,heartbeat=progress)
        native['handoff']=dict(schema='rnj_wzzt_dr_v2',training_sha256=observations.fingerprint(train),
            validation_sha256=observations.fingerprint(validation),parameter_rule='Configured training loss; unchanged strict-MSE selection')
        history.freeze_history(native,validation)
        io.write(directory/'native.json',native);progress.finish(native)
        result.update(status='complete',native=io.ref(directory/'native.json'),
            input_refs={k:entry['files'][k] for k in ('train.npz','validation.npz')},
            input_fingerprints=dict(train=observations.fingerprint(train),validation=observations.fingerprint(validation)),
            training_reference_loss=native['reference_loss'],reference_mse=native['reference_mse'],
            reference_mae=native['reference_loss'] if task['loss']=='mae' else None,
            method_seconds=native['total_seconds'],total_cap_exceeded=native['total_cap_exceeded'])
    except Exception as exc:
        result.update(status='failed',error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
            failure_details={k:getattr(exc,k) for k in ('receipt','qp_records','last_point') if hasattr(exc,k)})
        progress(dict(stage='failed',error=result['error']))
    result['seconds']=perf_counter()-started;io.write(directory/'result.json',result)
    return dict(task_id=task['task_id'],status=result['status'],result=io.ref(directory/'result.json'),seconds=result['seconds'])


def status(out):
    out=Path(out);protocol=io.read(out/'protocol.json');tasks=io.read(checked(protocol['tasks_ref']));rows=[]
    counts={p:dict(planned=0,complete=0,failed=0,interrupted=0,budget_unrun=0,pending=0,running=0) for p in PANELS}
    for task in tasks:
        directory=out/task['result_dir'];r=directory/'result.json';group=counts[task['panel']];group['planned']+=1
        if r.exists():s=io.read(r)['status'];group[s if s in group else 'failed']+=1
        elif (directory/'started.json').exists():
            group['running']+=1
            current=io.read(directory/'progress.json') if (directory/'progress.json').exists() else {}
            rows.append(dict(task_id=task['task_id'],**{k:current.get(k) for k in ('stage','wall_seconds','objective','bound','gap','nodes')}))
        else:group['pending']+=1
    return dict(time_utc=utc(),panels=counts,running=rows,stop_requested=(out/'STOP').exists())


def run(out, panels=PANELS):
    out=Path(out).resolve();protocol=io.read(out/'protocol.json')
    # The normal checkout is only a launcher; the controller and workers run
    # inside the frozen package so later workspace edits cannot change a run.
    if Path(__file__).resolve()!=Path(protocol['private_driver']).resolve():
        return subprocess.call([sys.executable,'-B',protocol['private_driver'],'run','--out',str(out),'--panels',*panels])
    for ref in protocol['sources']:checked(ref)
    tasks=io.read(checked(protocol['tasks_ref']));entries={e['id']:e for e in io.read(checked(protocol['inputs_ref']))}
    lock=out/'running.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    schedule=out/'schedule.json'
    if schedule.exists():started_wall=__import__('datetime').datetime.fromisoformat(io.read(schedule)['started_utc']).timestamp()
    else:
        started_wall=datetime.now(timezone.utc).timestamp();io.write(schedule,dict(started_utc=utc(),pid=os.getpid()))
    stop=False
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            for panel in panels:
                chosen=[t for t in tasks if t['panel']==panel];pending=[]
                for task in chosen:
                    directory=out/task['result_dir']
                    if (directory/'result.json').exists():
                        reuse_receipt(out,task);continue
                    if directory.exists():
                        io.write(directory/'result.json',dict(task_id=task['task_id'],status='interrupted',
                            error='Partial attempt retained; no automatic budget reset',source_protocol=io.ref(out/'protocol.json')))
                    else:pending.append(task)
                queue=iter(pending);active={};exhausted=False
                while active or not exhausted:
                    now=datetime.now(timezone.utc).timestamp()
                    stop=stop or (out/'STOP').exists() or now-started_wall>=protocol['dispatch_wall_seconds']
                    while not stop and len(active)<protocol['workers'] and not exhausted:
                        try:task=next(queue)
                        except StopIteration:exhausted=True;break
                        active[pool.submit(execute,str(out),task,entries[task['input_id']])]=task
                    if stop:exhausted=True
                    if not active:break
                    done,_=wait(active,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:
                        task=active.pop(future)
                        try:record=future.result()
                        except Exception as exc:
                            directory=out/task['result_dir'];directory.mkdir(parents=True,exist_ok=True)
                            io.write(directory/'result.json',dict(task_id=task['task_id'],status='failed',error=repr(exc),
                                source_protocol=io.ref(out/'protocol.json')))
                            record=dict(task_id=task['task_id'],status='failed',error=repr(exc))
                        print(json.dumps(record,ensure_ascii=False),flush=True)
                    io.write(out/'status.json',status(out))
                io.write(out/(panel+'_terminal.json'),dict(time_utc=utc(),panel=panel,stopped_dispatch=stop,status=status(out)['panels'][panel]))
                report(out,allow_partial=True)
                if stop:break
        if datetime.now(timezone.utc).timestamp()-started_wall>=protocol['dispatch_wall_seconds']:
            for task in tasks:
                directory=out/task['result_dir']
                if not directory.exists():
                    io.write(directory/'result.json',dict(task_id=task['task_id'],status='budget_unrun',
                        error='Global dispatch budget exhausted before starting; no new solve was attempted',
                        source_protocol=io.ref(out/'protocol.json')))
        final=status(out);terminal=sum(v[k] for v in final['panels'].values() for k in ('complete','failed','interrupted','budget_unrun'))
        io.write(out/'execution.json',dict(status='all_terminal' if terminal==len(tasks) else 'stopped_with_pending',terminal=terminal,planned=len(tasks),details=final))
        if terminal==len(tasks):
            report(out,allow_partial=False)
        return 0
    finally:lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--data',type=Path,required=True);q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('run');q.add_argument('--out',type=Path,required=True);q.add_argument('--panels',nargs='+',choices=PANELS,default=list(PANELS))
    q=sub.add_parser('status');q.add_argument('--out',type=Path,required=True)
    q=sub.add_parser('report');q.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':r=prepare(args.data,args.out);print({k:r[k] for k in ('planned_inputs','planned_tasks','workers','private_driver')})
    elif args.command=='run':raise SystemExit(run(args.out,args.panels))
    elif args.command=='status':print(json.dumps(status(args.out),ensure_ascii=False,indent=2))
    else:
        print(report(args.out,allow_partial=True))


if __name__=='__main__':main()
