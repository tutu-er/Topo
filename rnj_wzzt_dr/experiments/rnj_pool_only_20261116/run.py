"""Paired old/new fits from private, immutable source snapshots."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import importlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback

from ... import io
from ...evaluation_scope import load_catalog

ROOT=Path(__file__).resolve().parents[3]
OLD_COMMIT='7f8c9cbf'


def utc():return datetime.now(timezone.utc).isoformat()


def runtime_path(relative):
    return not ({'experiments','tests','evidence','validation','__pycache__'} & set(Path(relative).parts))


def check(reference):
    if io.ref(reference['path'])['sha256']!=reference['sha256']:
        raise ValueError('Changed sealed file: '+str(reference['path']))
    return Path(reference['path'])


def snapshot(destination,version):
    destination=Path(destination); refs=[]
    if version=='old':
        names=subprocess.check_output(['git','ls-tree','-r','--name-only',OLD_COMMIT,'--','rnj_wzzt_dr'],cwd=ROOT,text=True).splitlines()
        for name in names:
            relative=Path(name).relative_to('rnj_wzzt_dr')
            if not runtime_path(relative):continue
            target=destination/'rnj_wzzt_dr'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(subprocess.check_output(['git','show',OLD_COMMIT+':'+name],cwd=ROOT))
            refs.append(io.ref(target))
    else:
        for source in sorted((ROOT/'rnj_wzzt_dr').rglob('*')):
            relative=source.relative_to(ROOT/'rnj_wzzt_dr')
            if not source.is_file() or not runtime_path(relative):continue
            target=destination/'rnj_wzzt_dr'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target);refs.append(io.ref(target))
    return refs


def prepare(out):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('Use a fresh output: '+str(out))
    catalog=load_catalog()
    selected=[e for e in catalog if e['source_group']=='history10' and e.get('train_days')==8 and e['history_repeat'] in (0,3)]
    if len(selected)!=4:raise ValueError('The four predeclared inputs are not available')
    entries=[]
    for e in selected:
        entries.append(dict(id=e['id'],case=e['case'],history_repeat=e['history_repeat'],source_group=e['source_group'],
            labels=e['labels'],base_kw=e['base_kw'],zbase_ohm=e['zbase_ohm'],
            files={'train.npz':e['train_ref'],'validation.npz':e['validation_ref'],'new_day.npz':e['new_day_ref']},
            evaluation_case_refs=e['evaluation_case_refs'],Hstar_ref=e['Hstar_ref']))
    out.mkdir(parents=True)
    root_protocol=dict(created_utc=utc(),old_commit=OLD_COMMIT,inputs=4,planned_fits=8,
        versions=['old','new'],method='101',method_seconds=120.,per_mip_seconds=120.,
        output_reserve_seconds=12.,qp_soft_seconds=300.,workers=3,
        environment=io.environment(),no_new_AC=True,no_DR=True,no_E60=True,
        source_document=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        active_scope=io.ref(ROOT/'rnj_wzzt_dr/evaluation_scope.json'),runner=io.ref(Path(__file__)))
    for version in ('old','new'):
        directory=out/version;directory.mkdir()
        source_root=out/(version+'_source')
        refs=snapshot(source_root,version)
        io.write(directory/'inputs.json',entries)
        io.write(directory/'protocol.json',dict(root_protocol,version=version,
            source_root=str(source_root/'rnj_wzzt_dr'),sources=refs,inputs_ref=io.ref(directory/'inputs.json')))
    io.write(out/'protocol.json',root_protocol)
    return root_protocol


def package_from_snapshot(source_root,version):
    key='_rnj_pool_'+version
    if key not in sys.modules:
        root=Path(source_root)
        spec=importlib.util.spec_from_file_location(key,root/'__init__.py',submodule_search_locations=[str(root)])
        package=importlib.util.module_from_spec(spec);sys.modules[key]=package;spec.loader.exec_module(package)
    return key


def fit_one(run,entry):
    run=Path(run);protocol=io.read(run/'protocol.json');directory=run/'cases'/entry['id']
    if directory.exists():raise FileExistsError('Partial/completed task is retained: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json',dict(created_utc=utc(),pid=os.getpid(),input_id=entry['id'],version=protocol['version']))
    started=perf_counter();stage='load_snapshot'
    try:
        for reference in protocol['sources']:check(reference)
        package=package_from_snapshot(protocol['source_root'],protocol['version'])
        algorithm=importlib.import_module(package+'.identification.algorithm')
        observations=importlib.import_module(package+'.observations')
        history=importlib.import_module(package+'.history')
        cfg=algorithm.SearchConfig(total_seconds=protocol['method_seconds'],per_mip_seconds=protocol['per_mip_seconds'],
            qp_seconds=protocol['qp_soft_seconds'],output_reserve_seconds=protocol['output_reserve_seconds'])
        stage='load_training_validation'
        train=observations.load(check(entry['files']['train.npz']))
        validation=observations.load(check(entry['files']['validation.npz']))
        if list(train.labels)!=entry['labels'] or list(validation.labels)!=entry['labels']:
            raise ValueError('Observation labels changed')
        stage='fit'
        def heartbeat(row):
            io.write(directory/'progress.json',dict(stage=row['stage'],elapsed_seconds=row['elapsed_seconds'],
                path_points=len(row.get('path',[])),created_utc=utc()))
        native=algorithm.fit(train,validation,method=protocol['method'],config=cfg,heartbeat=heartbeat)
        native['handoff']=dict(schema='rnj_wzzt_dr_v2',training_sha256=observations.fingerprint(train),
            validation_sha256=observations.fingerprint(validation),parameter_rule='Native output floor and unchanged validation rule')
        history.freeze_history(native,validation)
        io.write(directory/'native.json',native)
        result=dict(status='complete',native=io.ref(directory/'native.json'),config=native['config'],
            input_id=entry['id'],case=entry['case'],seconds=perf_counter()-started,
            method_total_seconds=native['total_seconds'],total_cap_exceeded=native['total_cap_exceeded'],
            source_protocol=io.ref(run/'protocol.json'),sources_unchanged=all(io.ref(x['path'])['sha256']==x['sha256'] for x in protocol['sources']),
            truth_used=False,new_day_loaded=False)
    except Exception as exc:
        result=dict(status='failed',input_id=entry['id'],case=entry['case'],stage=stage,seconds=perf_counter()-started,
            error=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),
            failure_details={k:getattr(exc,k) for k in ('receipt','qp_records','last_point') if hasattr(exc,k)})
    io.write(directory/'result.json',result)
    return dict(input_id=entry['id'],version=protocol['version'],status=result['status'],
        result=io.ref(directory/'result.json'),seconds=result['seconds'])


def run(out):
    out=Path(out).resolve();protocol=io.read(out/'protocol.json')
    if (out/'complete.json').exists():return io.read(out/'complete.json')
    lock=out/'running.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    records=[];started=perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            futures={}
            for version in ('old','new'):
                for entry in io.read(out/version/'inputs.json'):
                    futures[pool.submit(fit_one,str(out/version),entry)]=(version,entry)
            for future in as_completed(futures):
                version,entry=futures[future]
                try:row=future.result()
                except Exception as exc:
                    row=dict(input_id=entry['id'],version=version,status='worker_failed',error=repr(exc),seconds=None)
                records.append(row)
                io.write(out/'progress.json',dict(completed=len(records),planned=protocol['planned_fits'],records=records))
                print({k:row[k] for k in ('version','input_id','status')},flush=True)
        result=dict(status='complete',planned=protocol['planned_fits'],completed=len(records),
            successful=sum(r['status']=='complete' for r in records),failed=sum(r['status']!='complete' for r in records),
            seconds=perf_counter()-started,records=records)
        io.write(out/'complete.json',result)
        for version in ('old','new'):
            subset=[r for r in records if r['version']==version]
            io.write(out/version/'complete.json',dict(status='complete',planned=4,completed=len(subset),
                successful=sum(r['status']=='complete' for r in subset),records=subset))
        return result
    finally:lock.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not a.resume:prepare(a.out)
    if not a.prepare_only:run(a.out)


if __name__=='__main__':main()
