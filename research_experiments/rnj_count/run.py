"""Observation-only tuning, followed by frozen four-network replay."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter,sleep
import traceback

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'

from research_experiments.rebuild_benchmark.run import ROOT,read,write,sha

PARENT=ROOT/'outputs/joint_five_rnj_wzzt_20261031'
FRONT=ROOT/'outputs/rebuild_tuned_benchmark_20261030_final'
GRID=(0.,.0003,.001,.003,.01,.03,.1)


def config_id(value):
    return f'lambda_{value:g}'


def freeze(output):
    import gurobipy as gp
    import numpy as np
    import scipy
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('experiment protocol already frozen')
    old=read(PARENT/'protocol.json');dev=read(FRONT/'protocol.json')['development'];evaluation=old['evaluation']
    assert len(dev)==16 and len(evaluation)==120 and not set(dev)&set(evaluation)
    source=output/'source';shutil.copytree(PARENT/'source',source)
    for name,digest in old['source_hashes'].items():
        assert sha(source/name)==digest
    for file in Path(__file__).parent.glob('*.py'):
        target=source/'research_experiments/rnj_count'/file.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())
    catalog=read(PARENT/'catalog.json');reused={}
    for job in dev:
        catalog[job]=read(ROOT/'outputs/wzzt_paper_20261029_full/jobs'/job/'payload.json')
    for job in [*dev,*evaluation]:
        meta=catalog[job];common=(FRONT if job in dev else PARENT)/'common'/job/'result.json'
        assert read(common)['source_input_hashes']==dict(train=sha(meta['train']),validation=sha(meta['validation']))
        target=output/'common'/job/'result.json';target.parent.mkdir(parents=True);shutil.copy2(common,target);reused[job]=sha(target)
    write(output/'catalog.json',catalog)
    write(output/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),development=dev,evaluation=evaluation,
        grid=GRID,settings={config_id(v):dict(coefficient=v) for v in GRID},workers=4,worker_hard_seconds=180.,
        search_seconds=20.,pre_seconds=5.,per_MIP_seconds=2.,LP_seconds=5.,relative_gain=.001,output_resolution=.16,
        count='retained proper internal supports; protected coarse supports are fixed count constants; optional zero supports are canonicalized off',
        scale='penalty per selected line = coefficient * unregularized initial RNJ training MAE',
        domains='same full-RNJ initial, five-block eligible preselection/tail domains, 0..2 response bounds, no added positive floor',
        variants=['search_only','selection_only','both','both_strict_min_control'],
        tuning='min mean log outer-development predictive MAE; paired one-SE/0.1% band; prefer fewer lines, lower MAE, then smaller coefficient',
        development_inner='training archive fits coefficients; validation archive selects each candidate model',
        development_outer='historical bundles0/1 test_observed archives are repurposed as hyperparameter-tuning observations only',
        evaluation_test_used_for_tuning=False,truth_used_for_tuning=False,
        evaluation_scope='same known 108-main +12-stress four-network matrix; no retuning after frozen selection',
        frontend_reused=reused,parent_protocol_sha256=sha(PARENT/'protocol.json'),
        source_hashes={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py')},
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,gurobi=gp.gurobi.version())))
    print('FROZEN',len(dev)*len(GRID),'development searches;',len(evaluation),'evaluation inputs',flush=True)


def worker(payload):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from .engine import search,select_output
    p=read(payload);common=read(p['common']);then=perf_counter()
    assert common['source_input_hashes']==dict(train=sha(p['train']),validation=sha(p['validation']))
    train=prepare(observed(p['train'],p['N']));validation=prepare(observed(p['validation']))
    bank=FloorBank(train,common['pool'],common['calibration'])
    try:
        coefficient=p['config']['coefficient'];trajectory=search(bank,common['initial'],coefficient=coefficient)
        select_start=perf_counter();star=bank.fit([(i,) for i in range(train.n)])
        points=[*common['bank'],*trajectory['path'],star];path=[*trajectory['path'],star];reference=trajectory['reference_MAE']
        def choose(v):
            chosen=select_output(bank,points,validation,penalty=v*reference)
            chosen_path=select_output(bank,path,validation,penalty=v*reference)
            i=chosen['index'];b=len(common['bank'])
            chosen['origin']='complete_rnj' if i<b else 'path' if i<b+len(trajectory['path']) else 'singleton'
            return dict(bank=chosen,path=chosen_path)
        selected_plain=choose(0.);selected_regular=selected_plain if coefficient==0 else choose(coefficient)
        extra={config_id(v):choose(v) for v in p['output_coefficients'] if v not in (0.,coefficient)}
        write(p['result'],dict(job=p['job'],config_id=p['config_id'],config=p['config'],trajectory=trajectory,
            selected_plain=selected_plain,selected_regular=selected_regular,extra_selection=extra,
            LP=bank.lp.solves,LP_failures=bank.failures,selection_seconds=perf_counter()-select_start,
            elapsed_seconds=perf_counter()-then,common_seconds=common['seconds'],truth_used=False,test_opened=False,
            protocol_sha256=p['protocol_sha256'],frozen_sha256=p.get('frozen_sha256'),common_sha256=sha(p['common']),input_hashes=common['source_input_hashes']))
    finally:
        bank.close()


def result_choice(result,mode,coefficient):
    if mode=='search_only' or coefficient==0:
        return result['selected_plain']
    if mode=='both':
        return result['selected_regular']
    return result['extra_selection'][config_id(coefficient)]


def dispatch(output,phase):
    protocol=read(output/'protocol.json');catalog=read(output/'catalog.json');source=output/'source'
    if phase=='development':
        jobs=protocol['development'];settings=protocol['settings'];out_coefficients=list(protocol['grid'])
    else:
        frozen=read(output/'frozen.json');jobs=protocol['evaluation'];settings=frozen['evaluation_settings']
        out_coefficients=[0.,frozen['selected']['selection_only']['coefficient']]
    tasks=[]
    for job in jobs:
        meta=catalog[job]
        for cid,config in settings.items():
            target=output/phase/job/cid/'result.json'
            if target.exists() or target.with_name('error.json').exists():
                continue
            tasks.append(dict(job=job,N=meta['N'],train=meta['train'],validation=meta['validation'],config_id=cid,config=config,
                common=str(output/'common'/job/'result.json'),result=str(target),protocol_sha256=sha(output/'protocol.json'),
                output_coefficients=out_coefficients if config['coefficient']==0 else [],
                frozen_sha256=sha(output/'frozen.json') if phase=='evaluation' else None))
    env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core')
    pending=list(tasks);active={};done=0;then=perf_counter()
    while pending or active:
        while pending and len(active)<protocol['workers']:
            p=pending.pop(0);target=Path(p['result']);payload=target.with_name('payload.json');write(payload,p)
            log=target.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.rnj_count.run','worker',str(payload)],
                cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[str(target)]=proc,log,perf_counter(),p
        for key,(proc,log,start,p) in list(active.items()):
            target=Path(p['result'])
            if proc.poll() is None and perf_counter()-start>protocol['worker_hard_seconds']:
                proc.kill();proc.wait();write(target.with_name('timeout.json'),dict(reason='worker hard timeout'))
            if proc.poll() is not None:
                log.close()
                if not target.exists() and not target.with_name('error.json').exists():
                    write(target.with_name('error.json'),dict(exit_code=proc.returncode))
                done+=1;del active[key]
                print(phase,done,'/',len(tasks),p['job'],p['config_id'],target.exists(),flush=True)
        write(output/f'{phase}_progress.json',dict(planned=len(tasks),finished=done,active=len(active),pending=len(pending),seconds=perf_counter()-then))
        if active:
            sleep(.5)
    returned=sum((output/phase/j/c/'result.json').exists() for j in jobs for c in settings)
    write(output/f'{phase}_completion.json',dict(planned=len(jobs)*len(settings),returned=returned,seconds=perf_counter()-then))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','development','select','evaluation','worker','analyze'))
    parser.add_argument('path');a=parser.parse_args();path=Path(a.path).resolve()
    if a.command=='worker':
        try:
            worker(path)
        except Exception:
            write(Path(read(path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()));raise
    elif a.command=='freeze':
        freeze(path)
    elif a.command in ('development','evaluation'):
        dispatch(path,a.command)
    elif a.command=='select':
        from .tune import select
        select(path)
    else:
        from .analyze import analyze
        analyze(path)


if __name__=='__main__':
    main()
