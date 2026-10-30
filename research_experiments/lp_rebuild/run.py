"""Freeze, develop and evaluate LP hierarchy enumeration without truth tuning."""
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

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from research_experiments.rebuild_benchmark.run import ROOT,CASES,read,write,sha,ident
import numpy as np

PARENT=ROOT/'outputs/rebuild_tuned_benchmark_20261030_final'


def configs():
    result={}
    for floor in (0.,.04,.08,.16):result[f'depth_{floor:g}']=dict(strategy='joint',mode='depth',floor=floor,sparsity=0.)
    for penalty in (.001,.005):result[f'sparse_{penalty:g}']=dict(strategy='joint',mode='depth',floor=.001,sparsity=penalty)
    for floor in (.5,1.):result[f'signal_{floor:g}']=dict(strategy='joint',mode='signal',floor=floor,sparsity=0.)
    result['hybrid']=dict(strategy='pool_joint',mode='depth',floor=0.,sparsity=0.)
    result['hybrid_regularized']=dict(strategy='pool_joint',mode='depth',floor=.08,sparsity=.001)
    result['pool']=dict(strategy='pool',mode='depth',floor=0.,sparsity=0.)
    return result


def freeze(output):
    import gurobipy,scipy,platform
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    old=read(PARENT/'protocol.json');dev=old['development']
    evaluation=[ident('main',c,b,'reference',v,48) for c in CASES for b in (5,6) for v in (.0002,.001)]
    assert not set(dev)&set(evaluation) and not set(old['evaluation'])&set(evaluation)
    snapshot=output/'source';shutil.copytree(PARENT/'source',snapshot)
    for folder in ('rebuild_benchmark','lp_rebuild'):
        for file in (ROOT/'research_experiments'/folder).glob('*.py'):
            target=snapshot/'research_experiments'/folder/file.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())
    metadata={};reused={}
    for job in [*dev,*evaluation]:
        p=read(ROOT/'outputs/wzzt_paper_20261029_full/jobs'/job/'payload.json');metadata[job]=p
        target=output/'common'/job
        payload=dict(job=job,N=p['N'],train=p['train'],validation=p['validation'],train_sha256=sha(p['train']),
                     validation_sha256=sha(p['validation']),destination=str(target),result=str(target/'result.json'))
        write(target/'payload.json',payload)
        if job in dev:
            source=PARENT/'common'/job/'result.json';shutil.copy2(source,target/'result.json');reused[job]=sha(source)
    write(output/'catalog.json',metadata)
    write(output/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),development=dev,evaluation=evaluation,
        configurations=configs(),workers=4,worker_hard_seconds=240.,search_seconds=20.,LP_seconds=5.,
        relative_gain=.001,blocks=4,output_resolution=.16,output_validation_fraction=0.,
        selection='min mean log development validation ratio; within 0.1% prefer fewer internals then lower runtime then fixed ID',
        tuning_truth_used=False,tuning_test_used=False,development_frontend_reused=reused,
        evaluation_scope='16 existing AC simulations from bundles 5/6, untouched by prior local-rebuild tuning/evaluation; not new field data or globally unseen historical simulations',
        model='26 legal four-block hierarchies; a virtual proper region edge is enumerated absent/present; each family refits every R/X weight',
        root_correction='full-set response is available in every enumerated neighborhood, even when its old weight was zero; native_root control uses the same corrected domain',
        feature_prefetch='one batch feature-bank extension per region before enumerating fixed families',
        floor_depth='(a*r+b*x)/median_root_depth >= delta for active proper internal supports, including final refits',
        floor_signal='fixed centered-response projection >= delta*unconstrained_initial_MAE; projection is a true lower bound on this support response RMS, not on unique evidence after nuisance refitting',
        sparsity='raw training MAE + sparsity*unconstrained_initial_MAE*active proper internal count',
        sparse_activation_floor=.001,
        hybrid='penalized pool single moves first, then local joint enumeration with the same total wall budget; all path points retained',
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,gurobi=gurobipy.gurobi.version(),platform=platform.platform()),
        source_hashes={p.relative_to(snapshot).as_posix():sha(p) for p in snapshot.rglob('*.py')}))
    print('FROZEN',len(dev),'development',len(evaluation),'evaluation',len(configs()),'development settings',flush=True)


def worker(payload):
    from research_experiments.rebuild_benchmark.run import prepare_worker
    p=read(payload)
    if p['kind']=='prepare':prepare_worker(payload);return
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare,Bank,search,loss,pool_repair
    from .engine import FloorBank,InfeasibleFamily,enumerate_search,select_output,criterion,native_root_search
    common=read(p['common']);then=perf_counter();config=p['config']
    assert sha(p['train'])==common['source_input_hashes']['train'] and sha(p['validation'])==common['source_input_hashes']['validation']
    train=prepare(observed(p['train'],p['N']));validation=prepare(observed(p['validation']))
    base=Bank(train,common['pool'])
    try:raw_initial=base.fit(common['initial'])
    finally:base.close()
    reference=raw_initial['objective']
    bank=FloorBank(train,common['pool'],common['calibration'],floor=config['floor'],mode=config['mode'],reference=reference)
    try:
        initial=raw_initial['supports'];fallback=False
        try:bank.fit(initial)
        except InfeasibleFamily:initial=[(i,) for i in range(train.n)];fallback=True
        if p['config_id']=='pool':trajectory=pool_repair(bank,initial,seconds=20.,relative=.001)
        elif p['config_id']=='native_root':trajectory=native_root_search(bank,initial,np.asarray(common['kernel']),seconds=20.)
        elif p['config_id'] in ('native','original_fullwarm'):
            initial=common['initial'] if p['config_id']=='native' else [(i,) for i in range(train.n)]
            trajectory=search(bank,initial,np.asarray(common['kernel']),kind='rebuild' if p['config_id']=='native' else 'forward',
                              b=4,relative=.001,seconds=20.,per_MIP=2.,warm_limit=None)
        else:
            trajectory=enumerate_search(bank,initial,np.asarray(common['kernel']),seconds=20.,relative=.001,
                penalty=config['sparsity']*reference,strategy=config['strategy'],b=4)
        selection_start=perf_counter()
        star=bank.fit([(i,) for i in range(train.n)])
        points=[*common['bank'],*trajectory['path'],star]
        chosen=select_output(bank,points,validation);path_choice=select_output(bank,[*trajectory['path'],star],validation)
        write(p['result'],dict(job=p['job'],config_id=p['config_id'],config=config,trajectory=trajectory,selected=chosen,
            selected_path=path_choice,bank_count=len(common['bank']),initial_fallback=fallback,reference_MAE=reference,
            initial_validation=loss(validation,raw_initial),selection_seconds=perf_counter()-selection_start,
            elapsed_seconds=perf_counter()-then,common_seconds=common['seconds'],LP=bank.lp.solves,
            LP_failures=bank.failures,truth_used=False,test_opened=False,common_sha256=sha(p['common']),
            input_hashes=common['source_input_hashes'],frozen_sha256=p.get('frozen_sha256')))
    finally:bank.close()


def dispatch(output,phase,tasks):
    output=Path(output).resolve();protocol=read(output/'protocol.json');snapshot=output/'source'
    env=os.environ.copy();env['PYTHONPATH']=str(snapshot)+os.pathsep+str(snapshot/'rnj_wzzt_core')
    pending=list(tasks);active={};done=0;then=perf_counter()
    while pending or active:
        while pending and len(active)<4:
            p=pending.pop(0);target=Path(p['result']);file=target.with_name('payload.json');write(file,p)
            log=target.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.lp_rebuild.run','worker',str(file)],cwd=snapshot,env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[str(target)]=(proc,log,perf_counter(),p)
        for key,(proc,log,start,p) in list(active.items()):
            target=Path(p['result'])
            if proc.poll() is None and perf_counter()-start>protocol['worker_hard_seconds']:
                proc.kill();proc.wait();write(target.with_name('timeout.json'),dict(reason='worker hard timeout'))
            if proc.poll() is not None:
                log.close()
                if not target.exists() and not target.with_name('error.json').exists():write(target.with_name('error.json'),dict(exit_code=proc.returncode))
                done+=1;del active[key]
                print(phase,done,'/',len(tasks),p['job'],p.get('config_id',''),target.exists(),flush=True)
        write(output/f'{phase}_progress.json',dict(planned=len(tasks),finished=done,pending=len(pending),active=len(active),seconds=perf_counter()-then))
        if active:sleep(.5)
    write(output/f'{phase}_completion.json',dict(planned=len(tasks),returned=sum(Path(p['result']).exists() for p in tasks),seconds=perf_counter()-then))


def tasks_for(output,jobs,settings,phase,frozen=None):
    tasks=[]
    for job in jobs:
        source=read(output/'common'/job/'payload.json')
        for cid,config in settings.items():
            tasks.append(dict(source,kind='search',config_id=cid,config=config,common=str(output/'common'/job/'result.json'),
                result=str(output/phase/job/cid/'result.json'),frozen_sha256=frozen))
    return tasks


def select(output):
    output=Path(output).resolve();protocol=read(output/'protocol.json');rows=[]
    for cid,config in protocol['configurations'].items():
        data=[]
        for job in protocol['development']:
            file=output/'development'/job/cid/'result.json'
            if file.exists():
                r=read(file);data.append(dict(job=job,log_ratio=float(np.log(r['selected']['validation']/r['initial_validation'])),
                    internal=r['selected']['internal'],seconds=r['elapsed_seconds'],result_sha256=sha(file)))
        rows.append(dict(config_id=cid,config=config,available=len(data),mean_log_ratio=float(np.mean([q['log_ratio'] for q in data])) if data else None,
            mean_internal=float(np.mean([q['internal'] for q in data])) if data else None,mean_seconds=float(np.mean([q['seconds'] for q in data])) if data else None,records=data))
    def pick(predicate):
        complete=[r for r in rows if r['available']==len(protocol['development']) and predicate(r['config'])]
        if not complete:raise RuntimeError('no complete development configuration')
        best=min(r['mean_log_ratio'] for r in complete);eligible=[r for r in complete if r['mean_log_ratio']<=best+np.log(1.001)]
        return min(eligible,key=lambda r:(r['mean_internal'],r['mean_log_ratio'],r['mean_seconds'],r['config_id']))
    selected=dict(best_joint=pick(lambda c:c['strategy']=='joint'),best_floor=pick(lambda c:c['strategy']=='joint' and c['floor']>0),
                  best_hybrid=pick(lambda c:c['strategy']=='pool_joint'))
    methods={name:q['config_id'] for name,q in selected.items()}
    methods.update(enum_control='depth_0',pool='pool',native_joint='native',native_corrected='native_root',original_fullwarm='original_fullwarm')
    configs=dict(protocol['configurations']);configs['native']=configs['native_root']=configs['original_fullwarm']=dict(strategy='baseline',mode='depth',floor=0.,sparsity=0.)
    file=output/'frozen.json'
    if file.exists():raise FileExistsError('parameters already frozen')
    write(file,dict(created_utc=datetime.now(timezone.utc).isoformat(),selected=selected,grid=rows,methods=methods,
        evaluation_settings={cid:configs[cid] for cid in sorted(set(methods.values()))},protocol_sha256=sha(output/'protocol.json'),truth_used=False,test_used=False))
    print('SELECTED',{k:q['config_id'] for k,q in selected.items()},flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','prepare','development','select','evaluation','worker'))
    parser.add_argument('output');args=parser.parse_args();output=Path(args.output).resolve()
    if args.command=='worker':
        try:worker(output)
        except Exception as exc:
            p=read(output);write(Path(p['result']).with_name('error.json'),dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()));raise
        return
    if args.command=='freeze':freeze(output);return
    protocol=read(output/'protocol.json')
    if args.command=='prepare':
        tasks=[dict(read(output/'common'/job/'payload.json'),kind='prepare') for job in protocol['evaluation']]
        dispatch(output,'prepare',tasks)
    elif args.command=='development':dispatch(output,'development',tasks_for(output,protocol['development'],protocol['configurations'],'development'))
    elif args.command=='select':select(output)
    else:
        frozen=read(output/'frozen.json')
        dispatch(output,'evaluation',tasks_for(output,protocol['evaluation'],frozen['evaluation_settings'],'evaluation',sha(output/'frozen.json')))


if __name__=='__main__':main()
