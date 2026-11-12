"""Separate 24-input warm-start check; immutable main results are not changed."""
from __future__ import annotations
import argparse,os,sys,subprocess,shutil,traceback
from pathlib import Path
from time import perf_counter,sleep
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from .run import ROOT,read,write,sha


def worker(payload):
    import numpy as np
    from .engine import exit_repair
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.joint_exit.run import selections
    p=read(payload);old=read(p['prefix_result']);common=read(p['common']);t=old['trajectory']
    assert sha(p['prefix_result'])==p['prefix_sha256'] and sha(p['common'])==old['common_sha256']
    data=prepare(observed(p['train'],p['N']));val=prepare(observed(p['validation']))
    bank=FloorBank(data,common['pool'],common['calibration']);bank.joint_exit_seed_pool=tuple(map(tuple,common['pool']))
    try:
        q,event=exit_repair(bank,common['initial'],t['point'],np.asarray(common['kernel']),reference=t['reference_MAE'],
            prefix_seconds=t['seconds'],warm_candidates=common['bank'])
        path=[*t['path'],q['point']] if q['accepted'] else t['path']
        write(p['result'],dict(job=p['job'],exit=q,event=event,choices=selections(bank,common,path,val,resolutions=(.16,.24)),
            prefix_result_sha256=p['prefix_sha256'],protocol_sha256=p['protocol_sha256'],truth_used=False,test_opened=False,
            LP=bank.lp.solves,LP_failures=bank.failures,source_hash=sha(Path(__file__).with_name('engine.py')),
            scope='continuation ablation: reuse frozen main prefix; same nominal 40 s structural cutoff, improved known-feasible warm start'))
    finally:bank.close()


def freeze(output):
    p=read(output/'protocol.json');dest=output/'warm_protocol.json'
    if dest.exists():raise FileExistsError('immutable warm ablation protocol')
    source=output/'warm_source';shutil.copytree(output/'source',source)
    for name in ('engine.py','warm.py'):
        shutil.copy2(Path(__file__).with_name(name),source/'research_experiments/rnj_small_seed'/name)
    write(dest,dict(jobs=p['mechanism_controls'],reason='development small3 exit worse than an already available feasible RNJ score on 8/16 inputs',
        scope='predeclared 24 metadata-selected controls; original main search untouched; no formal F1 used to choose this solver check',
        source_hashes={f.relative_to(source).as_posix():sha(f) for f in source.rglob('*.py')},main_protocol_sha256=sha(output/'protocol.json')))


def run(output):
    p=read(output/'warm_protocol.json');cat=read(output/'catalog.json');mode=read(output/'frozen.json')['mode'];source=output/'warm_source';tasks=[]
    for job in p['jobs']:
        result=output/'warm'/job/'result.json'
        if result.exists() or result.with_name('error.json').exists():continue
        old=output/'evaluation'/job/mode/'result.json';meta=cat[job]
        tasks.append(dict(job=job,train=meta['train'],validation=meta['validation'],N=meta['N'],prefix_result=str(old),prefix_sha256=sha(old),
            common=str(output/'common'/job/'result.json'),protocol_sha256=sha(output/'warm_protocol.json'),result=str(result)))
    active={};env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core');done=0;then=perf_counter()
    while tasks or active:
        while tasks and len(active)<8:
            task=tasks.pop(0);file=Path(task['result']);payload=file.with_name('payload.json');write(payload,task);log=file.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.rnj_small_seed.warm','worker',str(payload)],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active[task['job']]=proc,log,perf_counter(),task
        for job,(proc,log,start,task) in list(active.items()):
            result=Path(task['result'])
            if proc.poll() is None and perf_counter()-start>180:proc.kill();proc.wait();write(result.with_name('error.json'),dict(reason='hard warm-worker timeout'))
            if proc.poll() is not None:
                log.close();del active[job];done+=1
                if not result.exists() and not result.with_name('error.json').exists():write(result.with_name('error.json'),dict(exit_code=proc.returncode))
                print('warm',done,'/24',job,result.exists(),flush=True)
        write(output/'warm_progress.json',dict(done=done,active=len(active),pending=len(tasks),seconds=perf_counter()-then))
        if active:sleep(.5)
    failures=[j for j in p['jobs'] if not (output/'warm'/j/'result.json').exists()]
    write(output/'warm_completion.json',dict(planned=24,returned=24-len(failures),failures=failures))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('path');a=parser.parse_args();path=Path(a.path).resolve()
    if a.command=='worker':
        try:worker(path)
        except Exception:
            write(Path(read(path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()));raise
    elif a.command=='freeze':freeze(path)
    else:run(path)
