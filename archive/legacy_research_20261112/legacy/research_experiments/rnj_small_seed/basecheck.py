"""24-input original-WZZT control, distinct from the half-frontier heuristic."""
from __future__ import annotations
import argparse,os,sys,subprocess,shutil,traceback
from pathlib import Path
from time import perf_counter,sleep
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from .run import ROOT,PARENT,read,write,sha


def freeze(output):
    if (output/'base_protocol.json').exists():raise FileExistsError('immutable original-base check')
    source=output/'base_source';shutil.copytree(output/'long_source',source)
    for name in ('engine.py','basecheck.py'):shutil.copy2(Path(__file__).with_name(name),source/'research_experiments/rnj_small_seed'/name)
    p=read(output/'protocol.json')
    write(output/'base_protocol.json',dict(jobs=p['mechanism_controls'],limits=read(output/'long_protocol.json')['limits'],
        scope='post-audit original-foundation control on fixed 24 inputs; original unrestricted one-new-support WZZT, without half-frontier or atomic-union constraints',
        reason='half-frontier priority is an additional heuristic and may exclude all true admissible merges; it must not be conflated with sparse RNJ initialization',
        seed_mode='small3',coefficient=.003,workers=8,main_protocol_sha256=sha(output/'protocol.json'),
        source_hashes={f.relative_to(source).as_posix():sha(f) for f in source.rglob('*.py')}))


def worker(payload):
    import numpy as np
    from .engine import prefix,exit_repair
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.joint_exit.run import selections
    p=read(payload);common=read(p['common']);limits=p['limits'];then=perf_counter()
    assert common['source_input_hashes']==dict(train=sha(p['train']),validation=sha(p['validation']))
    train=prepare(observed(p['train'],p['N']));val=prepare(observed(p['validation']));bank=FloorBank(train,common['pool'],common['calibration'],seconds=limits['LP'])
    bank.joint_exit_seed_pool=tuple(map(tuple,common['pool']));bank.seed_start_pool=bank.joint_exit_seed_pool
    try:
        t=prefix(bank,common['initial'],mode='small3',reference=p['reference'],seconds=limits['prefix'],per_mip=limits['per_mip'],strategy='original')
        q,e=exit_repair(bank,common['initial'],t['point'],np.asarray(common['kernel']),reference=p['reference'],prefix_seconds=t['seconds'],
            warm_candidates=common['bank'],total_seconds=limits['total'])
        path=[*t['path'],q['point']] if q['accepted'] else t['path']
        write(p['result'],dict(job=p['job'],method='small3_original_wzzt',trajectory=t,exit=q,event=e,
            choices=selections(bank,common,path,val,resolutions=(.16,.24)),prefix_choices=selections(bank,common,t['path'],val,resolutions=(.16,.24)),
            protocol_sha256=p['protocol_sha256'],common_sha256=sha(p['common']),truth_used=False,test_opened=False,
            LP=bank.lp.solves,LP_failures=bank.failures,elapsed_seconds=perf_counter()-then))
    finally:bank.close()


def run(output):
    p=read(output/'base_protocol.json');cat=read(output/'catalog.json');source=output/'base_source';tasks=[]
    for job in p['jobs']:
        result=output/'base'/job/'result.json'
        if result.exists() or result.with_name('error.json').exists():continue
        m=cat[job];reference=read(PARENT/'evaluation'/job/'baseline20.json')['trajectory']['reference_MAE']
        tasks.append(dict(job=job,N=m['N'],train=m['train'],validation=m['validation'],reference=reference,limits=p['limits'],
            common=str(output/'common'/job/'result.json'),result=str(result),protocol_sha256=sha(output/'base_protocol.json')))
    active={};env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core');done=0;then=perf_counter()
    while tasks or active:
        while tasks and len(active)<p['workers']:
            task=tasks.pop(0);file=Path(task['result']);payload=file.with_name('payload.json');write(payload,task);log=file.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.rnj_small_seed.basecheck','worker',str(payload)],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active[task['job']]=proc,log,perf_counter(),task
        for job,(proc,log,start,task) in list(active.items()):
            file=Path(task['result'])
            if proc.poll() is None and perf_counter()-start>p['limits']['hard_worker']:proc.kill();proc.wait();write(file.with_name('error.json'),dict(reason='hard original-base worker timeout'))
            if proc.poll() is not None:
                log.close();del active[job];done+=1
                if not file.exists() and not file.with_name('error.json').exists():write(file.with_name('error.json'),dict(exit_code=proc.returncode))
                print('original-base',done,'/24',job,file.exists(),flush=True)
        write(output/'base_progress.json',dict(planned=24,done=done,active=len(active),pending=len(tasks),seconds=perf_counter()-then))
        if active:sleep(.5)
    failures=[j for j in p['jobs'] if not (output/'base'/j/'result.json').exists()]
    write(output/'base_completion.json',dict(planned=24,returned=24-len(failures),failures=failures))


def queued(output):
    while True:
        try:state=read(output/'pipeline_state.json')
        except (ValueError,OSError):sleep(1);continue
        if not state['running']:
            if not state.get('passed'):raise RuntimeError('preceding timed pipeline needs attention')
            break
        write(output/'base_state.json',dict(stage='waiting for paired-long pipeline and verification',running=True));sleep(5)
    write(output/'base_state.json',dict(stage='original foundation 300-second search',running=True));run(output)
    if read(output/'base_completion.json')['returned']!=24:
        write(output/'base_state.json',dict(stage='search incomplete',running=False,failed=True));return
    write(output/'base_state.json',dict(stage='independent original-base verification',running=True))
    from .baseanalyze import analyze
    analyze(output);write(output/'base_state.json',dict(stage='complete',running=False,passed=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('path');a=parser.parse_args();path=Path(a.path).resolve()
    if a.command=='worker':
        try:worker(path)
        except Exception:
            write(Path(read(path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()));raise
    elif a.command=='freeze':freeze(path)
    elif a.command=='queued':
        try:queued(path)
        except Exception:
            write(path/'base_state.json',dict(stage='needs attention',running=False,failed=True,traceback=traceback.format_exc()));raise
    else:run(path)
