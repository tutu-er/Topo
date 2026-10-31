"""User-authorized 300-second paired full matrix; preserve all short runs."""
from __future__ import annotations
import argparse,os,sys,subprocess,shutil,traceback
from pathlib import Path
from time import perf_counter,sleep
from datetime import datetime,timezone
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from .run import ROOT,PARENT,read,write,sha
LIMITS=dict(prefix=180.,per_mip=30.,rnj_pre=60.,total=300.,LP=30.,hard_worker=900.)


def freeze(output):
    p=read(output/'protocol.json');dest=output/'long_protocol.json'
    if dest.exists():raise FileExistsError('immutable long-run protocol')
    source=output/'long_source';shutil.copytree(output/'source',source)
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,source/'research_experiments/rnj_small_seed'/f.name)
    write(dest,dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=p['evaluation'],methods=['full_rnj','small3'],workers=8,limits=LIMITS,
        rationale='user explicitly requested substantially longer solver time; preserve 40-second main and short warm-control results',
        time_only_controls='compare short-known-warm versus long-small3 on predeclared 24 controls; both use same prefix algorithm and exit domain',
        coefficient=.003,resolutions=[.16,.24],warm='best available prefix or common RNJ training-penalized feasible solution for both methods',
        main_protocol_sha256=sha(output/'protocol.json'),frozen_sha256=sha(output/'frozen.json'),
        source_hashes={f.relative_to(source).as_posix():sha(f) for f in source.rglob('*.py')}))


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.rnj_count.engine import search as full_search
    from research_experiments.joint_exit.run import selections
    from .engine import prefix,exit_repair
    p=read(payload);common=read(p['common']);limits=p['limits'];then=perf_counter()
    assert common['source_input_hashes']==dict(train=sha(p['train']),validation=sha(p['validation']))
    data=prepare(observed(p['train'],p['N']));val=prepare(observed(p['validation']));bank=FloorBank(data,common['pool'],common['calibration'],seconds=limits['LP'])
    bank.joint_exit_seed_pool=tuple(map(tuple,common['pool']));bank.seed_start_pool=bank.joint_exit_seed_pool
    try:
        if p['method']=='full_rnj':
            t=full_search(bank,common['initial'],coefficient=.003,seconds=limits['prefix'],pre_seconds=limits['rnj_pre'],per_mip=limits['per_mip'])
        else:
            t=prefix(bank,common['initial'],mode=p['method'],reference=p['reference'],seconds=limits['prefix'],per_mip=limits['per_mip'])
        assert abs(t['reference_MAE']-p['reference'])<1e-8
        q,event=exit_repair(bank,common['initial'],t['point'],np.asarray(common['kernel']),reference=p['reference'],prefix_seconds=t['seconds'],
            warm_candidates=common['bank'],total_seconds=limits['total'],extra_repair=p['method']!='full_rnj')
        path=[*t['path'],q['point']] if q['accepted'] else t['path']
        write(p['result'],dict(job=p['job'],method=p['method'],limits=limits,trajectory=t,exit=q,event=event,
            choices=selections(bank,common,path,val,resolutions=(.16,.24)),prefix_choices=selections(bank,common,t['path'],val,resolutions=(.16,.24)),
            protocol_sha256=p['protocol_sha256'],common_sha256=sha(p['common']),input_hashes=common['source_input_hashes'],truth_used=False,test_opened=False,
            LP=bank.lp.solves,LP_failures=bank.failures,elapsed_seconds=perf_counter()-then))
    finally:bank.close()


def run(output):
    p=read(output/'long_protocol.json');cat=read(output/'catalog.json');source=output/'long_source';tasks=[]
    for job in p['jobs']:
        for method in p['methods']:
            result=output/'long'/job/method/'result.json'
            if result.exists() or result.with_name('error.json').exists():continue
            meta=cat[job];reference=read(PARENT/'evaluation'/job/'baseline20.json')['trajectory']['reference_MAE']
            tasks.append(dict(job=job,method=method,N=meta['N'],train=meta['train'],validation=meta['validation'],reference=reference,limits=p['limits'],
                common=str(output/'common'/job/'result.json'),result=str(result),protocol_sha256=sha(output/'long_protocol.json')))
    active={};env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core');done=0;then=perf_counter();total=len(tasks)
    while tasks or active:
        while tasks and len(active)<p['workers']:
            task=tasks.pop(0);result=Path(task['result']);payload=result.with_name('payload.json');write(payload,task)
            log=result.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.rnj_small_seed.longrun','worker',str(payload)],cwd=source,env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0));active[(task['job'],task['method'])]=proc,log,perf_counter(),task
        for key,(proc,log,start,task) in list(active.items()):
            result=Path(task['result'])
            if proc.poll() is None and perf_counter()-start>p['limits']['hard_worker']:
                proc.kill();proc.wait();write(result.with_name('error.json'),dict(reason='long worker hard timeout',seconds=perf_counter()-start))
            if proc.poll() is not None:
                log.close();del active[key];done+=1
                if not result.exists() and not result.with_name('error.json').exists():write(result.with_name('error.json'),dict(exit_code=proc.returncode))
                print('long',done,'/',total,*key,result.exists(),flush=True)
        write(output/'long_progress.json',dict(planned=240,finished_this_run=done,active=len(active),pending=len(tasks),seconds=perf_counter()-then))
        if active:sleep(.5)
    failures=[dict(job=j,method=m) for j in p['jobs'] for m in p['methods'] if not (output/'long'/j/m/'result.json').exists()]
    write(output/'long_completion.json',dict(planned=240,returned=240-len(failures),failures=failures))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command');parser.add_argument('path');a=parser.parse_args();path=Path(a.path).resolve()
    if a.command=='worker':
        try:worker(path)
        except Exception:
            write(Path(read(path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()));raise
    elif a.command=='freeze':freeze(path)
    else:run(path)
