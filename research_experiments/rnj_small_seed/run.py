"""Frozen four-case experiment; current full-RNJ method remains untouched."""
from __future__ import annotations
import argparse,os,sys,subprocess,shutil,traceback
from pathlib import Path
from time import perf_counter,sleep
from datetime import datetime,timezone
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from research_experiments.rebuild_benchmark.run import ROOT,read,write,sha
PARENT=ROOT/'outputs/joint_exit_20261031'
MODES=('small2','small3','none','large')


def freeze(output):
    import numpy as np,scipy,gurobipy as gp
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise FileExistsError('immutable experiment protocol')
    parent=read(PARENT/'protocol.json');catalog=read(PARENT/'catalog.json');write(output/'catalog.json',catalog)
    source=output/'source';shutil.copytree(PARENT/'source',source)
    for directory in ('joint_exit','rnj_small_seed'):
        for f in (ROOT/'research_experiments'/directory).glob('*.py'):
            dest=source/'research_experiments'/directory/f.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    hashes={}
    for job in [*parent['development'],*parent['evaluation']]:
        dest=output/'common'/job/'result.json';dest.parent.mkdir(parents=True);shutil.copy2(PARENT/'common'/job/'result.json',dest);hashes[job]=sha(dest)
    controls=[j for j in parent['evaluation'] if catalog[j]['panel']=='main' and catalog[j]['bundle'] in (2,3) and catalog[j]['N']==48]
    assert len(controls)==24
    write(output/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),development=parent['development'],evaluation=parent['evaluation'],
        mechanism_controls=controls,modes=MODES,small_modes=['small2','small3'],workers=8,hard_worker_seconds=180.,
        budgets=dict(prefix=20,total=40,per_MIP=2),coefficient=.003,max_seeds=2,seed_ranking='training full-RNJ deletion profile; positive gain above .003*f0',
        resolution_choices=[.16,.24],frozen_output_rule='strict validation; resolution .24; separately report .16',
        search='maximum-arity region; >=half children first then all proper unions if no accepted gain; all original terminal residuals',
        exit='all provisional supports optional; compact plus all terminal pairs and fixed training-RNJ local windows',
        truth_used_for_tuning=False,test_used_for_search=False,hard_priors=[],positive_floor=None,
        common_hashes=hashes,parent_protocol_sha256=sha(PARENT/'protocol.json'),parent_commit='a46691ed',
        parent_result_hashes={j:sha(PARENT/'evaluation'/j/'compact_0.003/result.json') for j in parent['evaluation']},
        source_hashes={f.relative_to(source).as_posix():sha(f) for f in source.rglob('*.py')},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,gurobi=gp.gurobi.version(),logical_processors=os.cpu_count())))
    print('frozen 16 x 4 development; 120 selected-small + 24 x 2 formal controls',flush=True)


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.joint_exit.run import selections
    from .engine import prefix,exit_repair
    p=read(payload);common=read(p['common']);dest=Path(p['result']);then=perf_counter()
    assert common['source_input_hashes']==dict(train=sha(p['train']),validation=sha(p['validation']))
    train=prepare(observed(p['train'],p['N']));val=prepare(observed(p['validation']))
    bank=FloorBank(train,common['pool'],common['calibration']);bank.joint_exit_seed_pool=tuple(map(tuple,common['pool']));bank.seed_start_pool=bank.joint_exit_seed_pool
    try:
        t=prefix(bank,common['initial'],mode=p['mode'],reference=p['reference'])
        q,e=exit_repair(bank,common['initial'],t['point'],np.asarray(common['kernel']),reference=p['reference'],prefix_seconds=t['seconds'])
        path=[*t['path'],q['point']] if q['accepted'] else t['path']
        choices=selections(bank,common,path,val,resolutions=(.16,.24))
        prefix_choices=selections(bank,common,t['path'],val,resolutions=(.16,.24))
        write(dest,dict(job=p['job'],mode=p['mode'],protocol_sha256=p['protocol_sha256'],frozen_sha256=p.get('frozen_sha256'),
            common_sha256=sha(p['common']),input_hashes=common['source_input_hashes'],truth_used=False,test_opened=False,
            trajectory=t,exit=q,event=e,choices=choices,prefix_choices=prefix_choices,LP=bank.lp.solves,LP_failures=bank.failures,
            elapsed_seconds=perf_counter()-then,postprocessing_excluded_from_structural_budget=True))
    finally:bank.close()


def dispatch(output,phase):
    p=read(output/'protocol.json');cat=read(output/'catalog.json');source=output/'source';tasks=[]
    jobs=p['development'] if phase=='development' else p['evaluation']
    modes=MODES if phase=='development' else [read(output/'frozen.json')['mode']]
    for job in jobs:
        requested=list(modes)
        if phase=='evaluation' and job in p['mechanism_controls']:requested+=['none','large']
        for mode in requested:
            result=output/phase/job/mode/'result.json'
            if result.exists() or result.with_name('error.json').exists():continue
            meta=cat[job];reference=read(PARENT/'evaluation'/job/'baseline20.json')['trajectory']['reference_MAE'] if phase=='evaluation' else read(PARENT/'development'/job/'baseline20.json')['trajectory']['reference_MAE']
            tasks.append(dict(job=job,mode=mode,N=meta['N'],train=meta['train'],validation=meta['validation'],reference=reference,
                common=str(output/'common'/job/'result.json'),result=str(result),protocol_sha256=sha(output/'protocol.json'),
                frozen_sha256=sha(output/'frozen.json') if phase=='evaluation' else None))
    env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core');active={};done=0;total=len(tasks);then=perf_counter()
    while tasks or active:
        while tasks and len(active)<p['workers']:
            task=tasks.pop(0);result=Path(task['result']);payload=result.with_name('payload.json');write(payload,task)
            log=result.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.rnj_small_seed.run','worker',str(payload)],cwd=source,env=env,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[(task['job'],task['mode'])]=proc,log,perf_counter(),task
        for key,(proc,log,start,task) in list(active.items()):
            result=Path(task['result'])
            if proc.poll() is None and perf_counter()-start>p['hard_worker_seconds']:
                proc.kill();proc.wait();write(result.with_name('error.json'),dict(reason='hard worker timeout',seconds=perf_counter()-start))
            if proc.poll() is not None:
                log.close();done+=1;del active[key]
                if not result.exists() and not result.with_name('error.json').exists():write(result.with_name('error.json'),dict(exit_code=proc.returncode))
                print(phase,done,'/',total,*key,result.exists(),flush=True)
        write(output/f'{phase}_progress.json',dict(dispatched=total,finished_this_run=done,active=len(active),pending=len(tasks),seconds=perf_counter()-then))
        if active:sleep(.5)
    expected=[(j,m) for j in jobs for m in (list(modes)+(['none','large'] if phase=='evaluation' and j in p['mechanism_controls'] else []))]
    failures=[dict(job=j,mode=m,error=read(output/phase/j/m/'error.json')) for j,m in expected if not (output/phase/j/m/'result.json').exists()]
    write(output/f'{phase}_completion.json',dict(planned=len(expected),returned=len(expected)-len(failures),failures=failures))


def select(output):
    import numpy as np,pandas as pd
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare,loss
    p=read(output/'protocol.json');cat=read(output/'catalog.json');assert read(output/'development_completion.json')['returned']==64
    rows=[]
    for mode in p['small_modes']:
        values=[];lines=[]
        for job in p['development']:
            r=read(output/'development'/job/mode/'result.json');q=r['choices']['strict_0.24']['bank'];data=prepare(observed(cat[job]['test']))
            base=read(PARENT/'development'/job/'compact_0.003/result.json')['outputs']['40']['choices']['strict_0.24']['bank']['point']
            values.append(float(np.log(max(loss(data,q['point']),1e-12)/max(loss(data,base),1e-12))));lines.append(q['internal'])
        rows.append(dict(mode=mode,mean_log_ratio=float(np.mean(values)),mean_internal=float(np.mean(lines))))
    best=min(r['mean_log_ratio'] for r in rows)
    chosen=min((r for r in rows if r['mean_log_ratio']<=best+np.log(1.001)),key=lambda r:(r['mean_internal'],r['mean_log_ratio'],r['mode']))
    write(output/'frozen.json',dict(**chosen,rule='strict',resolution=.24,coefficient=.003,truth_used=False,evaluation_used=False,
        protocol_sha256=sha(output/'protocol.json'),development_outer_hashes={j:sha(cat[j]['test']) for j in p['development']},grid=rows))
    pd.DataFrame(rows).to_csv(output/'tuning.csv',index=False);print('selected',chosen,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','worker','development','evaluation','select','analyze'));parser.add_argument('path');a=parser.parse_args();path=Path(a.path).resolve()
    if a.command=='worker':
        try:worker(path)
        except Exception:
            write(Path(read(path)['result']).with_name('error.json'),dict(traceback=traceback.format_exc()));raise
    elif a.command=='freeze':freeze(path)
    elif a.command in ('development','evaluation'):dispatch(path,a.command)
    elif a.command=='select':select(path)
    else:
        from .analyze import analyze
        analyze(path)


if __name__=='__main__':main()
