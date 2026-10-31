"""Freeze observations, tune without topology labels, replay the full matrix."""
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
PARENT=ROOT/'outputs/rnj_small_count_20261031'
GRID=(0.,.001,.003,.01)


def freeze(output):
    import gurobipy as gp
    import numpy as np
    import scipy
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('protocol is immutable')
    old=read(PARENT/'protocol.json');source=output/'source';shutil.copytree(PARENT/'source',source)
    for name,digest in old['source_hashes'].items():
        assert sha(source/name)==digest
    for p in Path(__file__).parent.glob('*.py'):
        q=source/'research_experiments/revocable'/p.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    shutil.copy2(PARENT/'catalog.json',output/'catalog.json')
    common={}
    for job in [*old['development'],*old['evaluation']]:
        q=output/'common'/job/'result.json';q.parent.mkdir(parents=True);shutil.copy2(PARENT/'common'/job/'result.json',q);common[job]=sha(q)
    write(output/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        development=old['development'],evaluation=old['evaluation'],grid=GRID,workers=4,
        search_seconds=20.,pre_seconds=5.,per_MIP_seconds=2.,worker_hard_seconds=180.,
        relative_gain=.001,output_resolution=.16,response_bounds=[0.,2.],positive_floor=None,
        objective='raw training MAE + coefficient * fixed initial RNJ training MAE * internal support count',
        neighborhoods='five-block complete-subtree frontiers; bounded training-geometry sibling subsets for large polytomies',
        zero_cleanup='<=1e-9 response candidates removed only when refitted MAE increases by at most 1e-10; mandatory singleton AND full-set response bases retained',
        correction='v2: full-set basis is always available and never penalized as a newly added internal support; first exploratory batch is retained separately',
        hard_priors=[],RNJ_initial='complete provisional RNJ family, not hard physical priors',
        stopping='20s soft budget; solver bounds distinguish ruled-out gain from unresolved neighborhoods; no global topology claim',
        tuning='revocable grid; mean predictive MAE log-ratio on development outer observations; paired one-SE/0.1% then fewer lines',
        truth_used_for_tuning=False,evaluation_test_used_for_tuning=False,
        source_hashes={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py')},
        common_hashes=common,parent_protocol_sha256=sha(PARENT/'protocol.json'),
        environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,scipy=scipy.__version__,gurobi=gp.gurobi.version())))
    print('FROZEN',len(old['development'])*len(GRID),'development searches',len(old['evaluation']),'evaluation inputs',flush=True)


def worker(payload):
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.rnj_count.engine import select_output
    from .engine import search
    p=read(payload);common=read(p['common']);then=perf_counter()
    assert common['source_input_hashes']==dict(train=sha(p['train']),validation=sha(p['validation']))
    train=prepare(observed(p['train'],p['N']));validation=prepare(observed(p['validation']))
    bank=FloorBank(train,common['pool'],common['calibration'])
    try:
        result=search(bank,common['initial'],np_kernel(common),**p['config'])
        star=bank.fit([(i,) for i in range(train.n)])
        def choose(points):
            q=select_output(bank,points,validation,penalty=0.,resolution=.16)
            q['origin']='complete_rnj' if q['index']<len(common['bank']) else 'path' if q['index']<len(points)-1 else 'singleton'
            return q
        pooled=choose([*common['bank'],*result['path'],star])
        path=select_output(bank,[*result['path'],star],validation,penalty=0.,resolution=.16)
        write(p['result'],dict(job=p['job'],config_id=p['config_id'],config=p['config'],trajectory=result,
            selected=pooled,selected_path=path,LP=bank.lp.solves,LP_failures=bank.failures,
            elapsed_seconds=perf_counter()-then,common_seconds=common['seconds'],truth_used=False,test_opened=False,
            protocol_sha256=p['protocol_sha256'],frozen_sha256=p.get('frozen_sha256'),
            common_sha256=sha(p['common']),input_hashes=common['source_input_hashes']))
    finally:
        bank.close()


def np_kernel(common):
    import numpy as np
    # The frozen frontend stores its training-only RNJ geometry.
    return np.asarray(common['kernel'])


def dispatch(output,phase):
    protocol=read(output/'protocol.json');catalog=read(output/'catalog.json');source=output/'source'
    if phase=='development':
        jobs=protocol['development'];settings={f'lambda_{v:g}':dict(kind='revocable',coefficient=v) for v in protocol['grid']}
    else:
        frozen=read(output/'frozen.json');jobs=protocol['evaluation'];settings=frozen['settings']
    tasks=[]
    for job in jobs:
        meta=catalog[job]
        for cid,config in settings.items():
            target=output/phase/job/cid/'result.json'
            if target.exists() or target.with_name('error.json').exists():
                continue
            tasks.append(dict(job=job,N=meta['N'],train=meta['train'],validation=meta['validation'],config_id=cid,config=config,
                common=str(output/'common'/job/'result.json'),result=str(target),protocol_sha256=sha(output/'protocol.json'),
                frozen_sha256=sha(output/'frozen.json') if phase=='evaluation' else None))
    env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core')
    pending=list(tasks);active={};done=0;then=perf_counter()
    while pending or active:
        while pending and len(active)<protocol['workers']:
            p=pending.pop(0);target=Path(p['result']);payload=target.with_name('payload.json');write(payload,p)
            log=target.with_name('worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.revocable.run','worker',str(payload)],
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
        write(output/f'{phase}_progress.json',dict(planned=len(jobs)*len(settings),finished=done,active=len(active),pending=len(pending),seconds=perf_counter()-then))
        if active:
            sleep(.5)
    write(output/f'{phase}_completion.json',dict(planned=len(jobs)*len(settings),
        returned=sum((output/phase/j/c/'result.json').exists() for j in jobs for c in settings),seconds=perf_counter()-then))


def select(output):
    import numpy as np
    import pandas as pd
    from research_experiments.rnj_count.tune import choose
    from research_experiments.rebuild_benchmark.engine import prepare,loss
    from research_experiments.wzzt_paper.common import observed
    p=read(output/'protocol.json');catalog=read(output/'catalog.json');rows=[];records=[]
    assert read(output/'development_completion.json')['returned']==len(p['development'])*len(p['grid'])
    for coefficient in p['grid']:
        fits=[]
        for job in p['development']:
            meta=catalog[job];outer=prepare(observed(meta['test']))
            base=read(output/'development'/job/'lambda_0'/'result.json')
            file=output/'development'/job/f'lambda_{coefficient:g}'/'result.json';r=read(file)
            value=loss(outer,r['selected']['point']);reference=loss(outer,base['selected']['point'])
            fits.append(dict(job=job,case=meta['case'],bundle=meta['bundle'],coefficient=coefficient,
                tuning_MAE=value,log_ratio=float(np.log(max(value,1e-12)/max(reference,1e-12))),
                internal=r['selected']['internal'],result_sha256=sha(file)))
        grouped=pd.DataFrame(fits).groupby(['case','bundle']).log_ratio.mean()
        rows.append(dict(coefficient=coefficient,mean_log_ratio=float(np.mean([q['log_ratio'] for q in fits])),
            mean_internal=float(np.mean([q['internal'] for q in fits])),group_log_ratios=grouped.tolist()))
        records+=fits
    selected,minimum=choose(rows)
    settings=dict(step_locked=dict(kind='locked',coefficient=.003),step_zero=dict(kind='zero_only',coefficient=.003),
        revocable_zero=dict(kind='revocable',coefficient=0.))
    if selected['coefficient']!=0:
        settings['revocable_tuned']=dict(kind='revocable',coefficient=selected['coefficient'])
    write(output/'frozen.json',dict(selected=selected,strict_min=minimum,settings=settings,
        tuned_method='revocable_tuned' if selected['coefficient'] else 'revocable_zero',
        protocol_sha256=sha(output/'protocol.json'),truth_used=False,evaluation_test_used=False,
        development_outer_hashes={j:sha(catalog[j]['test']) for j in p['development']},
        created_utc=datetime.now(timezone.utc).isoformat()))
    pd.DataFrame([{k:v for k,v in r.items() if k!='group_log_ratios'} for r in rows]).to_csv(output/'tuning_grid.csv',index=False)
    pd.DataFrame(records).to_csv(output/'tuning_records.csv',index=False)
    print('SELECTED',selected,'evaluation settings',settings,flush=True)


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
        select(path)
    else:
        from .analyze import analyze
        analyze(path)


if __name__=='__main__':
    main()
