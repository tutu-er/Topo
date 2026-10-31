"""Frozen paired budgets, observation-only tuning, and full four-case replay."""
from __future__ import annotations
import argparse,os,shutil,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from time import perf_counter,sleep
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
from research_experiments.rebuild_benchmark.run import ROOT,read,write,sha
PARENT=ROOT/'outputs/rnj_small_count_20261031'
GRID=[dict(policy=policy,coefficient=v) for policy in ('compact','expanded') for v in (.001,.003,.01)]
RESOLUTIONS=(.08,.16,.24)
def cid(config):return f"{config['policy']}_{config['coefficient']:g}"


def freeze(output):
    import numpy as np,scipy,gurobipy as gp
    output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists():raise FileExistsError('immutable protocol')
    old=read(PARENT/'protocol.json');source=output/'source';shutil.copytree(PARENT/'source',source)
    # The corrected regional implementation is used only for time ablation.
    for directory in ('joint_exit','revocable'):
        for file in (ROOT/'research_experiments'/directory).glob('*.py'):
            dest=source/'research_experiments'/directory/file.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(file.read_bytes())
    catalog=read(PARENT/'catalog.json');write(output/'catalog.json',catalog);common={}
    for job in [*old['development'],*old['evaluation']]:
        dest=output/'common'/job/'result.json';dest.parent.mkdir(parents=True);shutil.copy2(PARENT/'common'/job/'result.json',dest);common[job]=sha(dest)
    ablation=[j for j in old['development'] if (catalog[j]['bundle']==0 and catalog[j]['voltage_sigma']==.001) or
        (catalog[j]['bundle']==1 and catalog[j]['voltage_sigma']==.0002)]
    assert len(ablation)==8
    write(output/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),development=old['development'],evaluation=old['evaluation'],
        ablation=ablation,grid=GRID,resolutions=RESOLUTIONS,rules=['strict','paired'],workers=8,worker_hard_seconds=900.,
        budgets=[20.,40.],prefix_coefficient=.003,prefix_seconds=20.,
        model='old multiregion prefix; one pooled MILP with optional provisional internal supports; mandatory singleton/full basis',
        checkpoint='raw feasible incumbents and bounds available at wall-clock 20/40 seconds; fixed-family closure and common validation selection are postprocessing',
        penalty='exit coefficient * fixed unregularized complete-RNJ initial MAE; prefix uses historical .003; monotonicity only claimed within each phase',
        tuning='40-second paired predictive outer-development log ratio, paired one-SE/0.1% band then fewer lines; truth never used',
        ablation_configs={'regional20_2':dict(kind='revocable',seconds=20.,per_mip=2.,coefficient=.003),
                          'regional40_2':dict(kind='revocable',seconds=40.,per_mip=2.,coefficient=.003),
                          'regional40_5':dict(kind='revocable',seconds=40.,per_mip=5.,coefficient=.003)},
        truth_used_for_tuning=False,evaluation_test_used_for_tuning=False,hard_priors=[],positive_floor=None,
        common_hashes=common,parent_protocol_sha256=sha(PARENT/'protocol.json'),
        source_hashes={p.relative_to(source).as_posix():sha(p) for p in source.rglob('*.py')},
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,gurobi=gp.gurobi.version(),logical_processors=os.cpu_count())))
    print('FROZEN: 16 development inputs x 6 searches; 8 time-ablation inputs; 120 final inputs',flush=True)


def selections(bank,common,path,validation,resolutions=RESOLUTIONS):
    from .selection import library,choose
    star=bank.fit([(i,) for i in range(bank.data.n)])
    result={}
    for resolution in resolutions:
        own=library(bank,[*path,star],validation,resolution=resolution)
        pooled=library(bank,[*common['bank'],*path,star],validation,resolution=resolution)
        for rule in ('strict','paired'):
            q=choose(pooled,rule);i=q['index'];b=len(common['bank'])
            q['origin']='complete_rnj' if i<b else 'path' if i<b+len(path) else 'singleton'
            result[f'{rule}_{resolution:g}']=dict(bank=q,path=choose(own,rule))
    return result


def worker(payload):
    import numpy as np
    from research_experiments.wzzt_paper.common import observed
    from research_experiments.rebuild_benchmark.engine import prepare
    from research_experiments.lp_rebuild.engine import FloorBank
    from research_experiments.rnj_count.engine import search as old_search
    from .model import joint_exit
    p=read(payload);common=read(p['common']);then=perf_counter();dest=Path(p['destination']);dest.mkdir(parents=True,exist_ok=True)
    assert common['source_input_hashes']==dict(train=sha(p['train']),validation=sha(p['validation']))
    train=prepare(observed(p['train'],p['N']));validation=prepare(observed(p['validation']));kernel=np.asarray(common['kernel'])
    provenance=dict(job=p['job'],protocol_sha256=p['protocol_sha256'],frozen_sha256=p.get('frozen_sha256'),
        common_sha256=sha(p['common']),input_hashes=common['source_input_hashes'],truth_used=False,test_opened=False)
    if p['phase']=='ablation':
        from research_experiments.revocable.engine import search as regional
        for name,config in p['configs'].items():
            file=dest/name/'result.json'
            if file.exists():continue
            bank=FloorBank(train,common['pool'],common['calibration'])
            try:
                t=regional(bank,common['initial'],kernel,**config)
                q=selections(bank,common,t['path'],validation,resolutions=(.16,))
                write(file,dict(**provenance,config=config,trajectory=t,choices=q,LP=bank.lp.solves,LP_failures=bank.failures))
            finally:bank.close()
        return
    bank=FloorBank(train,common['pool'],common['calibration']);bank.joint_exit_seed_pool=tuple(map(tuple,common['pool']))
    try:
        prefix_file=dest/'baseline20.json'
        if prefix_file.exists():prefix=read(prefix_file)['trajectory']
        else:
            prefix=old_search(bank,common['initial'],coefficient=.003,seconds=20.)
            write(prefix_file,dict(**provenance,trajectory=prefix,choices=selections(bank,common,prefix['path'],validation),LP=bank.lp.solves))
        for config in p['configs']:
            file=dest/cid(config)/'result.json'
            if file.exists():continue
            outputs,event=joint_exit(bank,common['initial'],prefix['point'],kernel,**config,
                reference=prefix['reference_MAE'],prefix_seconds=prefix['seconds'])
            for budget,q in outputs.items():
                path=[*prefix['path'],q['point']] if q['accepted'] else prefix['path']
                q['choices']=selections(bank,common,path,validation)
            write(file,dict(**provenance,config=config,outputs=outputs,event=event,prefix_seconds=prefix['seconds'],
                reference_MAE=prefix['reference_MAE'],LP=bank.lp.solves,LP_failures=bank.failures,
                elapsed_seconds=perf_counter()-then))
    finally:bank.close()
    # A fresh LP bank prevents new candidates leaking into the old long-budget baseline.
    if not (dest/'baseline40.json').exists():
        bank=FloorBank(train,common['pool'],common['calibration'])
        try:
            baseline=old_search(bank,common['initial'],coefficient=.003,seconds=40.)
            write(dest/'baseline40.json',dict(**provenance,trajectory=baseline,
                choices=selections(bank,common,baseline['path'],validation),LP=bank.lp.solves,LP_failures=bank.failures))
        finally:bank.close()


def dispatch(output,phase):
    p=read(output/'protocol.json');catalog=read(output/'catalog.json');source=output/'source'
    jobs=p['ablation'] if phase=='ablation' else p[phase]
    configs=p['ablation_configs'] if phase=='ablation' else p['grid'] if phase=='development' else read(output/'frozen.json')['evaluation_configs']
    tasks=[]
    for job in jobs:
        dest=output/phase/job
        if (dest/'completed.json').exists() or (dest/'error.json').exists():continue
        meta=catalog[job]
        tasks.append(dict(job=job,N=meta['N'],train=meta['train'],validation=meta['validation'],phase=phase,configs=configs,
            common=str(output/'common'/job/'result.json'),destination=str(dest),protocol_sha256=sha(output/'protocol.json'),
            frozen_sha256=sha(output/'frozen.json') if phase=='evaluation' else None))
    env=os.environ.copy();env['PYTHONPATH']=str(source)+os.pathsep+str(source/'rnj_wzzt_core')
    pending=list(tasks);active={};then=perf_counter();done=0
    while pending or active:
        while pending and len(active)<p['workers']:
            task=pending.pop(0);dest=Path(task['destination']);payload=dest/'payload.json';write(payload,task)
            log=(dest/'worker.log').open('w',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-B','-m','research_experiments.joint_exit.run','worker',str(payload)],
                cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            active[task['job']]=proc,log,perf_counter(),task
        for job,(proc,log,start,task) in list(active.items()):
            dest=Path(task['destination'])
            if proc.poll() is None and perf_counter()-start>p['worker_hard_seconds']:
                proc.kill();proc.wait();write(dest/'error.json',dict(reason='worker hard timeout'))
            if proc.poll() is not None:
                log.close()
                if proc.returncode==0:write(dest/'completed.json',dict(passed=True,seconds=perf_counter()-start))
                elif not (dest/'error.json').exists():write(dest/'error.json',dict(exit_code=proc.returncode))
                del active[job];done+=1;print(phase,done,'/',len(tasks),job,proc.returncode,flush=True)
        write(output/f'{phase}_progress.json',dict(planned=len(jobs),finished=sum((output/phase/j/'completed.json').exists() for j in jobs),
            active=len(active),pending=len(pending),seconds=perf_counter()-then))
        if active:sleep(.5)
    write(output/f'{phase}_completion.json',dict(planned=len(jobs),returned=sum((output/phase/j/'completed.json').exists() for j in jobs),seconds=perf_counter()-then))


def select(output):
    import numpy as np,pandas as pd
    from research_experiments.rebuild_benchmark.engine import prepare,loss
    from research_experiments.wzzt_paper.common import observed
    p=read(output/'protocol.json');catalog=read(output/'catalog.json');assert read(output/'development_completion.json')['returned']==16
    records=[];rows=[]
    for config in p['grid']:
        for resolution in p['resolutions']:
            for rule in p['rules']:
                fits=[]
                for job in p['development']:
                    meta=catalog[job];data=prepare(observed(meta['test']))
                    base=read(output/'development'/job/'baseline20.json')['choices']['strict_0.16']['bank']
                    file=output/'development'/job/cid(config)/'result.json';r=read(file)
                    q=r['outputs']['40']['choices'][f'{rule}_{resolution:g}']['bank'];value=loss(data,q['point'])
                    fits.append(dict(job=job,case=meta['case'],bundle=meta['bundle'],config_id=cid(config),resolution=resolution,rule=rule,
                        log_ratio=float(np.log(max(value,1e-12)/max(loss(data,base['point']),1e-12))),internal=q['internal'],tuning_MAE=value))
                g=pd.DataFrame(fits).groupby(['case','bundle']).log_ratio.mean()
                rows.append(dict(**config,config_id=cid(config),resolution=resolution,rule=rule,mean_log_ratio=float(np.mean([r['log_ratio'] for r in fits])),
                    mean_internal=float(np.mean([r['internal'] for r in fits])),group_log_ratios=g.tolist()))
                records+=fits
    best=min(rows,key=lambda r:(r['mean_log_ratio'],r['mean_internal']))
    for r in rows:
        d=np.asarray(r['group_log_ratios'])-np.asarray(best['group_log_ratios']);se=float(d.std(ddof=1)/np.sqrt(len(d)))
        r.update(paired_SE=se,eligible=float(d.mean())<=max(np.log(1.001),se)+1e-12)
    picked=min((r for r in rows if r['eligible']),key=lambda r:(r['mean_internal'],r['mean_log_ratio'],r['policy']!='compact',r['coefficient']))
    selected_config={k:picked[k] for k in ('policy','coefficient')}
    matched_config=dict(policy=picked['policy'],coefficient=.003)
    eval_configs=[selected_config] if selected_config==matched_config else [selected_config,matched_config]
    write(output/'frozen.json',dict(config=selected_config,evaluation_configs=eval_configs,selection={k:picked[k] for k in ('rule','resolution')},
        picked=picked,strict_min=best,truth_used=False,evaluation_test_used=False,protocol_sha256=sha(output/'protocol.json'),
        development_outer_hashes={j:sha(catalog[j]['test']) for j in p['development']},created_utc=datetime.now(timezone.utc).isoformat()))
    pd.DataFrame([{k:v for k,v in r.items() if k!='group_log_ratios'} for r in rows]).to_csv(output/'tuning_grid.csv',index=False)
    pd.DataFrame(records).to_csv(output/'tuning_records.csv',index=False)
    print('FROZEN SELECTION',picked,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','ablation','development','evaluation','worker','select','analyze','information'))
    parser.add_argument('path');a=parser.parse_args();path=Path(a.path).resolve()
    if a.command=='worker':
        try:worker(path)
        except Exception:
            write(Path(read(path)['destination'])/'error.json',dict(traceback=traceback.format_exc()));raise
    elif a.command=='freeze':freeze(path)
    elif a.command in ('ablation','development','evaluation'):dispatch(path,a.command)
    elif a.command=='select':select(path)
    elif a.command=='information':
        from .information import run
        run(path)
    else:
        from .analyze import analyze
        analyze(path)


if __name__=='__main__':main()
