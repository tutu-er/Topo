"""Bounded follow-up; failed history gate never becomes a new price score.

Prices below are declared exploration baselines with independently randomized
ties. The history audit owns structural conclusions. Truth enters evaluation only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
from datetime import datetime, timezone
import traceback
import numpy as np
import pandas as pd
from research_experiments.wzzt_dr_selection.dr_multiblock import model
from research_experiments.wzzt_dr_selection.dr_multiblock import experiment as old
from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import fit_observed, source_hashes

PLANNING_SEEDS=(2026102911,2026102912)
RESPONSE_SEED=2026102919
PATHS=8
write,read,sha,digest=old.write,old.read,old.sha,old.digest


def code_hashes():
    out=old.code_hashes()
    for f in (Path(__file__).parent/n for n in ('__init__.py','experiment.py','history_audit.py')):
        out[str(f.resolve())]=sha(f)
    helper=Path(old.__file__).parent.parent/'dr_pool_v2/runner.py'
    out[str(helper.resolve())]=sha(helper)
    return out


def policies(planning,seed,law):
    rng=np.random.default_rng(np.random.SeedSequence([seed,0 if law=='local' else 1]))
    out=[dict(name='no_reward',kind='no_reward',plan=[0]*10,tie_count=1)]
    for name,cls,key in [('single','single',law+'_score'),('sync','sync',law+'_score'),
                         ('free','full',law+'_score'),('mean_free','full','mean_score')]:
        eligible=[r for r in planning['plans'] if r['classes'][cls]]
        best=max(r[key] for r in eligible)
        ties=[r for r in eligible if best-r[key]<=model.TIE_TOLERANCE]
        budget=min(r['budget_bound'] for r in ties)
        ties=[r for r in ties if r['budget_bound']==budget]
        chosen=int(rng.integers(len(ties)))
        out.append(dict(name=name,kind=cls,plan=ties[chosen]['plan'],tie_count=len(ties),
                        tie_index=chosen,score_key=key,maximum_score=best,chosen_score=ties[chosen][key],
                        tie_rule='uniform among score-tied minimum-reservation plans; independent planning RNG'))
    positions=rng.choice(5,size=2,replace=False)
    p=np.zeros((2,5),dtype=int);p[0,positions[0]]=2;p[1,positions[1]]=2
    out.append(dict(name='rotation',kind='predeclared_rotation',plan=p.ravel().tolist(),tie_count=None,
                    role='two distinct locations defines this exploration baseline only; not a constraint on optimized policies'))
    return out


def prepare(root):
    root=Path(root)
    if (root/'protocol.json').exists():
        p=read(root/'protocol.json')
        if p['source_sha256']!=code_hashes():raise RuntimeError('source changed after freeze')
        for name,h in p['input_sha256'].items():
            if sha(root/name)!=h:raise RuntimeError('frozen input changed: '+name)
        return p
    gate=read(root/'history_audit.json')
    if gate['status']!='complete':raise RuntimeError('history audit incomplete')
    if not gate['history_input_analysis']['analytic_singleton_witness_applicable']:
        raise RuntimeError('this bounded follow-up was predeclared only for the verified history gate')
    from .history_audit import load_history,witness
    _,_,past,_,_=load_history(root/'observed_history.json')
    all_past=witness(past,1e-8)
    if not all_past['analytic_singleton_witness_applicable']:
        raise RuntimeError('held-out historical day invalidates the proposed gate')
    write(root/'all_past_witness.json',all_past)
    planning=model.plan_all()
    selected={str(seed):{law:policies(planning,seed,law) for law in ('local','common')} for seed in PLANNING_SEEDS}
    write(root/'planning.json',planning);write(root/'policies.json',selected)
    # No evaluator future data/response has been constructed before prices.
    draws=np.random.default_rng(RESPONSE_SEED).random((PATHS,4,5))
    write(root/'test_draws.json',dict(seed=RESPONSE_SEED,uniforms=draws))
    p=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),source_sha256=code_hashes(),
       input_sha256={n:sha(root/n) for n in ['observed_history.json','observed_training_history.json','all_past_witness.json','history_audit.json','planning.json','policies.json','test_draws.json']},
       constants=model.constants(),planning_seeds=PLANNING_SEEDS,response_seed=RESPONSE_SEED,paths=PATHS,
       main_cells=192,control_cells=18,total_cells=210,max_unique_fits=160,search_seconds=25.,
       history_gate='compatible positive singleton witness; no uniform positive greedy-acceptance target',
       price_status='randomized exploration baselines only; no WZZT-specific or truth-conditioned price claim',
       main='2 planning seeds x 2 dependence laws x 6 policies x 8 response paths, fixed normal case',
       controls='2 planning seeds x 3 frozen local policies(single/sync/free) x normal/weak/zero, all participate',
       independence='8 response paths per law, shared between planning seeds and policies; two independent price tie draws',
       truth_to_planner=False,truth_to_identifier=False,unknown_meter_gains_added=False,
       evaluation='all effective block false+missing; complete family; payment; RNJ/primary/multistart origins',
       stop='finish this bounded control; no new policy or case selected from test results',
       interpretation='tests random tie artifact and exploration; not a completed history-conditioned optimal price method',
       environment=dict(python=sys.version,executable=sys.executable,numpy=np.__version__,pandas=pd.__version__))
    write(root/'protocol.json',p)
    return p


def tasks(root):
    selected=read(root/'policies.json')
    for seed,bylaw in selected.items():
        for case in ('normal','weak','zero'):
            for policy in bylaw['local']:
                if policy['name'] not in ('single','sync','free'):continue
                yield dict(key=f'control__{seed}__{case}__{policy["name"]}',phase='control',
                    planning_seed=int(seed),law='all_participate',case=case,policy=policy['name'],plan=policy['plan'],path=None)
        for law,pp in bylaw.items():
            for policy in pp:
                for path in range(PATHS):
                    yield dict(key=f'{seed}__{law}__{policy["name"]}__path{path}',phase='main',
                        planning_seed=int(seed),law=law,case='normal',policy=policy['name'],plan=policy['plan'],path=path)


def summarize(root,p):
    rows=[read(f) for f in sorted((root/'records').glob('*.json'))]
    groups=[]
    for seed in PLANNING_SEEDS:
        for law in ('local','common'):
            for policy in ('no_reward','single','sync','free','mean_free','rotation'):
                rr=[r for r in rows if r['phase']=='main' and r['planning_seed']==seed and r['law']==law and r['policy']==policy]
                ok=[r for r in rr if r['status']=='evaluated'];models={}
                for name in old.METHODS:
                    vals=[r['evaluation']['models'][name] for r in ok]
                    if vals:
                        models[name]=dict(mean_errors=float(np.mean([v['final']['all_block_errors'] for v in vals])),
                            full_count=sum(v['final']['complete_effective_family'] for v in vals),
                            origins={origin:sum(v['selected_origin']==origin for v in vals) for origin in sorted({v['selected_origin'] for v in vals})})
                groups.append(dict(seed=seed,law=law,policy=policy,recorded=len(rr),evaluated=len(ok),denominator=PATHS,
                    mean_payment=float(np.mean([r['physical']['payment_currency'] for r in ok])) if ok else None,models=models))
    out=dict(status='complete' if len(rows)==p['total_cells'] else 'partial',planned=p['total_cells'],
        recorded=len(rows),main=sum(r['phase']=='main' for r in rows),controls=sum(r['phase']=='control' for r in rows),
        evaluated=sum(r['status']=='evaluated' for r in rows),failed=sum(r['status']!='evaluated' for r in rows),
        wall_limited=sum(r.get('evaluation',{}).get('search_wall_limited',False) for r in rows),
        unique_fit_count=len(list((root/'fits').glob('*/result.json'))),groups=groups)
    write(root/'summary.json',out);return out


def run(root,max_new_fits):
    root=Path(root);p=prepare(root)
    lock=root/'run.lock'
    if lock.exists():raise RuntimeError('existing run lock')
    write(lock,dict(pid=os.getpid(),source=sha(__file__)))
    for name in ('records','fits','inputs'):(root/name).mkdir(exist_ok=True)
    draws=np.asarray(read(root/'test_draws.json')['uniforms']);new=0
    try:
        for task in tasks(root):
            target=root/'records'/(task['key']+'.json')
            if target.exists():continue
            z=(np.vstack([np.asarray(task['plan']).reshape(2,5)]*2)>0).astype(int) if task['phase']=='control' else old.response(task['plan'],draws[task['path']],task['law'])
            data=model.make_data(task['plan'],z,task['case']);training,validation=old.observations(data)
            key,identity=old.keys(training,validation,p);folder=root/'fits'/key;cached=(folder/'result.json').exists()
            if not cached and (new>=max_new_fits or len(list((root/'fits').glob('*/result.json')))>=p['max_unique_fits']):break
            row=dict(task,participation=z,observation_key=key,cache_reused=cached,physical=data['physical_audit'],fit_result=str(folder/'result.json'))
            if not (root/'inputs'/(key+'.json')).exists():write(root/'inputs'/(key+'.json'),dict(identity=identity,data=data))
            try:
                if not data['physical_audit']['valid']:raise ValueError('physical/contract check failed')
                if cached:fit=read(folder/'result.json')
                else:
                    new+=1;fit=fit_observed(training,validation,folder,root_label=1,search_seconds=25.)
                    print(json.dumps(dict(new_fit=new,key=task['key'],seconds=fit['elapsed_seconds'])),flush=True)
                if fit['status']!='complete':raise RuntimeError('failed cache retained')
                row.update(status='evaluated',fit_sha256=sha(folder/'result.json'),evaluation=old.fit_metrics(fit,data['truth']),
                           true_supports=data['truth']['supports'])
            except Exception as exc:row.update(status='failed',error=str(exc),traceback=traceback.format_exc())
            write(target,row);summarize(root,p)
    finally:lock.unlink(missing_ok=True)
    return summarize(root,p)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--max-new-fits',type=int,default=8)
    args=parser.parse_args();out=prepare(args.output) if args.prepare_only else run(args.output,args.max_new_fits)
    print(json.dumps({k:v for k,v in out.items() if k not in ('source_sha256','input_sha256','groups')},ensure_ascii=False))
