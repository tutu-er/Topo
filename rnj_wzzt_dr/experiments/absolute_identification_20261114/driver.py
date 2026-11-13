"""Paired large-population AC/physical-linear experiment; private outputs only."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[_key]='1'
import argparse, atexit, shutil, subprocess, traceback
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
from ... import io
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence,solve as old_solve
from ..flexibility_sweep_20261113.fitting import calibrated_means
from ..flexibility_sweep_20261113.execution import native_selection,fixed_rotation,margins
from ..flexibility_sweep_20261113.simulation import observe
from ..unified_posterior_dr_20261112.physical import open_runtime,generate_background
from ..linear_ac_population_20261113.study import evaluate
from ..astra_review_20261113.experiment import rejection_scores
from .data import load_entry
from .linear_truth import observe_linear
ROOT=Path(__file__).resolve().parents[3]
WORK={}

def utc(): return datetime.now(timezone.utc).isoformat()

def resources(entry,scale=1.):
    from .catalog import resources as catalog
    return catalog(entry,scale)

def physical_background(module,case,entry,replicate):
    parameters=dict(entry['physical_parameters'],replicate=replicate,study_seed=20261114)
    b=generate_background(module,parameters)
    if b['metadata']['physical_branch_sha256']!=entry['physical_branch_sha256']:
        raise ValueError('Physical network changed')
    if tuple(b['labels'])!=case.labels: raise ValueError('Physical labels changed')
    for key in ('p','q','root','voltage'): b[key]=np.roll(b[key],-32,axis=0)
    b['noise']=[np.roll(z,-32,axis=0) for z in b['noise']]
    dp,dq=case.device_baseline_pq_kw()
    b['p']+=dp/b['network']['base_kw']; b['q']+=dq/b['network']['base_kw']
    b.pop('voltage')
    return b

def forecast(b,replicate):
    P=np.column_stack((b['p'],b['q']))
    rng=np.random.default_rng(np.random.SeedSequence([20261114,replicate,991]))
    return P*(1+.05*rng.normal(size=P.shape))

def prepare(args):
    out=args.out.resolve()
    if out.exists(): raise ValueError('Preserve previous run; choose new directory or --resume')
    out.mkdir(parents=True)
    entries=io.read(args.inputs)
    if args.limit: entries=entries[:args.limit]
    io.write(out/'inputs.json',entries)
    io.write(out/'requests.json',[e['request'] for e in entries])
    for cohort in sorted({e['physical_cohort'] for e in entries}): open_runtime(out,cohort)
    for entry in entries:
        dest=out/'resource_configs'/(entry['case']+'.json')
        if not dest.exists():
            io.write(dest,resources(entry,args.resource_scale).config)
            io.write(out/'installed_configs'/(entry['case']+'.json'),resources(entry,args.baseline_scale).config)
    sources={}
    for p in sorted((ROOT/'rnj_wzzt_dr').rglob('*.py')):
        relative=p.relative_to(ROOT);dest=out/'source_snapshot'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        sources[relative.as_posix()]=io.ref(p)['sha256']
    p=dict(created_utc=utc(),environment=io.environment(),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        sources=sources,input_ref=io.ref(args.inputs),entries=len(entries),days=args.days,solver_seconds=args.seconds,
        workers=args.workers,modes=args.modes.split(','),replicate_base=args.replicate_base,
        resource_scale=args.resource_scale,baseline_scale=args.baseline_scale,
        replicate_rule='base + original_index*100 + day',study_seed=20261114,forecast_relative_sd=.05,
        primary='Original complete native frozen bank strict validation-history+new-day MSE recovery',
        physics=['original_AC','physical_LinDistFlow'],new_identifier_fits=0,parameter_refits=0,
        final_scoring_history='Original validation history only. The identifier training observations are not rescored; all original validation rows are retained.',
        devices='Fixed12 PV/EV/thermal contract, original power/energy bounds and quoted activation fees; node ordinal mapping. DER baseline added only to new days.',
        failure_rule='Unresolved/failed procurement executes zero; every strategy/path remains in denominator',
        planning_inputs='Frozen candidate bank, observed historical train/validation, noisy forecast, contract and power base only',
        stream='All controls sealed before response and execution observation; paired meters, background, response across strategies',
        shared_initial_points=args.common_seeds,linear_ablation='Same true network branches and exact P/Q/meters; history remains original AC fit',
        rejection='External factor2*8*voltage_sigma² per scalar cell; sensitivity1.25/2/4; not calibrated error-rate control',
        trusted_directions='pq_trust1/2 retain symmetric_guard evidence and physical devices; add historical-train P/Q departure energy bound at amplitude scale1/2. Planning-only restriction, not a physical device constraint. All methods receive the same raw and trust-scaled seed pool.',
        development='This round is development unless --phase confirmation. Old source histories already existed; future streams are new.',phase=args.phase)
    io.write(out/'protocol.json',p);io.write(out/'lock.json',dict(pid=os.getpid(),protocol=io.ref(out/'protocol.json')))
    return p

def initialize(out):
    import gurobipy as gp
    out=Path(out);env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start();atexit.register(env.dispose)
    WORK.update(out=out,p=io.read(out/'protocol.json'),env=env,modules={},cache={})

def context(entry):
    if entry['id'] not in WORK['cache']:
        score,train,val,truth=load_entry(entry);score['p_map']=entry['p_map']
        case=resources(entry,WORK['p'].get('resource_scale',1.));pattern,rotation_receipt=fixed_rotation(case)
        WORK['cache'][entry['id']]=dict(score=score,train=train,val=val,truth=truth,case=case,
            pattern=pattern,rotation_receipt=rotation_receipt,means=calibrated_means(score,val),voltage_sigma=entry['voltage_sigma'])
    return WORK['cache'][entry['id']]

def planning_context(ctx):
    """Keep execution-only truth outside even the quote adapter's inputs."""
    return {key:ctx[key] for key in ('case','score','train','val','means','voltage_sigma')}


def quote(mode,ctx,prediction,base,seconds,seeds=None):
    case,s=ctx['case'],ctx['score'];H=np.asarray([p['H'] for p in s['parameters']]);target=s['map_index']
    other=[j for j in range(len(H)) if j!=target]
    if not other:return dict(feasible=True,status='single_candidate_zero',source='zero',cost_yuan=0.,lower_bound_yuan=0.,global_optimality_verified=True),np.zeros(case.shape)
    if mode=='old_gamma2':
        e=CandidateEvidence(case,H,target,other,prediction,s['sigma2'],base,ctx['means'])
        if seeds is not None:
            from .commonseeds import solve_old_with_seeds
            return solve_old_with_seeds(e,2.,seed_points=seeds,seconds=seconds,env=WORK['env'])
        return old_solve(e,2.,seconds=seconds,env=WORK['env'])
    if mode in ('pq_trust1','pq_trust2'):
        from .trusted_direction import PQTrustedCase
        from .solver import plan
        began=perf_counter()
        trusted=PQTrustedCase(case,ctx['train'],base_kw=base,scale=float(mode[-1]))
        setup_seconds=perf_counter()-began
        receipt,u=plan(trusted,s,ctx['val'],prediction,base,mode='symmetric_guard',
            seconds=max(.001,seconds-setup_seconds),env=WORK['env'],seed_points=seeds)
        receipt.update(mode=mode,trust=trusted.trust_receipt,trust_setup_seconds=setup_seconds)
        receipt['seconds']+=setup_seconds
        if u is not None:
            receipt['trust_audit']=trusted.trust_audit(u)
        return receipt,u
    if mode in ('nominal','calibrated'):
        from ..astra_review_20261113.solver import plan
        return plan(case,s,ctx['val'],prediction,base,mode=mode,seconds=seconds,env=WORK['env'],training=ctx['train'],voltage_sigma=ctx['voltage_sigma'])
    if mode in ('multiplicative_absolute','capped_absolute'):
        from .solver_v2 import plan
        return plan(case,s,ctx['val'],prediction,base,mode=mode,seconds=seconds,env=WORK['env'],seed_points=seeds)
    if mode in ('weighted_nominal','weighted_capped'):
        from .solver_v3 import plan
        return plan(case,s,ctx['val'],prediction,base,mode=mode,seconds=seconds,env=WORK['env'],seed_points=seeds)
    from .solver import plan
    return plan(case,s,ctx['val'],prediction,base,mode='symmetric_guard' if 'guard' in mode else 'symmetric',
        target_mass=1. if 'full' in mode else .99,seconds=seconds,env=WORK['env'],seed_points=seeds)

def run_day(task):
    entry,day=task;out,p=WORK['out'],WORK['p'];d=out/'days'/entry['id']/f'day{day:03d}'
    if (d/'complete.json').exists():return io.read(d/'complete.json')
    if (d/'started.json').exists():raise ValueError('Review unfinished lock before resuming: '+str(d))
    d.mkdir(parents=True,exist_ok=True);io.write(d/'started.json',dict(pid=os.getpid(),created_utc=utc()))
    tick=perf_counter();ctx=context(entry);s,case,truth=ctx['score'],ctx['case'],ctx['truth'];target=s['map_index']
    cohort=entry['physical_cohort']
    if cohort not in WORK['modules']:WORK['modules'][cohort]=open_runtime(out,cohort)[0]
    module=WORK['modules'][cohort];replicate=p['replicate_base']+entry['original_index']*100+day
    installed=resources(entry,p.get('baseline_scale',1.))
    b=physical_background(module,installed,entry,replicate);prediction=forecast(b,replicate);base=b['network']['base_kw']
    controls={'no_dfs':np.zeros(case.shape)};plans={};seeds=None;seed_receipt={'seconds':0.}
    if p['shared_initial_points']:
        from .commonseeds import build_portfolio
        seeds,seed_receipt=build_portfolio(case,s,prediction,base,env=WORK['env'],seconds=min(2.,p['solver_seconds']/3))
        if any(mode.startswith('pq_trust') for mode in p['modes']):
            from .trusted_direction import PQTrustedCase
            tick_seeds=perf_counter();original_seeds=list(seeds.items())
            for scale in (1.,2.):
                trusted=PQTrustedCase(case,ctx['train'],base_kw=base,scale=scale)
                for name,point in original_seeds:
                    info=trusted.trust_audit(point)
                    factor=min(1.,np.sqrt(info['limit']/info['energy'])) if info['energy']>0 else 1.
                    adjusted=point*factor
                    seed_name=f'pq{int(scale)}_'+name
                    seeds[seed_name]=adjusted
                    seed_receipt['records'].append(dict(name=seed_name,accepted=True,
                        cost_yuan=case.fee(adjusted),source='radial_trust_scaling',parent=name,scale=factor))
            seed_receipt['trust_scaling_seconds']=perf_counter()-tick_seeds
            seed_receipt['seconds']+=seed_receipt['trust_scaling_seconds']
            seed_receipt['points']=len(seeds)
        np.savez_compressed(d/'shared_points.npz',**seeds)
    for mode in p['modes']:
        began=perf_counter()
        try:
            charged=seed_receipt['seconds'] if mode not in ('nominal','calibrated') else 0.
            receipt,u=quote(mode,planning_context(ctx),prediction,base,max(.001,p['solver_seconds']-charged),seeds)
            if (u is not None)!=bool(receipt['feasible']):raise ValueError('Returned policy disagrees with flag')
            if u is not None and not case.audit(u)['passed']:raise ValueError('Returned resource-infeasible control')
        except Exception as exc:
            receipt,u=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc()),None
        receipt.update(driver_planning_seconds=perf_counter()-began,shared_seed_seconds=charged if 'charged' in locals() else 0.,zero_fallback=u is None)
        controls[mode]=np.zeros(case.shape) if u is None else u
        fee=case.fee(controls[mode]);fraction=min(1.,fee/case.fee(ctx['pattern']))
        controls[mode+'_rotation']=fraction*ctx['pattern']
        receipt['rotation_same_fee']=abs(case.fee(controls[mode+'_rotation'])-fee)<=1e-7;plans[mode]=receipt
    np.savez_compressed(d/'planner.npz',forecast=prediction,**controls)
    io.write(d/'plans.json',plans);io.write(d/'shared_seeds.json',seed_receipt)
    io.write(d/'seal.json',dict(created_utc=utc(),planner=io.ref(d/'planner.npz'),plans=io.ref(d/'plans.json'),protocol=io.ref(out/'protocol.json'),truth_not_passed_to_planner=True,
        shared_points=None if seeds is None else io.ref(d/'shared_points.npz')))
    rng=np.random.default_rng(np.random.SeedSequence([20261114,replicate,773]));eta=rng.uniform(.9,1.,case.R)
    delivered_trust={}
    for mode in p['modes']:
        if mode.startswith('pq_trust'):
            from .trusted_direction import PQTrustedCase
            delivered_trust[mode]=PQTrustedCase(case,ctx['train'],base_kw=base,scale=float(mode[-1])).trust_audit(controls[mode]*eta[None])
    # Keep sealed plans immutable; post-seal delivered-direction audits are separate.
    io.write(d/'delivered_trust.json',delivered_trust)
    np.savez_compressed(d/'background.npz',P=b['p'],Q=b['q'],root=b['root'],response=eta,**{'noise'+str(i):z for i,z in enumerate(b['noise'])})
    io.write(d/'generation.json',b['metadata']);rows=[]
    for strategy,u in controls.items():
        receipt=plans.get(strategy.removesuffix('_rotation'))
        for physics,observer in [('AC',observe),('linear',observe_linear)]:
            info=dict(id=entry['id'],day=day,replicate=replicate,role=entry['role'],case=entry['case'],history_partition=entry['history_partition'],strategy=strategy,physics=physics,
                historical_p_map=entry['p_map'],historical_truth_present=entry['truth_covered'],target=target,
                planned_fee_yuan=case.fee(u),settled_fee_yuan=case.fee(u*eta[None]),delivered_abs_kwh=float(np.abs(u*eta[None]).sum()*case.dt),
                planning_feasible=None if receipt is None else receipt['feasible'],zero_fallback=False if receipt is None else receipt['zero_fallback'])
            try:
                observation,audit,physical=observer(module,b,u,case,eta)
                values=evaluate(s,observation,truth,target);ll=values.pop('log_probability')
                values.update(native_selection(s,observation,truth));reject,losses=rejection_scores(s,observation,truth,target,8*entry['voltage_sigma']**2)
                info.update(status='complete',**values,**audit,rejection=reject)
                np.savez_compressed(d/(strategy+'_'+physics+'.npz'),P=observation.p,Q=observation.q,Y=observation.y,V=observation.voltage,root=observation.root,
                    true_P=physical[0],true_Q=physical[1],true_V=physical[2],log_probability=ll,losses=losses)
            except Exception as exc:info.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(info)
    io.write(d/'results.json',rows)
    complete=dict(id=entry['id'],day=day,rows=len(rows),failed=sum(r['status']!='complete' for r in rows),seconds=perf_counter()-tick)
    io.write(d/'complete.json',complete);return complete

def main():
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,required=True);a.add_argument('--inputs',type=Path);a.add_argument('--days',type=int,default=2)
    a.add_argument('--seconds',type=float,default=6.);a.add_argument('--workers',type=int,default=4);a.add_argument('--replicate-base',type=int,default=2000000)
    a.add_argument('--modes',default='old_gamma2,nominal,symmetric,symmetric_guard');a.add_argument('--common-seeds',action='store_true')
    a.add_argument('--resource-scale',type=float,default=1.);a.add_argument('--baseline-scale',type=float,default=1.)
    a.add_argument('--resume',action='store_true');a.add_argument('--limit',type=int);a.add_argument('--phase',default='development');args=a.parse_args();out=args.out.resolve()
    # Resolve every solver module before locking a large batch.
    from .solver import plan as _v1
    from .solver_v2 import plan as _v2
    from .solver_v3 import plan as _v3
    from .trusted_direction import PQTrustedCase as _trusted
    valid={'old_gamma2','nominal','calibrated','symmetric','symmetric_guard','symmetric_full','symmetric_full_guard','multiplicative_absolute','capped_absolute','weighted_nominal','weighted_capped','pq_trust1','pq_trust2'}
    if not set(args.modes.split(','))<=valid:raise ValueError('Unknown planning mode')
    if not 0<args.resource_scale<=args.baseline_scale:raise ValueError('Contracted device scale must be positive and no larger than installed catalog')
    p=io.read(out/'protocol.json') if args.resume else prepare(args)
    entries=io.read(out/'inputs.json');tasks=[(e,day) for e in entries for day in range(p['days'])]
    finished=[];failed=[]
    with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(out),)) as ex:
        futures={ex.submit(run_day,t):t for t in tasks}
        for f in as_completed(futures):
            e,day=futures[f]
            try:r=f.result();finished.append(r);print(f"done {len(finished)}/{len(tasks)} {e['id']} day{day} failures={r['failed']}",flush=True)
            except Exception as exc:
                failed.append(dict(id=e['id'],day=day,error=repr(exc),traceback=traceback.format_exc()));print('FAILED '+e['id']+' '+repr(exc),flush=True)
            io.write(out/'progress.json',dict(completed=len(finished),total=len(tasks),failed=failed,updated=utc()))
    io.write(out/'complete.json',dict(completed=len(finished),total=len(tasks),failed=failed,updated=utc()))

if __name__=='__main__':main()
