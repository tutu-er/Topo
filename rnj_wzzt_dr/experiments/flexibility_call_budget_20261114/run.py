"""Prelocked activation-budget correction on the untouched confirmation worlds."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import shutil
from time import perf_counter
import traceback
import numpy as np
from ... import io
from ...observations import Observation
from ..measured_pq_flex_20261114 import runner as old
from ..flexibility_stability_20261114 import domain,solver as prices
from ..absolute_identification_20261114.solver import SymmetricRecovery
from . import solver

POLICY='budget_capacity'

def prepare(source,out,*,smoke=False):
    source=Path(source).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    p=io.read(source/'protocol.json')
    if not (source/'complete.json').exists():raise ValueError('Completed source required')
    if not smoke and (p['phase']!='confirmation' or p['seed']!=2026111416 or p['repeats']!=64 or p['forecasts']!=4):
        raise ValueError('Only the prelocked confirmation worlds are permitted')
    entries=[e for e in io.read(old.checked(p['inputs'])) if e['resource_variant']=='original_p10_q36']
    if len(entries)!=1:raise ValueError('Exactly one original resource case required')
    out.mkdir(parents=True)
    io.write(out/'inputs.json',entries)
    p.update(phase='call_budget_confirmation' if not smoke else 'integration_smoke',
        source_run=str(source),source_protocol=io.ref(source/'protocol.json'),
        source_completion=io.ref(source/'complete.json'),
        locked_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        inputs=io.ref(out/'inputs.json'),policies=[POLICY],modes=[POLICY],
        source_policies_unchanged=True,solver_seconds=2. if smoke else 300.,
        workers=1 if smoke else 4,expected_rows=len(entries)*p['forecasts']*p['repeats'],
        allowance_yuan=.01,original_shared_budget_seconds=150.,
        initialization='Only same-forecast shared initial points and anchor, without execution records',
        interpretation='Additional policy chosen from development before confirmation; same confirmation worlds and original seven-P resources',
        fallback='Validated same-forecast anchor if the bounded solve returns no accepted point; unresolved remains explicit')
    known={Path(r['path']).resolve() for r in p['sources']}
    for file in (Path(__file__),Path(solver.__file__)):
        if file.resolve() not in known:p['sources'].append(io.ref(file));known.add(file.resolve())
    for item in p['sources']:
        path=old.checked(item);target=out/'source_snapshot'/path.relative_to(old.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    p.update(old._git_receipt(sorted(known)))
    io.write(out/'protocol.json',p)
    return p

def run_forecast(task):
    entry,draw=task;p=old.WORK['p'];out=old.WORK['out'];began=perf_counter()
    directory=out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():return io.read(directory/'complete.json')
    if directory.exists():raise RuntimeError('Inspect unfinished task before reuse')
    directory.mkdir(parents=True)
    io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=old.utc()))
    source=Path(p['source_run'])/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    seal=io.read(source/'seal.json');anchor_receipt=io.read(old.checked(seal['anchor']))
    with np.load(old.checked(seal['planner']),allow_pickle=False) as z:
        prediction=z['forecast'].copy();anchor=z['anchor'].copy() if anchor_receipt['feasible'] else None
    source_plans=io.read(old.checked(seal['plans']))
    shared_points_ref=source_plans['activation']['initial_points']
    with np.load(old.checked(shared_points_ref),allow_pickle=False) as z:seeds={k:z[k].copy() for k in z.files}
    shared=io.read(source/'shared_initial_receipt.json')
    ctx=old.context(entry);planning={key:ctx[key] for key in ('score','val','base')}
    case=domain.create_case(entry,ctx,prediction)
    io.write(directory/'resource_case.json',case.config)
    quote_began=perf_counter()
    try:
        receipt,point=solver.plan(case,planning['score'],planning['val'],prediction,planning['base'],
            anchor=anchor,env=old.WORK['env'],seed_points=seeds,seconds=p['solver_seconds'],
            record_directory=directory/'diagnostics',allowance_yuan=p['allowance_yuan'],anchor_source=seal['anchor'])
    except Exception as exc:
        receipt,point=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc()),None
    elapsed=perf_counter()-quote_began
    verified_anchor=False
    if anchor is not None:
        evidence=SymmetricRecovery(case,ctx['score'],ctx['val'],prediction,ctx['base'],'symmetric_guard',.99)
        verified_anchor=bool(case.audit(anchor)['passed'] and evidence.verify_expansion(anchor)['feasible'])
    fallback=bool((point is None and verified_anchor) or
        (receipt.get('fallback_used') and receipt.get('fallback_seed_name')==solver.ANCHOR_NAME))
    action=anchor.copy() if fallback else np.zeros(case.shape) if point is None else point
    receipt.update(quote_wall_seconds=elapsed,charged_total_seconds=shared['shared_preparation_seconds']+elapsed,
        shared_preparation_seconds=shared['shared_preparation_seconds'],requested_quote_budget_seconds=p['solver_seconds'],
        anchor_fallback=fallback,zero_fallback=point is None and not verified_anchor,
        executed_source='validated_anchor_fallback' if fallback else 'zero_unresolved' if point is None else 'bounded_quote',
        executed_activation_yuan=case.fee(action),executed_cost=prices.cost_breakdown(case,action))
    cap=None if anchor is None else case.fee(anchor)+p['allowance_yuan']
    receipt.update(activation_limit_yuan=cap,executed_activation_slack_yuan=None if cap is None else cap-case.fee(action))
    if cap is not None and case.fee(action)>cap:raise ArithmeticError('Executed action exceeds the exact call budget')
    io.write(directory/'plans.json',{POLICY:receipt})
    np.savez_compressed(directory/'planner.npz',forecast=prediction,anchor=np.zeros(case.shape) if anchor is None else anchor,**{POLICY:action})
    np.savez_compressed(directory/'reservation.npz',**{POLICY:np.max(abs(action),axis=0)})
    io.write(directory/'seal.json',dict(created_utc=old.utc(),planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'),reservation=io.ref(directory/'reservation.npz'),
        source_seal=io.ref(source/'seal.json'),source_anchor=seal['anchor'],
        source_shared_points=shared_points_ref,
        source_shared_receipt=io.ref(source/'shared_initial_receipt.json'),
        score=entry['score'],protocol=io.ref(out/'protocol.json'),planner_actual_voltage_or_network_inputs=False))
    # Actual power/voltage and counterpart execution data are used only AFTER the seal.
    H,C,hist,sigma2=ctx['H'],ctx['common_H'],ctx['history'],ctx['score']['sigma2']
    pred0=old.pair_components(H,C,hist,sigma2,prediction)
    pred=old.pair_components(H,C,hist,sigma2,prediction+case.shift_pq_pu(action,ctx['base']))
    folder=directory/'exact_requested'/POLICY;folder.mkdir(parents=True)
    exact=None;exact0=None
    try:
        pp,qq,state,checks,ref=old.physical_state(ctx,case,action,folder/'full_ac.npz')
        v=state['terminal_voltage_pu'];y=ctx['root'][:,None]**2-v**2
        obs=Observation(pp,qq,y,ctx['labels'],v,ctx['root'])
        exact=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],y,obs,ctx['history_without_common'])
        vb=ctx['baseline_state']['terminal_voltage_pu'];yb=ctx['root'][:,None]**2-vb**2
        ob=Observation(ctx['p'],ctx['q'],yb,ctx['labels'],vb,ctx['root'])
        exact0=old.pair_components(H,C,hist,sigma2,np.c_[ctx['p'],ctx['q']],yb,ob,ctx['history_without_common'])
        np.savez_compressed(folder/'pair_evidence.npz',**exact)
        np.savez_compressed(folder/'forecast_pair_evidence.npz',**pred)
        np.savez_compressed(folder/'clean_observation.npz',P=pp,Q=qq,V=v,Y=y,root=ctx['root'],labels=ctx['labels'])
        io.write(folder/'receipt.json',dict(status='complete',physical_state=ref,checks=checks))
    except Exception as exc:io.write(folder/'receipt.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
    rows=[]
    for repeat in range(p['repeats']):
        rd=directory/f'execution{repeat:02d}';rd.mkdir()
        eta=np.r_[old.rng(p['seed'],entry['index'],draw,773,repeat).uniform(.9,1.,case.P_R),old.rng(p['seed'],entry['index'],draw,774,repeat).uniform(.9,1.,case.Q_R)]
        noise=[old.rng(p['seed'],entry['index'],draw,k,repeat).normal(size=shape) for k,shape in ((2,ctx['p'].shape),(3,ctx['q'].shape),(4,ctx['p'].shape),(5,ctx['root'].shape))]
        np.savez_compressed(rd/'paired_randomness.npz',eta=eta,**{f'noise{i}':x for i,x in enumerate(noise)})
        folder=rd/POLICY;folder.mkdir()
        row=dict(id=entry['id'],original_id=entry['original_id'],resource_variant=entry['resource_variant'],case=entry['case'],
            start_date=entry['start_date'],forecast_index=draw,repeat=repeat,contract_fraction=.1,strategy=POLICY,
            quote_zero_fallback=receipt['zero_fallback'],quote_anchor_fallback=fallback,
            historical_p_map=ctx['score']['p_map'],historical_native_index=ctx['score']['native_index'])
        try:
            delivered,delivery=case.execute(action,ctx['p']*ctx['base'],eta)
            pp,qq,state,checks,ref=old.physical_state(ctx,case,delivered,folder/'full_ac.npz')
            v=state['terminal_voltage_pu'];y0=ctx['root'][:,None]**2-v**2
            pm,qm,y,vm,rm=old.meter(pp,qq,v,ctx['root'],noise)
            obs=Observation(pm,qm,y,ctx['labels'],vm,rm)
            values,ll,new_sse=old.score_observation(ctx,obs)
            components=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],y0,obs,ctx['history_without_common'])
            vb=ctx['baseline_state']['terminal_voltage_pu'];yb=ctx['root'][:,None]**2-vb**2
            pb,qb,ym,vmb,rmb=old.meter(ctx['p'],ctx['q'],vb,ctx['root'],noise)
            ob=Observation(pb,qb,ym,ctx['labels'],vmb,rmb)
            baseline=old.pair_components(H,C,hist,sigma2,np.c_[ctx['p'],ctx['q']],yb,ob,ctx['history_without_common'])
            changes,summary=old.evidence_summary(ctx,baseline,components,pred0,pred,exact0,exact)
            np.savez_compressed(folder/'pair_evidence.npz',**components);np.savez_compressed(folder/'pair_changes.npz',**changes)
            np.savez_compressed(folder/'observations.npz',P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                true_P=pp,true_Q=qq,true_V=v,true_root=ctx['root'],requested_action=action,delivered_action=delivered,
                native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,representative_log_probability=ll)
            requested_cost=prices.cost_breakdown(case,action);settled_cost=prices.cost_breakdown(case,action,delivered)
            quantities=case.amounts(delivered)
            row.update(status='complete',**values,delivery=delivery,evidence=summary,physical_state=ref,AC=checks,
                planned_fee_yuan=requested_cost['total_yuan'],settled_fee_yuan=settled_cost['total_yuan'],
                planned_cost=requested_cost,settled_cost=settled_cost,requested_quantities=case.amounts(action),delivered_quantities=quantities,
                delivered_abs_kwh=quantities['active_abs_kwh'],delivered_abs_kvarh=quantities['reactive_abs_kvarh'],
                all_bus_voltage_min=float(state['voltage_magnitude_pu'].min()),all_bus_voltage_max=float(state['voltage_magnitude_pu'].max()),
                voltage_outside_095_105_cells=int(((state['voltage_magnitude_pu']<.95)|(state['voltage_magnitude_pu']>1.05)).sum()),
                baseline_root_energy_value_yuan=old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                root_energy_value_change_yuan=old.grid_value(case,state['root_supply_kw'])-old.grid_value(case,ctx['baseline_state']['root_supply_kw']))
        except Exception as exc:row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
        rows.append(row);io.write(folder/'receipt.json',row);io.write(directory/'results.json',rows)
    result=dict(status='complete',id=entry['id'],forecast_index=draw,rows=len(rows),failed_rows=sum(r['status']!='complete' for r in rows),
        quote_exceptions=int(receipt.get('status')=='exception' or receipt.get('engine_exception') is not None),
        seconds=perf_counter()-began,results=io.ref(directory/'results.json'),plans=io.ref(directory/'plans.json'))
    io.write(directory/'complete.json',result);return result

def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['inputs'],p['locked_protocol'],p['source_protocol']]:old.checked(ref)
    if (out/'complete.json').exists():return io.read(out/'complete_fraction0.10.json')
    if (out/'lock.json').exists():raise RuntimeError('Review existing lock')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    try:
        entries=io.read(old.checked(p['inputs']));tasks=[(e,d) for e in entries for d in range(p['forecasts'])];receipts=[];failures=[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=old.initialize,initargs=(str(out),)) as pool:
            jobs={pool.submit(run_forecast,t):t for t in tasks}
            for job in as_completed(jobs):
                e,d=jobs[job]
                try:receipts.append(job.result())
                except Exception as exc:failures.append(dict(id=e['id'],forecast_index=d,error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(completed=len(receipts),failed_tasks=failures,total=len(tasks)))
                print(f'{len(receipts)+len(failures)}/{len(tasks)} tasks, {len(failures)} failures',flush=True)
        result=dict(total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,receipts=receipts,
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),execution_failed_rows=sum(r['failed_rows'] for r in receipts))
        io.write(out/'complete_fraction0.10.json',result)
        io.write(out/'complete.json',dict(status='complete',main=io.ref(out/'complete_fraction0.10.json'),source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources'])))
        return result
    finally:(out/'lock.json').unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not args.resume:prepare(args.source,args.out,smoke=args.smoke)
    if not args.prepare_only:print(run(args.out))
