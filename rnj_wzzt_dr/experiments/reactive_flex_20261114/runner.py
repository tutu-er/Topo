"""Paired placement experiment using the existing optimizers and AC evaluator."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np

from ... import io
from ...observations import Observation
from ..measured_pq_flex_20261114 import runner as old
from ..absolute_identification_20261114.commonseeds import build_portfolio
from ..flexibility_sweep_20261113.execution import fixed_rotation
from .domain import PQIncrementCase


def prepare(index, ac_index, placements, out, *, seed=20261114, forecasts=4,
            repeats=8, seconds=30., workers=4):
    out = Path(out).resolve()
    p = old.prepare(index, ac_index, out, seed=seed, forecasts=forecasts,
                    repeats=repeats, seconds=seconds, workers=workers)
    placement = io.read(placements)
    configs = placement['configurations_by_input']
    originals = io.read(out/'inputs.json'); entries = []
    for entry in originals:
        for variant, resources in configs[entry['id']].items():
            item = deepcopy(entry)
            item.update(original_id=entry['id'], id=entry['id']+'__'+variant,
                        resource_variant=variant, q_resources=resources)
            entries.append(item)
    io.write(out/'inputs.json', entries)
    io.write(out/'placements.json', placement)
    p.update(inputs=io.ref(out/'inputs.json'), placement=io.ref(out/'placements.json'),
             placement_source=io.ref(placements),
             phase='paired active/reactive flexibility placement development' if seed==20261114
                   else 'locked-placement confirmation with fresh forecast/response/meter streams',
             expansion=dict(automatic_only_for_full_16x4=False),
             interpretation=f'{len(originals)} fixed network/day histories, {len(entries)} placement/history combinations; '
                            f'{forecasts*repeats} common paths per history across placements, not independent field days',
             original_histories=len(originals), placement_variants=sorted({e['resource_variant'] for e in entries}),
             response='P stream 773 unchanged; appended independent Q stream 774; each whole-day Uniform(.9,1).',
             units='P columns kW/kWh; Q columns kvar/kvarh; shared per-unit Sbase. No mixed-unit energy totals.',
             solver_scope='Existing representative-parameter planners unchanged except general independent P/Q response mapping; all native outcomes retained.',
             source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')))
    io.write(out/'protocol.json', p)
    return p


def initialize(out):
    old.initialize(out)


def case_for(entry, ctx, prediction):
    return PQIncrementCase(ctx['resource'], ctx['labels'], prediction, ctx['base'],
                           q_resources=entry['q_resources'], fraction=.1)


def run_forecast(task):
    entry, draw = task; p=old.WORK['p']; out=old.WORK['out']
    directory=out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():
        return io.read(directory/'complete.json')
    if directory.exists():
        raise RuntimeError('Review partial task before resuming: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json', dict(pid=os.getpid(), created_utc=old.utc()))
    began=perf_counter(); ctx=old.context(entry); score=ctx['score']
    prediction=old.forecast_power(np.c_[ctx['p'],ctx['q']], p['seed'],entry['index'],draw)
    case=case_for(entry,ctx,prediction)
    io.write(directory/'resource_case.json',case.config)
    seeds, seed_receipt=build_portfolio(case,score,prediction,ctx['base'],env=old.WORK['env'],
                                        seconds=min(2.,p['solver_seconds']/4))
    np.savez_compressed(directory/'shared_initial_points.npz',**seeds)
    io.write(directory/'shared_initial_receipt.json',seed_receipt)
    pattern, pattern_receipt=fixed_rotation(case); pattern_fee=case.fee(pattern)
    controls={'no_dfs':np.zeros(case.shape)}; plans={}
    for mode in old.MODES:
        tick=perf_counter()
        try:
            receipt,u=old.quote(mode,case,score,ctx['val'],ctx['means'],prediction,ctx['base'],seeds,
                max(.001,p['solver_seconds']-seed_receipt['seconds']),old.WORK['env'])
            if (u is not None)!=bool(receipt['feasible']):
                raise ArithmeticError('Quote feasibility flag mismatch')
            if u is not None and not case.audit(u)['passed']:
                raise ArithmeticError('Quote violates resource domain')
        except Exception as exc:
            receipt,u=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc()),None
        controls[mode]=np.zeros(case.shape) if u is None else u
        fee=case.fee(controls[mode])
        controls[mode+'_rotation']=min(1.,fee/pattern_fee)*pattern if pattern_fee>0 else np.zeros(case.shape)
        total=seed_receipt['seconds']+perf_counter()-tick
        receipt.update(shared_seed_seconds=seed_receipt['seconds'],charged_total_seconds=total,
            charged_total_budget_exceeded=total>p['solver_seconds']+.05,zero_fallback=u is None,
            nonzero_verified_feasible=bool(u is not None and fee>1e-9),
            rotation_fee_yuan=case.fee(controls[mode+'_rotation']),
            rotation_same_fee=abs(case.fee(controls[mode+'_rotation'])-fee)<=1e-7,
            rotation_full_pattern_fee_yuan=pattern_fee,
            quantities=case.quantity_summary(controls[mode]))
        plans[mode]=receipt
    np.savez_compressed(directory/'planner.npz',forecast=prediction,**controls)
    io.write(directory/'plans.json',plans); io.write(directory/'rotation_receipt.json',pattern_receipt)
    io.write(directory/'seal.json',dict(created_utc=old.utc(),planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'),score=entry['score'],protocol=io.ref(out/'protocol.json'),
        planner_actual_voltage_or_network_inputs=False))
    H,C,hist,sigma2=ctx['H'],ctx['common_H'],ctx['history'],score['sigma2']
    predicted={name:old.pair_components(H,C,hist,sigma2,prediction+case.shift_pq_pu(u,ctx['base']))
               for name,u in controls.items()}
    exact={}; rows=[]
    for name,u in controls.items():
        d=directory/'exact_requested'/name;d.mkdir(parents=True)
        try:
            pp,qq,state,checks,ref=old.physical_state(ctx,case,u,d/'full_ac.npz')
            v=state['terminal_voltage_pu'];y=ctx['root'][:,None]**2-v**2
            obs=Observation(pp,qq,y,ctx['labels'],v,ctx['root'])
            exact[name]=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],y,obs,ctx['history_without_common'])
            _,availability=case.execute(u,ctx['p']*ctx['base'],np.ones(case.R))
            np.savez_compressed(d/'pair_evidence.npz',**exact[name])
            np.savez_compressed(d/'forecast_pair_evidence.npz',**predicted[name])
            np.savez_compressed(d/'clean_observation.npz',P=pp,Q=qq,V=v,Y=y,root=ctx['root'],labels=ctx['labels'])
            io.write(d/'receipt.json',dict(status='complete',physical_state=ref,checks=checks,
                requested_quantities=case.quantity_summary(u),
                PV_net_export_proxy_excess_kwh=availability['PV_clipped_kwh'],
                interpretation='Exact requested action, no response or PV clipping; ideal delivery diagnostic'))
        except Exception as exc:
            io.write(d/'receipt.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
    for repeat in range(p['repeats']):
        rd=directory/f'execution{repeat:02d}';rd.mkdir()
        eta=np.r_[old.rng(p['seed'],entry['index'],draw,773,repeat).uniform(.9,1.,case.P_R),
                  old.rng(p['seed'],entry['index'],draw,774,repeat).uniform(.9,1.,case.Q_R)]
        noise=[old.rng(p['seed'],entry['index'],draw,k,repeat).normal(size=shape) for k,shape in
               ((2,ctx['p'].shape),(3,ctx['q'].shape),(4,ctx['p'].shape),(5,ctx['root'].shape))]
        np.savez_compressed(rd/'paired_randomness.npz',eta=eta,**{f'noise{i}':z for i,z in enumerate(noise)})
        baseline=None
        for name,u in controls.items():
            d=rd/name;d.mkdir()
            row=dict(id=entry['id'],original_id=entry['original_id'],resource_variant=entry['resource_variant'],
                case=entry['case'],start_date=entry['start_date'],forecast_index=draw,repeat=repeat,
                contract_fraction=.1,strategy=name,planned_fee_yuan=case.fee(u),
                historical_p_map=score['p_map'],historical_native_index=score['native_index'],
                quote_zero_fallback=False if name=='no_dfs' else plans[name.removesuffix('_rotation')]['zero_fallback'])
            try:
                action,delivery=case.execute(u,ctx['p']*ctx['base'],eta)
                pp,qq,state,checks,ref=old.physical_state(ctx,case,action,d/'full_ac.npz')
                v=state['terminal_voltage_pu'];clean_y=ctx['root'][:,None]**2-v**2
                pm,qm,y,vm,rm=old.meter(pp,qq,v,ctx['root'],noise)
                obs=Observation(pm,qm,y,ctx['labels'],vm,rm)
                values,ll,new_sse=old.score_observation(ctx,obs)
                components=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],clean_y,obs,ctx['history_without_common'])
                if name=='no_dfs':baseline=components
                if baseline is None:
                    change,summary={},dict(status='unavailable',reason='Paired zero failed')
                else:
                    change,summary=old.evidence_summary(ctx,baseline,components,predicted['no_dfs'],predicted[name],
                                                        exact.get('no_dfs'),exact.get(name))
                np.savez_compressed(d/'pair_evidence.npz',**components)
                np.savez_compressed(d/'pair_changes.npz',**change)
                np.savez_compressed(d/'observations.npz',P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=ctx['root'],requested_action=u,delivered_action=action,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,
                    representative_log_probability=ll)
                amounts=case.quantity_summary(action)
                row.update(status='complete',**values,delivery=delivery,evidence=summary,physical_state=ref,AC=checks,
                    settled_fee_yuan=case.fee(action),requested_quantities=case.quantity_summary(u),
                    delivered_quantities=amounts,delivered_abs_kwh=amounts['active_abs_kwh'],
                    delivered_abs_kvarh=amounts['reactive_abs_kvarh'],
                    baseline_root_energy_value_yuan=old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    root_energy_value_change_yuan=old.grid_value(case,state['root_supply_kw'])-old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    all_bus_voltage_min=float(state['voltage_magnitude_pu'].min()),
                    all_bus_voltage_max=float(state['voltage_magnitude_pu'].max()),
                    voltage_outside_095_105_cells=int(((state['voltage_magnitude_pu']<.95)|(state['voltage_magnitude_pu']>1.05)).sum()))
            except Exception as exc:
                row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(row);io.write(d/'receipt.json',row)
        io.write(directory/'results.json',rows)
    receipt=dict(status='complete',id=entry['id'],forecast_index=draw,contract_fraction=.1,
        rows=len(rows),failed_rows=sum(r['status']!='complete' for r in rows),seconds=perf_counter()-began,
        plans=io.ref(directory/'plans.json'),results=io.ref(directory/'results.json'),
        quote_exceptions=sum(r['status']=='exception' for r in plans.values()),
        old_nonzero_feasible=plans['old_gamma2']['nonzero_verified_feasible'],
        symmetric_nonzero_feasible=plans['symmetric_guard']['nonzero_verified_feasible'])
    io.write(directory/'complete.json',receipt)
    return receipt


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in p['sources']:old.checked(ref)
    old.checked(p['placement']);old.checked(p['source_protocol'])
    if (out/'lock.json').exists():raise RuntimeError('Review existing run lock')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    try:
        entries=io.read(old.checked(p['inputs']));tasks=[(e,d) for e in entries for d in range(p['forecasts'])]
        receipts=[];failures=[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(out),)) as pool:
            pending={pool.submit(run_forecast,t):t for t in tasks}
            for future in as_completed(pending):
                entry,draw=pending[future]
                try:receipts.append(future.result())
                except Exception as exc:failures.append(dict(id=entry['id'],forecast_index=draw,error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(total=len(tasks),completed=len(receipts),failed_tasks=failures,receipts=receipts))
                print(f'{len(receipts)+len(failures)}/{len(tasks)} tasks, {len(failures)} failures',flush=True)
        completion=dict(fraction=.1,total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),execution_failed_rows=sum(r['failed_rows'] for r in receipts),
            old_nonzero_feasible=sum(r['old_nonzero_feasible'] for r in receipts),
            symmetric_nonzero_feasible=sum(r['symmetric_nonzero_feasible'] for r in receipts),receipts=receipts)
        io.write(out/'complete_fraction0.10.json',completion)
        io.write(out/'complete.json',dict(status='complete',source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources']),
            main=io.ref(out/'complete_fraction0.10.json'),expansion=None))
        return completion
    finally:(out/'lock.json').unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path);parser.add_argument('--ac-index',type=Path)
    parser.add_argument('--placements',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261114);parser.add_argument('--forecasts',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=8);parser.add_argument('--seconds',type=float,default=30.)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if not args.resume:
        prepare(args.index,args.ac_index,args.placements,args.out,seed=args.seed,forecasts=args.forecasts,
                repeats=args.repeats,seconds=args.seconds,workers=args.workers)
    if not args.prepare_only:
        done=run(args.out);print({k:v for k,v in done.items() if k!='receipts'})
