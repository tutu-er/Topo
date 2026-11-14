"""Recover interrupted forecasts without resetting the original quote budget."""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_k]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
from datetime import datetime
import json
import hashlib
import math
from pathlib import Path
import shutil
from time import perf_counter
import traceback
import numpy as np
from . import run as original
from .run import (io,old,evidence,solver,prices,PQIncrementCase,planning_context,
                  diagnostic_check,arrays,MODES,POLICIES,rotation_actions,execute)


def trace_rows(path):
    rows=[]
    for line in Path(path).read_text(encoding='utf8').splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:pass
    return rows


def finite_bound(x):return isinstance(x,(int,float)) and np.isfinite(x) and 0<=x<1e90


def budget_state(directory,mode,stop_epoch,total=600.):
    quote=directory/(mode+'_plan.json')
    if quote.exists():
        q=io.read(quote);elapsed=float(q.get('quote_wall_seconds',q.get('seconds',0.)))
        return dict(kind='completed_quote_action_unavailable',original_quote=io.ref(quote),
            prior_budget_charge_seconds=min(total,elapsed),prior_measured_wall_lower_bound=elapsed,
            prior_time_exact=True,remaining_native_budget_seconds=0.,
            prior_bound=None if q.get('lower_bound_withdrawn') else q.get('lower_bound_yuan'))
    trace=directory/(mode+'_diagnostics')/'progress.jsonl'
    if not trace.exists():return dict(kind='unstarted',prior_budget_charge_seconds=0.,
        prior_measured_wall_lower_bound=0.,prior_time_exact=True,remaining_native_budget_seconds=total,prior_bound=None)
    rows=trace_rows(trace)
    prior=directory/('shared_initial_receipt.json' if mode==MODES[0] else MODES[MODES.index(mode)-1]+'_plan.json')
    charge=min(total,float(math.ceil(max(0.,stop_epoch-prior.stat().st_mtime))))
    bounds=[r['bound'] for r in rows if finite_bound(r.get('bound'))]
    return dict(kind='interrupted_search',interrupted_trace=io.ref(trace),
        interrupted_native_log=io.ref(trace.with_name('native.log')),preceding_completed_file=io.ref(prior),
        preceding_file_mtime_epoch=prior.stat().st_mtime,confirmed_stop_epoch=stop_epoch,
        prior_budget_charge_seconds=charge,prior_measured_wall_lower_bound=max((r.get('wall_seconds',0.) for r in rows),default=0.),
        prior_time_exact=False,remaining_native_budget_seconds=max(0.,total-charge),prior_bound=max(bounds) if bounds else None)


def merge_bounds(cost,prior,current):
    bound=max([0.,*[float(x) for x in (prior,current) if finite_bound(x)]])
    conflict=cost is not None and bound>cost+1e-7*max(1.,cost)
    if conflict:bound=0.
    return bound,None if cost is None else (cost-bound)/max(cost,1e-12),conflict


def select_saved_seed(case,model,seeds):
    point=None;chosen=None;rows=[];began=perf_counter()
    for name,value in seeds.items():
        u=case.array(value);check=solver.direct_audit(case,model,u);cost=float(case.fee(u))
        rows.append(dict(name=name,feasible=check['passed_acceptance'],cost_yuan=cost))
        if check['passed_acceptance'] and (point is None or cost<case.fee(point)):point,chosen=u.copy(),name
    return point,dict(source='original_saved_common_pool',feasible=point is not None,name=chosen,
        cost_yuan=None if point is None else float(case.fee(point)),candidates=rows,audit_seconds=perf_counter()-began,
        scope='Recover a saved seed, not the lost native incumbent')


def atomic_action(path,point):
    path=Path(path);temp=path.with_name(path.name+'.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,u=point)
    os.replace(temp,path);return io.ref(path)


def recover_quote(case,planning,prediction,seeds,anchor,shared_seconds,mode,state,p,directory):
    tick=perf_counter();state=deepcopy(state)
    model=evidence.DistributionEvidence(case,planning['score'],planning['val'],prediction,planning['base'],mode=mode,target_mass=.99)
    saved=directory/(mode+'_resume01_plan.json')
    if saved.exists():
        q=io.read(saved);action=arrays(q['executed_action_checkpoint'])['u']
        if diagnostic_check(model,action)['feasible']!=q['executed_gate_feasible']:raise ValueError('Recovery checkpoint changed')
        return q,model,action
    if (directory/(mode+'_resume01_diagnostics')).exists():
        raise RuntimeError('Recovery search was itself interrupted; audit its used budget before any further attempt')
    remaining=state['remaining_native_budget_seconds'];point=None
    if remaining>max(.001,perf_counter()-tick):
        try:
            q,point=solver.plan(case,planning['score'],planning['val'],prediction,planning['base'],
                mode=mode,seconds=max(.001,remaining-(perf_counter()-tick)),env=old.WORK['env'],
                seed_points={k:v.copy() for k,v in seeds.items()},target_mass=.99,
                record_directory=directory/(mode+'_resume01_diagnostics'))
        except Exception as exc:
            q,point=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc(),
                         optimizer_invoked=False,native_status=None,lower_bound_yuan=None),None
        additional=perf_counter()-tick;verification=0.
    else:
        point,initial=select_saved_seed(case,model,seeds)
        q=dict(mode=mode,feasible=point is not None,status='recovered_saved_seed' if point is not None else 'no_recoverable_feasible_point',
            source='saved_seed_after_interruption',optimizer_invoked=False,native_status=None,initial=initial,
            original_nominal_gates_retained=False,original_scalar_guards_retained=True,
            bound_cost_object='activation_yuan',objective_cost_object='activation_yuan',gate_rows=model.rows,
            gate_count=len(model.rows),target_mass=.99,numerical_gate_buffer=solver.BUFFER,targets=model.targets,
            global_optimality_verified=False,lower_bound_yuan=None)
        additional=0.;verification=perf_counter()-tick
    accepted=point is not None
    if accepted and not solver.direct_audit(case,model,point)['passed_acceptance']:raise ArithmeticError('Invalid recovered point')
    action=point.copy() if accepted else np.zeros(case.shape) if anchor is None else anchor.copy()
    cost=float(case.fee(point)) if accepted else None
    lower,gap,conflict=merge_bounds(cost,state.get('prior_bound'),q.get('lower_bound_yuan'))
    state.update(original_action_unavailable=state['kind']!='unstarted',recovery_verification_seconds=verification,
        additional_quote_wall_seconds=additional,current_bound_before_merge=q.get('lower_bound_yuan'),
        inherited_bound=state.get('prior_bound'),bound_conflict_with_current_verified_point=conflict,
        timing_scope='Prior interruption debit is a conservative budget upper charge, not measured runtime; verification and resumed solve are measured separately.',
        same_model_scope='Original source hashes, saved forecast/resources/history, mode, target mass, risk/lambda rule, guards, buffer and activation objective unchanged.')
    bound_sources=[]
    if finite_bound(state.get('prior_bound')):
        bound_sources.append(dict(reference=state.get('original_quote',state.get('interrupted_trace')),
            lower_bound_yuan=state['prior_bound'],same_problem_fingerprint=state['same_problem_fingerprint'],origin='prior_same_mode'))
    if finite_bound(q.get('lower_bound_yuan')):
        bound_sources.append(dict(reference=q.get('diagnostics_receipt'),lower_bound_yuan=q['lower_bound_yuan'],
            same_problem_fingerprint=state['same_problem_fingerprint'],origin='current_same_mode'))
    check=diagnostic_check(model,action)
    q.update(feasible=accepted,cost_yuan=cost,lower_bound_yuan=lower,relative_gap=gap,lower_bound_withdrawn=conflict,
        lower_bound_source='max_valid_prior_and_current_same_qcqp' if not conflict else 'nonnegative_cost_after_conflict',
        global_optimality_verified=False,recovery=state,bound_sources=bound_sources,new_gate_feasible=accepted,new_constraints_feasible=accepted,
        anchor_fallback=not accepted and anchor is not None,zero_fallback=not accepted and anchor is None,
        executed_source='accepted_recovery_quote' if accepted else 'validated_anchor_fallback' if anchor is not None else 'zero_fallback',
        executed_gate_feasible=bool(check['feasible']),executed_gate_check=check,executed_cost=prices.cost_breakdown(case,action),
        requested_quote_budget_seconds=p['solver_seconds'],quote_budget_exceeded=state['prior_budget_charge_seconds']+additional>p['solver_seconds']+.05,
        shared_preparation_seconds=shared_seconds,charged_total_seconds=shared_seconds+state['prior_budget_charge_seconds']+additional+verification,
        initial_points=io.ref(directory/'shared_initial_points.npz'),
        feasibility_scope='Current executable recovery point; lost original incumbent not replayed',
        executed_action_checkpoint=atomic_action(directory/(mode+'_resume01_action.npz'),action))
    q.pop('quote_wall_seconds',None)
    io.write(saved,q);return q,model,action


def initialize(out):
    from .continuation import initialize as init
    init(out)


def prepare(out):
    out=Path(out).resolve();pause=io.read(out/'paused.json');p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['inputs'],pause['progress']]:old.checked(ref)
    stop=datetime.fromisoformat(pause['recorded_utc']).timestamp();tasks=[]
    entries={e['id']:e for e in io.read(out/'inputs.json')}
    for item in pause['partial_tasks']:
        d=out/'fraction0.10'/item['id']/item['forecast']
        for ref in [item['started'],*item['saved_plans'],*item['native_logs']]:old.checked(ref)
        states={m:budget_state(d,m,stop,p['solver_seconds']) for m in MODES};entry=entries[item['id']]
        for mode,state in states.items():
            problem=dict(mode=mode,original_protocol=io.ref(out/'protocol.json'),score=entry['score'],
                validation=entry['files']['validation.npz'],forecast=entry['forecast_ref'],
                resource_case=io.ref(d/'resource_case.json'),shared_initial_points=io.ref(d/'shared_initial_points.npz'),
                target_mass=.99,buffer=solver.BUFFER)
            state.update(same_problem_inputs=problem,pause_manifest=io.ref(out/'paused.json'),
                task_id=item['id'],forecast=int(item['forecast'].removeprefix('forecast')),
                same_problem_fingerprint=hashlib.sha256(json.dumps(problem,sort_keys=True).encode()).hexdigest())
        tasks.append(dict(id=item['id'],forecast=int(item['forecast'].removeprefix('forecast')),modes=states))
    own=out/'resume01_partial_protocol.json'
    if own.exists():raise FileExistsError(own)
    source=Path(__file__).resolve();copy=out/'resume01_source_snapshot'/source.relative_to(old.ROOT)
    copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,copy)
    record=dict(stage='partial_recovery',created_utc=old.utc(),source=io.ref(source),source_snapshot=io.ref(copy),
        original_protocol=io.ref(out/'protocol.json'),pause=io.ref(out/'paused.json'),tasks=tasks,workers=4,
        policy='Preserve original quotes/logs; no new budget for finished search. Recheck saved seeds and use conservative remaining budget for interrupted search; all four forecasts are separately reported.')
    io.write(own,record);return record


def run(out):
    out=Path(out).resolve();p=io.read(out/'protocol.json');rp=io.read(out/'resume01_partial_protocol.json')
    for ref in [*p['sources'],p['inputs'],rp['source'],rp['source_snapshot']]:old.checked(ref)
    if (out/'resume01_partial_complete.json').exists():return io.read(out/'resume01_partial_complete.json')
    if not (out/'resume01_unstarted_complete.json').exists():raise ValueError('Wait for other four-worker continuation pool')
    cp=io.read(out/'resume01_unstarted_protocol.json')
    for ref in (cp['source'],cp['source_snapshot']):old.checked(ref)
    entries={e['id']:e for e in io.read(out/'inputs.json')};receipts=[];failures=[]
    lock=out/'resume01_partial_lock.json'
    with lock.open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),created_utc=old.utc()),f)
    try:
        with ProcessPoolExecutor(max_workers=4,initializer=initialize,initargs=(str(out),)) as pool:
            jobs={pool.submit(run_forecast,(entries[t['id']],t['forecast'])):t for t in rp['tasks']}
            for future in as_completed(jobs):
                t=jobs[future]
                try:receipts.append(future.result())
                except Exception as exc:failures.append(dict(id=t['id'],forecast_index=t['forecast'],status='task_failed',error=repr(exc),traceback=traceback.format_exc(),interruption_affected=True))
                io.write(out/'resume01_partial_progress.json',dict(completed=len(receipts),failed_tasks=failures))
                print('partial',len(receipts),len(failures),flush=True)
    finally:lock.unlink(missing_ok=True)
    unchanged=all(io.ref(ref['path'])['sha256']==ref['sha256'] for ref in [*p['sources'],rp['source'],cp['source']])
    done=dict(status='complete',receipts=receipts,failed_tasks=failures,source_unchanged=unchanged)
    io.write(out/'resume01_partial_complete.json',done);return done


def finalize(out):
    """Assemble all original identities without overwriting paused progress."""
    out=Path(out).resolve();p=io.read(out/'protocol.json');pause=io.read(out/'paused.json')
    if (out/'complete.json').exists():raise FileExistsError('Retain existing main completion')
    for ref in [*p['sources'],p['inputs'],pause['protocol'],pause['progress']]:old.checked(ref)
    original_progress=io.read(old.checked(pause['progress']))
    stages=[io.read(out/(name+'_complete.json')) for name in ('resume01_unstarted','resume01_partial')]
    protocols=[out/(name+'_protocol.json') for name in ('resume01_unstarted','resume01_partial')]
    for stage,path in zip(stages,protocols,strict=True):
        if stage['status']!='complete' or not stage['source_unchanged']:raise ValueError('Incomplete or changed continuation stage')
        doc=io.read(path)
        for ref in (doc['source'],doc['source_snapshot'],doc['original_protocol']):old.checked(ref)
    entries=io.read(out/'inputs.json');expected={(e['id'],d) for e in entries for d in range(p['forecasts'])}
    receipts=original_progress['receipts']+[r for stage in stages for r in stage['receipts']]
    failures=original_progress['failed_tasks']+[r for stage in stages for r in stage['failed_tasks']]
    identities=[(r['id'],r['forecast_index']) for r in receipts+failures]
    if len(identities)!=len(set(identities)) or set(identities)!=expected:raise ValueError('Incomplete or duplicate continuation identities')
    for row in receipts:
        directory=out/'fraction0.10'/row['id']/f"forecast{row['forecast_index']:02d}"
        if io.read(directory/'complete.json')!=row:raise ValueError('Task completion receipt changed')
        old.checked(row['plans']);old.checked(row['results'])
    result=dict(total_forecasts=len(expected),completed_forecasts=len(receipts),failed_tasks=failures,
        preparation_failed_histories=len(io.read(out/'preparation_failures.json')),
        quote_exceptions=sum(r['quote_exceptions'] for r in receipts),execution_failed_rows=sum(r['failed_rows'] for r in receipts),
        feasible_quotes_by_mode={m:sum(r['new_gate_feasible'][m] for r in receipts) for m in MODES},
        anchor_fallbacks=sum(r['anchor_fallbacks'] for r in receipts),zero_fallbacks=sum(r['zero_fallbacks'] for r in receipts),
        receipts=receipts,interruption_affected_forecasts=pause['partial_tasks'],
        continuation_protocols=[io.ref(x) for x in protocols],
        original_paused_progress_retained=True,original_lock_retained_as_historical_record=True)
    io.write(out/'complete_summary.json',result)
    io.write(out/'resume01_final_progress.json',dict(total=len(expected),completed=len(receipts),failed_tasks=failures))
    done=dict(status='complete',main=io.ref(out/'complete_summary.json'),source_unchanged=True,
        continuation_sources_unchanged=True,continuation_protocols=[io.ref(x) for x in protocols],
        original_driver_scope='All 85 original source files unchanged; persistence/recovery changes are explicitly recorded by the two continuation protocols')
    io.write(out/'complete.json',done);return done


# run_forecast below is copied from the frozen driver, replacing only directory
# admission, shared-state reload and quote recovery. Execution stays shared.
def run_forecast(task):
    entry, draw = task
    p, out = old.WORK['p'], old.WORK['out']
    directory = out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():
        return io.read(directory/'complete.json')
    if not directory.exists():raise ValueError('Only explicitly recorded partial forecasts are admitted')
    if (directory/'seal.json').exists():raise ValueError('Already sealed partial execution requires separate review')
    began=perf_counter()
    rp=io.read(out/'resume01_partial_protocol.json')
    item=next(t for t in rp['tasks'] if t['id']==entry['id'] and t['forecast']==draw)
    planning=planning_context(entry,entry['base_kw'])
    prediction=arrays(entry['forecast_ref'])[f'forecast{draw:02d}']
    case=PQIncrementCase(planning['resource'],planning['labels'],prediction,planning['base'],q_resources=entry['q_resources'],fraction=.1)
    if case.config!=io.read(directory/'resource_case.json'):raise ValueError('Recovered resource case differs')
    seeds=arrays(io.ref(directory/'shared_initial_points.npz'))
    anchor_info=io.read(directory/'anchor.json');anchor_receipt=io.read(directory/'anchor_plan.json')
    anchor=seeds.get('shared_150_second_anchor') if anchor_info['feasible'] else None
    if anchor_info['feasible'] and anchor is None:raise ValueError('Saved anchor missing')
    shared_seconds=float(anchor_info['shared_seconds'])
    plans, models = {}, {}
    controls = dict(no_dfs=np.zeros(case.shape), anchor=np.zeros(case.shape) if anchor is None else anchor.copy())
    for mode in MODES:
        receipt,model,action=recover_quote(case,planning,prediction,seeds,anchor,shared_seconds,mode,item['modes'][mode],p,directory)
        plans[mode],models[mode],controls[mode]=receipt,model,action
    rotations = rotation_actions(case, controls)
    controls = {name:controls[name] for name in POLICIES}
    io.write(directory/'rotation_receipt.json', rotations)
    audits = {mode:{name:diagnostic_check(model, u) for name,u in controls.items()} for mode,model in models.items()}
    io.write(directory/'prediction_audits.json', audits)
    np.savez_compressed(directory/'planner.npz', forecast=prediction, **controls)
    np.savez_compressed(directory/'reservation.npz', **{name:np.max(abs(u), axis=0) for name,u in controls.items()})
    io.write(directory/'plans.json', plans)
    io.write(directory/'seal.json', dict(created_utc=old.utc(), planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'), reservation=io.ref(directory/'reservation.npz'),
        anchor=io.ref(directory/'anchor_plan.json'), prediction_audits=io.ref(directory/'prediction_audits.json'),
        score=entry['score'], validation=entry['files']['validation.npz'], protocol=io.ref(out/'protocol.json'),
        resource_case=io.ref(directory/'resource_case.json'), rotation=io.ref(directory/'rotation_receipt.json'),
        planner_actual_voltage_or_network_inputs=False,continuation_protocol=io.ref(out/'resume01_partial_protocol.json')))
    ctx = old.context(entry)  # Actual future network/voltage becomes available only after the seal.
    if ctx['base'] != planning['base'] or ctx['score'] != planning['score']:
        raise ArithmeticError('Planning and evaluation contract differ')
    rows = execute(entry, draw, p, directory, ctx, case, prediction, controls, plans, anchor_receipt)
    completion = dict(status='complete', id=entry['id'], forecast_index=draw, rows=len(rows),
        failed_rows=sum(r['status'] != 'complete' for r in rows), seconds=perf_counter()-began,
        plans=io.ref(directory/'plans.json'), results=io.ref(directory/'results.json'),
        quote_exceptions=sum(r.get('status') == 'exception' for r in plans.values()),
        new_gate_feasible={m:plans[m]['new_gate_feasible'] for m in MODES},
        anchor_fallbacks=sum(r['anchor_fallback'] for r in plans.values()),
        zero_fallbacks=sum(r['zero_fallback'] for r in plans.values()))
    completion.update(interruption_affected=True,continuation_protocol=io.ref(out/'resume01_partial_protocol.json'),seconds_scope='Continuation attempt only; prior quote accounting remains separate')
    io.write(directory/'complete.json', completion)
    return completion

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--finalize-only',action='store_true')
    args=parser.parse_args()
    if args.finalize_only:print(finalize(args.run))
    else:
        if not args.resume:prepare(args.run)
        if not args.prepare_only:print(run(args.run))
