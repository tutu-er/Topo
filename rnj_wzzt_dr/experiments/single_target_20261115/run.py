"""Single history winner, all frozen native competitors, paired original AC paths."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import difflib
from pathlib import Path
import shutil
from time import perf_counter
import traceback
import numpy as np

from ... import io
from ..frozen_rx_guard_20261115 import run as paired
from ..distribution_gate_20261114 import solver as original_solver
from ..flexibility_stability_20261114 import solver as prices
from . import evidence,solver

MODES=paired.MODES


def baseline_sources(source,original):
    changed=[]
    allowed={'rnj_wzzt_dr/experiments/measured_pq_20261114/validate.py',
             'rnj_wzzt_dr/experiments/distribution_gate_20261114/solver.py'}
    for ref in original['sources']:
        path=Path(ref['path']);relative=path.relative_to(paired.old.ROOT)
        snapshot=source/'source_snapshot'/relative
        paired.old.checked(dict(path=str(snapshot),sha256=ref['sha256']))
        current=io.ref(path)
        if current==ref:continue
        if relative.as_posix() not in allowed:raise ValueError('Unreviewed baseline source difference: '+str(path))
        difference=''.join(difflib.unified_diff(snapshot.read_text(encoding='utf8').splitlines(True),path.read_text(encoding='utf8').splitlines(True),fromfile='executed_snapshot',tofile='current'))
        changed.append(dict(old_source=ref,old_snapshot=io.ref(snapshot),new_source=current,diff=difference,
            reason='Historical label repair or opt-in evidence_factory; original defaults preserved and tested'))
    return changed


def prepare(out,*,smoke=False,seconds=600.,preflight=None):
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    source=paired.old.ROOT/'outputs/flexibility_grid_execution_20261115_v1'
    source_audit=paired.old.ROOT/'outputs/flexibility_grid_audit_20261115_v1/report.json'
    original=io.read(source/'protocol.json');complete=io.read(source/'complete.json');audit=io.read(source_audit)
    if complete['status']!='complete' or not complete['source_unchanged'] or audit['status']!='passed' or audit['audited_rows']!=4608:raise ValueError('Completed audited original grid required')
    paired.check_refs([original['inputs'],audit['source_protocol'],audit['source_inputs']])
    changes=baseline_sources(source,original)
    if not smoke and (preflight is None or io.read(preflight)['status']!='passed'):raise ValueError('Independent smoke audit required')
    by_id={e['id']:e for e in io.read(original['inputs']['path'])};entries=[]
    for identity in paired.IDS:
        entry=deepcopy(by_id[identity]);prior=source/'fraction0.10'/identity/'forecast00'
        done=io.read(prior/'complete.json')
        if done['status']!='complete' or done['rows']!=128 or done['failed_rows']:raise ValueError('Incomplete source case')
        entry.update(prior_directory=str(prior),prior_completion=io.ref(prior/'complete.json'),
            prior_seal=io.ref(prior/'seal.json'),prior_results=io.ref(prior/'results.json'),initial_points=io.ref(prior/'shared_initial_points.npz'))
        if any(k.startswith('oracle') for k in entry):raise ValueError('Oracle-selected fields forbidden')
        planning,forecast,case,_=paired.context_planning(entry)
        # Constructor independently verifies strict historical native selection.
        probe=evidence.SingleTargetEvidence(case,planning['score'],planning['val'],forecast,planning['base'],mode='normal_fixed',
            meter_relative_sd=entry['meter']['additional_V_relative_sd'])
        entry['planned_history_target_native']=int(probe.native_indices[probe.targets[0]])
        entries.append(entry)
    out.mkdir(parents=True);io.write(out/'inputs.json',entries)
    refs=paired.panel.snapshot(out,entries)
    for path in (Path(__file__),Path(evidence.__file__),Path(solver.__file__),Path(__file__).with_name('PROTOCOL.md')):
        path=path.resolve();dest=out/'source_snapshot'/path.relative_to(paired.old.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        if not any(ref['path']==str(path) for ref in refs):refs.append(io.ref(path))
    p=dict(created_utc=paired.old.utc(),phase='history_single_target_development',smoke=smoke,oracle=False,
        inputs=io.ref(out/'inputs.json'),baseline_audit=io.ref(source_audit),source_grid_audit=io.ref(source_audit),
        source_run=str(source),source_protocol=io.ref(source/'protocol.json'),source_completion=io.ref(source/'complete.json'),
        baseline_source_differences=changes,preflight_audit=None if preflight is None else io.ref(preflight),
        modes=list(MODES),policies=['single_'+m for m in MODES],seed=original['seed'],forecasts=1,repeats=1 if smoke else 16,
        workers=3,solver_seconds=2. if smoke else float(seconds),expected_quotes=9,expected_rows=9*(1 if smoke else 16),
        expected_reused_baseline_rows=3*8*(1 if smoke else 16),target_rule='Strict original historical validation native MSE winner, no true label',
        competitor_rule='All native points of a different internal topology; no competitor pruned',
        guard_rule='Unchanged original representative forward guards from the historical target only',
        objective='min original activation cost; requested capacity plus delivered activation reported separately',
        reservation_prices=dict(P=.5,Q=.5),forecast_relative_sd=.05,
        shared_seed_source='Exact original common LP portfolio and original150s anchor; no final-policy/oracle/cross-mode seeds',
        zero_rule='Original direct zero gate first, then nonnegative-cost zero sufficient certificate',
        fallback='No verified gate-feasible quote executes zero and is explicitly recorded as failed quote',
        pairing='Exact original forecast, whole-trajectory eta, P/Q/V/root standard normal arrays, original AC and clipping',
        independence='Known development physical days and repeated original random paths, not independent confirmation',
        sources=refs,environment=io.environment())
    p.update(paired.old._git_receipt([Path(r['path']) for r in refs]));io.write(out/'protocol.json',p);return p


def run_case(entry):
    p,out=paired.old.WORK['p'],paired.old.WORK['out'];directory=out/'cases'/entry['id']
    if directory.exists():raise FileExistsError(directory)
    directory.mkdir(parents=True);began=perf_counter();io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=paired.old.utc()))
    planning,forecast,case,seeds=paired.context_planning(entry)
    io.write(directory/'resource_case.json',case.config);shutil.copyfile(entry['initial_points']['path'],directory/'shared_initial_points.npz')
    controls={};plans={};models={}
    for mode in MODES:
        name='single_'+mode;tick=perf_counter()
        io.write(directory/(name+'_started.json'),dict(pid=os.getpid(),mode=mode,allowed_seconds=p['solver_seconds']))
        try:
            receipt,point=solver.plan(case,planning['score'],planning['val'],forecast,planning['base'],mode=mode,
                seconds=p['solver_seconds'],env=paired.old.WORK['env'],seed_points=seeds,record_directory=directory/(name+'_diagnostics'),
                meter_relative_sd=entry['meter']['additional_V_relative_sd'],forecast_relative_sd=p['forecast_relative_sd'])
            if bool(receipt['feasible'])!=(point is not None):raise ValueError('Point and quote disagree')
        except Exception as exc:receipt,point=dict(status='exception',feasible=False,error=repr(exc),traceback=traceback.format_exc()),None
        models[mode]=evidence.SingleTargetEvidence(case,planning['score'],planning['val'],forecast,planning['base'],mode=mode,
            meter_relative_sd=entry['meter']['additional_V_relative_sd'],forecast_relative_sd=p['forecast_relative_sd'])
        action=np.zeros(case.shape) if point is None else np.asarray(point).copy();check=models[mode].verify_expansion(action)
        if point is not None and not(case.audit(action)['passed'] and check['feasible']):raise ValueError('Final direct acceptance failed')
        receipt.update(mode=mode,guard_kind='single_target',new_gate_feasible=point is not None,zero_fallback=point is None,
            quote_wall_seconds=perf_counter()-tick,requested_quote_budget_seconds=p['solver_seconds'],initial_points=entry['initial_points'],
            executed_source='zero_fallback' if point is None else 'verified_history_single_target',executed_probability_check=check,
            executed_cost=prices.cost_breakdown(case,action),target_native=entry['planned_history_target_native'])
        controls[name]=action;plans[name]=receipt;np.savez_compressed(directory/(name+'_action.npz'),action=action)
        io.write(directory/(name+'_plan.json'),receipt)
        print(entry['id'],name,receipt['status'],'fee',receipt.get('cost_yuan'),'fallback',receipt['zero_fallback'],flush=True)
    np.savez_compressed(directory/'planner.npz',forecast=forecast,**controls);io.write(directory/'plans.json',plans)
    io.write(directory/'prediction_audits.json',{m:{name:models[m].verify_expansion(u) for name,u in controls.items()} for m in MODES})
    io.write(directory/'seal.json',dict(created_utc=paired.old.utc(),planner=io.ref(directory/'planner.npz'),plans=io.ref(directory/'plans.json'),
        score=entry['score'],train=entry['files']['train.npz'],validation=entry['files']['validation.npz'],
        shared_initial_points=io.ref(directory/'shared_initial_points.npz'),shared_initial_points_source=entry['initial_points'],
        protocol=io.ref(out/'protocol.json'),resource_case=io.ref(directory/'resource_case.json'),prediction_audits=io.ref(directory/'prediction_audits.json'),
        planned_history_target_native=entry['planned_history_target_native'],planner_actual_voltage_or_network_inputs=False))
    # Empty geometry deliberately adds no PQ/W/oracle protection; original AC execution is reused.
    rows=paired.execute(entry,p,directory,planning,case,forecast,controls,plans,{},dict(guards=[]))
    good=[r for r in rows if r['status']=='complete']
    result=dict(status='complete',id=entry['id'],rows=len(rows),failed_rows=len(rows)-len(good),
        quote_exceptions=sum(r['status']=='exception' for r in plans.values()),feasible_quotes=sum(r['feasible'] for r in plans.values()),
        zero_fallbacks=sum(r['zero_fallback'] for r in plans.values()),zero_sufficient_quotes=sum(r['status']=='zero_sufficient' for r in plans.values()),
        reused_ac_rows=sum('reused_action_source' in r for r in good),new_ac_rows=sum('reused_action_source' not in r for r in good),
        seconds=perf_counter()-began,plans=io.ref(directory/'plans.json'),results=io.ref(directory/'results.json'))
    io.write(directory/'complete.json',result);return result


def run(out):
    out=Path(out).resolve();p=io.read(out/'protocol.json');paired.check_refs([p['inputs'],*p['sources']])
    if (out/'lock.json').exists() or (out/'complete.json').exists():raise RuntimeError('Preserve existing run/locks')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=paired.old.utc()));receipts=[];failures=[]
    try:
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=paired.old.initialize,initargs=(str(out),)) as pool:
            tasks={pool.submit(run_case,e):e for e in io.read(p['inputs']['path'])}
            for task in as_completed(tasks):
                try:receipts.append(task.result())
                except Exception as exc:failures.append(dict(id=tasks[task]['id'],error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(receipts=receipts,failed_tasks=failures))
        result=dict(status='complete' if not failures else 'incomplete',receipts=receipts,failed_tasks=failures,
            **{k:sum(r[k] for r in receipts) for k in ('rows','failed_rows','quote_exceptions','feasible_quotes','zero_fallbacks','zero_sufficient_quotes','reused_ac_rows','new_ac_rows')},
            source_unchanged=all(io.ref(r['path'])==r for r in p['sources']))
        io.write(out/'complete.json',result);return result
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--seconds',type=float,default=600.);parser.add_argument('--preflight',type=Path)
    a=parser.parse_args();prepare(a.out,smoke=a.smoke,seconds=a.seconds,preflight=a.preflight);print(run(a.out))


if __name__=='__main__':main()
