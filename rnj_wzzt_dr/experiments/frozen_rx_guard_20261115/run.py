"""Keep frozen identifiers and execute only newly planned guarded actions."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...observations import Observation, load
from ..distribution_panel_20261115 import run as panel
from ..distribution_gate_20261114.run import planning_context
from ..distribution_gate_20261114.evidence import DistributionEvidence
from ..measured_pq_flex_20261114 import runner as old
from ..reactive_flex_20261114.domain import PQIncrementCase
from ..flexibility_stability_20261114 import solver as prices
from . import geometry, solver

IDS = ('pengwah18__2024-01-04__s2__m2p0',
       'soumalas11__2024-01-04__s2__m0p5',
       'soumalas11__2024-01-04__s2__m1p0')
MODES = panel.MODES
KINDS = ('pq', 'response')


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def check_refs(refs):
    for ref in refs:
        old.checked(ref)


def baseline_sources(source, references):
    """Verify old snapshots, explicitly account for the later label-only repair."""
    changes=[]
    before='scoring_rows="96 historical validation + 96 new-day rows; training is not rescored",'
    after='scoring_rows=f"{splits[\'validation\'].points} historical validation + {splits[\'new_day\'].points} new-day rows; training is not rescored",'
    for ref in references:
        path=Path(ref['path']); relative=path.relative_to(old.ROOT)
        snapshot=source/'source_snapshot'/relative
        old.checked(dict(path=str(snapshot),sha256=ref['sha256']))
        current=io.ref(path)
        if current['sha256']==ref['sha256']:
            continue
        text=snapshot.read_text(encoding='utf-8')
        if (relative.as_posix()!='rnj_wzzt_dr/experiments/measured_pq_20261114/validate.py'
                or text.count(before)!=1 or text.replace(before,after)!=path.read_text(encoding='utf-8')):
            raise ValueError('Unreviewed baseline source change: '+str(path))
        changes.append(dict(prior=ref,snapshot=io.ref(snapshot),current=current,
            reason='Previously committed scoring_rows description repair only; computation unchanged'))
    return changes


def context_planning(entry):
    prior = Path(entry['prior_directory'])
    sealed = io.read(old.checked(entry['prior_seal']))
    check_refs([sealed['planner'], sealed['plans'], sealed['resource_case'], entry['score'],
                entry['files']['train.npz'], entry['files']['validation.npz'], entry['initial_points']])
    planning = planning_context(entry, entry['base_kw'])
    prediction = arrays(sealed['planner']['path'])['forecast']
    case = PQIncrementCase(planning['resource'], planning['labels'], prediction,
        planning['base'], q_resources=entry['q_resources'], fraction=.1)
    if case.config != io.read(sealed['resource_case']['path']):
        raise ValueError('Resource offers or domain changed')
    if not np.array_equal(prediction, arrays(entry['forecast_ref']['path'])['forecast00']):
        raise ValueError('Forecast differs from original grid')
    return planning, prediction, case, arrays(entry['initial_points']['path'])


def source_snapshot(out, entries):
    refs = panel.snapshot(out, entries)
    known = {r['path'] for r in refs}
    for path in (Path(__file__), Path(geometry.__file__), Path(solver.__file__),
                 Path(__file__).with_name('PROTOCOL.md')):
        path = path.resolve()
        if str(path) not in known:
            target = out/'source_snapshot'/path.relative_to(old.ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            refs.append(io.ref(path))
    return refs


def prepare(source, source_audit, out, *, smoke=False, pq_rho=4., response_rms=.25,
            seconds=600., parent_run=None, modes=MODES, preflight_audit=None):
    source, source_audit, out = map(lambda x: Path(x).resolve(), (source, source_audit, out))
    if out.exists():
        raise FileExistsError('Keep all previous outputs: '+str(out))
    if pq_rho <= 0 or response_rms <= 0 or seconds <= 0:
        raise ValueError('Positive budget values required')
    if not smoke:
        if preflight_audit is None:
            raise ValueError('An independently passed integration audit is required')
        preflight=io.read(preflight_audit)
        if preflight['status']!='passed' or preflight['audited_rows']<12:
            raise ValueError('Preflight audit did not pass')
    original = io.read(source/'protocol.json')
    complete = io.read(source/'complete.json')
    audit = io.read(source_audit)
    if complete['status'] != 'complete' or not complete['source_unchanged']:
        raise ValueError('Complete unchanged baseline run required')
    if audit['status'] != 'passed' or audit['audited_rows'] != 4608:
        raise ValueError('Original 4608-row independent audit required')
    if (original['seed'], original['repeats'], original['forecasts']) != (2026111504, 16, 1):
        raise ValueError('Expected the frozen 36-grid execution stream')
    source_changes=baseline_sources(source,original['sources'])
    check_refs([original['inputs'], audit['source_protocol'], audit['source_inputs']])
    by_id = {e['id']: e for e in io.read(original['inputs']['path'])}
    entries = []
    for identity in IDS:
        entry = deepcopy(by_id[identity])
        prior = source/'fraction0.10'/identity/'forecast00'
        done = io.read(prior/'complete.json')
        if done['status'] != 'complete' or done['rows'] != 128 or done['failed_rows']:
            raise ValueError('Incomplete paired source: '+identity)
        entry.update(prior_directory=str(prior), prior_completion=io.ref(prior/'complete.json'),
            prior_seal=io.ref(prior/'seal.json'), prior_results=io.ref(prior/'results.json'),
            initial_points=io.ref(prior/'shared_initial_points.npz'))
        context_planning(entry)
        entries.append(entry)
    if smoke:
        entries = [entries[0], entries[-1]]
    out.mkdir(parents=True)
    io.write(out/'inputs.json', entries)
    protocol = dict(created_utc=old.utc(), phase='frozen_rx_guard_development', smoke=smoke,
        source_run=str(source), source_audit=io.ref(source_audit),
        source_protocol=io.ref(source/'protocol.json'), source_completion=io.ref(source/'complete.json'),
        baseline_source_label_repairs=source_changes,
        preflight_audit=None if preflight_audit is None else io.ref(preflight_audit),
        inputs=io.ref(out/'inputs.json'), modes=list(modes), kinds=list(KINDS),
        seed=original['seed'], forecasts=1, repeats=1 if smoke else 16,
        solver_seconds=2. if smoke else float(seconds), workers=1 if smoke else 3,
        pq_rho=float(pq_rho), response_rms=float(response_rms), target_mass=.99,
        expected_quotes=len(entries)*len(modes)*len(KINDS),
        expected_rows=len(entries)*len(modes)*len(KINDS)*2*(1 if smoke else 16),
        shared_seed_source='Unchanged old common LP pool plus original150s anchor; no old final policy or cross-mode warm starts',
        previous_shared_work='Old shared2s and anchor150s are sunk costs; reported separately, not repeated',
        reservation_prices=dict(P=.5, Q=.5), forecast_relative_sd=.05,
        budget_scope='Additional homogeneous PSD bounds on the unchanged probability QCQP; same activation objective',
        response_scope='Empirical historical parameter-uncertainty index of the action increment; not a total model-error bound or additional90% guarantee',
        frozen_scope='No refitting, no scoring/calibration/target/risk/response/price/domain changes',
        fallback='No verified guarded point executes zero, recorded as failure; zero fallback is not a new gate certificate',
        pairing='Exact old forecast, device multipliers and P/Q/V/root standard normal arrays',
        independent_unit='Three known development settings on two physical network-days; not independent confirmation',
        parent_run=None if parent_run is None else str(Path(parent_run).resolve()),
        environment=io.environment(), sources=source_snapshot(out, entries))
    protocol.update(old._git_receipt([Path(r['path']) for r in protocol['sources']]))
    io.write(out/'protocol.json', protocol)
    return protocol


def build_geometry(entry, planning, case, forecast, p):
    """The only place that builds new budgets; historical observations only."""
    train = load(old.checked(entry['files']['train.npz']))
    probe = DistributionEvidence(case, planning['score'], planning['val'], forecast,
        planning['base'], mode='normal_fixed', target_mass=p['target_mass'],
        meter_relative_sd=entry['meter']['additional_V_relative_sd'])
    return geometry.build_budgets(case, planning['score'], train, planning['val'],
        planning['base'], targets=probe.targets,
        meter_relative_sd=entry['meter']['additional_V_relative_sd'],
        target_mass=p['target_mass'], pq_rho=p['pq_rho'], response_rms_fraction=p['response_rms'])


def get_guards(kind, matrices, metadata, p):
    keys = ['pq'] if kind == 'pq' else [k for k in matrices if k.startswith('response_')]
    if not keys:
        raise ValueError('Missing guard geometry')
    limit = p['pq_rho']**2 if kind == 'pq' else p['response_rms']**2
    return [dict(id=k, Q=matrices[k], limit=limit, source=metadata,
        units='mean squared standardized PQ deviation' if kind=='pq' else 'mean squared action-response uncertainty / frozen score variance') for k in keys]


def rotations(case, controls):
    phase = np.arange(case.R)
    direction = np.where((np.arange(case.H)[:, None]//2+phase[None, :])%2 == 0, 1., -1.)
    pattern, raw = case.linear_optimum(direction, least_fee=True)
    full = prices.cost_breakdown(case, pattern)['total_yuan']
    receipts = {}
    for name, action in list(controls.items()):
        target = prices.cost_breakdown(case, action)['total_yuan']
        factor = min(1., target/full) if full > 0 else 0.
        controls[name+'_rotation'] = factor*pattern
        fee = prices.cost_breakdown(case, factor*pattern)['total_yuan']
        receipts[name] = dict(raw, target_total_yuan=target, full_pattern_total_yuan=full,
            rotation_total_yuan=fee, same_planned_fee=abs(fee-target)<=1e-7,
            scale=factor, scope='Same original candidate-independent pattern; budget constraints do not apply to the rotation control')
    return receipts


def execute(entry, p, directory, planning, case, forecast, controls, plans, matrices, metadata):
    """Open evaluation data after sealing; reuse old baseline evidence and streams."""
    ctx = old.context(entry)
    if ctx['score'] != planning['score'] or ctx['base'] != planning['base']:
        raise ValueError('Frozen planning and evaluation score differ')
    prior = Path(entry['prior_directory'])
    source_planner = arrays(io.read(entry['prior_seal']['path'])['planner']['path'])
    H, C, hist, sigma2 = ctx['H'], ctx['common_H'], ctx['history'], ctx['score']['sigma2']
    predicted = {name: old.pair_components(H,C,hist,sigma2,forecast+case.shift_pq_pu(u,ctx['base']))
        for name,u in controls.items()}
    zero_prediction = old.pair_components(H,C,hist,sigma2,forecast)
    all_rows = []
    for repeat in range(p['repeats']):
        rd = directory/f'execution{repeat:02d}'; rd.mkdir()
        source_rd = prior/f'execution{repeat:02d}'
        random_ref = io.ref(source_rd/'paired_randomness.npz')
        shutil.copyfile(random_ref['path'],rd/'paired_randomness.npz')
        random = arrays(random_ref['path'])
        baseline = arrays(source_rd/'no_dfs'/'pair_evidence.npz')
        zero_ob = arrays(source_rd/'no_dfs'/'observations.npz')
        zero_clean = np.asarray(zero_ob['true_root'])[:,None]**2-zero_ob['true_V']**2
        for name,u in controls.items():
            folder=rd/name; folder.mkdir()
            parent=name.removesuffix('_rotation'); quote=plans[parent]
            kind=quote['guard_kind']; mode=quote['mode']
            row=dict(id=entry['id'],case=entry['case'],start_date=entry['start_date'],
                original_id=entry['original_id'],index=entry['index'],forecast_index=0,repeat=repeat,
                strategy=name,parent_strategy=parent,mode=mode,guard_kind=kind,
                is_rotation=name.endswith('_rotation'),quote_zero_fallback=quote['zero_fallback'],
                quote_new_constraints_feasible=quote['new_gate_feasible'],
                paired_randomness_source=random_ref,baseline_source=io.ref(source_rd/'no_dfs'/'receipt.json'))
            try:
                delivered,delivery=case.execute(u,ctx['p']*ctx['base'],random['eta'])
                reused=next((s for s in panel.POLICIES if np.array_equal(u,source_planner[s])),None)
                if reused is not None:
                    source_folder=source_rd/reused; ob=arrays(source_folder/'observations.npz')
                    old_row=io.read(source_folder/'receipt.json')
                    if not np.array_equal(delivered,ob['delivered_action']):
                        raise ValueError('Cached delivery differs from paired response')
                    pp,qq,v,root=(ob[k] for k in ('true_P','true_Q','true_V','true_root'))
                    pm,qm,y,vm,rm=(ob[k] for k in ('P','Q','Y','V','root'))
                    ref=old_row['physical_state'];checks=old_row['AC']
                    vmin,vmax=old_row['all_bus_voltage_min'],old_row['all_bus_voltage_max']
                    outside=old_row['voltage_outside_095_105_cells']
                    row['reused_action_source']=io.ref(source_folder/'observations.npz')
                else:
                    pp,qq,state,checks,ref=old.physical_state(ctx,case,delivered,folder/'full_ac.npz')
                    v,root=state['terminal_voltage_pu'],ctx['root']
                    pm,qm,y,vm,rm=old.meter(pp,qq,v,root,[random[f'noise{i}'] for i in range(4)],
                        pq_sigma=entry['meter']['additional_PQ_relative_sd'],
                        voltage_sigma=entry['meter']['additional_V_relative_sd'])
                    vmin,vmax=float(state['voltage_magnitude_pu'].min()),float(state['voltage_magnitude_pu'].max())
                    outside=int(((state['voltage_magnitude_pu']<.95)|(state['voltage_magnitude_pu']>1.05)).sum())
                clean_y=root[:,None]**2-v**2
                obs=Observation(pm,qm,y,ctx['labels'],vm,rm)
                values,ll,new_sse=old.score_observation(ctx,obs)
                components=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],clean_y,obs,ctx['history_without_common'])
                change,summary=old.evidence_summary(ctx,baseline,components,zero_prediction,predicted[name],None,None)
                np.savez_compressed(folder/'pair_evidence.npz',**components)
                np.savez_compressed(folder/'pair_changes.npz',**change)
                np.savez_compressed(folder/'observations.npz',P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=root,requested_action=u,delivered_action=delivered,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,
                    representative_log_probability=ll)
                request_cost=prices.cost_breakdown(case,u);settled_cost=prices.cost_breakdown(case,u,delivered)
                amounts=case.amounts(delivered)
                empirical={g['matrix_key']:float(delivered.ravel()@matrices[g['matrix_key']]@delivered.ravel())
                    for g in metadata['guards']}
                # H is the scoring response; K is the row-scaled planning mean.
                actual_power=np.c_[pp,qq];base_power=np.c_[zero_ob['true_P'],zero_ob['true_Q']]
                increment_y=clean_y-zero_clean;delta_power=actual_power-base_power
                model_increment=np.einsum('tc,knc->ktn',delta_power,ctx['means'])
                actual_response_error=np.mean((increment_y[None]-model_increment)**2,axis=(1,2))/sigma2
                representative_H=np.array([r['H'] for r in ctx['score']['parameters']])
                H_increment=np.einsum('tc,knc->ktn',delta_power,representative_H)
                H_response_error=np.mean((increment_y[None]-H_increment)**2,axis=(1,2))/sigma2
                row.update(status='complete',**values,delivery=delivery,evidence=summary,
                    physical_state=ref,AC=checks,planned_cost=request_cost,settled_cost=settled_cost,
                    planned_fee_yuan=request_cost['total_yuan'],settled_fee_yuan=settled_cost['total_yuan'],
                    requested_quantities=case.amounts(u),delivered_quantities=amounts,
                    delivered_abs_kwh=amounts['active_abs_kwh'],delivered_abs_kvarh=amounts['reactive_abs_kvarh'],
                    all_bus_voltage_min=vmin,all_bus_voltage_max=vmax,voltage_outside_095_105_cells=outside,
                    delivered_budget_values=empirical,
                    actual_increment_response_error_H_over_sigma2_by_representative=H_response_error.tolist(),
                    actual_increment_response_error_K_over_sigma2_by_representative=actual_response_error.tolist())
            except Exception as exc:
                row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            all_rows.append(row);io.write(folder/'receipt.json',row)
        io.write(directory/'results.json',all_rows)
    return all_rows


def run_case(entry):
    p,out=old.WORK['p'],old.WORK['out']
    directory=out/'cases'/entry['id']
    if (directory/'complete.json').exists():
        return io.read(directory/'complete.json')
    if directory.exists():
        raise RuntimeError('Review unfinished case before reuse: '+str(directory))
    directory.mkdir(parents=True)
    began=perf_counter();io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=old.utc()))
    planning,forecast,case,seeds=context_planning(entry)
    io.write(directory/'resource_case.json',case.config)
    shutil.copyfile(entry['initial_points']['path'],directory/'shared_initial_points.npz')
    tick=perf_counter();matrices,metadata=build_geometry(entry,planning,case,forecast,p)
    geometry_seconds=perf_counter()-tick
    np.savez_compressed(directory/'guard_geometry.npz',**matrices)
    io.write(directory/'guard_geometry.json',dict(metadata,construction_seconds=geometry_seconds,
        arrays=io.ref(directory/'guard_geometry.npz'),train=entry['files']['train.npz'],score=entry['score']))
    controls,plans={},{}
    tick=perf_counter()
    models={m:DistributionEvidence(case,planning['score'],planning['val'],forecast,planning['base'],
        mode=m,target_mass=p['target_mass'],meter_relative_sd=entry['meter']['additional_V_relative_sd'],
        forecast_relative_sd=.05) for m in p['modes']}
    shared_models_seconds=perf_counter()-tick
    for mode in p['modes']:
        for kind in p['kinds']:
            name=mode+'__'+kind;tick=perf_counter()
            guards=get_guards(kind,matrices,io.ref(directory/'guard_geometry.json'),p)
            io.write(directory/(name+'_started.json'),dict(pid=os.getpid(),created_utc=old.utc(),
                allowed_seconds=p['solver_seconds'],guard_kind=kind,mode=mode))
            try:
                receipt,point=solver.plan(case,planning['score'],planning['val'],forecast,planning['base'],
                    guards=guards,mode=mode,seconds=max(.001,p['solver_seconds']-geometry_seconds),
                    env=old.WORK['env'],seed_points=seeds,target_mass=p['target_mass'],
                    record_directory=directory/(name+'_diagnostics'),
                    meter_relative_sd=entry['meter']['additional_V_relative_sd'],forecast_relative_sd=.05)
                if bool(receipt['feasible']) != (point is not None):
                    raise ValueError('Quote and action disagree')
                if point is not None and not (case.audit(point)['passed'] and models[mode].verify_expansion(point)['feasible']):
                    raise ValueError('Returned point fails original constraints')
            except Exception as exc:
                receipt,point=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc()),None
            action=np.zeros(case.shape) if point is None else point.copy()
            receipt.update(mode=mode,guard_kind=kind,new_gate_feasible=point is not None,
                zero_fallback=point is None,geometry_seconds=geometry_seconds,
                shared_prediction_models_seconds=shared_models_seconds,
                quote_wall_seconds=perf_counter()-tick,requested_quote_budget_seconds=p['solver_seconds'],
                initial_points=entry['initial_points'],
                executed_source='zero_fallback' if point is None else 'verified_guarded_quote',
                executed_probability_check=models[mode].verify_expansion(action),
                executed_guard_values={g['id']:float(action.ravel()@g['Q']@action.ravel()) for g in guards},
                executed_cost=prices.cost_breakdown(case,action))
            controls[name],plans[name]=action,receipt
            np.savez_compressed(directory/(name+'_action.npz'),action=action)
            io.write(directory/(name+'_plan.json'),receipt)
            print(f"{entry['id']} {name}: {receipt['status']}, source={receipt.get('source')}, fee={receipt.get('cost_yuan')}",flush=True)
    rotation=rotations(case,controls)
    io.write(directory/'rotation_receipt.json',rotation)
    np.savez_compressed(directory/'planner.npz',forecast=forecast,**controls)
    io.write(directory/'plans.json',plans)
    checks={mode:{name:models[mode].verify_expansion(u) for name,u in controls.items()} for mode in p['modes']}
    io.write(directory/'prediction_audits.json',checks)
    io.write(directory/'seal.json',dict(created_utc=old.utc(),planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'),score=entry['score'],train=entry['files']['train.npz'],
        validation=entry['files']['validation.npz'],shared_initial_points=io.ref(directory/'shared_initial_points.npz'),
        shared_initial_points_source=entry['initial_points'],
        protocol=io.ref(out/'protocol.json'),resource_case=io.ref(directory/'resource_case.json'),
        geometry=io.ref(directory/'guard_geometry.json'),prediction_audits=io.ref(directory/'prediction_audits.json'),
        rotation=io.ref(directory/'rotation_receipt.json'),planner_actual_voltage_or_network_inputs=False))
    rows=execute(entry,p,directory,planning,case,forecast,controls,plans,matrices,metadata)
    completion=dict(status='complete',id=entry['id'],rows=len(rows),
        failed_rows=sum(r['status']!='complete' for r in rows),results=io.ref(directory/'results.json'),
        quote_exceptions=sum(q['status']=='exception' for q in plans.values()),
        feasible_quotes=sum(q['feasible'] for q in plans.values()),zero_fallbacks=sum(q['zero_fallback'] for q in plans.values()),
        seconds=perf_counter()-began,geometry_seconds=geometry_seconds,
        shared_prediction_models_seconds=shared_models_seconds,plans=io.ref(directory/'plans.json'))
    io.write(directory/'complete.json',completion)
    return completion


def run(out):
    out=Path(out).resolve();p=io.read(out/'protocol.json')
    check_refs([*p['sources'],p['inputs']])
    if (out/'complete.json').exists():
        return io.read(out/'complete.json')
    if (out/'lock.json').exists():
        raise RuntimeError('Inspect running process and locks before resuming')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    receipts,failures=[],[]
    try:
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=old.initialize,initargs=(str(out),)) as pool:
            jobs={pool.submit(run_case,e):e for e in io.read(p['inputs']['path'])}
            for task in as_completed(jobs):
                entry=jobs[task]
                try:receipts.append(task.result())
                except Exception as exc:
                    failures.append(dict(id=entry['id'],status='task_failed',error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(receipts=receipts,failed_tasks=failures))
        result=dict(status='complete' if not failures else 'incomplete',receipts=receipts,failed_tasks=failures,
            rows=sum(r['rows'] for r in receipts),failed_rows=sum(r['failed_rows'] for r in receipts),
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),
            feasible_quotes=sum(r['feasible_quotes'] for r in receipts),zero_fallbacks=sum(r['zero_fallbacks'] for r in receipts),
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources']))
        io.write(out/'complete.json',result)
        return result
    finally:
        (out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=Path('outputs/flexibility_grid_execution_20261115_v1'))
    parser.add_argument('--source-audit',type=Path,default=Path('outputs/flexibility_grid_audit_20261115_v1/report.json'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--pq-rho',type=float,default=4.)
    parser.add_argument('--response-rms',type=float,default=.25)
    parser.add_argument('--seconds',type=float,default=600.)
    parser.add_argument('--parent-run',type=Path)
    parser.add_argument('--preflight-audit',type=Path)
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    a=parser.parse_args()
    if not a.resume:
        prepare(a.source_run,a.source_audit,a.out,smoke=a.smoke,pq_rho=a.pq_rho,
            response_rms=a.response_rms,seconds=a.seconds,parent_run=a.parent_run,modes=tuple(a.modes),
            preflight_audit=a.preflight_audit)
    if not a.prepare_only:
        print(run(a.out))


if __name__=='__main__':
    main()
