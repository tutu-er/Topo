"""Eight paired strategies on new, sealed histories; three unchanged QCQPs."""
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
from ...observations import Observation
from ..measured_pq_flex_20261114 import runner as old
from ..absolute_identification_20261114.commonseeds import build_portfolio
from ..reactive_flex_20261114.domain import PQIncrementCase
from ..flexibility_stability_20261114 import solver as prices
from ..flexibility_stability_20261114.run import safe_plan
from ..distribution_gate_20261114 import evidence, solver
from ..distribution_gate_20261114.run import planning_context, diagnostic_check
from .resources import build_resources

MODES = ('normal_fixed', 'mgf_fixed', 'mgf_forecast')
POLICIES = ('no_dfs', 'anchor', *MODES, *(m+'_rotation' for m in MODES))
SETTINGS = dict(development=dict(seed=2026111501, forecasts=2, repeats=16),
                confirmation=dict(seed=2026111502, forecasts=2, repeats=32))


def arrays(ref):
    with np.load(old.checked(ref), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def snapshot(out, entries):
    paths = {Path(__file__).resolve(), Path(__file__).with_name('PROTOCOL.md').resolve()}
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(old.ROOT) and 'outputs' not in path.parts:
                paths.add(path)
    for name in sorted({entry['case'] for entry in entries}):
        for filename in ('metadata.json', 'branches.csv', 'buses.csv'):
            paths.add(old.ROOT/'topology_identification_standalone'/'cases'/name/filename)
    for path in sorted(paths):
        dest = out/'source_snapshot'/path.relative_to(old.ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    return [io.ref(path) for path in sorted(paths)]


def prepare(args):
    out = Path(args.out).resolve()
    if out.exists():
        raise FileExistsError('Preserve previous panel outputs: '+str(out))
    source = io.read(args.index)
    phase = args.phase
    if any(e.get('phase') != phase for e in source['inputs']):
        raise ValueError('Input phase differs from requested phase')
    p = dict(SETTINGS[phase], phase=phase, validation_phase=phase, smoke=bool(args.smoke),
             created_utc=old.utc(), solver_seconds=600., anchor_seconds=150.,
             shared_seed_seconds=2., workers=4, target_mass=.99,
             policies=list(POLICIES), modes=list(MODES),
             reservation_prices=dict(P=.5, Q=.5),
             training_rows=192, validation_rows=96, new_rows=96,
             input_index=io.ref(args.index), source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
             environment=io.environment(),
             driver_relative_path=Path(__file__).resolve().relative_to(old.ROOT).as_posix(),
             forecast_relative_sd=.05,
             meter=dict(additional_PQ_relative_sd=.005, additional_V_relative_sd=.0005),
             response='Uniform(.9,1) multiplier per resource and whole day; original PV export clipping',
             scoring='Frozen full native bank; validation96 + new96 equal residual-cell MSE',
             cost='Activation objective; requested daily peak capacity plus delivered activation settlement',
             time_scope='600s includes each new evidence/model/seed check/solve; shared seed generation and150s anchor charged separately once',
             confidence_scope='Declared fixed Gaussian law conditional on historical calibrated mean; physical AC evaluated separately',
             physical_constraints='Original P linear contracts and independent Q offers; full-network voltage checked after sealing; thermal ratings unavailable',
             independent_unit='Date cluster; four networks share source traces, path repetitions are conditional',
             fallback='New quote without verified point uses verified original anchor, otherwise zero; preserve failure status',
             confirmation_source=None if args.development_audit is None else io.ref(args.development_audit))
    if phase == 'confirmation' and not args.smoke:
        if args.development_audit is None:
            raise ValueError('Confirmation requires a saved development audit and readiness receipt')
        readiness = io.read(args.development_audit)
        if not readiness.get('confirmation_ready', False):
            raise ValueError('Development readiness receipt does not authorize confirmation')
        for ref in readiness.get('checked_refs', []):
            old.checked(ref)
    if args.smoke:
        p.update(seed=2026111500, forecasts=1, repeats=1, solver_seconds=2.,
                 anchor_seconds=2., shared_seed_seconds=.5, workers=1)
    out.mkdir(parents=True)
    ledger = deepcopy(source.get('attempts', []))
    io.write(out/'qualification_ledger.json', ledger)
    p['qualification_ledger'] = io.ref(out/'qualification_ledger.json')
    entries, failures = [], []
    selected = source['inputs'][:1] if args.smoke else source['inputs']
    for original in selected:
        entry = deepcopy(original)
        entry.setdefault('original_id', entry['id'])
        entry.update(resource_variant='history_registered_q95_focus', physical_contract_fraction=.1,
                     phase=phase, date_cluster=entry['start_date'])
        directory = out/'histories'/entry['id']
        directory.mkdir(parents=True)
        try:
            for ref in (entry['score'], entry['native_copy'], entry['resource_context'],
                        entry['files']['train.npz'], entry['files']['validation.npz']):
                old.checked(ref)
            context = io.read(entry['resource_context']['path'])
            score = io.read(entry['score']['path'])
            train = arrays(entry['files']['train.npz'])
            # Public normalization base, not true impedances, determines units.
            base = float(entry['base_kw'])
            if not np.isfinite(base) or base <= 0:
                raise ValueError('Explicit positive normalization base required')
            resource = build_resources(entry, score, context, train, base)
            io.write(directory/'resource_spec.json', resource)
            if resource.get('status') != 'ready':
                raise ValueError('Resource preparation unresolved: '+str(resource.get('problems',resource.get('reason'))))
            entry.update(base_kw=base, p_resources=resource['p_resources'], q_resources=resource['q_resources'],
                         resource_spec=io.ref(directory/'resource_spec.json'))
            # Evaluation-side forecast adapter reads only future P/Q; optimizer gets the resulting forecast.
            with np.load(old.checked(entry['physical_cache']['original']), allow_pickle=False) as z:
                true_power = np.c_[z['P'][288:384], z['Q'][288:384]]
            forecasts = {f'forecast{i:02d}': old.forecast_power(true_power, p['seed'], entry['index'], i)
                         for i in range(p['forecasts'])}
            np.savez_compressed(directory/'forecasts.npz', **forecasts)
            entry['forecast_ref'] = io.ref(directory/'forecasts.npz')
            io.write(directory/'forecast_receipt.json', dict(seed=p['seed'], index=entry['index'],
                physical_power_source=entry['physical_cache']['original'], forecast_ref=entry['forecast_ref'],
                method='true P/Q times independent(1+.05Z); future voltage/impedance not used',
                created_utc=old.utc()))
            entries.append(entry)
        except Exception as exc:
            failure = dict(id=entry['id'], case=entry['case'], start_date=entry['start_date'],
                           stage='resource_or_forecast_preparation', status='failed',
                           error=repr(exc), traceback=traceback.format_exc())
            failures.append(failure)
            io.write(directory/'preparation_failure.json', failure)
    io.write(out/'inputs.json', entries)
    io.write(out/'preparation_failures.json', failures)
    p.update(inputs=io.ref(out/'inputs.json'), preparation_failures=io.ref(out/'preparation_failures.json'),
             original_histories=len(selected), runnable_histories=len(entries),
             planned_histories=len(ledger) if not args.smoke else 1,
             expected_rows=len(entries)*p['forecasts']*p['repeats']*len(POLICIES),
             planned_rows=(len(ledger) if not args.smoke else 1)*p['forecasts']*p['repeats']*len(POLICIES))
    p['sources'] = snapshot(out, entries)
    p.update(old._git_receipt([Path(r['path']) for r in p['sources']]))
    io.write(out/'protocol.json', p)
    return p


def rotation_actions(case, controls):
    """One candidate-independent LP pattern; three homogeneous total-bill matches."""
    began = perf_counter()
    phase = np.arange(case.R)
    direction = np.where((np.arange(case.H)[:, None]//2+phase[None, :]) % 2 == 0, 1., -1.)
    pattern, raw = case.linear_optimum(direction, least_fee=True)
    full = prices.cost_breakdown(case, pattern)['total_yuan']
    receipts = {}
    for mode in MODES:
        target = prices.cost_breakdown(case, controls[mode])['total_yuan']
        factor = min(1., target/full) if full > 0 else 0.
        controls[mode+'_rotation'] = factor*pattern
        actual = prices.cost_breakdown(case, controls[mode+'_rotation'])['total_yuan']
        receipts[mode] = dict(raw, target_total_yuan=target, full_pattern_total_yuan=full,
            rotation_total_yuan=actual, same_planned_fee=abs(actual-target) <= 1e-7,
            seconds=perf_counter()-began, scale=factor, target_policy=mode,
            scope='Candidate-independent fixed pattern; cap at full feasible pattern, report budget mismatch')
    return receipts


def run_forecast(task):
    entry, draw = task
    p, out = old.WORK['p'], old.WORK['out']
    directory = out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():
        return io.read(directory/'complete.json')
    if directory.exists():
        raise RuntimeError('Inspect unfinished task before reuse: '+str(directory))
    directory.mkdir(parents=True)
    began = perf_counter()
    io.write(directory/'started.json', dict(pid=os.getpid(), created_utc=old.utc()))
    shared_began = perf_counter()
    planning = planning_context(entry, entry['base_kw'])
    meter_config = entry.get('meter', p.get('meter', {}))
    distribution_config = dict(
        meter_relative_sd=float(meter_config.get('additional_V_relative_sd', .0005)),
        forecast_relative_sd=float(entry.get('forecast_relative_sd', p.get('forecast_relative_sd', .05))))
    prediction = arrays(entry['forecast_ref'])[f'forecast{draw:02d}']
    case = PQIncrementCase(planning['resource'], planning['labels'], prediction, planning['base'],
                           q_resources=entry['q_resources'], fraction=.1)
    if case.resources[:case.P_R] != entry['p_resources']:
        raise ArithmeticError('Automatically created P contracts differ from sealed historical catalogue')
    io.write(directory/'resource_case.json', case.config)
    try:
        seeds, seed_receipt = build_portfolio(case, planning['score'], prediction, planning['base'],
            seconds=p['shared_seed_seconds'], env=old.WORK['env'])
    except Exception as exc:
        seeds = {'zero': np.zeros(case.shape)}
        seed_receipt = dict(status='generation_failed_zero_pool_retained', error=repr(exc), traceback=traceback.format_exc())
    np.savez_compressed(directory/'generated_initial_points.npz', **seeds)
    anchor_receipt, anchor = safe_plan(case, planning, prediction, seeds, 'activation',
                                      p['anchor_seconds'], directory/'anchor_diagnostics')
    if anchor is not None:
        seeds['shared_150_second_anchor'] = anchor.copy()
    shared_seconds = perf_counter()-shared_began
    np.savez_compressed(directory/'shared_initial_points.npz', **seeds)
    io.write(directory/'anchor_plan.json', anchor_receipt)
    io.write(directory/'anchor.json', dict(feasible=anchor is not None, shared_seconds=shared_seconds,
        source='current_forecast_150s_anchor', receipt=io.ref(directory/'anchor_plan.json')))
    seed_receipt.update(shared_preparation_seconds=shared_seconds, points_after_anchor=len(seeds),
        generated_points=io.ref(directory/'generated_initial_points.npz'),
        initial_points=io.ref(directory/'shared_initial_points.npz'),
        anchor_receipt=io.ref(directory/'anchor_plan.json'), physical_or_truth_inputs=False)
    io.write(directory/'shared_initial_receipt.json', seed_receipt)
    plans, models = {}, {}
    controls = dict(no_dfs=np.zeros(case.shape), anchor=np.zeros(case.shape) if anchor is None else anchor.copy())
    for mode in MODES:
        tick = perf_counter()
        model = point = None
        try:
            model = evidence.DistributionEvidence(case, planning['score'], planning['val'], prediction,
                                                   planning['base'], mode=mode, target_mass=.99, **distribution_config)
            receipt, point = solver.plan(case, planning['score'], planning['val'], prediction, planning['base'],
                mode=mode, seconds=max(.001, p['solver_seconds']-(perf_counter()-tick)),
                env=old.WORK['env'], seed_points={k:v.copy() for k,v in seeds.items()},
                target_mass=.99, record_directory=directory/(mode+'_diagnostics'), **distribution_config)
            if bool(receipt['feasible']) != (point is not None):
                raise ArithmeticError('Quote/point feasibility mismatch')
            if point is not None and not (case.audit(point)['passed'] and model.verify_expansion(point)['feasible']):
                raise ArithmeticError('Returned quote fails original resource/probability checks')
        except Exception as exc:
            receipt, point = dict(feasible=False, status='exception', error=repr(exc),
                traceback=traceback.format_exc(), cost_yuan=None), None
        accepted = point is not None
        action = point.copy() if accepted else controls['anchor'].copy()
        executed = diagnostic_check(model, action)
        elapsed = perf_counter()-tick
        receipt.update(new_gate_feasible=accepted, new_constraints_feasible=accepted,
            anchor_fallback=not accepted and anchor is not None,
            zero_fallback=not accepted and anchor is None,
            executed_source='accepted_new_quote' if accepted else 'validated_anchor_fallback' if anchor is not None else 'zero_fallback',
            executed_gate_feasible=bool(executed['feasible']), executed_gate_check=executed,
            executed_cost=prices.cost_breakdown(case, action), quote_wall_seconds=elapsed,
            requested_quote_budget_seconds=p['solver_seconds'], quote_budget_exceeded=elapsed > p['solver_seconds']+.05,
            shared_preparation_seconds=shared_seconds, charged_total_seconds=shared_seconds+elapsed,
            initial_points=io.ref(directory/'shared_initial_points.npz'),
            feasibility_scope='Fallback preserves failed new-constraint status')
        plans[mode], models[mode], controls[mode] = receipt, model, action
        io.write(directory/(mode+'_plan.json'), receipt)
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
        planner_actual_voltage_or_network_inputs=False))
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
    io.write(directory/'complete.json', completion)
    return completion


def execute(entry, draw, p, directory, ctx, case, prediction, controls, plans, anchor_receipt):
    H, C, hist, sigma2 = ctx['H'], ctx['common_H'], ctx['history'], ctx['score']['sigma2']
    meter_config = entry.get('meter', p.get('meter', {}))
    pq_sigma = float(meter_config.get('additional_PQ_relative_sd', .005))
    voltage_sigma = float(meter_config.get('additional_V_relative_sd', .0005))
    predicted = {name:old.pair_components(H, C, hist, sigma2, prediction+case.shift_pq_pu(u, ctx['base']))
                 for name,u in controls.items()}
    exact, rows = {}, []
    for name,u in controls.items():
        folder = directory/'exact_requested'/name
        folder.mkdir(parents=True)
        try:
            pp, qq, state, checks, ref = old.physical_state(ctx, case, u, folder/'full_ac.npz')
            v = state['terminal_voltage_pu']; y = ctx['root'][:, None]**2-v**2
            obs = Observation(pp, qq, y, ctx['labels'], v, ctx['root'])
            exact[name] = old.pair_components(H, C, hist, sigma2, np.c_[pp, qq], y, obs, ctx['history_without_common'])
            _, availability = case.execute(u, ctx['p']*ctx['base'], np.ones(case.R))
            np.savez_compressed(folder/'pair_evidence.npz', **exact[name])
            np.savez_compressed(folder/'forecast_pair_evidence.npz', **predicted[name])
            np.savez_compressed(folder/'clean_observation.npz', P=pp,Q=qq,V=v,Y=y,root=ctx['root'],labels=ctx['labels'])
            io.write(folder/'receipt.json', dict(status='complete',physical_state=ref,checks=checks,
                PV_net_export_proxy_excess_kwh=availability['PV_clipped_kwh'],
                interpretation='Post-seal ideal requested action diagnostic; actual delivery evaluated separately'))
        except Exception as exc:
            io.write(folder/'receipt.json', dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
    for repeat in range(p['repeats']):
        rd = directory/f'execution{repeat:02d}'
        rd.mkdir()
        eta = np.r_[old.rng(p['seed'],entry['index'],draw,773,repeat).uniform(.9,1.,case.P_R),
                    old.rng(p['seed'],entry['index'],draw,774,repeat).uniform(.9,1.,case.Q_R)]
        noise = [old.rng(p['seed'],entry['index'],draw,k,repeat).normal(size=shape)
                 for k,shape in ((2,ctx['p'].shape),(3,ctx['q'].shape),(4,ctx['p'].shape),(5,ctx['root'].shape))]
        np.savez_compressed(rd/'paired_randomness.npz', eta=eta, **{f'noise{i}':x for i,x in enumerate(noise)})
        baseline = None
        for name,u in controls.items():
            folder = rd/name; folder.mkdir()
            mode = name.removesuffix('_rotation') if name.removesuffix('_rotation') in MODES else None
            quote = plans.get(mode, anchor_receipt if name == 'anchor' else {})
            row = dict(id=entry['id'], original_id=entry['original_id'], resource_variant=entry['resource_variant'],
                case=entry['case'], start_date=entry['start_date'], date_cluster=entry['date_cluster'],
                forecast_index=draw, repeat=repeat, contract_fraction=.1, strategy=name,
                quote_zero_fallback=quote.get('zero_fallback',False),
                quote_anchor_fallback=quote.get('anchor_fallback',False),
                quote_new_constraints_feasible=plans[name]['new_gate_feasible'] if name in MODES else None,
                historical_p_map=ctx['score']['p_map'], historical_native_index=ctx['score']['native_index'])
            try:
                delivered, delivery = case.execute(u, ctx['p']*ctx['base'], eta)
                pp, qq, state, checks, ref = old.physical_state(ctx, case, delivered, folder/'full_ac.npz')
                v = state['terminal_voltage_pu']; clean_y = ctx['root'][:,None]**2-v**2
                pm,qm,y,vm,rm = old.meter(pp,qq,v,ctx['root'],noise,pq_sigma=pq_sigma,voltage_sigma=voltage_sigma)
                obs = Observation(pm,qm,y,ctx['labels'],vm,rm)
                values,ll,new_sse = old.score_observation(ctx,obs)
                components = old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],clean_y,obs,ctx['history_without_common'])
                if name == 'no_dfs':
                    baseline = components
                if baseline is None:
                    change,summary = {},dict(status='unavailable',reason='Paired zero execution failed')
                else:
                    change,summary = old.evidence_summary(ctx,baseline,components,predicted['no_dfs'],predicted[name],exact.get('no_dfs'),exact.get(name))
                np.savez_compressed(folder/'pair_evidence.npz', **components)
                np.savez_compressed(folder/'pair_changes.npz', **change)
                np.savez_compressed(folder/'observations.npz',P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=ctx['root'],requested_action=u,delivered_action=delivered,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,representative_log_probability=ll)
                requested_cost = prices.cost_breakdown(case,u)
                settled_cost = prices.cost_breakdown(case,u,delivered)
                requested,actual = case.amounts(u),case.amounts(delivered)
                row.update(status='complete',**values,delivery=delivery,evidence=summary,physical_state=ref,AC=checks,
                    planned_fee_yuan=requested_cost['total_yuan'],settled_fee_yuan=settled_cost['total_yuan'],
                    planned_cost=requested_cost,settled_cost=settled_cost,requested_quantities=requested,delivered_quantities=actual,
                    delivered_abs_kwh=actual['active_abs_kwh'],delivered_abs_kvarh=actual['reactive_abs_kvarh'],
                    baseline_root_energy_value_yuan=old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    root_energy_value_change_yuan=old.grid_value(case,state['root_supply_kw'])-old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    all_bus_voltage_min=float(state['voltage_magnitude_pu'].min()),
                    all_bus_voltage_max=float(state['voltage_magnitude_pu'].max()),
                    voltage_outside_095_105_cells=int(((state['voltage_magnitude_pu'] < .95)|(state['voltage_magnitude_pu'] > 1.05)).sum()))
            except Exception as exc:
                row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(row)
            io.write(folder/'receipt.json',row)
        io.write(directory/'results.json',rows)
    return rows


def run(out):
    out = Path(out).resolve(); p = io.read(out/'protocol.json')
    for ref in [*p['sources'], p['inputs'], p['qualification_ledger'], p['preparation_failures']]:
        old.checked(ref)
    if (out/'complete.json').exists():
        return io.read(out/'complete_summary.json')
    if (out/'lock.json').exists():
        raise RuntimeError('Inspect actual process before removing a stale lock')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    try:
        entries = io.read(out/'inputs.json')
        tasks = [(entry,i) for entry in entries for i in range(p['forecasts'])]
        receipts,failures = [],[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=old.initialize,initargs=(str(out),)) as pool:
            pending = {pool.submit(run_forecast,task):task for task in tasks}
            for future in as_completed(pending):
                entry,draw = pending[future]
                try:
                    receipts.append(future.result())
                except Exception as exc:
                    failures.append(dict(id=entry['id'],case=entry['case'],start_date=entry['start_date'],
                        forecast_index=draw,status='task_failed',error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(total=len(tasks),completed=len(receipts),failed_tasks=failures,receipts=receipts))
                print(f'{len(receipts)+len(failures)}/{len(tasks)} tasks; {len(failures)} failed',flush=True)
        result = dict(total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,
            preparation_failed_histories=len(io.read(out/'preparation_failures.json')),
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),
            execution_failed_rows=sum(r['failed_rows'] for r in receipts),
            feasible_quotes_by_mode={m:sum(r['new_gate_feasible'][m] for r in receipts) for m in MODES},
            anchor_fallbacks=sum(r['anchor_fallbacks'] for r in receipts),
            zero_fallbacks=sum(r['zero_fallbacks'] for r in receipts),receipts=receipts)
        io.write(out/'complete_summary.json',result)
        io.write(out/'complete.json',dict(status='complete',main=io.ref(out/'complete_summary.json'),
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources'])))
        return result
    finally:
        (out/'lock.json').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=SETTINGS,default='development')
    parser.add_argument('--development-audit',type=Path)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if not args.resume:
        if args.index is None:
            parser.error('--index required for preparation')
        prepare(args)
    else:
        frozen = io.read(args.out/'protocol.json')
        if frozen['phase']!=args.phase or frozen['smoke']!=args.smoke:
            raise ValueError('Resume phase/smoke differs from frozen protocol')
    if not args.prepare_only:
        result = run(args.out)
        print({k:v for k,v in result.items() if k!='receipts'})


if __name__ == '__main__':
    main()
