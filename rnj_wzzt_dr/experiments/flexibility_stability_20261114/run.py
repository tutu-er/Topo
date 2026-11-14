"""Sealed, paired P/Q availability-price and action-stability experiment."""
from __future__ import annotations

import os
for _variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_variable] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import shutil
from time import perf_counter
import traceback

import numpy as np

from ... import io
from ...observations import Observation
from ..measured_pq_flex_20261114 import runner as old
from ..absolute_identification_20261114.commonseeds import build_portfolio
from ..absolute_identification_20261114.solver import SymmetricRecovery
from . import domain, solver

POLICIES = ('no_dfs', 'anchor', 'activation', 'stable_activation', 'capacity', 'capacity_rotation')
VARIANTS = ('original_p10_q36', 'expanded_p10_q36')
SETTINGS = dict(development=dict(seed=2026111415, repeats=32),
                confirmation=dict(seed=2026111416, repeats=64))


def prepare(args):
    setting = SETTINGS[args.phase]
    spec = io.read(args.resources)
    out = args.out.resolve()
    p = old.prepare(args.index, args.ac_index, out, seed=setting['seed'],
        forecasts=4, repeats=setting['repeats'], seconds=300., workers=4)
    originals = io.read(out/'inputs.json'); entries = []
    for original in originals:
        offered = spec['configurations_by_input'][original['id']]
        if set(offered) != set(VARIANTS):
            raise ValueError('Resource variants differ from the locked protocol')
        for variant in VARIANTS:
            config = offered[variant]; entry = deepcopy(original)
            entry.update(original_id=original['id'], id=original['id']+'__'+variant,
                resource_variant=variant, p_resources=config['p_resources'], q_resources=config['q_resources'],
                physical_contract_fraction=config['physical_contract_fraction'])
            entries.append(entry)
    if len(originals) != 1 or len(entries) != 2:
        raise ValueError('This locked experiment contains one history and two resource variants')
    io.write(out/'inputs.json', entries); io.write(out/'placements.json', spec)
    p.update(phase=args.phase, validation_phase=args.phase, modes=['activation', 'capacity'],
        policies=list(POLICIES), placement_variants=list(VARIANTS), original_histories=1,
        inputs=io.ref(out/'inputs.json'), placement=io.ref(out/'placements.json'),
        placement_source=io.ref(args.resources), source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        anchor_seconds=150., shared_seed_seconds=2., stable_fee_threshold_yuan=.01,
        reservation_prices=dict(P=.5, Q=.5, P_unit='yuan/(kW service-day)', Q_unit='yuan/(kvar service-day)',
            convention='One-sided maximum adjustment rating; signed Q interval [-s,s] is charged once'),
        maximum_resource_capability='Offers, not already purchased reserves; each policy purchases its own requested peak',
        settlement='Requested peak availability payment plus delivered activation payment, for every policy',
        objective_scope=dict(anchor='activation only', activation='activation only',
            stable_activation='activation only, with explicit .01-yuan acceptance rule', capacity='activation plus purchased capacity'),
        planning_input_boundary='Current noisy forecast, historical observations, frozen candidate H and device offers only',
        seed_source='Current-forecast LP portfolio followed by a shared 150-second activation solve; no previous quotes or executions',
        source_quote_inputs=[], historical_refits=0,
        expansion=dict(automatic_only_for_full_16x4=False),
        interpretation='New forecast and response/meter streams on the same known physical day; resource variants share worlds',
        response='Original seven P devices retain prefix of stream773; seven added P append; Q stream774 and all measurement streams unchanged',
        confirmation_rule='All policy, price, resource and budget settings locked before development; confirmation has independent streams',
        expected_rows=len(entries)*4*setting['repeats']*len(POLICIES),
        timing='Each refined method reports shared portfolio/anchor cost plus its own 300-second call; shared work executed only once')
    known = {Path(r['path']).resolve() for r in p['sources']}
    for path in (Path(__file__), Path(domain.__file__), Path(solver.__file__), Path(__file__).with_name('resources.py')):
        path = path.resolve()
        if path not in known:
            p['sources'].append(io.ref(path)); known.add(path)
            destination = out/'source_snapshot'/path.relative_to(old.ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    p.update(old._git_receipt(sorted(known)))
    io.write(out/'protocol.json', p)
    return p


def safe_plan(case, ctx, prediction, points, mode, seconds, directory):
    began = perf_counter()
    try:
        receipt, point = solver.plan(case, ctx['score'], ctx['val'], prediction, ctx['base'],
            mode=mode, seconds=seconds, env=old.WORK['env'], seed_points=points,
            record_directory=directory)
        if bool(receipt['feasible']) != (point is not None):
            raise ArithmeticError('Planning feasibility flag differs from returned point')
        if point is not None and not case.audit(point)['passed']:
            raise ArithmeticError('Returned action violates the physical resource domain')
    except Exception as exc:
        receipt, point = dict(feasible=False, status='exception', error=repr(exc),
            traceback=traceback.format_exc(), objective_kind=mode, cost_yuan=None), None
    elapsed = perf_counter()-began
    receipt.update(quote_wall_seconds=elapsed, requested_quote_budget_seconds=seconds,
        quote_budget_exceeded=elapsed>seconds+.05, zero_fallback=point is None)
    if point is not None:
        receipt.update(planned_cost=solver.cost_breakdown(case, point),
            requested_reserve=np.max(np.abs(point), axis=0).tolist())
    return receipt, point


def run_forecast(task):
    entry, draw = task; p = old.WORK['p']; out = old.WORK['out']
    directory = out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():
        return io.read(directory/'complete.json')
    if directory.exists():
        raise RuntimeError('Inspect unfinished task before reuse: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json', dict(pid=os.getpid(), created_utc=old.utc()))
    began = perf_counter(); ctx = old.context(entry); score = ctx['score']
    planning = {key:ctx[key] for key in ('score', 'val', 'base')}
    prediction = old.forecast_power(np.c_[ctx['p'], ctx['q']], p['seed'], entry['index'], draw)
    shared_began = perf_counter()
    case = domain.create_case(entry, ctx, prediction)
    io.write(directory/'resource_case.json', case.config)
    seeds, seed_receipt = build_portfolio(case, score, prediction, ctx['base'],
        seconds=p['shared_seed_seconds'], env=old.WORK['env'])
    np.savez_compressed(directory/'generated_initial_points.npz', **seeds)
    anchor_receipt, anchor = safe_plan(case, planning, prediction, seeds, 'activation',
        max(.001, p['anchor_seconds']-(perf_counter()-shared_began)), directory/'anchor_diagnostics')
    shared_seconds = perf_counter()-shared_began
    if anchor is not None:
        seeds['shared_150_second_anchor'] = anchor.copy()
    np.savez_compressed(directory/'shared_initial_points.npz', **seeds)
    io.write(directory/'anchor_plan.json', anchor_receipt)
    io.write(directory/'anchor.json', dict(feasible=anchor is not None,
        source='shared_150_second_current_forecast_activation_plan',
        activation_yuan=None if anchor is None else case.fee(anchor),
        receipt=io.ref(directory/'anchor_plan.json'), shared_seconds=shared_seconds))
    seed_receipt.update(shared_preparation_seconds=shared_seconds, anchor_receipt=io.ref(directory/'anchor_plan.json'),
        generated_points=io.ref(directory/'generated_initial_points.npz'), points_after_anchor=len(seeds),
        evidence_feasibility='Each solver rechecks every shared point; original portfolio only promises resource feasibility',
        time_scope='Current domain construction, current-forecast portfolio and 150-second anchor stage; charged to each method')
    io.write(directory/'shared_initial_receipt.json', seed_receipt)
    controls = {'no_dfs':np.zeros(case.shape), 'anchor':np.zeros(case.shape) if anchor is None else anchor}
    plans = {}
    for mode in ('activation', 'capacity'):
        receipt, point = safe_plan(case, planning, prediction, seeds, mode, p['solver_seconds'], directory/(mode+'_diagnostics'))
        receipt.update(shared_preparation_seconds=shared_seconds,
            charged_total_seconds=shared_seconds+receipt['quote_wall_seconds'],
            initial_points=io.ref(directory/'shared_initial_points.npz'))
        controls[mode] = np.zeros(case.shape) if point is None else point
        plans[mode] = receipt
    refined = None if plans['activation']['zero_fallback'] else controls['activation']
    saving = None if anchor is None or refined is None else case.fee(anchor)-case.fee(refined)
    retain = anchor is not None and (refined is None or saving < p['stable_fee_threshold_yuan'])
    stable = anchor.copy() if retain else (None if refined is None else refined.copy())
    controls['stable_activation'] = np.zeros(case.shape) if stable is None else stable
    evidence = SymmetricRecovery(case, score, ctx['val'], prediction, ctx['base'], 'symmetric_guard', .99)
    stable_direct = None if stable is None else evidence.verify_expansion(stable)
    if stable is not None and (not stable_direct['feasible'] or not case.audit(stable)['passed']):
        raise ArithmeticError('Stable decision is not original-form feasible')
    lower = plans['activation'].get('lower_bound_yuan')
    stable_fee = None if stable is None else case.fee(stable)
    if lower is not None and stable_fee is not None and lower>stable_fee+1e-7:
        raise ArithmeticError('Refinement bound conflicts with the stable feasible action')
    plans['stable_activation'] = dict(feasible=stable is not None, zero_fallback=stable is None,
        status='feasible' if stable is not None else 'unresolved', objective_kind='activation',
        source='anchor_retained' if retain else 'refinement_selected', retained_anchor=retain,
        anchor_available=anchor is not None, anchor=io.ref(directory/'anchor.json'),
        saving_over_anchor_yuan=saving, threshold_yuan=p['stable_fee_threshold_yuan'],
        cost_yuan=stable_fee, lower_bound_yuan=lower,
        relative_gap=None if lower is None or stable_fee is None else (stable_fee-lower)/max(stable_fee,1e-12),
        global_optimality_verified=False, final=stable_direct,
        planned_cost=solver.cost_breakdown(case, controls['stable_activation']),
        charged_total_seconds=plans['activation']['charged_total_seconds'], additional_optimization=False,
        bound_scope='Inherited valid activation-objective lower bound; total billed costs are reported separately')
    pattern_began = perf_counter()
    pattern, pattern_receipt = case.linear_optimum(case.fixed_rotation_direction(), least_fee=True)
    pattern_total = solver.cost_breakdown(case, pattern)['total_yuan']
    capacity_total = solver.cost_breakdown(case, controls['capacity'])['total_yuan']
    if pattern_total <= 0:
        raise ArithmeticError('Candidate-independent rotation has no payable capability')
    controls['capacity_rotation'] = min(1., capacity_total/pattern_total)*pattern
    pattern_receipt.update(seconds=perf_counter()-pattern_began, full_pattern_total_yuan=pattern_total,
        rotation_total_yuan=solver.cost_breakdown(case, controls['capacity_rotation'])['total_yuan'],
        target_total_yuan=capacity_total, same_planned_fee=abs(solver.cost_breakdown(case, controls['capacity_rotation'])['total_yuan']-capacity_total)<=1e-7,
        scope='Candidate-independent physical-resource LP, outside both procurement budgets; fixed original Q rotation identity')
    io.write(directory/'rotation_receipt.json', pattern_receipt)
    controls = {name:controls[name] for name in POLICIES}
    np.savez_compressed(directory/'planner.npz', forecast=prediction, **controls)
    np.savez_compressed(directory/'reservation.npz', **{name:np.max(np.abs(u),axis=0) for name,u in controls.items()})
    io.write(directory/'plans.json', plans)
    io.write(directory/'seal.json', dict(created_utc=old.utc(), planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'), reservation=io.ref(directory/'reservation.npz'),
        anchor=io.ref(directory/'anchor_plan.json'), score=entry['score'], protocol=io.ref(out/'protocol.json'),
        planner_actual_voltage_or_network_inputs=False))
    H,C,hist,sigma2 = ctx['H'],ctx['common_H'],ctx['history'],score['sigma2']
    predicted = {name:old.pair_components(H,C,hist,sigma2,prediction+case.shift_pq_pu(u,ctx['base'])) for name,u in controls.items()}
    exact = {}; rows = []
    for name,u in controls.items():
        folder = directory/'exact_requested'/name; folder.mkdir(parents=True)
        try:
            pp,qq,state,checks,ref = old.physical_state(ctx,case,u,folder/'full_ac.npz')
            v = state['terminal_voltage_pu']; y = ctx['root'][:,None]**2-v**2
            obs = Observation(pp,qq,y,ctx['labels'],v,ctx['root'])
            exact[name] = old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],y,obs,ctx['history_without_common'])
            _,availability = case.execute(u,ctx['p']*ctx['base'],np.ones(case.R))
            np.savez_compressed(folder/'pair_evidence.npz', **exact[name])
            np.savez_compressed(folder/'forecast_pair_evidence.npz', **predicted[name])
            np.savez_compressed(folder/'clean_observation.npz', P=pp,Q=qq,V=v,Y=y,root=ctx['root'],labels=ctx['labels'])
            io.write(folder/'receipt.json', dict(status='complete', physical_state=ref, checks=checks,
                PV_net_export_proxy_excess_kwh=availability['PV_clipped_kwh'],
                interpretation='Ideal delivery diagnostic after sealing: eta=1 and no actual-PV clipping'))
        except Exception as exc:
            io.write(folder/'receipt.json', dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
    for repeat in range(p['repeats']):
        rd = directory/f'execution{repeat:02d}'; rd.mkdir()
        eta = np.r_[old.rng(p['seed'],entry['index'],draw,773,repeat).uniform(.9,1.,case.P_R),
                    old.rng(p['seed'],entry['index'],draw,774,repeat).uniform(.9,1.,case.Q_R)]
        noise = [old.rng(p['seed'],entry['index'],draw,k,repeat).normal(size=shape) for k,shape in
            ((2,ctx['p'].shape),(3,ctx['q'].shape),(4,ctx['p'].shape),(5,ctx['root'].shape))]
        np.savez_compressed(rd/'paired_randomness.npz', eta=eta, **{f'noise{i}':x for i,x in enumerate(noise)})
        baseline = None
        for name,u in controls.items():
            folder = rd/name; folder.mkdir()
            zero_fallback = (anchor_receipt['zero_fallback'] if name=='anchor' else
                plans[name.removesuffix('_rotation')]['zero_fallback'] if name!='no_dfs' else False)
            row = dict(id=entry['id'],original_id=entry['original_id'],resource_variant=entry['resource_variant'],
                case=entry['case'],start_date=entry['start_date'],forecast_index=draw,repeat=repeat,
                contract_fraction=.1,strategy=name,quote_zero_fallback=zero_fallback,
                historical_p_map=score['p_map'],historical_native_index=score['native_index'])
            try:
                action,delivery = case.execute(u,ctx['p']*ctx['base'],eta)
                pp,qq,state,checks,ref = old.physical_state(ctx,case,action,folder/'full_ac.npz')
                v = state['terminal_voltage_pu']; clean_y = ctx['root'][:,None]**2-v**2
                pm,qm,y,vm,rm = old.meter(pp,qq,v,ctx['root'],noise)
                obs = Observation(pm,qm,y,ctx['labels'],vm,rm)
                values,ll,new_sse = old.score_observation(ctx,obs)
                components = old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],clean_y,obs,ctx['history_without_common'])
                if name=='no_dfs': baseline=components
                if baseline is None:
                    change,summary = {},dict(status='unavailable',reason='Paired zero execution failed')
                else:
                    change,summary = old.evidence_summary(ctx,baseline,components,predicted['no_dfs'],predicted[name],
                        exact.get('no_dfs'),exact.get(name))
                np.savez_compressed(folder/'pair_evidence.npz', **components)
                np.savez_compressed(folder/'pair_changes.npz', **change)
                np.savez_compressed(folder/'observations.npz', P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=ctx['root'],requested_action=u,delivered_action=action,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,
                    representative_log_probability=ll)
                requested_cost = solver.cost_breakdown(case,u)
                settled_cost = solver.cost_breakdown(case,u,delivered=action)
                requested,delivered = case.amounts(u),case.amounts(action)
                row.update(status='complete',**values,delivery=delivery,evidence=summary,physical_state=ref,AC=checks,
                    planned_fee_yuan=requested_cost['total_yuan'],settled_fee_yuan=settled_cost['total_yuan'],
                    planned_cost=requested_cost,settled_cost=settled_cost,
                    requested_quantities=requested,delivered_quantities=delivered,
                    delivered_abs_kwh=delivered['active_abs_kwh'],delivered_abs_kvarh=delivered['reactive_abs_kvarh'],
                    baseline_root_energy_value_yuan=old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    root_energy_value_change_yuan=old.grid_value(case,state['root_supply_kw'])-old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    all_bus_voltage_min=float(state['voltage_magnitude_pu'].min()),
                    all_bus_voltage_max=float(state['voltage_magnitude_pu'].max()),
                    voltage_outside_095_105_cells=int(((state['voltage_magnitude_pu']<.95)|(state['voltage_magnitude_pu']>1.05)).sum()))
            except Exception as exc:
                row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(row);io.write(folder/'receipt.json',row)
        io.write(directory/'results.json',rows)
    receipt = dict(status='complete',id=entry['id'],forecast_index=draw,rows=len(rows),
        failed_rows=sum(r['status']!='complete' for r in rows), seconds=perf_counter()-began,
        plans=io.ref(directory/'plans.json'),results=io.ref(directory/'results.json'),
        quote_exceptions=sum(r.get('status')=='exception' for r in [anchor_receipt,*plans.values()]),
        feasible_quotes=sum(r['feasible'] for r in [anchor_receipt,plans['activation'],plans['capacity']]))
    io.write(directory/'complete.json',receipt)
    return receipt


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['inputs'],p['placement'],p['source_protocol']]: old.checked(ref)
    if (out/'complete.json').exists(): return io.read(out/'complete_fraction0.10.json')
    if (out/'lock.json').exists(): raise RuntimeError('Inspect the existing run lock before resume')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    try:
        entries=io.read(old.checked(p['inputs'])); tasks=[(entry,draw) for entry in entries for draw in range(p['forecasts'])]
        receipts=[];failures=[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=old.initialize,initargs=(str(out),)) as pool:
            pending={pool.submit(run_forecast,task):task for task in tasks}
            for future in as_completed(pending):
                entry,draw=pending[future]
                try: receipts.append(future.result())
                except Exception as exc:
                    failures.append(dict(id=entry['id'],forecast_index=draw,error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(total=len(tasks),completed=len(receipts),failed_tasks=failures,receipts=receipts))
                print(f'{len(receipts)+len(failures)}/{len(tasks)} tasks, {len(failures)} failures',flush=True)
        completion=dict(total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),
            execution_failed_rows=sum(r['failed_rows'] for r in receipts),receipts=receipts)
        io.write(out/'complete_fraction0.10.json',completion)
        io.write(out/'complete.json',dict(status='complete',main=io.ref(out/'complete_fraction0.10.json'),
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources'])))
        return completion
    finally: (out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=SETTINGS,required=True)
    parser.add_argument('--index',type=Path,default=Path('outputs/measured_pq_pengwah_enriched_20261114_v1/qualified_enriched_input.json'))
    parser.add_argument('--ac-index',type=Path,default=Path('outputs/measured_pq_full_ac_20261114_v1/index.json'))
    parser.add_argument('--resources',type=Path,default=Path('outputs/flexibility_stability_resources_20261114_v1/placements.json'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve()
    if not args.resume: prepare(args)
    elif io.read(args.out/'protocol.json')['phase']!=args.phase: raise ValueError('Frozen phase mismatch')
    if not args.prepare_only:
        result=run(args.out);print({k:v for k,v in result.items() if k!='receipts'})


if __name__=='__main__': main()
