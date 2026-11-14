"""Paired and fresh-error validation of a fixed updated P/Q resource contract.

The original representative absolute-recovery constraints, historical mean
protection, resource execution, frozen native-bank scoring and AC bridge are
reused. Only the isolated driver, resource configuration and initialization
source differ. Historical fitting and completed old executions are not rerun.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter
import traceback

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np

from ... import io
from ...observations import Observation
from ..measured_pq_flex_20261114 import runner as old
from ..reactive_flex_20261114 import runner as reactive
from ..absolute_identification_20261114.commonseeds import build_portfolio as original_portfolio
from ..flexibility_sweep_20261113.execution import fixed_rotation

MODES = ('symmetric_guard',)


def array_hash(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def score_hash(score):
    return hashlib.sha256(json.dumps(score, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def previous_points(case, score, forecast):
    """Map sealed old 23-kvar plans; read no execution, voltage, or truth file."""
    p = old.WORK['p']
    root = Path(p['previous_run']['path'])
    previous_protocol = io.read(old.checked(p['previous_protocol']))
    entries = io.read(old.checked(previous_protocol['inputs']))
    current_score, current_forecast = score_hash(score), array_hash(forecast)
    new_ids = [(r['id'], int(r['node_label']), unit)
               for r, unit in zip(case.resources, case.action_units, strict=True)]
    positions = {identity:i for i, identity in enumerate(new_ids)}
    if len(positions) != len(new_ids):
        raise ValueError('Duplicate resource identity')
    points, records = {}, []
    for entry in entries:
        if entry.get('resource_variant') != p['previous_variant']:
            continue
        prior_score = io.read(old.checked(entry['score']))
        if score_hash(prior_score) != current_score:
            continue
        for folder in sorted((root/'fraction0.10'/entry['id']).glob('forecast*')):
            if not (folder/'complete.json').exists():
                raise ValueError('Prior forecast lacks completion receipt: '+str(folder))
            seal = io.read(folder/'seal.json')
            with np.load(old.checked(seal['planner']), allow_pickle=False) as saved:
                same_forecast = array_hash(saved['forecast']) == current_forecast
                if p['validation_phase'] == 'paired' and not same_forecast:
                    continue
                plans = io.read(old.checked(seal['plans']))
                config = io.read(folder/'resource_case.json')
                old_ids = [(r['id'], int(r['node_label']), unit)
                    for r, unit in zip(config['resources'], config['action_units'], strict=True)]
                # Positional component 774 is intentionally retained: existing
                # Q resources must keep their order, and new devices may append.
                old_q = [key for key in old_ids if key[2] == 'kvar']
                new_q = [key for key in new_ids if key[2] == 'kvar']
                old_p = [key for key in old_ids if key[2] == 'kW']
                new_p = [key for key in new_ids if key[2] == 'kW']
                if p['validation_phase'] == 'paired' and (new_p != old_p or new_q[:len(old_q)] != old_q):
                    raise ValueError('Paired resource identities/order changed the old P/Q response streams')
                for mode in MODES:
                    prior = plans[mode]
                    record = dict(mode=mode, original_entry=entry['id'], original_forecast=folder.name,
                        planner=seal['planner'], plans=seal['plans'], resource_config=io.ref(folder/'resource_case.json'),
                        original_status=prior['status'], original_feasible=prior['feasible'],
                        original_cost_yuan=prior.get('cost_yuan'),
                        original_charged_seconds=prior.get('charged_total_seconds'),
                        original_forecast_sha256=array_hash(saved['forecast']),
                        current_forecast_sha256=current_forecast, same_forecast=same_forecast,
                        transfer_seed=not same_forecast, current_score_sha256=current_score,
                        accepted=False, new_evidence_feasibility='Rechecked by unchanged solver, not assumed from old quote')
                    if not prior['feasible']:
                        record['reason'] = 'Previous quote was not feasible'
                        records.append(record)
                        continue
                    original = saved[mode].copy()
                    mapped = np.zeros(case.shape)
                    unavailable = []
                    for j, identity in enumerate(old_ids):
                        if identity in positions:
                            mapped[:, positions[identity]] = original[:, j]
                        elif np.max(np.abs(original[:, j]), initial=0.) > 1e-12:
                            unavailable.append(identity)
                    if unavailable:
                        record.update(reason='An active old device is absent', unavailable=unavailable)
                    else:
                        audit = case.audit(mapped)
                        record.update(resource_audit=audit, mapped_cost_yuan=case.fee(mapped))
                        if audit['passed']:
                            name = 'previous180_'+entry['resource_variant']+'_'+folder.name+'_'+mode
                            points[name] = mapped
                            record.update(accepted=True, seed_name=name,
                                reason='Mapped action passes new resource domain; unchanged optimizer checks new evidence')
                        else:
                            record['reason'] = 'Mapped action violates new resource contract'
                    records.append(record)
    if p['validation_phase'] == 'paired' and not records:
        raise ValueError('No matching old frozen score/forecast for paired initialization')
    return points, records


def build_portfolio(case, score, forecast, base_kw, **kwargs):
    began = perf_counter()
    budget = float(old.WORK['p']['shared_seed_seconds'])
    previous, records = previous_points(case, score, forecast)
    loading = perf_counter()-began
    options = dict(kwargs, seconds=max(.001, budget-loading))
    points, receipt = original_portfolio(case, score, forecast, base_kw, **options)
    points.update(previous)
    elapsed = perf_counter()-began
    receipt.update(seconds=elapsed, budget_seconds=budget, budget_exceeded=elapsed > budget+.05,
        points=len(points), previous_plan_loading_seconds=loading,
        previous_plan_records=records, previous_plans_added=len(previous),
        source='shared_domain_portfolio_plus_sealed_previous_23kvar_actions',
        previous_compute_cost='Old 180s quotes and their previous 30s starts were already computed; their times are retained and are excluded from this additional budget',
        charge_rule='Current loading, mapping, domain auditing and fresh portfolio generation are charged to this quote',
        future_execution_inputs=False)
    return points, receipt


def initialize(out):
    old.initialize(out)


def case_for(entry, ctx, prediction):
    return reactive.case_for(entry, ctx, prediction)


def prepare(args):
    prior = Path(args.previous_run).resolve()
    prior_protocol = io.read(prior/'protocol.json')
    if not (prior/'complete.json').exists():
        raise ValueError('Previous run lacks a completion receipt')
    completion = io.read(prior/'complete.json')
    if not completion.get('source_unchanged', False):
        raise ValueError('Previous completed run reports source drift')
    if args.seed is None:
        args.seed = int(prior_protocol['seed']) if args.phase == 'paired' else 2026111409
    if args.phase == 'paired' and args.seed != prior_protocol['seed']:
        raise ValueError('Paired phase must reuse the previous forecast/response/meter seed')
    if args.phase == 'fresh' and args.seed in (20261114, int(prior_protocol['seed'])):
        raise ValueError('Fresh phase requires a seed not used by prior development/confirmation')
    if args.seconds < 180.:
        raise ValueError('This protocol requires at least 180 seconds per quote')
    p = reactive.prepare(args.index, args.ac_index, args.placements, args.out,
        seed=args.seed, forecasts=args.forecasts, repeats=args.repeats,
        seconds=args.seconds, workers=args.workers)
    expected = ({'registered_q95_36'} if args.phase == 'paired'
                else {'registered_plus1', 'registered_q95_36'})
    if set(p['placement_variants']) != expected:
        raise ValueError('Placement variants differ from the locked paired/fresh resource protocol')
    p.update(modes=list(MODES), validation_phase=args.phase,
        previous_run=dict(path=str(prior)), previous_protocol=io.ref(prior/'protocol.json'),
        previous_complete=io.ref(prior/'complete.json'), previous_variant='registered_plus1',
        shared_seed_seconds=10., source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        phase=('paired resource validation on previously used forecast/response/meter paths'
               if args.phase == 'paired' else 'fixed-resource validation with fresh forecast/response/meter paths on the same known physical day'),
        interpretation=f"{p['original_histories']} fixed previously evaluated historical input(s); "
            f"{len(p['placement_variants'])} resource variants share {args.forecasts} forecasts x {args.repeats} paired executions. "
            + ('Reuses previously evaluated error streams for resource comparison.' if args.phase == 'paired'
               else 'Fresh error streams after resource locking; the physical load day and candidate history are already known.'),
        solver_scope='Unchanged representative symmetric_guard constraints, risks, historical means and prices; all original native candidates retained in final scoring',
        budget_comparison_scope=f'{args.seconds:g}s additional planning includes current seed work, excluding already-spent old quotes',
        previous_plan_policy='Read only same-score sealed planning actions from registered_plus1; paired requires identical forecast, fresh treats old actions as transfer templates; new domain and evidence are rechecked',
        policies=['no_dfs','symmetric_guard','symmetric_guard_rotation'],
        exact_policy_count=3, historical_refits=0,
        outcome_dependent_resource_selection=False,
        capacity_feasibility='A reserved Q box is an explicit capability contract; registered kW alone does not prove spare inverter kVA')
    # The independent resource builder may have produced the placement before
    # this process starts; include it even if it was not imported by prepare.
    known={Path(ref['path']).resolve() for ref in p['sources']}
    for path in (Path(__file__).resolve(), Path(__file__).with_name('resources.py').resolve()):
        if path.exists() and path not in known:
            p['sources'].append(io.ref(path))
            destination=Path(args.out)/'source_snapshot'/path.relative_to(old.ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path,destination)
            known.add(path)
    p.update(old._git_receipt(sorted(known)))
    io.write(Path(args.out)/'protocol.json',p)
    return p


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
    pattern_started=perf_counter()
    pattern, pattern_receipt=fixed_rotation(case); pattern_fee=case.fee(pattern)
    pattern_receipt.update(seconds=perf_counter()-pattern_started,
        time_scope='Separate candidate-independent comparison-policy LP; included in task elapsed time, outside the procurement quote budget')
    controls={'no_dfs':np.zeros(case.shape)}; plans={}
    for mode in MODES:
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
        symmetric_nonzero_feasible=plans['symmetric_guard']['nonzero_verified_feasible'])
    io.write(directory/'complete.json',receipt)
    return receipt


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in p['sources']:old.checked(ref)
    old.checked(p['placement']);old.checked(p['source_protocol'])
    old.checked(p['previous_protocol']);old.checked(p['previous_complete'])
    if (out/'complete.json').exists():
        return io.read(out/'complete_fraction0.10.json')
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
            symmetric_nonzero_feasible=sum(r['symmetric_nonzero_feasible'] for r in receipts),receipts=receipts)
        io.write(out/'complete_fraction0.10.json',completion)
        io.write(out/'complete.json',dict(status='complete',source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources']),
            main=io.ref(out/'complete_fraction0.10.json'),expansion=None))
        return completion
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=Path('outputs/measured_pq_pengwah_enriched_20261114_v1/qualified_enriched_input.json'))
    parser.add_argument('--ac-index',type=Path,default=Path('outputs/measured_pq_full_ac_20261114_v1/index.json'))
    parser.add_argument('--placements',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--phase',choices=('paired','fresh'),required=True)
    parser.add_argument('--previous-run',type=Path,default=Path('outputs/flexibility_reaudit_long180_20261114_v1'))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--forecasts',type=int,default=4)
    parser.add_argument('--repeats',type=int,default=8)
    parser.add_argument('--seconds',type=float,default=300.)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve()
    if not args.resume:
        if args.placements is None:
            parser.error('--placements is required when preparing a new run')
        prepare(args)
    else:
        p=io.read(args.out/'protocol.json')
        if p['validation_phase'] != args.phase:
            raise ValueError('Resume phase differs from frozen protocol')
    if not args.prepare_only:
        result=run(args.out)
        print({key:value for key,value in result.items() if key!='receipts'})


if __name__=='__main__':
    main()
