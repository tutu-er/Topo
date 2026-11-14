"""Locked single-history-mean planning experiment with frozen native scoring."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import shutil
from time import perf_counter
import traceback
from types import SimpleNamespace

import numpy as np

from ... import io
from ...observations import Observation
from ..measured_pq_flex_20261114 import runner as old
from ..flexibility_stability_20261114 import run as base
from ..flexibility_stability_20261114 import domain,solver as prices
from ..absolute_identification_20261114.commonseeds import build_portfolio
from ..absolute_identification_20261114.solver import SymmetricRecovery
from . import solver,history_means

POLICIES=('no_dfs','anchor','mean_guard','mean_guard_rotation')


def means_for(protocol,score):
    receipt=io.read(old.checked(protocol['means_source']))
    fitting=io.read(old.checked(receipt['protocol']))
    recorded=io.read(old.checked(fitting['score']))
    if recorded!=score:
        raise ValueError('Auxiliary means belong to another frozen score')
    old.checked(receipt['means'])
    if receipt['means']['sha256']!=protocol['auxiliary_means']['sha256']:
        raise ValueError('Copied auxiliary means differ from the completed fit artifact')
    for ref in receipt['fits']:
        old.checked(ref)
    with np.load(old.checked(protocol['auxiliary_means']),allow_pickle=False) as z:
        means=z['means'].copy();indices=z['native_indices'].copy();labels=z['labels'].copy()
    expected=np.array([m['index'] for m in score['models']])
    if not np.array_equal(indices,expected) or not np.array_equal(labels,score['labels']):
        raise ValueError('Auxiliary means are not in frozen representative/label order')
    if means.shape!=(len(expected),len(labels),2*len(labels)) or not np.isfinite(means).all():
        raise ValueError('Invalid auxiliary mean matrices')
    return means


def prepare(args):
    out=args.out.resolve()
    p=base.prepare(SimpleNamespace(phase='development',index=args.index,ac_index=args.ac_index,
        resources=args.resources,out=out))
    entries=[e for e in io.read(out/'inputs.json') if e['resource_variant']=='original_p10_q36']
    if len(entries)!=1:
        raise ValueError('Exactly one original-resource history required')
    fitting=io.read(args.means_source)
    if fitting['status']!='complete':
        raise ValueError('Completed historical means required')
    fitting_protocol=io.read(old.checked(fitting['protocol']))
    entry=entries[0]
    for a,b in ((fitting_protocol['score'],entry['score']),
                (fitting_protocol['train'],entry['files']['train.npz']),
                (fitting_protocol['validation'],entry['files']['validation.npz'])):
        old.checked(a);old.checked(b)
        if a['sha256']!=b['sha256']:
            raise ValueError('Historical mean lineage differs from this frozen problem')
    shutil.copyfile(old.checked(fitting['means']),out/'auxiliary_means.npz')
    io.write(out/'inputs.json',entries)
    p.update(phase='integration_smoke' if args.smoke else 'rx_history_mean_guard',
        validation_phase='integration_smoke' if args.smoke else 'new_error_stream_development',
        smoke=bool(args.smoke),seed=2026111499 if args.smoke else 2026111417,
        forecasts=1 if args.smoke else 4,repeats=1 if args.smoke else 32,
        anchor_seconds=2. if args.smoke else 150.,solver_seconds=2. if args.smoke else 600.,
        workers=1 if args.smoke else 4,shared_seed_seconds=2.,
        policies=list(POLICIES),modes=['mean_guard'],placement_variants=['original_p10_q36'],
        inputs=io.ref(out/'inputs.json'),means_source=io.ref(args.means_source),
        auxiliary_means=io.ref(out/'auxiliary_means.npz'),auxiliary_means_original=fitting['means'],
        auxiliary_fit_protocol=fitting['protocol'],source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        auxiliary_mean_fit_seconds_once=float(fitting['seconds']),
        auxiliary_mean_fit_timing_scope='Previously completed historical fit, shared once per case; no refit during these forecasts and not included in this run wall time',
        expected_rows=4 if args.smoke else 512,original_histories=1,
        interpretation='Same known physical load day; new forecast/response/meter streams. Historical288-point Ki is auxiliary and original Hi/native scoring remains frozen.',
        objective_scope=dict(anchor='activation only',mean_guard='activation only with additional fixed-history-mean constraints'),
        historical_refits=0,refits='No fitting during this run; reuse sealed auxiliary Ki. Native scoring matrices remain frozen.',
        seed_source='Current-forecast LP portfolio and same-forecast original150s anchor only',
        fallback='If no accepted augmented quote, execute the independently validated old anchor; mark anchor_fallback and new_constraints_feasible=False. If no old anchor, execute explicit zero fallback.',
        confirmation_rule='Settings prelocked for this new-error-stream development check; smoke reuses the existing seed99 world',
        planning_input_boundary='Only case offers,score,historical validation,current forecast,base,and sealed auxiliary means; no actual voltage/network passed to either quote',
        timing='New method reports shared domain/portfolio/original anchor work plus its600s quote; rotation is separately timed')
    p.pop('stable_fee_threshold_yuan',None)
    means_for(p,io.read(old.checked(entry['score'])))
    known={Path(r['path']).resolve() for r in p['sources']}
    for path in (Path(__file__),Path(solver.__file__),Path(history_means.__file__)):
        path=path.resolve()
        if path not in known:
            p['sources'].append(io.ref(path));known.add(path)
        destination=out/'source_snapshot'/path.relative_to(old.ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    p.update(old._git_receipt(sorted(known)))
    io.write(out/'protocol.json',p)
    return p


def run_forecast(task):
    entry,draw=task;p=old.WORK['p'];out=old.WORK['out'];began=perf_counter()
    directory=out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():return io.read(directory/'complete.json')
    if directory.exists():raise RuntimeError('Inspect unfinished task before reuse: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=old.utc()))
    ctx=old.context(entry)
    means=means_for(p,ctx['score'])
    planning={key:ctx[key] for key in ('score','val','base')};planning['means']=means
    prediction=old.forecast_power(np.c_[ctx['p'],ctx['q']],p['seed'],entry['index'],draw)
    shared_began=perf_counter()
    case=domain.create_case(entry,ctx,prediction)
    io.write(directory/'resource_case.json',case.config)
    seeds,seed_receipt=build_portfolio(case,planning['score'],prediction,planning['base'],
        seconds=p['shared_seed_seconds'],env=old.WORK['env'])
    np.savez_compressed(directory/'generated_initial_points.npz',**seeds)
    anchor_receipt,anchor=base.safe_plan(case,planning,prediction,seeds,'activation',
        max(.001,p['anchor_seconds']-(perf_counter()-shared_began)),directory/'anchor_diagnostics')
    old_evidence=SymmetricRecovery(case,planning['score'],planning['val'],prediction,planning['base'],'symmetric_guard',.99)
    if anchor is not None and not (case.audit(anchor)['passed'] and old_evidence.verify_expansion(anchor)['feasible']):
        raise ArithmeticError('Original anchor failed independent old-constraint check')
    shared_seconds=perf_counter()-shared_began
    if anchor is not None:seeds['shared_150_second_anchor']=anchor.copy()
    np.savez_compressed(directory/'shared_initial_points.npz',**seeds)
    io.write(directory/'anchor_plan.json',anchor_receipt)
    io.write(directory/'anchor.json',dict(feasible=anchor is not None,
        source='shared_current_forecast_original_activation_plan',requested_budget_seconds=p['anchor_seconds'],
        activation_yuan=None if anchor is None else case.fee(anchor),
        receipt=io.ref(directory/'anchor_plan.json'),shared_seconds=shared_seconds))
    seed_receipt.update(shared_preparation_seconds=shared_seconds,anchor_receipt=io.ref(directory/'anchor_plan.json'),
        generated_points=io.ref(directory/'generated_initial_points.npz'),points_after_anchor=len(seeds),
        evidence_feasibility='Original and augmented solvers independently recheck supplied points; augmented feasibility is not required to start native search')
    io.write(directory/'shared_initial_receipt.json',seed_receipt)
    quote_began=perf_counter()
    augmented=solver.MeanRecovery(case,planning['score'],planning['val'],prediction,planning['base'],planning['means'],target_mass=.99)
    outer_evidence_seconds=perf_counter()-quote_began
    try:
        receipt,point=solver.plan(case,planning['score'],planning['val'],prediction,planning['base'],
            means=planning['means'],mean_source=p['means_source'],mode='mean_absolute_guard',
            seconds=max(.001,p['solver_seconds']-outer_evidence_seconds),env=old.WORK['env'],seed_points=seeds,
            target_mass=.99,record_directory=directory/'mean_guard_diagnostics')
        if bool(receipt['feasible'])!=(point is not None):raise ArithmeticError('Augmented quote feasibility mismatch')
        if point is not None and not (case.audit(point)['passed'] and augmented.verify_expansion(point)['feasible']):
            raise ArithmeticError('Augmented quote fails original/mean/resource direct checks')
    except Exception as exc:
        receipt,point=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc(),cost_yuan=None),None
    new_feasible=point is not None
    anchor_fallback=point is None and anchor is not None
    action=(point.copy() if new_feasible else anchor.copy() if anchor_fallback else np.zeros(case.shape))
    executed_check=augmented.verify_expansion(action)
    quote_seconds=perf_counter()-quote_began
    receipt.update(quote_wall_seconds=quote_seconds,requested_quote_budget_seconds=p['solver_seconds'],
        quote_budget_exceeded=quote_seconds>p['solver_seconds']+.05,
        outer_evidence_seconds=outer_evidence_seconds,
        auxiliary_mean_fit_seconds_once=p['auxiliary_mean_fit_seconds_once'],
        auxiliary_mean_fit_timing_scope=p['auxiliary_mean_fit_timing_scope'],
        shared_preparation_seconds=shared_seconds,charged_total_seconds=shared_seconds+quote_seconds,
        initial_points=io.ref(directory/'shared_initial_points.npz'),means_source=p['means_source'],
        new_constraints_feasible=new_feasible,anchor_fallback=anchor_fallback,
        zero_fallback=not new_feasible and not anchor_fallback,
        executed_source='augmented_quote' if new_feasible else 'validated_original_anchor_fallback' if anchor_fallback else 'zero_unresolved',
        executed_original_constraints_feasible=bool(executed_check['original_feasible']),
        executed_mean_constraints_passed_direct=bool(executed_check['mean_feasible']),
        executed_cost=prices.cost_breakdown(case,action),
        fallback_scope='Execution fallback is separate from augmented optimization feasibility and its bounds')
    plans={'mean_guard':receipt}
    controls=dict(no_dfs=np.zeros(case.shape),anchor=np.zeros(case.shape) if anchor is None else anchor,mean_guard=action)
    tick=perf_counter();pattern,pattern_receipt=case.linear_optimum(case.fixed_rotation_direction(),least_fee=True)
    pattern_total=prices.cost_breakdown(case,pattern)['total_yuan'];target_total=prices.cost_breakdown(case,action)['total_yuan']
    if pattern_total<=0:raise ArithmeticError('Candidate-independent rotation has no payable capability')
    controls['mean_guard_rotation']=min(1.,target_total/pattern_total)*pattern
    rotation_total=prices.cost_breakdown(case,controls['mean_guard_rotation'])['total_yuan']
    pattern_receipt.update(seconds=perf_counter()-tick,full_pattern_total_yuan=pattern_total,
        target_total_yuan=target_total,rotation_total_yuan=rotation_total,
        same_planned_fee=abs(rotation_total-target_total)<=1e-7,
        scope='Candidate-independent original physical-resource LP; separately timed, uses executed plan total bill')
    io.write(directory/'rotation_receipt.json',pattern_receipt)
    audits={name:dict(resource=case.audit(u),evidence=augmented.verify_expansion(u)) for name,u in controls.items()}
    io.write(directory/'mean_prediction_audits.json',audits)
    np.savez_compressed(directory/'planner.npz',forecast=prediction,**controls)
    np.savez_compressed(directory/'reservation.npz',**{name:np.max(abs(u),axis=0) for name,u in controls.items()})
    io.write(directory/'plans.json',plans)
    io.write(directory/'seal.json',dict(created_utc=old.utc(),planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'),reservation=io.ref(directory/'reservation.npz'),
        anchor=io.ref(directory/'anchor_plan.json'),mean_prediction_audits=io.ref(directory/'mean_prediction_audits.json'),
        score=entry['score'],means_source=p['means_source'],auxiliary_means=p['auxiliary_means'],protocol=io.ref(out/'protocol.json'),
        planner_actual_voltage_or_network_inputs=False))
    # Shared frozen scoring and original AC execution begin only after all plans are sealed.
    H,C,hist,sigma2=ctx['H'],ctx['common_H'],ctx['history'],ctx['score']['sigma2']
    predicted={name:old.pair_components(H,C,hist,sigma2,prediction+case.shift_pq_pu(u,ctx['base'])) for name,u in controls.items()}
    exact={};rows=[]
    for name,u in controls.items():
        folder=directory/'exact_requested'/name;folder.mkdir(parents=True)
        try:
            pp,qq,state,checks,ref=old.physical_state(ctx,case,u,folder/'full_ac.npz')
            v=state['terminal_voltage_pu'];y=ctx['root'][:,None]**2-v**2
            obs=Observation(pp,qq,y,ctx['labels'],v,ctx['root'])
            exact[name]=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],y,obs,ctx['history_without_common'])
            _,availability=case.execute(u,ctx['p']*ctx['base'],np.ones(case.R))
            np.savez_compressed(folder/'pair_evidence.npz',**exact[name])
            np.savez_compressed(folder/'forecast_pair_evidence.npz',**predicted[name])
            np.savez_compressed(folder/'clean_observation.npz',P=pp,Q=qq,V=v,Y=y,root=ctx['root'],labels=ctx['labels'])
            io.write(folder/'receipt.json',dict(status='complete',physical_state=ref,checks=checks,
                PV_net_export_proxy_excess_kwh=availability['PV_clipped_kwh'],
                interpretation='Ideal delivery diagnostic after sealing; no response error or actual-PV clipping'))
        except Exception as exc:io.write(folder/'receipt.json',dict(status='failed',error=repr(exc),traceback=traceback.format_exc()))
    for repeat in range(p['repeats']):
        rd=directory/f'execution{repeat:02d}';rd.mkdir()
        eta=np.r_[old.rng(p['seed'],entry['index'],draw,773,repeat).uniform(.9,1.,case.P_R),
                  old.rng(p['seed'],entry['index'],draw,774,repeat).uniform(.9,1.,case.Q_R)]
        noise=[old.rng(p['seed'],entry['index'],draw,k,repeat).normal(size=shape)
            for k,shape in ((2,ctx['p'].shape),(3,ctx['q'].shape),(4,ctx['p'].shape),(5,ctx['root'].shape))]
        np.savez_compressed(rd/'paired_randomness.npz',eta=eta,**{f'noise{i}':x for i,x in enumerate(noise)})
        baseline=None
        for name,u in controls.items():
            folder=rd/name;folder.mkdir()
            is_new=name.startswith('mean_guard')
            row=dict(id=entry['id'],original_id=entry['original_id'],resource_variant=entry['resource_variant'],
                case=entry['case'],start_date=entry['start_date'],forecast_index=draw,repeat=repeat,contract_fraction=.1,strategy=name,
                quote_zero_fallback=receipt['zero_fallback'] if is_new else anchor_receipt['zero_fallback'] if name=='anchor' else False,
                quote_anchor_fallback=anchor_fallback if is_new else False,
                quote_new_constraints_feasible=new_feasible if name=='mean_guard' else None,
                historical_p_map=ctx['score']['p_map'],historical_native_index=ctx['score']['native_index'])
            try:
                delivered,delivery=case.execute(u,ctx['p']*ctx['base'],eta)
                pp,qq,state,checks,ref=old.physical_state(ctx,case,delivered,folder/'full_ac.npz')
                v=state['terminal_voltage_pu'];clean_y=ctx['root'][:,None]**2-v**2
                pm,qm,y,vm,rm=old.meter(pp,qq,v,ctx['root'],noise)
                obs=Observation(pm,qm,y,ctx['labels'],vm,rm)
                values,ll,new_sse=old.score_observation(ctx,obs)
                components=old.pair_components(H,C,hist,sigma2,np.c_[pp,qq],clean_y,obs,ctx['history_without_common'])
                if name=='no_dfs':baseline=components
                if baseline is None:change,summary={},dict(status='unavailable',reason='Paired zero execution failed')
                else:change,summary=old.evidence_summary(ctx,baseline,components,predicted['no_dfs'],predicted[name],exact.get('no_dfs'),exact.get(name))
                np.savez_compressed(folder/'pair_evidence.npz',**components);np.savez_compressed(folder/'pair_changes.npz',**change)
                np.savez_compressed(folder/'observations.npz',P=pm,Q=qm,Y=y,V=vm,root=rm,labels=ctx['labels'],
                    true_P=pp,true_Q=qq,true_V=v,true_root=ctx['root'],requested_action=u,delivered_action=delivered,
                    native_history_sse=hist,native_new_sse=new_sse,native_total_sse=hist+new_sse,representative_log_probability=ll)
                requested_cost=prices.cost_breakdown(case,u);settled_cost=prices.cost_breakdown(case,u,delivered)
                requested,actual=case.amounts(u),case.amounts(delivered)
                row.update(status='complete',**values,delivery=delivery,evidence=summary,physical_state=ref,AC=checks,
                    planned_fee_yuan=requested_cost['total_yuan'],settled_fee_yuan=settled_cost['total_yuan'],
                    planned_cost=requested_cost,settled_cost=settled_cost,requested_quantities=requested,delivered_quantities=actual,
                    delivered_abs_kwh=actual['active_abs_kwh'],delivered_abs_kvarh=actual['reactive_abs_kvarh'],
                    baseline_root_energy_value_yuan=old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    root_energy_value_change_yuan=old.grid_value(case,state['root_supply_kw'])-old.grid_value(case,ctx['baseline_state']['root_supply_kw']),
                    all_bus_voltage_min=float(state['voltage_magnitude_pu'].min()),all_bus_voltage_max=float(state['voltage_magnitude_pu'].max()),
                    voltage_outside_095_105_cells=int(((state['voltage_magnitude_pu']<.95)|(state['voltage_magnitude_pu']>1.05)).sum()))
            except Exception as exc:row.update(status='execution_failed',error=repr(exc),traceback=traceback.format_exc())
            rows.append(row);io.write(folder/'receipt.json',row)
        io.write(directory/'results.json',rows)
    completion=dict(status='complete',id=entry['id'],forecast_index=draw,rows=len(rows),
        failed_rows=sum(r['status']!='complete' for r in rows),seconds=perf_counter()-began,
        plans=io.ref(directory/'plans.json'),results=io.ref(directory/'results.json'),
        quote_exceptions=sum(r.get('status')=='exception' for r in (anchor_receipt,receipt)),
        original_anchor_feasible=anchor is not None,new_constraints_feasible=new_feasible,
        anchor_fallback=anchor_fallback,feasible_quotes=int(anchor is not None)+int(new_feasible))
    io.write(directory/'complete.json',completion)
    return completion


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['inputs'],p['placement'],p['source_protocol'],p['means_source'],p['auxiliary_means'],p['auxiliary_fit_protocol']]:old.checked(ref)
    if (out/'complete.json').exists():return io.read(out/'complete_fraction0.10.json')
    if (out/'lock.json').exists():raise RuntimeError('Inspect existing run lock before resume')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    try:
        entries=io.read(old.checked(p['inputs']));tasks=[(entry,draw) for entry in entries for draw in range(p['forecasts'])]
        receipts=[];failures=[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=old.initialize,initargs=(str(out),)) as pool:
            pending={pool.submit(run_forecast,task):task for task in tasks}
            for future in as_completed(pending):
                entry,draw=pending[future]
                try:receipts.append(future.result())
                except Exception as exc:failures.append(dict(id=entry['id'],forecast_index=draw,error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(total=len(tasks),completed=len(receipts),failed_tasks=failures,receipts=receipts))
                print(f'{len(receipts)+len(failures)}/{len(tasks)} tasks, {len(failures)} failures',flush=True)
        completion=dict(total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),execution_failed_rows=sum(r['failed_rows'] for r in receipts),
            new_constraints_feasible_quotes=sum(r['new_constraints_feasible'] for r in receipts),
            anchor_fallbacks=sum(r['anchor_fallback'] for r in receipts),receipts=receipts)
        io.write(out/'complete_fraction0.10.json',completion)
        io.write(out/'complete.json',dict(status='complete',main=io.ref(out/'complete_fraction0.10.json'),
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources'])))
        return completion
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=Path('outputs/measured_pq_pengwah_enriched_20261114_v1/qualified_enriched_input.json'))
    parser.add_argument('--ac-index',type=Path,default=Path('outputs/measured_pq_full_ac_20261114_v1/index.json'))
    parser.add_argument('--resources',type=Path,default=Path('outputs/flexibility_stability_resources_20261114_v1/placements.json'))
    parser.add_argument('--means-source',type=Path,default=Path('outputs/flexibility_history_mean_fit_20261114_v1/fit_complete.json'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve()
    if not args.resume:prepare(args)
    elif bool(io.read(args.out/'protocol.json')['smoke'])!=args.smoke:raise ValueError('Frozen smoke/formal setting mismatch')
    if not args.prepare_only:
        result=run(args.out);print({k:v for k,v in result.items() if k!='receipts'})


if __name__=='__main__':main()
