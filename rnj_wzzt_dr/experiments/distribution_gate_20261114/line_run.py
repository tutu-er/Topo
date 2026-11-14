"""Paired replay using old planning-only seed lines and one resource-removal pass."""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import shutil
from time import perf_counter
import traceback
import numpy as np

from ... import io
from ..measured_pq_flex_20261114 import runner as old
from ..flexibility_stability_20261114 import domain,solver as prices
from ..absolute_identification_20261114.solver import SymmetricRecovery
from . import run as replay,evidence,solver,line_seed,line_polish

MODES=replay.MODES
POLICIES=replay.POLICIES
read=replay.read
diagnostic_check=replay.diagnostic_check
planning_context=replay.planning_context
execute=replay.execute


def prepare(args):
    p=replay.prepare(args)
    p.update(phase='line_search_integration_smoke_replay' if args.smoke else 'line_search_distribution_gate_paired_replay',
        driver_relative_path='rnj_wzzt_dr/experiments/distribution_gate_20261114/line_run.py',
        source_protocol=io.ref(Path(__file__).with_name('LINE_PROTOCOL.md')),
        parent_distribution_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        solver_seconds=None,native_QCQP_calls=0,
        calculation='All original anchor-to-seed lines, then one fixed activation-fee-descending column removal pass for each feasible mode',
        seed_source='Exact RX same-forecast shared LP pool and150s anchor only; no current600s outcomes or controls',
        timing='Full evidence construction, line enumeration, polishing and direct checks timed; finite fixed pass count, no native QCQP. Previous shared150s work sunk and separate',
        bound_scope='Zero from nonnegative activation fees only; no native global-optimality or line-search completeness certificate',
        interpretation='Same known RX seed17 worlds and fixed probability constraints; planning-only initialization/fee-reduction iteration, not fresh confirmation')
    known={Path(ref['path']).resolve() for ref in p['sources']}
    known.update(Path(module.__file__).resolve() for module in (line_seed,line_polish))
    known.add(Path(__file__).resolve())
    p['sources']=[io.ref(path) for path in sorted(known)]
    for path in sorted(known):
        target=args.out/'source_snapshot'/path.relative_to(old.ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    p.update(old._git_receipt(sorted(known)))
    io.write(args.out/'protocol.json',p)
    return p


def initialize(out):
    out=Path(out)
    old.WORK.update(out=out,p=io.read(out/'protocol.json'),env=None,contexts={})


def run_forecast(task):
    entry,draw=task;p=old.WORK['p'];out=old.WORK['out'];began=perf_counter()
    directory=out/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
    if (directory/'complete.json').exists():return io.read(directory/'complete.json')
    if directory.exists():raise RuntimeError('Inspect unfinished task before reuse: '+str(directory))
    directory.mkdir(parents=True)
    io.write(directory/'started.json',dict(pid=os.getpid(),created_utc=old.utc()))
    shared_began=perf_counter();prior=entry['replay_forecasts'][str(draw)]
    planning=planning_context(entry,p['base_kw'])
    with np.load(old.checked(prior['planner']),allow_pickle=False) as z:
        prediction=z['forecast'].copy();anchor=z['anchor'].copy()
    with np.load(old.checked(prior['initial_points']),allow_pickle=False) as z:
        seeds={name:z[name].copy() for name in z.files}
    if 'shared_150_second_anchor' not in seeds or not np.array_equal(anchor,seeds['shared_150_second_anchor']):
        raise ValueError('Previous anchor and recorded seed array differ')
    case=domain.create_case(entry,planning,prediction)
    if case.config!=read(prior['resource_case']):raise ValueError('Physical resource domain differs from previous forecast')
    anchor_receipt=read(prior['anchor']);anchor_lineage=read(prior['anchor_receipt'])
    if not anchor_receipt['feasible']:raise ValueError('Verified old anchor required')
    original=SymmetricRecovery(case,planning['score'],planning['val'],prediction,planning['base'],'symmetric_guard',.99)
    if not (case.audit(anchor)['passed'] and original.verify_expansion(anchor)['feasible']):
        raise ArithmeticError('Old anchor fails original gate/resource checks')
    io.write(directory/'resource_case.json',case.config)
    shutil.copyfile(old.checked(prior['initial_points']),directory/'shared_initial_points.npz')
    io.write(directory/'anchor_plan.json',anchor_receipt)
    io.write(directory/'anchor.json',dict(anchor_lineage,source_previous=prior,
        newly_optimized=False,old_compute_is_sunk=True))
    shared_seconds=perf_counter()-shared_began
    io.write(directory/'shared_initial_receipt.json',dict(source_previous=prior,seconds=shared_seconds,
        points=len(seeds),array_hashes={k:io.contract_hash(v) for k,v in seeds.items()},
        previous_shared_preparation_seconds=anchor_lineage['shared_seconds'],
        previous_anchor_requested_seconds=150.,old_compute_is_sunk=True,
        physical_or_truth_inputs=False,validation_voltage_preserved=True))
    plans={};models={};controls=dict(no_dfs=np.zeros(case.shape),anchor=anchor.copy())
    for mode in MODES:
        tick=perf_counter();model=None;point=None;receipt={};scan=None;polish=None
        try:
            model=evidence.DistributionEvidence(case,planning['score'],planning['val'],prediction,planning['base'],mode=mode,target_mass=.99)
            baseline=model.verify_expansion(np.zeros(case.shape))
            scan,point=line_seed.search_lines(case,model,anchor,{k:v.copy() for k,v in seeds.items()})
            if point is not None:
                polish,point=line_polish.polish_resources(case,model,point)
            final=None if point is None else solver.direct_audit(case,model,point)
            if point is not None and not final['passed_acceptance']:
                raise ArithmeticError('Returned line point fails original gate/resource check')
            receipt=dict(feasible=point is not None,
                status='feasible' if point is not None else 'no_verified_line_point',
                source='line_search_polished' if point is not None else 'line_search_no_verified_point',
                cost_yuan=None if point is None else float(case.fee(point)),
                lower_bound_yuan=0.,relative_gap=None,native_status=None,optimizer_invoked=False,
                global_optimality_verified=False,infeasibility_proved=False,initial=None,
                mode=mode,objective_kind='activation',objective_cost_object='activation_yuan',bound_cost_object='activation_yuan',
                targets=model.targets,gate_rows=model.rows,gate_count=len(model.rows),guard_count=len(model.guards),
                evidence_metadata=model.metadata,
                baseline=baseline,final=None if final is None else final['evidence'],direct_audit=final,
                buffered_direct_audit=final,target_mass=.99,numerical_gate_buffer=solver.BUFFER,
                original_nominal_gates_retained=False,original_scalar_guards_retained=True,
                constrained_bank='Chosen frozen representative versus all original native points in other internal-topology families',
                line_search=scan,resource_polish=polish,
                solver_source=io.ref(__file__),line_seed_source=io.ref(line_seed.__file__),
                line_polish_source=io.ref(line_polish.__file__),evidence_source=io.ref(evidence.__file__),
                raw_bound_scope='Zero solely from globally nonnegative activation cost; no native QCQP bound',
                global_optimality_scope='No full-problem or numerical-line completeness claim',
                confidence_scope='Same declared distribution and original scalar guards; actual AC evaluated after sealing',
                time_scope='Evidence construction, original seed-line enumeration, one resource-removal pass and direct checks included',
                native_attributes={},diagnostic_files={},trace_status='not_applicable_no_native_optimizer',logging_errors=[])
        except Exception as exc:
            receipt,point=dict(feasible=False,status='exception',source='line_search_exception',
                error=repr(exc),traceback=traceback.format_exc(),cost_yuan=None,
                lower_bound_yuan=0.,relative_gap=None,native_status=None,optimizer_invoked=False,
                global_optimality_verified=False,infeasibility_proved=False,initial=None,
                line_search=scan,resource_polish=polish,
                evidence_metadata=None if model is None else model.metadata),None
        accepted=point is not None;action=point.copy() if accepted else anchor.copy()
        executed_gate=diagnostic_check(model,action);elapsed=perf_counter()-tick
        receipt.update(new_gate_feasible=accepted,new_constraints_feasible=accepted,anchor_fallback=not accepted,zero_fallback=False,
            executed_source='accepted_new_quote' if accepted else 'validated_original_anchor_fallback',
            executed_gate_feasible=bool(executed_gate['feasible']),executed_gate_check=executed_gate,
            executed_cost=prices.cost_breakdown(case,action),quote_wall_seconds=elapsed,seconds=elapsed,
            requested_quote_budget_seconds=None,quote_budget_exceeded=False,
            shared_preparation_seconds=shared_seconds,charged_total_seconds=shared_seconds+elapsed,
            previous_shared_preparation_seconds_sunk=anchor_lineage['shared_seconds'],
            total_method_seconds_including_previous_shared_work=anchor_lineage['shared_seconds']+shared_seconds+elapsed,
            previous_anchor_requested_seconds_sunk=150.,initial_points=io.ref(directory/'shared_initial_points.npz'),
            source_previous=prior,feasibility_scope='Failed line search executes old anchor without claiming new-model feasibility')
        plans[mode]=receipt;models[mode]=model;controls[mode]=action
        io.write(directory/(mode+'_plan.json'),receipt)
    tick=perf_counter();pattern,rotation=case.linear_optimum(case.fixed_rotation_direction(),least_fee=True)
    target_total=prices.cost_breakdown(case,controls['mgf_forecast'])['total_yuan']
    full_total=prices.cost_breakdown(case,pattern)['total_yuan']
    if full_total<=0:raise ArithmeticError('Rotation has no payable capability')
    controls['mgf_forecast_rotation']=min(1.,target_total/full_total)*pattern
    rotation_total=prices.cost_breakdown(case,controls['mgf_forecast_rotation'])['total_yuan']
    rotation.update(seconds=perf_counter()-tick,target_total_yuan=target_total,full_pattern_total_yuan=full_total,
        rotation_total_yuan=rotation_total,same_planned_fee=abs(rotation_total-target_total)<=1e-7,
        scope='Candidate-independent physical-resource LP, matched to total request capacity-plus-activation bill; separately timed')
    io.write(directory/'rotation_receipt.json',rotation)
    audits={mode:{name:diagnostic_check(model,u) for name,u in controls.items()} for mode,model in models.items()}
    io.write(directory/'prediction_audits.json',audits)
    np.savez_compressed(directory/'planner.npz',forecast=prediction,**controls)
    np.savez_compressed(directory/'reservation.npz',**{name:np.max(abs(u),axis=0) for name,u in controls.items()})
    io.write(directory/'plans.json',plans)
    io.write(directory/'seal.json',dict(created_utc=old.utc(),planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'),reservation=io.ref(directory/'reservation.npz'),
        anchor=io.ref(directory/'anchor_plan.json'),prediction_audits=io.ref(directory/'prediction_audits.json'),
        score=entry['score'],validation=entry['files']['validation.npz'],source_previous=prior,
        protocol=io.ref(out/'protocol.json'),planner_actual_voltage_or_network_inputs=False))
    ctx=old.context(entry)
    if ctx['base']!=planning['base'] or ctx['score']!=planning['score']:
        raise ArithmeticError('Evaluation and frozen planning contracts differ')
    rows=execute(entry,draw,p,directory,prior,ctx,case,prediction,controls,plans)
    completion=dict(status='complete',id=entry['id'],forecast_index=draw,rows=len(rows),
        failed_rows=sum(r['status']!='complete' for r in rows),seconds=perf_counter()-began,
        plans=io.ref(directory/'plans.json'),results=io.ref(directory/'results.json'),
        quote_exceptions=sum(r.get('status')=='exception' for r in plans.values()),
        new_gate_feasible={name:plans[name]['new_gate_feasible'] for name in MODES},
        anchor_fallbacks=sum(r['anchor_fallback'] for r in plans.values()))
    io.write(directory/'complete.json',completion)
    return completion


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['inputs'],p['placement'],p['source_protocol'],p['parent_distribution_protocol'],*p['source_previous'].values()]:old.checked(ref)
    if (out/'complete.json').exists():return io.read(out/'complete_fraction0.10.json')
    if (out/'lock.json').exists():raise RuntimeError('Inspect existing lock before resume')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    try:
        entries=read(p['inputs']);tasks=[(entry,draw) for entry in entries for draw in range(p['forecasts'])]
        receipts=[];failures=[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(out),)) as pool:
            pending={pool.submit(run_forecast,task):task for task in tasks}
            for future in as_completed(pending):
                entry,draw=pending[future]
                try:receipts.append(future.result())
                except Exception as exc:failures.append(dict(id=entry['id'],forecast_index=draw,error=repr(exc),traceback=traceback.format_exc()))
                io.write(out/'progress.json',dict(total=len(tasks),completed=len(receipts),failed_tasks=failures,receipts=receipts))
                print(f'{len(receipts)+len(failures)}/{len(tasks)} tasks, {len(failures)} failures',flush=True)
        result=dict(total_forecasts=len(tasks),completed_forecasts=len(receipts),failed_tasks=failures,
            quote_exceptions=sum(r['quote_exceptions'] for r in receipts),execution_failed_rows=sum(r['failed_rows'] for r in receipts),
            feasible_quotes_by_mode={name:sum(r['new_gate_feasible'][name] for r in receipts) for name in MODES},
            anchor_fallbacks=sum(r['anchor_fallbacks'] for r in receipts),receipts=receipts)
        io.write(out/'complete_fraction0.10.json',result)
        io.write(out/'complete.json',dict(status='complete',main=io.ref(out/'complete_fraction0.10.json'),
            source_unchanged=all(io.ref(r['path'])['sha256']==r['sha256'] for r in p['sources'])))
        return result
    finally:(out/'lock.json').unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=Path('outputs/rx_mean_guard_formal_20261114_v1'))
    parser.add_argument('--index',type=Path,default=Path('outputs/measured_pq_pengwah_enriched_20261114_v1/qualified_enriched_input.json'))
    parser.add_argument('--ac-index',type=Path,default=Path('outputs/measured_pq_full_ac_20261114_v1/index.json'))
    parser.add_argument('--resources',type=Path,default=Path('outputs/flexibility_stability_resources_20261114_v1/placements.json'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve()
    if not args.resume:prepare(args)
    elif bool(io.read(args.out/'protocol.json')['smoke'])!=args.smoke:raise ValueError('Frozen smoke/formal setting mismatch')
    if not args.prepare_only:
        result=run(args.out);print({k:v for k,v in result.items() if k!='receipts'})


if __name__=='__main__':main()
