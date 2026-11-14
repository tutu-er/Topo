"""Additional solve time and paired replay of already completed resource runs.

Only matching source planning files are used for initialization. The 300-second
phase's shared arrays and final point are retained without new seed generation.
Physical execution and full-native scoring remain in the frozen parent driver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

for _thread_variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_thread_variable]='1'

import numpy as np

from ... import io
from ..absolute_identification_20261114.solver import SymmetricRecovery
from ..flexibility_resource_validation_20261114 import run as base
from . import solver


ORIGINAL_CASE_FOR=base.case_for


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_quote_inputs(root, protocol):
    """Freeze only completed quote inputs; never open old execution files."""
    root=Path(root)
    records=[]
    for entry in io.read(base.old.checked(protocol['inputs'])):
        score=io.read(base.old.checked(entry['score']))
        for draw in range(protocol['forecasts']):
            directory=root/'fraction0.10'/entry['id']/f'forecast{draw:02d}'
            if not (directory/'complete.json').exists():
                raise ValueError('Source forecast is unfinished: '+str(directory))
            seal=io.read(directory/'seal.json')
            for key in ('planner','plans','score','protocol'):
                base.old.checked(seal[key])
            config=io.read(directory/'resource_case.json')
            with np.load(seal['planner']['path'],allow_pickle=False) as z:
                forecast_sha=base.array_hash(z['forecast'])
            records.append(dict(entry_id=entry['id'],forecast_index=draw,
                source_score=entry['score'],score_sha256=base.score_hash(score),
                config_sha256=canonical_hash(config),forecast_sha256=forecast_sha,
                resource_config=io.ref(directory/'resource_case.json'),
                planner=seal['planner'],plans=seal['plans'],seal=io.ref(directory/'seal.json'),
                shared_points=io.ref(directory/'shared_initial_points.npz'),
                shared_receipt=io.ref(directory/'shared_initial_receipt.json'),
                forecast_complete=io.ref(directory/'complete.json')))
    return records


def retained_portfolio(case,score,forecast,base_kw,**unused):
    """Load exact source arrays and final action; no LP or outcome selection."""
    began=perf_counter();p=base.old.WORK['p']
    signature=(base.score_hash(score),canonical_hash(case.config),base.array_hash(forecast))
    records=[r for r in p['source_quote_inputs']
             if (r['score_sha256'],r['config_sha256'],r['forecast_sha256'])==signature]
    if len(records)!=1:
        raise ValueError('Expected one identical source score/config/forecast, found '+str(len(records)))
    record=records[0]
    for key in ('source_score','resource_config','planner','plans','shared_points','shared_receipt'):
        base.old.checked(record[key])
    if base.score_hash(io.read(record['source_score']['path']))!=signature[0]:
        raise ValueError('Source score content mismatch')
    if canonical_hash(io.read(record['resource_config']['path']))!=signature[1]:
        raise ValueError('Source resource domain/config content mismatch')
    with np.load(record['shared_points']['path'],allow_pickle=False) as z:
        points={name:z[name].copy() for name in z.files}
    original_names=list(points)
    with np.load(record['planner']['path'],allow_pickle=False) as z:
        if not np.array_equal(z['forecast'],forecast):
            raise ValueError('Source forecast array is not identical')
        previous=z['symmetric_guard'].copy()
    name='source300_final_symmetric_guard'
    if name in points:
        raise ValueError('Reserved final-incumbent seed name already exists')
    points[name]=previous
    for key,value in points.items():
        if value.shape!=case.shape or not np.isfinite(value).all():
            raise ValueError('Invalid source seed shape/values: '+key)
    prior=io.read(record['plans']['path'])['symmetric_guard']
    domain=case.audit(previous)
    if prior.get('feasible') and not domain['passed']:
        raise ArithmeticError('Previously feasible final action no longer satisfies unchanged resource domain')
    prior_fee=case.fee(previous)
    if prior.get('feasible') and abs(prior_fee-prior['cost_yuan'])>1e-7:
        raise ArithmeticError('Previous fee changed on the identical configuration')
    directory=base.old.WORK['out']/'fraction0.10'/record['entry_id']/f"forecast{record['forecast_index']:02d}"
    base.old.WORK['extended_seed_context']=dict(record=record,prior=prior,previous=previous,
        previous_domain=domain,directory=directory,score_sha256=signature[0],forecast_sha256=signature[2])
    loading_seconds=perf_counter()-began
    case_seconds=float(base.old.WORK.pop('extended_case_construction_seconds',0.))
    elapsed=loading_seconds+case_seconds
    receipt=dict(seconds=elapsed,budget_seconds=p['solver_seconds'],budget_exceeded=elapsed>p['solver_seconds'],
        source='identical_source300_shared_arrays_plus_latest_final_action',points=len(points),
        source_loading_seconds=loading_seconds,resource_case_construction_seconds=case_seconds,
        original_shared_point_count=len(original_names),original_shared_point_names=original_names,
        previous_final_seed_name=name,previous_final_feasible=bool(prior.get('feasible')),
        previous_final_fee_yuan=prior.get('cost_yuan'),previous_final_recomputed_fee_yuan=prior_fee,
        previous_final_resource_audit=domain,previous_charged_seconds=prior.get('charged_total_seconds'),
        source_planner=record['planner'],source_plans=record['plans'],source_shared_points=record['shared_points'],
        source_shared_receipt=record['shared_receipt'],source_resource_config=record['resource_config'],
        seed_array_sha256={key:base.array_hash(value) for key,value in points.items()},
        original_arrays_preserved=True,score_identical=True,resource_config_identical=True,forecast_identical=True,
        generated_new_seeds=False,LPs=[],evidence_constraints_checked=False,future_execution_inputs=False,
        charge_rule='All current file loading, hashing, copying and domain auditing is charged to the additional quote budget',
        previous_compute_cost='All source300 and earlier solves are already spent and excluded from this additional budget')
    return points,receipt


def extended_plan(case,score,validation,forecast,base_kw,*,mode='symmetric_guard',
                  seconds=900.,env,seed_points=None,target_mass=.99):
    """Runtime adapter for the exact equivalent solver; preserve a valid incumbent."""
    began=perf_counter()
    if mode!='symmetric_guard':
        raise ValueError('Only the unchanged absolute-plus-guard problem is authorized')
    context=base.old.WORK['extended_seed_context'];record=context['record'];prior=context['prior']
    if base.score_hash(score)!=context['score_sha256'] or base.array_hash(forecast)!=context['forecast_sha256']:
        raise ValueError('Initial-point context does not match the quote')
    previous=context['previous']
    evidence=SymmetricRecovery(case,score,validation,forecast,base_kw,mode,target_mass)
    previous_direct=evidence.verify_expansion(previous)
    previous_feasible=bool(prior.get('feasible') and context['previous_domain']['passed'] and previous_direct['feasible'])
    if prior.get('feasible') and not previous_feasible:
        raise ArithmeticError('Source300 feasible incumbent fails unchanged original-model verification')
    receipt,point=solver.plan(case,score,validation,forecast,base_kw,mode=mode,
        seconds=max(.001,seconds-(perf_counter()-began)),env=env,seed_points=seed_points,target_mass=target_mass,
        record_directory=context['directory']/'solver_diagnostics')
    native_result=dict(feasible=receipt.get('feasible'),status=receipt.get('status'),source=receipt.get('source'),
        cost_yuan=receipt.get('cost_yuan'),native_status=receipt.get('native_status'))
    old_cost=case.fee(previous)
    fallback=False
    # A solver failure must not erase an independently rechecked feasible point.
    # Retain the raw engine status and withdraw any contradictory infeasibility.
    if previous_feasible and (point is None or case.fee(point)>old_cost):
        fallback=True;point=previous.copy()
        bound=receipt.get('lower_bound_yuan')
        conflict=bound is not None and np.isfinite(bound) and bound>old_cost+1e-7*max(1.,old_cost)
        lower=0. if bound is None or not np.isfinite(bound) or conflict else max(0.,min(float(bound),old_cost))
        gap=(old_cost-lower)/max(old_cost,1e-12)
        receipt.update(feasible=True,status='feasible_prior_incumbent_retained',source='verified_source300_final_retained',
            cost_yuan=old_cost,lower_bound_yuan=lower,relative_gap=gap,global_optimality_verified=gap<=1e-5,
            final=previous_direct,resource_audit=context['previous_domain'],
            infeasibility_proved=False,native_infeasibility_withdrawn=bool(native_result['native_status']==3),
            lower_bound_withdrawn=bool(receipt.get('lower_bound_withdrawn',False) or conflict))
    receipt.update(engine='flexibility_extended_solve_20261114.solver.plan',
        engine_source=base.old.WORK['p']['engine_source'],previous_300_status=prior.get('status'),
        previous_300_source=prior.get('source'),previous_300_feasible=bool(prior.get('feasible')),
        previous_300_cost_yuan=prior.get('cost_yuan'),previous_300_charged_seconds=prior.get('charged_total_seconds'),
        source_planner=record['planner'],source_plans=record['plans'],source_shared_points=record['shared_points'],
        loaded_initial_points=io.ref(context['directory']/'shared_initial_points.npz'),
        previous_300_direct=previous_direct,previous_300_resource_audit=context['previous_domain'],
        engine_result_before_incumbent_retention=native_result,explicit_previous_incumbent_retention=fallback,
        old_computation_excluded=True,previous_execution_or_truth_read_for_initialization=False,
        mathematical_constraints_changed=False,engine_changed_by_exact_reformulation=True,
        seconds=perf_counter()-began,
        time_scope='Adapter original-model verification, equivalent model, seed audit and solve; caller adds retained-portfolio loading time')
    if point is not None:
        direct=evidence.verify_expansion(point);domain=case.audit(point)
        if not direct['feasible'] or not domain['passed']:
            raise ArithmeticError('Returned point fails original constraints or resource domain')
        new_cost=case.fee(point)
        receipt.update(action_bitwise_identical_to_source300=bool(np.array_equal(point,previous)),
            action_close_to_source300=bool(np.allclose(point,previous,atol=1e-8,rtol=1e-8)),
            maximum_P_action_change_kw=float(np.max(np.abs(point[:,:case.P_R]-previous[:,:case.P_R]),initial=0.)),
            maximum_Q_action_change_kvar=float(np.max(np.abs(point[:,case.P_R:]-previous[:,case.P_R:]),initial=0.)),
            absolute_P_action_change_kwh=float(np.abs(point[:,:case.P_R]-previous[:,:case.P_R]).sum()*case.dt),
            absolute_Q_action_change_kvarh=float(np.abs(point[:,case.P_R:]-previous[:,case.P_R:]).sum()*case.dt),
            fee_improvement_over_source300_yuan=old_cost-new_cost if previous_feasible else None,
            feasible_source300_fee_not_increased=bool(new_cost<=old_cost) if previous_feasible else None,
            final_original_direct=direct,final_original_resource_audit=domain)
        original_buffered=evidence.h+np.asarray(direct['J'])-evidence.z*np.sqrt(2*np.asarray(direct['J']))-evidence.tie-1e-8
        receipt['returned_point_buffered_audit']=dict(pair_margins=original_buffered.tolist(),
            minimum_buffered_pair_margin=float(original_buffered.min()) if len(original_buffered) else None,
            guard_gains=direct['guard_gains'],original_direct_feasible=direct['feasible'],resource_audit=domain,
            numerical_gate_buffer=1e-8,scope='Returned action, including explicit predecessor retention, evaluated by the unchanged original formula')
    receipt['seconds']=perf_counter()-began
    return receipt,point


def timed_case_for(entry,context,forecast):
    began=perf_counter()
    case=ORIGINAL_CASE_FOR(entry,context,forecast)
    base.old.WORK['extended_case_construction_seconds']=perf_counter()-began
    return case


def initialize(out):
    base.old.initialize(out)
    base.case_for=timed_case_for
    base.build_portfolio=retained_portfolio
    base.old.recovery_plan=extended_plan


def prepare(args):
    source=Path(args.source_run).resolve();out=Path(args.out).resolve()
    if not (source/'complete.json').exists():
        raise ValueError('Source300 run is not complete')
    source_complete=io.read(source/'complete.json')
    if not source_complete.get('source_unchanged',False):
        raise ValueError('Source300 run reported source drift')
    previous=io.read(source/'protocol.json')
    if previous['modes']!=['symmetric_guard']:
        raise ValueError('Expected the completed single-method resource-validation run')
    if args.seconds<=0 or args.workers<=0:
        raise ValueError('Positive additional budget and workers required')
    for key in ('input_index','AC_cache_index','placement','inputs'):
        base.old.checked(previous[key])
    records=source_quote_inputs(source,previous)
    p=base.reactive.prepare(previous['input_index']['path'],previous['AC_cache_index']['path'],
        previous['placement']['path'],out,seed=previous['seed'],forecasts=previous['forecasts'],
        repeats=previous['repeats'],seconds=args.seconds,workers=args.workers)
    old_entries={e['id']:e for e in io.read(previous['inputs']['path'])}
    new_entries={e['id']:e for e in io.read(out/'inputs.json')}
    if set(old_entries)!=set(new_entries):
        raise ValueError('Source and continued entry sets differ')
    for key,new in new_entries.items():
        old=old_entries[key]
        for field in ('index','case','start_date','original_id','resource_variant','q_resources','files','physical_cache'):
            if new[field]!=old[field]:raise ValueError('Continued input changed: '+field)
        for field in ('score','resource_context','native_copy'):
            if new[field]['sha256']!=old[field]['sha256']:raise ValueError('Continued frozen artifact changed: '+field)
    p.update(modes=['symmetric_guard'],policies=['no_dfs','symmetric_guard','symmetric_guard_rotation'],
        validation_phase=previous['validation_phase'],source_validation_phase=previous['validation_phase'],
        phase='Additional equivalent-form solve and paired replay of previously evaluated resource-validation paths',
        interpretation=f"The same {len(new_entries)} resource/history entries, {previous['forecasts']} forecasts and {previous['repeats']} execution paths per forecast as the completed source300 run. No fresh independent experiment.",
        previous_run=dict(path=str(source)),previous_protocol=io.ref(source/'protocol.json'),previous_complete=io.ref(source/'complete.json'),
        continuation_source_run=dict(path=str(source)),source_quote_inputs=records,
        source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        engine_source=io.ref(Path(solver.__file__)),
        engine='flexibility_extended_solve_20261114.solver.plan',
        engine_scope='Exact auxiliary elimination with original-form auditing and unchanged two-interval fallback; same frozen representative constraints and full-native final scoring',
        initialization='Only identical source300 shared arrays plus latest final action; no fresh LP seed generation and no old execution/outcome reads',
        source_solver_seconds=previous['solver_seconds'],shared_seed_seconds=0.,
        budget_comparison_scope=f'{args.seconds:g}s additional solve, including current source-array loading and original-model auditing; excludes all already-spent source300/earlier solves',
        no_new_error_streams=True,historical_refits=0,resource_config_unchanged=True,math_constraints_unchanged=True,
        source_entry_and_artifact_identity_verified=True)
    known={Path(r['path']).resolve() for r in p['sources']}
    for path in (Path(__file__).resolve(),Path(solver.__file__).resolve()):
        if path not in known:
            p['sources'].append(io.ref(path));known.add(path)
            destination=out/'source_snapshot'/path.relative_to(base.old.ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    p.update(base.old._git_receipt(sorted(known)))
    io.write(out/'protocol.json',p)
    return p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=900.)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();args.out=args.out.resolve()
    if not args.resume:
        if args.source_run is None:parser.error('--source-run is required for a new continuation')
        prepare(args)
    if not args.prepare_only:
        base.initialize=initialize
        result=base.run(args.out)
        print({key:value for key,value in result.items() if key!='receipts'})


if __name__=='__main__':
    main()
