"""Replay sealed RX worlds under three distribution-based procurement gates."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
from types import SimpleNamespace

import numpy as np

from ... import io
from ...observations import Observation,load
from ..measured_pq_flex_20261114 import runner as old
from ..flexibility_stability_20261114 import run as base
from ..flexibility_stability_20261114 import domain,solver as prices
from ..absolute_identification_20261114.solver import SymmetricRecovery
from . import evidence,solver

MODES=('normal_fixed','mgf_fixed','mgf_forecast')
POLICIES=('no_dfs','anchor',*MODES,'mgf_forecast_rotation')


def read(ref):
    return io.read(old.checked(ref))


def diagnostic_check(model,u):
    if model is None:return dict(feasible=False,status='evidence_construction_failed')
    try:return model.verify_expansion(u)
    except Exception as exc:
        return dict(feasible=False,status='evidence_check_exception',error=repr(exc),traceback=traceback.format_exc())


def planning_context(entry,base_kw):
    """Historical observations only; explicitly preserve observed V and root."""
    score=read(entry['score']);path=old.checked(entry['files']['validation.npz'])
    observed=load(path)
    with np.load(path,allow_pickle=False) as z:
        val=Observation(observed.p,observed.q,observed.y,observed.labels,z['V'].copy(),z['root'].copy())
        if not np.array_equal(val.y,z['Y']):
            raise ArithmeticError('Historical Y changed while restoring measured voltage channels')
    expected_rows=int(entry.get('validation_rows',96))
    if val.rows!=expected_rows or val.labels!=tuple(score['labels']):
        raise ValueError('Wrong historical validation split')
    return dict(score=score,val=val,base=float(base_kw),labels=val.labels,resource=read(entry['resource_context']))


def prepare(args):
    source=args.source_run.resolve();out=args.out.resolve()
    done=io.read(source/'complete.json');previous=io.read(source/'protocol.json')
    if done['status']!='complete' or not done['source_unchanged']:
        raise ValueError('Completed unchanged RX source required')
    prior_completion=read(done['main'])
    if prior_completion['failed_tasks'] or prior_completion['execution_failed_rows']:
        raise ValueError('Source RX execution is incomplete')
    if (previous['seed'],previous['forecasts'],previous['repeats'])!=(2026111417,4,32):
        raise ValueError('Expected the locked RX seed17 four-by32 source')
    for ref in [*previous['sources'],previous['inputs'],previous['placement'],previous['source_protocol']]:
        old.checked(ref)
    previous_entries=read(previous['inputs'])
    p=base.prepare(SimpleNamespace(phase='development',index=args.index,ac_index=args.ac_index,
        resources=args.resources,out=out))
    entries=[e for e in io.read(out/'inputs.json') if e['resource_variant']=='original_p10_q36']
    if len(entries)!=1 or len(previous_entries)!=1:
        raise ValueError('Exactly one original-resource history required')
    entry,prior=entries[0],previous_entries[0]
    for field in ('id','index','case','start_date','p_resources','q_resources','physical_contract_fraction'):
        if entry[field]!=prior[field]:raise ValueError('Changed source configuration '+field)
    for a,b in ((entry['score'],prior['score']),(entry['resource_context'],prior['resource_context']),
                (entry['files']['validation.npz'],prior['files']['validation.npz'])):
        old.checked(a);old.checked(b)
        if a['sha256']!=b['sha256']:raise ValueError('Changed frozen planning input')
    entry['replay_forecasts']={}
    for draw in range(4):
        d=source/'fraction0.10'/prior['id']/f'forecast{draw:02d}'
        seal=io.read(d/'seal.json');plans=read(seal['plans'])
        pool=plans['mean_guard']['initial_points'];old.checked(pool)
        for ref in (seal['planner'],seal['anchor'],seal['protocol']):old.checked(ref)
        entry['replay_forecasts'][str(draw)]=dict(directory=str(d),seal=io.ref(d/'seal.json'),
            planner=seal['planner'],initial_points=pool,anchor=seal['anchor'],
            resource_case=io.ref(d/'resource_case.json'),anchor_receipt=io.ref(d/'anchor.json'),
            shared_receipt=io.ref(d/'shared_initial_receipt.json'),completion=io.ref(d/'complete.json'))
    io.write(out/'inputs.json',entries)
    p.update(phase='integration_smoke_replay' if args.smoke else 'distribution_gate_paired_replay',
        validation_phase='known_RX_stream_replay',smoke=bool(args.smoke),seed=2026111417,
        forecasts=1 if args.smoke else 4,repeats=1 if args.smoke else 32,
        solver_seconds=2. if args.smoke else 600.,workers=1 if args.smoke else 4,
        policies=list(POLICIES),modes=list(MODES),placement_variants=['original_p10_q36'],
        inputs=io.ref(out/'inputs.json'),source_protocol=io.ref(Path(__file__).with_name('PROTOCOL.md')),
        previous_run=str(source),source_previous=dict(protocol=io.ref(source/'protocol.json'),
            inputs=previous['inputs'],completion=io.ref(source/'complete.json')),
        base_kw=200.,base_scope='Retained public normalization base; no physical line parameters in planning',
        expected_rows=6 if args.smoke else 768,original_histories=1,refits=0,historical_refits=0,
        previous_anchor_seconds=150.,anchor_seconds=0.,shared_seed_seconds=0.,
        objective_scope={name:'activation only' for name in MODES},
        interpretation='Replay of all preselected RX seed17 forecasts and response/meter paths on one known physical day; not fresh confirmation',
        seed_source='Exact previous same-forecast shared LP pool and150s anchor; identical pool for all three modes; no cross-mode warm starts',
        historical_voltage='Preserve observed validation.npz V/root and unchanged Y; future voltage does not enter noise estimation',
        planning_input_boundary='Only frozen score, historical validation P/Q/Y/V/root, sealed forecast, normalization base and resource offers; full physical context is opened only after seal',
        probability_check='Independent post-seal probability.py audit; no distribution draws or outcomes used to select plans',
        fallback='Each failed new quote executes the old independently checked anchor. New-model feasibility remains false; executed gate check is separately recorded',
        timing='Current source loading/domain checks charged once per forecast and reported with each method; each mode has600s additional quote time. Previous150s anchor/shared work is sunk and separately stated',
        confirmation_rule='Known-world paired method comparison; no new independent load day or random confirmation sample')
    p.pop('stable_fee_threshold_yuan',None)
    known={Path(r['path']).resolve() for r in p['sources']}
    # Include the actual imported research dependency closure, not future report files.
    for module in list(sys.modules.values()):
        file=getattr(module,'__file__',None)
        if file is None:continue
        path=Path(file).resolve()
        if path.suffix=='.py' and path.is_relative_to(old.ROOT) and 'outputs' not in path.relative_to(old.ROOT).parts:
            known.add(path)
    known.update(Path(m.__file__).resolve() for m in (evidence,solver));known.add(Path(__file__).resolve())
    p['sources']=[io.ref(path) for path in sorted(known)]
    for path in sorted(known):
        dest=out/'source_snapshot'/path.relative_to(old.ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
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
        tick=perf_counter();model=None;point=None
        try:
            model=evidence.DistributionEvidence(case,planning['score'],planning['val'],prediction,planning['base'],mode=mode,target_mass=.99)
            receipt,point=solver.plan(case,planning['score'],planning['val'],prediction,planning['base'],
                mode=mode,seconds=max(.001,p['solver_seconds']-(perf_counter()-tick)),env=old.WORK['env'],
                seed_points={k:v.copy() for k,v in seeds.items()},target_mass=.99,record_directory=directory/(mode+'_diagnostics'))
            if bool(receipt['feasible'])!=(point is not None):raise ArithmeticError('Quote/point feasibility mismatch')
            if point is not None and not (case.audit(point)['passed'] and model.verify_expansion(point)['feasible']):
                raise ArithmeticError('Returned point fails independent gate/resource check')
        except Exception as exc:
            receipt,point=dict(feasible=False,status='exception',error=repr(exc),traceback=traceback.format_exc(),cost_yuan=None),None
        accepted=point is not None;action=point.copy() if accepted else anchor.copy()
        executed_gate=diagnostic_check(model,action)
        elapsed=perf_counter()-tick
        receipt.update(new_gate_feasible=accepted,new_constraints_feasible=accepted,anchor_fallback=not accepted,zero_fallback=False,
            executed_source='accepted_new_quote' if accepted else 'validated_original_anchor_fallback',
            executed_gate_feasible=bool(executed_gate['feasible']),
            executed_gate_check=executed_gate,executed_cost=prices.cost_breakdown(case,action),
            quote_wall_seconds=elapsed,requested_quote_budget_seconds=p['solver_seconds'],
            quote_budget_exceeded=elapsed>p['solver_seconds']+.05,
            shared_preparation_seconds=shared_seconds,charged_total_seconds=shared_seconds+elapsed,
            previous_shared_preparation_seconds_sunk=anchor_lineage['shared_seconds'],
            total_method_seconds_including_previous_shared_work=anchor_lineage['shared_seconds']+shared_seconds+elapsed,
            previous_anchor_requested_seconds_sunk=150.,initial_points=io.ref(directory/'shared_initial_points.npz'),
            source_previous=prior,feasibility_scope='Failed quote executes old anchor without claiming new-model feasibility')
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
    audits={mode:{name:diagnostic_check(model,u) for name,u in controls.items()}
        for mode,model in models.items()}
    io.write(directory/'prediction_audits.json',audits)
    np.savez_compressed(directory/'planner.npz',forecast=prediction,**controls)
    np.savez_compressed(directory/'reservation.npz',**{name:np.max(abs(u),axis=0) for name,u in controls.items()})
    io.write(directory/'plans.json',plans)
    io.write(directory/'seal.json',dict(created_utc=old.utc(),planner=io.ref(directory/'planner.npz'),
        plans=io.ref(directory/'plans.json'),reservation=io.ref(directory/'reservation.npz'),
        anchor=io.ref(directory/'anchor_plan.json'),prediction_audits=io.ref(directory/'prediction_audits.json'),
        score=entry['score'],validation=entry['files']['validation.npz'],source_previous=prior,
        protocol=io.ref(out/'protocol.json'),planner_actual_voltage_or_network_inputs=False))
    # Full physical context and executions are opened only after the plans above are sealed.
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


def execute(entry,draw,p,directory,prior,ctx,case,prediction,controls,plans):
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
        with np.load(Path(prior['directory'])/f'execution{repeat:02d}'/'paired_randomness.npz',allow_pickle=False) as z:
            if not np.array_equal(eta,z['eta']) or not all(np.array_equal(x,z[f'noise{i}']) for i,x in enumerate(noise)):
                raise ArithmeticError('Execution randomness differs from the sealed RX paired world')
        baseline=None
        for name,u in controls.items():
            folder=rd/name;folder.mkdir()
            mode=name if name in MODES else 'mgf_forecast' if name=='mgf_forecast_rotation' else None
            quote=None if mode is None else plans[mode]
            row=dict(id=entry['id'],original_id=entry['original_id'],resource_variant=entry['resource_variant'],
                case=entry['case'],start_date=entry['start_date'],forecast_index=draw,repeat=repeat,contract_fraction=.1,strategy=name,
                quote_zero_fallback=False,
                quote_anchor_fallback=False if quote is None else quote['anchor_fallback'],
                quote_new_constraints_feasible=plans[name]['new_gate_feasible'] if name in MODES else None,
                historical_p_map=ctx['score']['p_map'],historical_native_index=ctx['score']['native_index'])
            try:
                delivered,delivery=case.execute(u,ctx['p']*ctx['base'],eta)
                pp,qq,state,checks,ref=old.physical_state(ctx,case,delivered,folder/'full_ac.npz')
                v=state['terminal_voltage_pu'];clean_y=ctx['root'][:,None]**2-v**2
                pm,qm,y,vm,rm=old.meter(pp,qq,v,ctx['root'],noise)
                if name in ('no_dfs','anchor'):
                    with np.load(Path(prior['directory'])/f'execution{repeat:02d}'/name/'observations.npz',allow_pickle=False) as z:
                        same=all(np.array_equal(z[k],a) for k,a in [('P',pm),('Q',qm),('Y',y),('V',vm),('root',rm),('requested_action',u),('delivered_action',delivered)])
                    if not same:raise ArithmeticError('Baseline/anchor replay differs from original RX observations')
                    row['previous_observation_bitwise_equal']=True
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
    return rows


def run(out):
    out=Path(out);p=io.read(out/'protocol.json')
    for ref in [*p['sources'],p['inputs'],p['placement'],p['source_protocol'],*p['source_previous'].values()]:old.checked(ref)
    if (out/'complete.json').exists():return io.read(out/'complete_fraction0.10.json')
    if (out/'lock.json').exists():raise RuntimeError('Inspect existing lock before resume')
    io.write(out/'lock.json',dict(pid=os.getpid(),created_utc=old.utc()))
    try:
        entries=read(p['inputs']);tasks=[(entry,draw) for entry in entries for draw in range(p['forecasts'])]
        receipts=[];failures=[]
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=old.initialize,initargs=(str(out),)) as pool:
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
