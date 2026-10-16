"""Bounded matched-window pilot for shared-line EV-MILP proposals.

Old v2 evaluation is diagnostic only. Fresh histories are generated from the
explicit specifications below. This pilot evaluates physical separation and
prediction ranking, without fitting or claiming new rejection thresholds.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import platform
import sys
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
import scipy
from . import ev_mobile_v3_cases as cases
from . import ev_mobile_v3_design as design
from . import ev_mobile_v2_design as v2design
from . import ev_mobile_v2_study as v2study
from . import ev_fixed_train_design as fixed
from . import ev_systematic_design as scalar
from . import ev_systematic_refit as refit
from . import ev_rx_study as rx
from . import gradient_price_study as io
from . import ev_dynamic_study as dynamic
from .ev_gradient_flow_study import public_curves
from .ev_rx_precision import solve_squared_precision
from .information_context import meter_to_record

HERE=Path(__file__).resolve().parent
METHODS=('constant','highlow','square_fixed','square_mobile',
         'scalar_fixed','scalar_mobile','projection_fixed','projection_mobile')
CONFIG=dict(version='mobile_v3_shared_projection_pilot2',T=12,
    H=[8,96],banks=['n3','n4'],candidate_count=4,
    domain_incremental_seconds=60.,solver_seconds=3.,segments=4,max_steps=1,support_seconds=.75,
    scalar_grids=[3,9],no_new_calibration=True,
    information_scope='training only, one trained reference line per candidate',
    inference='full AC H+validation+future shared-line refit',
    time_scope='60s incremental per method/domain; all ancestor costs also reported')


def specifications(phase):
    if phase=='development':
        return [dict(index=i,phase=phase,bank=bank,H=H,T=12,
            seed=2026101601+i,family_index=fi)
            for i,(bank,H,fi) in enumerate((('n3',8,0),('n3',96,3),('n4',8,0),('n4',96,3)))]
    if phase!='evaluation':raise ValueError('development or evaluation')
    return [dict(index=i,phase=phase,bank=bank,H=H,T=12,
        seed=2026101800+100*bi+10*hi+fi,family_index=fi)
        for i,(bi,bank,hi,H,fi) in enumerate((bi,bank,hi,H,fi)
        for bi,bank in enumerate(('n3','n4')) for hi,H in enumerate((8,96)) for fi in range(4))]


def hashes():
    result=v2study.source_hashes()
    names=('ev_mobile_v3_ac.py','ev_mobile_v3_cases.py','ev_mobile_v3_design.py',
        'ev_mobile_v3_domain.py','ev_mobile_v3_projection.py','ev_mobile_v3_support.py','ev_mobile_v3_study.py')
    result.update({name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names})
    return result


def freeze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    record=dict(config=CONFIG,specifications={k:specifications(k) for k in ('development','evaluation')},
        source_sha256=hashes(),methods=METHODS,old_frozen_science_unchanged=True,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        implementation_parent_commit='02fcc52c')
    record['policy_id']=io.identity(record['config'],record['source_sha256'])
    file=output/'freeze.json'
    if file.exists():
        old=io.read_json(file)
        if old!=io.serial(record):raise ValueError('freeze changed; use a new output directory')
        return old
    io.write_json(file,record)
    payload={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).decode()
        for name in record['source_sha256']}
    io.write_json(output/'source_snapshot.json.gz',dict(source_sha256=record['source_sha256'],source_base64=payload),compressed=True)
    return record


def select(training,tm,bank,fits,fm,forecast,domains,cfg,seed):
    """Explicitly accepts no case object, truth, validation or future noise."""
    T,n=np.asarray(forecast['p']).shape;curves=public_curves(T,n)
    plans={k:curves[k].copy() for k in ('constant','highlow')};meta={}
    common=dict(total_seconds=CONFIG['domain_incremental_seconds'],solver_seconds=CONFIG['solver_seconds'])
    for d in ('fixed','mobile'):
        start=perf_counter();initial={k:curves[k] for k in ('constant','highlow')}
        initial['public_square']=curves['analytic_square']
        deps=[]
        if d=='mobile':initial={'fixed_parent':plans['square_fixed'],**initial};deps=['square_fixed']
        row=solve_squared_precision(domains[d],fm,initial,exchange_options=dict(common,
            max_iterations=3,control_segments=4,residual_segments=4,max_segments=16))
        name='square_'+d
        if row.get('power_kw') is None:raise RuntimeError(name+' has no feasible incumbent')
        plans[name]=np.asarray(row['power_kw']);meta[name]=dict(raw=row,seconds=perf_counter()-start,dependencies=deps)
    context_began=perf_counter()
    ctx=fixed.build_fixed_contexts(training,bank,fits,tm,fm,
        -fixed.CONFIG['command_power_kw'],fixed.CONFIG['command_power_kw'],plans['square_fixed'],
        base_kw=cfg['base_kw'],forecast=forecast,**cfg['source'])
    if not ctx['success']:raise RuntimeError('fixed context unavailable')
    context_seconds=perf_counter()-context_began
    for d in ('fixed','mobile'):
        start=perf_counter()
        initials={'square_parent':plans['square_'+d],**{k:curves[k] for k in ('constant','highlow')}}
        deps=['square_'+d]
        if d=='mobile':initials={'fixed_parent':plans['scalar_fixed'],**initials};deps+=['scalar_fixed']
        u,row=scalar.scalar_policy(ctx,domains[d],initials,fit_options=cfg['profile'],seed=seed,
            grids=tuple(CONFIG['scalar_grids']),**common)
        name='scalar_'+d
        plans[name]=np.asarray(u if u is not None else initials[next(iter(initials))]).copy()
        meta[name]=dict(raw=row,seconds=perf_counter()-start,dependencies=deps)
    joint={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds'],'multistart':1}
    engine_began=perf_counter();engine=v2design.JointPredictiveEngine(ctx,fit_options=joint,seed=seed)
    engine_initialization_seconds=perf_counter()-engine_began
    for d in ('fixed','mobile'):
        start=perf_counter();initials={'square_parent':plans['square_'+d]};deps=['square_'+d]
        if d=='mobile':initials={'fixed_parent':plans['projection_fixed'],**initials};deps+=['projection_fixed']
        u,row=design.projected_policy(ctx,domains[d],initials,seed=seed,joint_options=joint,
            engine=engine,segments=CONFIG['segments'],max_steps=CONFIG['max_steps'],
            support_seconds=CONFIG['support_seconds'],**common)
        name='projection_'+d
        plans[name]=np.asarray(u if u is not None else initials[next(iter(initials))]).copy()
        meta[name]=dict(raw=row,seconds=perf_counter()-start,dependencies=deps)
    def ancestors(name):
        result=set()
        for p in meta.get(name,{}).get('dependencies',[]):result.add(p);result.update(ancestors(p))
        return result
    for name,row in meta.items():
        row['context_seconds']=context_seconds if not name.startswith('square') else 0.
        row['engine_initialization_seconds']=engine_initialization_seconds if name.startswith('projection') else 0.
        row['standalone_seconds']=row['seconds']+sum(meta[p]['seconds'] for p in ancestors(name))+row['context_seconds']+row['engine_initialization_seconds']
        row['soft_budget_exceeded']=row['seconds']>CONFIG['domain_incremental_seconds']
    return plans,meta,ctx


def run(spec):
    began=perf_counter();case=cases.make_case(spec['bank'],spec['seed'],spec['family_index'],H=spec['H'],T=spec['T'])
    cfg=case['config'];tr,tm,bank=case['train'],case['tm'],case['bank'];seed=spec['seed']
    truth=case['truth'];true_name=next(c.name for c in bank if c.supports==truth['candidate'].supports)
    result=dict(spec=spec,config=cfg,true_name=true_name,methods={},failures=[],
        training=io.observations_record(tr),training_meter=meter_to_record(tm),
        validation=io.observations_record(case['validation']),validation_meter=meter_to_record(case['vm']),
        future_meter=meter_to_record(case['fm']),truth=dict(r=truth['r'],x=truth['x']),provenance=case['provenance'])
    fits,training=rx.train_bank(bank,tr,tm,cfg,seed+700);result['training_result']=training
    if not training['all_complete']:
        result.update(status='training_failure',failure='training required candidates failed',seconds=perf_counter()-began)
        return result
    forecast=cases.forecast_for(tr,tm,fits,cfg);result['forecast']=forecast
    domains={d:cases.service_domain(spec['T'],tr.n,mobile=d=='mobile') for d in ('fixed','mobile')}
    before={k:fixed._line_id(f.r,f.x) for k,f in fits.items()}
    plans,metadata,ctx=select(tr,tm,bank,fits,case['fm'],forecast,domains,cfg,seed+800)
    if before!={k:fixed._line_id(f.r,f.x) for k,f in fits.items()}:raise AssertionError('trained line mutated')
    result['planning']=metadata;options={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds']}
    histories={c.name:refit.prepare_refit_history(c,tr,case['validation'],tm,case['vm'],
        training_fit=fits[c.name],fit_options=options,seed=seed+900+j) for j,c in enumerate(bank)}
    result['history_prefits']={k:dict(fixed=h.fixed_pre,joint=h.joint_pre,seconds=h.seconds) for k,h in histories.items()}
    cache={}
    for name in METHODS:
        u=plans[name];key=hashlib.sha256(np.asarray(u,dtype='<f8').tobytes()).hexdigest()
        d='mobile' if name.endswith('mobile') else 'fixed';row=dict(power_kw=u,action_key=key,success=False)
        result['methods'][name]=row
        # Preserve the optimized full route for projection, rather than silently
        # reconstructing a possibly different route from net powers.
        saved=metadata.get(name,{}).get('raw',{}).get('selected_operation_x')
        if saved is None:saved=metadata.get(name,{}).get('raw',{}).get('operation',{}).get('x')
        if saved is None:
            realized=scalar._realize(domains[d],u,seconds=3.);saved=realized.get('x')
        if saved is None:row['failure']='no executable operation';continue
        audit=domains[d].check(saved);decoded=domains[d].decode(saved)
        power_error=float(np.max(abs(decoded['u_kw']-u)))
        if power_error>1e-7:raise AssertionError('saved full route and executed powers disagree')
        row['operation']=dict(audit=audit,energy_kwh=decoded['energy_kwh'],driver_cost=decoded['driver_cost'],
            charge_kw=decoded['charge_kw'],discharge_kw=decoded['discharge_kw'],
            arcs=[dict(kind=a.kind,departure=a.departure,arrival=a.arrival,origin=a.origin,
                destination=a.destination,driving_energy_kwh=a.driving_energy_kwh) for a in decoded['selected_arcs']])
        if key not in cache:
            future=cases.observe_future(case,u);comp={};fail={};ts=perf_counter()
            for j,c in enumerate(bank):
                try:comp[c.name]=refit.compare_fixed_and_joint(c,tr,case['validation'],future,tm,case['vm'],case['fm'],
                    training_fit=fits[c.name],history_cache=histories[c.name],fit_options=options,seed=seed+1000+j)
                except Exception as exc:fail[c.name]=type(exc).__name__+': '+str(exc)
            losses={k:float(v['joint']['joint_total']) for k,v in comp.items()}
            complete=len(comp)==len(bank) and all(v['joint']['post']['success'] for v in comp.values())
            M=min(v for k,v in losses.items() if k!=true_name)-losses[true_name] if complete else None
            cache[key]=dict(complete=complete,losses=losses,M=M,comparisons=comp,failures=fail,
                future=io.observations_record(future),seconds=perf_counter()-ts)
        planning=metadata.get(name,{}).get('raw',{})
        row.update(evaluation=cache[key],success=bool(audit['feasible'] and cache[key]['complete']),
            route_power_error_kw=power_error,
            planning_success=bool(planning.get('success',True)),
            planning_failure=None if planning.get('success',True) else planning.get('failure','planning incomplete'))
    result.update(seconds=perf_counter()-began,status='complete' if all(r['success'] for r in result['methods'].values()) else 'partial_failure',
        one_training_line_preserved=True,unique_evaluated_actions=len(cache))
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--phase',choices=['development','evaluation'],default='development')
    p.add_argument('--indices',default='all');p.add_argument('--freeze-only',action='store_true');a=p.parse_args()
    out=Path(a.output);frozen=freeze(out)
    if a.freeze_only:return
    specs=specifications(a.phase)
    indices=set(range(len(specs))) if a.indices=='all' else {int(v) for v in a.indices.split(',')}
    for spec in specs:
        i=spec['index'];file=out/a.phase/(str(i).zfill(3)+'.json.gz')
        if i not in indices or file.exists():continue
        start=perf_counter();print('START',spec,flush=True)
        try:
            with dynamic.count_physics_calls() as counts:record=run(spec)
            record['physics_calls']=dict(counts)
        except Exception as exc:record=dict(spec=spec,status='fatal_failure',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc(),seconds=perf_counter()-start)
        if hashes()!=frozen['source_sha256']:raise RuntimeError('scientific source changed during trial')
        record['policy_id']=frozen['policy_id'];io.write_json(file,record,compressed=True)
        print('DONE',i,record['status'],round(record['seconds'],2),flush=True)


if __name__=='__main__':main()
