"""Frozen four-domain EV study; one plan, paired regular/shift future tests."""
from __future__ import annotations
import argparse
import base64
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
import numpy as np
import scipy

from . import ev_mobile_v3_study as previous
from . import ev_mobile_v4_cases as cases
from . import ev_mobile_v4_domain as nested
from . import ev_mobile_v4_policy as policy
from . import ev_mobile_v3_cases as public
from . import ev_rx_study as rx
from . import ev_systematic_refit as refit
from . import ev_fixed_train_design as fixed
from . import ev_fixed_train_fit as fixedfit
from .ev_rx_fit import fit_rx_inputs
from . import likelihood as lk
from . import gradient_price_study as io
from .ev_gradient_flow_study import public_curves
from .ev_rx_precision import solve_squared_precision,precision_weights
from .information_context import meter_to_record
from . import ev_dynamic_study as dynamic

HERE=Path(__file__).resolve().parent
FAMILIES=('square','aligned','guarded')
CONFIG=dict(version='observed_D_aligned_bounded_guarded_v4_dev2',H=96,V=4,T=12,
    family_total_seconds=300.,domain_soft_quota_seconds=75.,solver_seconds=3.,segments=4,max_exchange=2,qp_seconds=3.,
    max_nfev=400,multistart_predictive=1,reference_tolerance=1e-7,
    information='one reference per topology fitted to entire observed D=H+V',
    proxy='original physical QP, AC parent value/gradient affine correction',
    regimes=list(cases.REGIMES),paired_future_same_history_and_noise=True,
    inference='same D plus actual future, one shared line vector and per-time source at interface1',
    new_rejection_calibration=False,posthoc_diagnostic_evaluation_indices=[0,6])


def specifications(phase):
    if phase=='development':
        return [dict(index=i,phase=phase,bank=b,seed=2026101700+i,family_index=f,
                     regimes=[r],H=96,T=12) for i,(b,f,r) in enumerate(
            [('n3',1,'regular'),('n3',2,'shift'),('n4',1,'regular'),('n4',2,'shift')])]
    if phase!='evaluation':raise ValueError('unknown phase')
    return [dict(index=4*bi+f,phase=phase,bank=b,seed=2026101900+100*bi+f,
        family_index=f,regimes=list(cases.REGIMES),H=96,T=12)
        for bi,b in enumerate(('n3','n4')) for f in range(4)]


def source_hashes():
    hashes=previous.hashes()
    for suffix in ('domain','cases','bounded','master','policy','study'):
        name='ev_mobile_v4_'+suffix+'.py';hashes[name]=sha256((HERE/name).read_bytes()).hexdigest()
    return hashes


def freeze(directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    value=dict(config=CONFIG,specifications={p:specifications(p) for p in ('development','evaluation')},
        source_sha256=source_hashes(),domains=nested.DOMAIN_FLAGS,parents=nested.PARENTS,
        generator=cases.GENERATOR_CONFIG,forecast=cases.FORECAST_CONFIG,
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform()),
        seed_unit='8 independent evaluation histories; two futures within each are paired, not independent histories',
        old_scientific_sources_unchanged=122)
    value['policy_id']=io.identity(value['config'],value['source_sha256'])
    path=root/'freeze.json'
    if path.exists():
        old=io.read_json(path)
        if old!=io.serial(value):raise ValueError('preserve frozen run; use a new output directory')
        return old
    io.write_json(path,value)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).decode()
              for name in value['source_sha256']}
    io.write_json(root/'source_snapshot.json.gz',dict(source_sha256=value['source_sha256'],source_base64=snapshot),compressed=True)
    return value


def square_family(domains,meter):
    began=perf_counter();plans={};logs={};T,n=next(iter(domains.values())).indices.u.shape
    curves=public_curves(T,n);weights=precision_weights(meter,(T,n))
    for name,domain in domains.items():
        start=perf_counter();parents=[nested.lift_operation(domains[p],domain,plans[p]) for p in nested.PARENTS[name]]
        initial={'constant':curves['constant'],'highlow':curves['highlow']}
        if domain.config.allow_discharge:initial['analytic_square']=curves['analytic_square']
        initial={**{'parent_'+str(i):domain.decode(x)['u_kw'] for i,x in enumerate(parents)},**initial}
        remaining=CONFIG['family_total_seconds']-(perf_counter()-began)
        row=solve_squared_precision(domain,meter,initial,exchange_options=dict(
            total_seconds=max(.01,remaining),solver_seconds=CONFIG['solver_seconds'],max_iterations=2,
            control_segments=4,residual_segments=4,max_segments=8))
        options=list(parents)
        if row.get('success'):options.append(np.asarray(row['operation']['x']))
        if not options:raise RuntimeError('no executable square plan '+name)
        Q=lambda x:float(np.sum(weights*np.asarray(domain.decode(x)['u_kw'])**2))
        best=options[0]
        for x in options[1:]:
            if Q(x)>Q(best)+1e-9:best=x
        plans[name]=np.asarray(best).copy()
        logs[name]=dict(raw=row,Q=Q(best),parent_Q=[Q(x) for x in parents],
            scalar_monotonicity_verified=all(Q(best)>=Q(x)-1e-10 for x in parents),
            seconds=perf_counter()-start,selected_operation_x=plans[name])
    return plans,logs,perf_counter()-began


def evaluate_future(bank,D,dm,future,fm,fits,options,seed):
    """Exactly the same conditional output likelihood and shared line domain."""
    began=perf_counter();options=refit._options(options);obs=lk.concat_observations(D,future)
    model=refit.concat_models((D,future),(dm,fm));rows={};failures={}
    for j,c in enumerate(bank):
        try:
            fitted=fits[c.name];start=refit._state(fitted)
            f=fit_rx_inputs(c,future,fm,initial=start,fixed_r=fitted.r,fixed_x=fitted.x,
                           seed=seed+j,**options)
            state=refit._join_state(fitted.r,fitted.x,(start,f))
            fixed_score=refit._score(c,obs,model,state,options,(D.t,future.t),('observed','future'))
            fixed_score['state']={k:v.copy() for k,v in state.items()}
            joint,_=refit._stage(c,obs,model,options,[('D_reference_fixed_source',state)],
                (D.t,future.t),('observed','future'),seed+100+j,None)
            rows[c.name]=dict(fixed=fixed_score,joint=joint,fixed_source_success=bool(f.success))
        except Exception as exc:failures[c.name]=type(exc).__name__+': '+str(exc)
    complete=len(rows)==len(bank) and all(r['joint']['success'] for r in rows.values())
    return dict(complete=complete,comparisons=rows,failures=failures,
        losses={k:r['joint']['loss'] for k,r in rows.items()},seconds=perf_counter()-began,
        history_scope='same D=H+V used by planning',global_optimality_certified=False)


def run(spec):
    began=perf_counter();case=cases.make_case(spec['bank'],spec['seed'],spec['family_index'],'regular')
    D=lk.concat_observations(case['train'],case['validation'])
    dm=refit.concat_models((case['train'],case['validation']),(case['tm'],case['vm']))
    cfg=case['config'];bank=case['bank'];seed=spec['seed']
    result=dict(spec=spec,config=cfg,observed=io.observations_record(D),observed_meter=meter_to_record(dm),
        future_meter=meter_to_record(case['fm']),provenance=case['provenance'],methods={},families={},
        truth=dict(r=case['truth']['r'],x=case['truth']['x']),
        true_name=next(c.name for c in bank if c.supports==case['truth']['candidate'].supports))
    fits,training=rx.train_bank(bank,D,dm,cfg,seed+1000);result['training_result']=training
    print('REFERENCE_FITS',spec['index'],training['all_complete'],round(training['seconds'],2),flush=True)
    if not training['all_complete']:
        return dict(result,status='training_failure',failure='required observed-D reference fit failed',seconds=perf_counter()-began)
    forecast=cases.forecast_for(D,dm,fits,T=12,base_kw=case['base_kw']);result['forecast']=forecast
    template=public.service_domain(12,D.n,mobile=True).config
    domains=nested.build_nested_domains(template);result['operation_config']=io.serial(asdict(template))
    square,slogs,stime=square_family(domains,case['fm']);result['families']['square']=dict(logs=slogs,seconds=stime)
    print('SQUARE_DONE',spec['index'],round(stime,2),flush=True)
    options={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds'],
             'multistart':CONFIG['multistart_predictive'],'max_nfev':CONFIG['max_nfev']}
    context=policy.build_context(D,dm,bank,fits,case['fm'],forecast,base_kw=case['base_kw'],**cfg['source'])
    before={name:fixed._line_id(f.r,f.x) for name,f in fits.items()}
    families={}
    for label,guard in (('aligned',False),('guarded',True)):
        families[label]=policy.run_family(context,domains,square,{k:v['seconds'] for k,v in slogs.items()},
            options,guard=guard,seed=seed+3000,total_seconds=CONFIG['family_total_seconds'],
            solver_seconds=CONFIG['solver_seconds'],segments=CONFIG['segments'],max_exchange=CONFIG['max_exchange'],
            qp_seconds=CONFIG['qp_seconds'])
        result['families'][label]={k:v for k,v in families[label].items() if k!='plans'}
        print('POLICY_DONE',spec['index'],label,round(families[label]['standalone_planning_seconds'],2),flush=True)
    if before!={name:fixed._line_id(f.r,f.x) for name,f in fits.items()}:raise AssertionError('reference theta_D mutated')
    reporting=policy.CommonACObjective(context,options,seed=seed+3000)
    for family in families.values():
        for record in family['cache'].values():reporting.cache.remember(record['power_kw'],record)
    for label in FAMILIES:
        for domain in nested.DOMAIN_FLAGS:
            plan=None if label=='square' else families[label]['plans'].get(domain)
            x=square[domain] if label=='square' or plan is None else plan.x
            decoded=domains[domain].decode(x);power=np.asarray(decoded['u_kw'])
            key,nominal=reporting.cache.evaluate(power)
            result['methods'][label+'__'+domain]=dict(domain=domain,family=label,power_kw=power,
                operation_x=x,operation=domains[domain].check(x),action_key=key,
                scored_plan_available=bool(label=='square' or plan is not None),
                planning_success=bool(label=='square' or (plan is not None and families[label]['logs'][domain].get('planning_failure') is None)),
                planning_failure=None if label=='square' else families[label]['logs'][domain].get('planning_failure'),
                nominal=dict(success=nominal['success'],score=nominal.get('score'),nearest=nominal.get('nearest',[])))
    result['reporting_cache']=reporting.cache.records();result['actual']={}
    for regime in spec['regimes']:
        replay=cases.make_case(spec['bank'],seed,spec['family_index'],regime)
        if not np.array_equal(replay['train'].v,case['train'].v) or not np.array_equal(replay['validation'].v,case['validation'].v):
            raise AssertionError('paired regimes changed observed history')
        actions={}
        for method in result['methods'].values():
            key=method['action_key']
            if key in actions:continue
            future=cases.observe_future(replay,method['power_kw'])
            evaluation=evaluate_future(bank,D,dm,future,case['fm'],fits,options,seed+4000)
            if evaluation['complete']:
                loss=evaluation['losses'];truth=result['true_name']
                evaluation['M']=min(v for k,v in loss.items() if k!=truth)-loss[truth]
            else:evaluation['M']=None
            actions[key]=dict(future=io.observations_record(future),evaluation=evaluation)
        result['actual'][regime]=dict(actions=actions,paired_noise=True)
        print('ACTUAL_DONE',spec['index'],regime,len(actions),sum(a['evaluation']['complete'] for a in actions.values()),flush=True)
    all_actual=all(a['evaluation']['complete'] for r in result['actual'].values() for a in r['actions'].values())
    result.update(status='complete' if all_actual else 'actual_fit_failure',seconds=perf_counter()-began,reference_lines_preserved=True,
        selector_received_truth=False,regime_used_only_for_future_replay=True,
        expected_plan_slots=12,expected_actual_method_slots=12*len(spec['regimes']))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=('development','evaluation'),default='development')
    parser.add_argument('--indices',default='all');parser.add_argument('--freeze-only',action='store_true')
    args=parser.parse_args();root=Path(args.output);frozen=freeze(root)
    if args.freeze_only:return
    specs=specifications(args.phase)
    indices={s['index'] for s in specs} if args.indices=='all' else {int(v) for v in args.indices.split(',')}
    for spec in specs:
        path=root/args.phase/(str(spec['index']).zfill(3)+'.json.gz')
        if spec['index'] not in indices or path.exists():continue
        print('START',spec,flush=True)
        try:
            with dynamic.count_physics_calls() as counters:record=run(spec)
            record['physics_calls']=dict(counters)
        except Exception as exc:
            record=dict(spec=spec,status='fatal_failure',failure=type(exc).__name__+': '+str(exc),traceback=traceback.format_exc())
        if source_hashes()!=frozen['source_sha256']:raise RuntimeError('science changed during frozen execution')
        record['policy_id']=frozen['policy_id'];io.write_json(path,record,compressed=True)
        print('DONE',spec['index'],record['status'],record.get('seconds'),flush=True)


if __name__=='__main__':main()
