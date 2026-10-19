"""Broad paired EV design and fixed/shared-line inference experiment."""
from __future__ import annotations
import argparse,base64,hashlib,json,platform,traceback
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from . import ev_systematic_cases as cases
from . import ev_systematic_refit as refit
from . import ev_fixed_train_design as fixed
from . import ev_gradient_flow_study as previous
from . import ev_gradient_flow_design as gradient
from . import ev_response_metric_study as response
from . import ev_rx_study as rx
from . import gradient_price_study as io
from . import ev_dynamic_study as dynamic
from .ev_rx_precision import solve_squared_precision
from .information_context import meter_to_record

HERE=Path(__file__).resolve().parent
METHODS=('none','constant','highlow','square_fixed','square_mobile','nearest_fixed',
         'gradac_fixed','scalar_fixed','scalar_hist_fixed','scalar_mobile')
MODES=('fixed_total','fixed_holdout','joint_total')
CONFIG=dict(version='systematic_EV_fixed_joint_v1',methods=list(METHODS),modes=list(MODES),
    development_case_slots=64,formal_cases=list(cases.FORMAL_CASES),calibration_histories=108,
    evaluation_histories=72,alpha=.1,planning_seconds=15.,solver_seconds=2.,scalar_grids=[3,9],
    base_commit='1007559d',selector_uses_validation=False,selector_uses_truth=False,
    independent_unit='one new generated history; paired development conditions separately counted',
    action_planning='fixed trained lines for every policy; shared-line fitting compared after same action',
    formal_sampling='iid uniform case and uniform four truth structures from named selection seed',
    source_interface=1,PQ_direct_inputs=True,root_and_bias_fixed=True)


def source_hashes():
    values=response.source_hashes()
    for name in ('ev_systematic_cases.py','ev_systematic_refit.py','ev_systematic_design.py',
                 'ev_systematic_study.py','ev_systematic_protocol.md','ev_scalar_distance_milp.py',
                 'ev_scalar_distance_math.md','ev_physical_cases.py'):
        values[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    values['../../terminal_case33/data/case33bw_raw.py']=hashlib.sha256(
        (HERE/'../../terminal_case33/data/case33bw_raw.py').read_bytes()).hexdigest()
    return values


def freeze(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();policy_id=io.identity(CONFIG,hashes)
    frozen=dict(config=CONFIG,source_sha256=hashes,policy_id=policy_id,
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    path=directory/'freeze.json'
    if path.exists():
        if io.read_json(path)['policy_id']!=policy_id:raise ValueError('freeze changed; preserve old output')
        return frozen
    io.write_json(path,frozen)
    snapshots={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else
        HERE/name).read_bytes()).decode() for name in hashes}
    io.write_json(directory/'source_snapshot.json.gz',dict(source_sha256=hashes,source_base64=snapshots),compressed=True)
    return frozen


def specification(phase,index):
    index=int(index)
    if phase=='development':
        if not 0<=index<64:raise ValueError('development index outside64')
        k,fi=divmod(index,4);case_id=cases.CASES[k]
        group=0 if case_id in ('ordinary96','train32','train8') else 8 if case_id in ('weak_numeric96','weak_asset96') else k
        seed=100610001+100*group+fi
    elif phase in ('calibration','evaluation'):
        limit=CONFIG[phase+'_histories']
        if not 0<=index<limit:raise ValueError('formal index outside frozen count')
        first=100620001 if phase=='calibration' else 100630001
        seed=first+index;rng=np.random.default_rng(np.random.SeedSequence([seed,87231]))
        case_id=cases.FORMAL_CASES[int(rng.integers(len(cases.FORMAL_CASES)))];fi=int(rng.integers(4))
    elif phase=='pilot':
        if not 0<=index<3:raise ValueError('three interface pilots only')
        case_id=('ordinary96','case33local12','legacy_n5_12')[index];fi=index%4;seed=100600001+index
    else:raise ValueError('unknown phase')
    return dict(phase=phase,index=index,case_id=case_id,seed=seed,family_index=fi)


def _exception(exc):return type(exc).__name__+': '+str(exc)


def failed_methods(message):
    return [dict(name=m,power_kw=None,action_key=None,planning_failure=message,
        planning_seconds=None,total_seconds=None,operation=dict(success=False),parameter_error={},
        inference={mode:dict(losses={},complete=False,M=None,seconds=None,total_seconds=None)
            for mode in MODES}) for m in METHODS]


def select_plans(training,bank,fits,tm,fm,forecast,*,case_id,base_kw,seed,source_options,profile_options):
    """No truth, validation or simulator-case argument crosses this boundary."""
    from .ev_systematic_design import scalar_policy
    began=perf_counter();T,n=np.asarray(forecast['p']).shape
    curves=previous.public_curves(T,n)
    plans={k:curves[k].copy() for k in ('none','constant','highlow')}
    metadata={k:dict(planning_actual_seconds=0.,planning_seconds=0.,dependencies=[],success=True)
        for k in plans}
    domains={key:cases.service_domain(case_id,T,n,mobile=key=='mobile') for key in ('fixed','mobile')}
    setup=perf_counter()-began
    exchange=deepcopy(fixed.DEFAULT_EXCHANGE)
    exchange.update(total_seconds=CONFIG['planning_seconds'],solver_seconds=CONFIG['solver_seconds'],max_iterations=3)
    for d in ('fixed','mobile'):
        name='square_'+d;started=perf_counter()
        initial={k:curves[k] for k in ('constant','highlow')};initial['public_square']=curves['analytic_square']
        if d=='mobile':initial['fixed_parent']=plans['square_fixed']
        dependencies=['square_fixed'] if d=='mobile' else []
        dependency=sum(metadata[k]['planning_actual_seconds'] for k in dependencies)
        opts=dict(exchange,total_seconds=max(.01,CONFIG['planning_seconds']-dependency-setup))
        try:
            row=solve_squared_precision(domains[d],fm,initial,exchange_options=opts)
            u=row.get('power_kw');failure=None if u is not None else 'no square incumbent'
            # Uniform-device Q ties use a public temporal order, independent
            # of model scores or future noise. Do not apply its proof to8slots.
            if T==6 and len(set(np.asarray(fm.v_sigma).ravel()))==1:
                canonical=curves['analytic_square'];Q= float(np.sum(row['weights']*canonical**2)) if 'weights' in row else None
                if fixed._realize(domains[d],canonical)['success']:
                    actual_weights=np.mean(1/np.asarray(fm.v_sigma)**2,axis=1)[:,None]
                    canonical_Q=float(np.sum(actual_weights*canonical**2))
                    raw_Q=float(np.sum(actual_weights*np.asarray(u)**2)) if u is not None else -np.inf
                    if canonical_Q>=raw_Q-1e-6*max(1.,canonical_Q):u=canonical.copy()
            plans[name]=np.asarray(u).copy() if u is not None else curves['analytic_square'].copy()
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failure=_exception(exc);row=dict(failure=failure);plans[name]=curves['analytic_square'].copy()
        own=perf_counter()-started
        metadata[name]=dict(success=failure is None,failure=failure,raw=row,dependencies=dependencies,
            planning_actual_seconds=own,planning_seconds=own+dependency+setup)
    started=perf_counter()
    context=fixed.build_fixed_contexts(training,bank,fits,tm,fm,-fixed.CONFIG['command_power_kw'],
        fixed.CONFIG['command_power_kw'],plans['square_fixed'],forecast=forecast,base_kw=base_kw,**source_options)
    context_seconds=perf_counter()-started
    dependency=metadata['square_fixed']['planning_actual_seconds']+setup+context_seconds
    initials={k:plans[k] for k in ('constant','highlow','square_fixed')}
    for name in ('nearest_fixed','gradac_fixed','scalar_fixed','scalar_hist_fixed','scalar_mobile'):
        started=perf_counter();mobile=name.endswith('mobile');initial=dict(initials)
        dep=dependency;deps=['square_fixed']
        if mobile:
            initial.update(square_mobile=plans['square_mobile'],fixed_parent=plans['scalar_fixed'])
            # square_mobile includes fixed-square once; scalar_fixed shares context.
            dep+=metadata['square_mobile']['planning_actual_seconds']+metadata['scalar_fixed']['planning_actual_seconds']
            deps+=['square_mobile','scalar_fixed']
        seconds=max(.01,CONFIG['planning_seconds']-dep);domain=domains['mobile' if mobile else 'fixed']
        try:
            if not context['success']:raise RuntimeError('a required frozen context failed')
            if name=='nearest_fixed':
                cfg=dict(exchange=exchange,profile=profile_options,source=source_options)
                u,row=previous.fixed_surrogate_policy(context,domain,initial,cfg,seed=seed,seconds=seconds)
            elif name=='gradac_fixed':
                u,row=gradient.select_gradient_controls(context,domain,initial,method='ac',aggregation='nearest',
                    fit_options={**profile_options,**source_options},options=dict(total_seconds=seconds,
                        solver_seconds=CONFIG['solver_seconds'],max_starts=2,max_iterations=2),seed=seed)
            else:
                history={k.name:.5*float(fits[k.name].loss) for k in bank} if name=='scalar_hist_fixed' else None
                u,row=scalar_policy(context,domain,initial,history_losses=history,grids=tuple(CONFIG['scalar_grids']),
                    total_seconds=seconds,fit_options={**profile_options,**source_options},seed=seed,
                    solver_seconds=CONFIG['solver_seconds'])
            failure=None if u is not None and row.get('success',True) else 'no fully checked design; public fallback'
            plans[name]=np.asarray(u).copy() if u is not None else plans['square_mobile' if mobile else 'square_fixed'].copy()
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            failure=_exception(exc);row=dict(failure=failure)
            plans[name]=plans['square_mobile' if mobile else 'square_fixed'].copy()
        own=perf_counter()-started
        metadata[name]=dict(success=failure is None,failure=failure,raw=row,dependencies=deps,
            planning_actual_seconds=own,planning_seconds=own+dep,
            soft_budget_overrun_seconds=max(0.,own+dep-CONFIG['planning_seconds']))
    return plans,dict(methods=metadata,context=context,context_seconds=context_seconds,setup_seconds=setup,
        seconds=perf_counter()-began,uses_truth=False,uses_validation=False,uses_future_noise=False,
        source_profile_options={**profile_options,**source_options},base_kw=base_kw),domains


def _margin(losses,true_name):
    return None if true_name not in losses else min(v for k,v in losses.items() if k!=true_name)-losses[true_name]


def _parameter_error(state,truth):
    return {key:float(np.linalg.norm(np.asarray(state[key])-truth[key])/np.linalg.norm(truth[key])) for key in ('r','x')}


def _run(spec):
    began=perf_counter();case=cases.make_case(spec['case_id'],spec['seed'],spec['family_index'])
    cfg=case['config'];tr,tm,bank=case['train'],case['tm'],case['bank'];seed=spec['seed']
    truth=case['truth'];true_name=next((k.name for k in bank if k.supports==truth['candidate'].supports),None)
    record=dict(schema='systematic_EV_fixed_joint_v1',**spec,config=cfg,true_name=true_name,
        candidate_miss=true_name is None,candidates=[dict(name=k.name,supports=k.supports) for k in bank],
        training=io.observations_record(tr),training_meter=meter_to_record(tm),
        validation=io.observations_record(case['validation']),validation_meter=meter_to_record(case['vm']),
        future_meter=meter_to_record(case['fm']),truth=dict(r=truth['r'],x=truth['x'],supports=truth['candidate'].supports),
        provenance=case['provenance'],methods=[])
    started=perf_counter();fits,training=rx.train_bank(bank,tr,tm,cfg,seed+700)
    train_seconds=perf_counter()-started;record['training_result']=training
    if not training['all_complete']:
        message='required training fits incomplete: '+str(training['failures'])
        record.update(status='training_failure',failure=message,methods=failed_methods(message),
            seconds=perf_counter()-began,common_cost=dict(training_seconds=train_seconds))
        return record
    forecast=cases.forecast_for(tr,tm,fits,cfg);record['forecast']=forecast
    before_lines={k:fixed._line_id(f.r,f.x) for k,f in fits.items()}
    plans,selection,domains=select_plans(tr,bank,fits,tm,case['fm'],forecast,case_id=spec['case_id'],
        base_kw=case['base_kw'],seed=seed+800,source_options=cfg['source'],profile_options=cfg['profile'])
    assert before_lines=={k:fixed._line_id(f.r,f.x) for k,f in fits.items()}
    # Raw context retains trained objects; save its reproducible primitive data.
    record['selection']={k:v for k,v in selection.items() if k!='context'}
    record['selection']['frozen_pairs']=[dict(name=p.name,c=p.c,B=p.B,A0=p.A0,
        lower=p.lower,upper=p.upper,base_center=p.base_center) for p in selection['context'].get('pairs',[])]
    options={**cfg['training'],**cfg['source'],'ratio_bounds':cfg['ratio_bounds']}
    started=perf_counter();histories={}
    for j,k in enumerate(bank):
        histories[k.name]=refit.prepare_refit_history(k,tr,case['validation'],tm,case['vm'],
            training_fit=fits[k.name],fit_options=options,seed=seed+900+j)
    pre_seconds=perf_counter()-started
    record['history_prefits']={k:dict(fixed=h.fixed_pre,joint=h.joint_pre,seconds=h.seconds) for k,h in histories.items()}
    fixed_history_seconds=sum(max(0.,h.seconds-h.joint_pre['seconds']) for h in histories.values())
    joint_history_seconds=sum(h.joint_pre['seconds'] for h in histories.values())
    record['common_cost']=dict(training_seconds=train_seconds,joint_history_seconds=joint_history_seconds,
        fixed_history_seconds=fixed_history_seconds,combined_adapter_preparation_seconds=pre_seconds)
    cache={};future_records={}
    for name in METHODS:
        u=plans[name];mobile=name.endswith('mobile');domain=domains['mobile' if mobile else 'fixed']
        if name=='none':operation=dict(success=True,battery_task_applicable=False)
        else:
            realized=fixed._realize(domain,u);operation=dict(success=realized['success'],audit=realized['audit'])
            if realized['success']:
                decoded=domain.decode(realized['x']);operation.update(energy_kwh=decoded['energy_kwh'],
                    driver_cost=decoded['driver_cost'],arcs=[dict(kind=a.kind,driving_energy_kwh=a.driving_energy_kwh)
                        for a in decoded['selected_arcs']])
        key=hashlib.sha256(np.asarray(u,dtype='<f8').tobytes()).hexdigest()
        if key not in cache:
            started=perf_counter();future=cases.observe_future(case,u);comparisons={};failures={}
            for j,k in enumerate(bank):
                try:
                    comparisons[k.name]=refit.compare_fixed_and_joint(k,tr,case['validation'],future,tm,case['vm'],case['fm'],
                        training_fit=fits[k.name],history_cache=histories[k.name],fit_options=options,seed=seed+1000+j)
                except (ValueError,RuntimeError,AssertionError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    failures[k.name]=_exception(exc)
            inference={}
            for mode in MODES:
                losses={};complete=len(comparisons)==len(bank);own=0.
                for k,row in comparisons.items():
                    side='joint' if mode=='joint_total' else 'fixed';stage=row[side]['post']
                    complete=complete and bool(stage['success']);losses[k]=float(row[side][mode])
                    own+=row['cost']['fixed_future_seconds']
                    if side=='joint':own+=row['cost']['joint_post_seconds']
                inference[mode]=dict(losses=losses,complete=bool(complete),M=_margin(losses,true_name),seconds=own)
            perr={}
            if true_name in comparisons:
                te=_parameter_error(dict(r=fits[true_name].r,x=fits[true_name].x),truth)
                je=_parameter_error(comparisons[true_name]['joint']['post']['state'],truth)
                perr=dict(training_R=te['r'],training_X=te['x'],joint_R=je['r'],joint_X=je['x'])
            cache[key]=dict(inference=inference,comparisons=comparisons,failures=failures,
                parameter_error=perr,seconds=perf_counter()-started)
            future_records[key]=dict(observations=io.observations_record(future),power_kw=u,**cache[key])
        result=cache[key];timing=selection['methods'][name]
        inference=deepcopy(result['inference'])
        for mode,value in inference.items():
            value['total_seconds']=train_seconds+timing['planning_seconds']+value['seconds']+fixed_history_seconds+(
                joint_history_seconds if mode=='joint_total' else 0.)
            value['joint_pays_fixed_source_witness_dependencies']=mode=='joint_total'
        record['methods'].append(dict(name=name,power_kw=u,action_key=key,
            planning_failure=None if timing['success'] else timing.get('failure','planning incomplete'),
            planning_seconds=timing['planning_seconds'],planning_actual_seconds=timing['planning_actual_seconds'],
            operation=operation,inference=inference,parameter_error=result['parameter_error'],
            total_seconds=train_seconds+timing['planning_seconds']+result['seconds']+pre_seconds,
            evaluation_seconds=result['seconds'],evaluation_cache_reused=key in [m['action_key'] for m in record['methods']]))
    record.update(status='complete' if all(m['operation']['success'] and all(v['complete'] for v in m['inference'].values())
        for m in record['methods']) else 'partial_failure',future_records=future_records,
        unique_actual_curves=len(cache),seconds=perf_counter()-began,line_parameters_unchanged_during_planning=True)
    return record


def run_trial(spec):
    before=source_hashes();began=perf_counter()
    with dynamic.count_physics_calls() as counts:
        try:record=_run(spec)
        except Exception as exc:
            case=cases.make_case(spec['case_id'],spec['seed'],spec['family_index'])
            true_name=next((k.name for k in case['bank'] if k.supports==case['truth']['candidate'].supports),None)
            record=dict(schema='systematic_EV_fixed_joint_v1',**spec,config=case['config'],true_name=true_name,
                candidate_miss=true_name is None,candidates=[dict(name=k.name,supports=k.supports) for k in case['bank']],
                status='pipeline_failure',failure=_exception(exc),failure_kind='pipeline_or_implementation_error',
                traceback=traceback.format_exc(),
                methods=failed_methods(_exception(exc)),seconds=perf_counter()-began)
    record.update(source_sha256=before,policy_id=io.identity(CONFIG,before),source_unchanged=before==source_hashes(),
        actual_physics_calls=counts)
    return io.serial(record)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('pilot','development','calibration','evaluation'),required=True)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,required=True)
    args=parser.parse_args();frozen=freeze(args.output)
    for index in range(args.start,args.stop):
        spec=specification(args.phase,index);path=args.output/f"case_{index:03d}_{spec['seed']}.json.gz"
        if path.exists():raise ValueError('existing case retained; use a fresh output')
        record=run_trial(spec)
        if record['policy_id']!=frozen['policy_id'] or not record['source_unchanged']:raise RuntimeError('source drift')
        io.write_json(path,record,compressed=True)
        io.write_json(args.output/'checkpoint.json',dict(phase=args.phase,completed=index-args.start+1,
            planned=args.stop-args.start,last_index=index,status=record['status']))
        print(json.dumps(dict(index=index,case=spec['case_id'],seed=spec['seed'],status=record['status'],
            seconds=record.get('seconds'),failure=record.get('failure'),
            methods=[dict(name=m['name'],fixed=m['inference']['fixed_total'].get('M'),
                joint=m['inference']['joint_total'].get('M')) for m in record['methods']])),flush=True)


if __name__=='__main__':main()
