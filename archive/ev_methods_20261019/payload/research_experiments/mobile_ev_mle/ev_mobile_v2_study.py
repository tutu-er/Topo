"""Matched EV reliability/mobility experiment; source snapshots and truth isolation."""
from __future__ import annotations
import argparse,base64,hashlib,json,platform,traceback
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import numpy as np
import scipy
from . import ev_systematic_study as old
from . import ev_systematic_cases as oldcases
from . import ev_mobile_cases_v2 as newcases
from . import ev_mobile_v2_design as design
from . import ev_systematic_refit as refit
from . import ev_fixed_train_design as fixed
from . import ev_rx_study as rx
from . import ev_dynamic_study as dynamic
from . import gradient_price_study as io
from .information_context import meter_to_record

HERE=Path(__file__).resolve().parent
METHODS=design.METHODS
MODES=old.MODES
N3_CASES=('ordinary96','train32','train8','remote8_H96','remote8_H8')
CONFIG=dict(version='mobile_reliability_EV_v2_guarded_60',methods=list(METHODS),modes=list(MODES),
    calibration_histories_per_bank=96,evaluation_histories_per_bank=48,
    n3_formal_cases=list(N3_CASES),n4_formal_cases=['twinblock12'],
    planning_seconds=60.,solver_seconds=2.,scalar_grids=[3,9],task_cap=float(design.TASK_CAP),
    alpha=.1,selector_uses_validation=False,selector_uses_truth=False,base_commit='c522193b',
    capped_reference_floor=True,reference_floor_cap=float(design.TASK_CAP),
    PQ_direct_inputs=True,root_and_bias_fixed=True,source_interface=1,
    independent_unit='one independent generated history; paired development regimes counted separately',
    action_planning='fixed training lines for source-surrogates; shared-line AC predictive envelope prices for jointgrad',
    inference='same realized action, measurements and source interface for fixed and joint lines')

def source_hashes():
    values=old.source_hashes()
    for name in ('ev_mobile_v2_study.py','ev_mobile_v2_design.py','ev_mobile_cases_v2.py',
                 'ev_mobile_v2_protocol.md','ev_mobile_v2_math.md'):
        values[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return values

def freeze(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();value=dict(config=CONFIG,source_sha256=hashes,policy_id=io.identity(CONFIG,hashes),
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    path=directory/'freeze.json'
    if path.exists():
        if io.read_json(path)!=value:raise ValueError('frozen source/config changed; retain directory and use a new one')
        return value
    io.write_json(path,value)
    payload={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/') else HERE/name).read_bytes()).decode() for name in hashes}
    io.write_json(directory/'source_snapshot.json.gz',dict(source_sha256=hashes,source_base64=payload),compressed=True)
    return value

def specification(bank,phase,index):
    index=int(index)
    if bank not in ('n3','n4'):raise ValueError('physical bank must be n3 or n4')
    if phase=='pilot':
        cases=('train8','ordinary96','remote8_H8') if bank=='n3' else ('twinblock12',)
        if not 0<=index<len(cases):raise ValueError('pilot index outside declared list')
        case_id=cases[index];fi=(0,3,3)[index] if bank=='n3' else 3;seed=101100001+index+(100 if bank=='n4' else 0)
    elif phase=='development':
        if bank=='n3':
            if not 0<=index<20:raise ValueError('n3 development20 slots')
            k,fi=divmod(index,4);case_id=N3_CASES[k];group=0 if k<3 else 1;seed=101110001+100*group+fi
        else:
            if not 0<=index<8:raise ValueError('n4 development8 slots')
            case_id='twinblock12';fi=index%4;seed=101120001+index
    elif phase in ('calibration','evaluation'):
        count=CONFIG[phase+'_histories_per_bank']
        if not 0<=index<count:raise ValueError('index outside frozen formal count')
        first={('n3','calibration'):101210001,('n3','evaluation'):101220001,('n4','calibration'):101310001,('n4','evaluation'):101320001}[(bank,phase)]
        seed=first+index;rng=np.random.default_rng(np.random.SeedSequence([seed,94991]))
        case_id=N3_CASES[int(rng.integers(len(N3_CASES)))] if bank=='n3' else 'twinblock12';fi=int(rng.integers(4))
    else:raise ValueError('unknown phase')
    return dict(physical_bank=bank,phase=phase,index=index,case_id=case_id,seed=seed,family_index=fi)

def make_case(case_id,seed,family_index):
    if case_id.startswith('remote8_H'):
        case=newcases.make_case('remote8',seed,family_index,training_points=int(case_id.split('_H')[-1]))
    elif case_id=='twinblock12':case=newcases.make_case(case_id,seed,family_index)
    else:case=oldcases.make_case(case_id,seed,family_index)
    case['config']['experiment_case_id']=case_id
    return case

def domains_for(case):
    case_id=case['case_id'];T,n=case['config']['horizon'],case['train'].n
    if case_id in newcases.CASES:
        return {key:newcases.service_domain(case_id,mobile=key=='mobile') for key in ('fixed','mobile')}
    return {key:oldcases.service_domain(case_id,T,n,mobile=key=='mobile') for key in ('fixed','mobile')}

def observe_future(case,u):
    return newcases.observe_future(case,u) if case['case_id'] in newcases.CASES else oldcases.observe_future(case,u)

def _exception(exc):return type(exc).__name__+': '+str(exc)
def failed_methods(message):
    return [dict(name=name,power_kw=None,action_key=None,planning_failure=message,planning_seconds=None,
        total_seconds=None,operation=dict(success=False),parameter_error={},
        inference={mode:dict(losses={},complete=False,M=None,seconds=None,total_seconds=None) for mode in MODES}) for name in METHODS]
def _margin(losses,true_name):
    # A partial fit may retain only the true-labelled candidate in offline
    # evaluation. Keep the failed competitors and leave its margin undefined.
    rivals=[value for name,value in losses.items() if name!=true_name]
    return None if true_name not in losses or not rivals else min(rivals)-losses[true_name]
_parameter_error=old._parameter_error

def _run(spec):
    began=perf_counter();case=make_case(spec['case_id'],spec['seed'],spec['family_index'])
    cfg=case['config'];tr,tm,bank=case['train'],case['tm'],case['bank'];seed=spec['seed']
    truth=case['truth'];true_name=next((k.name for k in bank if k.supports==truth['candidate'].supports),None)
    record=dict(schema='mobile_reliability_EV_v2',**spec,config=cfg,true_name=true_name,
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
    forecast=oldcases.forecast_for(tr,tm,fits,cfg);record['forecast']=forecast
    before_lines={k:fixed._line_id(f.r,f.x) for k,f in fits.items()}
    plans,selection,domains=design.select_plans(tr,bank,fits,tm,case['fm'],forecast,domains=domains_for(case),
        base_kw=case['base_kw'],seed=seed+800,training_options={**cfg['training'],'ratio_bounds':cfg['ratio_bounds']},
        source_options=cfg['source'],profile_options=cfg['profile'],total_seconds=CONFIG['planning_seconds'],
        capped_reference_floor=CONFIG['capped_reference_floor'],
        solver_seconds=CONFIG['solver_seconds'],grids=tuple(CONFIG['scalar_grids']))
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
            started=perf_counter();future=observe_future(case,u);comparisons={};failures={}
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
            case=make_case(spec['case_id'],spec['seed'],spec['family_index'])
            true_name=next((k.name for k in case['bank'] if k.supports==case['truth']['candidate'].supports),None)
            record=dict(schema='mobile_reliability_EV_v2',**spec,config=case['config'],true_name=true_name,
                candidate_miss=true_name is None,candidates=[dict(name=k.name,supports=k.supports) for k in case['bank']],
                status='pipeline_failure',failure=_exception(exc),traceback=traceback.format_exc(),
                methods=failed_methods(_exception(exc)),seconds=perf_counter()-began)
    record.update(source_sha256=before,policy_id=io.identity(CONFIG,before),source_unchanged=before==source_hashes(),actual_physics_calls=counts)
    return io.serial(record)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bank',choices=('n3','n4'),required=True)
    parser.add_argument('--phase',choices=('pilot','development','calibration','evaluation'),required=True)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();frozen=freeze(args.output)
    for index in range(args.start,args.stop):
        spec=specification(args.bank,args.phase,index);path=args.output/f"case_{index:03d}_{spec['seed']}.json.gz"
        if path.exists():raise ValueError('existing record retained; choose new output')
        record=run_trial(spec)
        if record['policy_id']!=frozen['policy_id'] or not record['source_unchanged']:raise RuntimeError('source changed during frozen run')
        io.write_json(path,record,compressed=True);io.write_json(args.output/'checkpoint.json',dict(phase=args.phase,bank=args.bank,completed=index-args.start+1,planned=args.stop-args.start,last_index=index,status=record['status']))
        print(json.dumps(dict(index=index,bank=args.bank,case=spec['case_id'],seed=spec['seed'],status=record['status'],seconds=record.get('seconds'),failure=record.get('failure'),
            methods=[dict(name=m['name'],planning_failure=m.get('planning_failure'),joint_M=m['inference']['joint_total'].get('M'),planning_seconds=m.get('planning_seconds')) for m in record['methods']])),flush=True)

if __name__=='__main__':main()
