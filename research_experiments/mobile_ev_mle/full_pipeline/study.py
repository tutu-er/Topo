"""Complete observed-only passive proposal, AC fitting, EV planning and execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
from .. import likelihood as lk
from .. import ev_ac_repair_refit as refit
from .. import ev_systematic_refit as legacy
from .. import ev_mobile_v4_domain as nested
from ..ev_mobile_service_domain import MobileServiceConfig
from ..ev_bayes_small_study import complete_curve
from ..information_context import meter_to_record
from ..gradient_price_study import observations_record
from ..g_precision.context import FixedResponses
from ..g_precision.coefficients import QuadraticTask,build_tasks_from_responses,score_tasks
from ..g_precision.study import precision_task,plan_family as old_family
from ..block_joint_margin.study import _plan_record
from . import cases,data,forecast,passive,solver
from .config import CONFIG,CALENDAR
from .io import ROOT,read,write,freeze,serial,compact_fits,source_hashes,digest

DEFAULT_OUTPUT=ROOT/'outputs/full_pipeline_v1'
RAW=ROOT/'data_external/opsd_household_20200415/household_data_15min_singleindex.csv'

def prepare(output,raw=RAW):
    output=Path(output);path=output/'prepared.json.gz'
    if path.exists():return read(path)
    series=data.load_opsd_power(raw,data.MIXED8_COLUMNS,expected_sha256=CONFIG['raw_csv_sha256'])
    splits=[data.CalendarSplit(**x) for x in CALENDAR];data.validate_calendar_splits(splits)
    windows={};inventory=[]
    for split in splits[1:]:
        count=CONFIG[split.name+'_clusters']
        starts=data.chronological_windows(series,CONFIG['H'],CONFIG['T'],split=split,count=count)
        if len(starts)!=count:raise RuntimeError('not enough declared calendar windows for '+split.name)
        windows[split.name]=[data.select_window(series,phase=split.name,start_utc=data.utc_string(series.timestamps[i]),
            history_slots=CONFIG['H'],future_slots=CONFIG['T'],split=split) for i in starts]
        for i,window in enumerate(windows[split.name]):
            inventory.append(dict(phase=split.name,cluster_index=i,start_utc=window['provenance']['start_utc'],
                end_utc=window['provenance']['end_utc'],csv_lines=window['provenance']['csv_line_numbers']))
    data.assert_windows_disjoint([v for w in windows.values() for v in w])
    # Inner rolling CV receives raw historical imports: its prefixes must not
    # inherit the final 96-slot simulator scaling or synthetic meter noise.
    development=[]
    for i,window in enumerate(windows['development']):
        development.append(dict(phase='development',values=np.asarray(window['history']['p_kw']),
            timestamps=window['history']['timestamps_utc']))
    # H=96 cannot provide an observed day-lag at internal rolling origins.
    candidates=tuple(c for c in forecast.DEFAULT_CANDIDATES if c.method!='seasonal_naive')
    fc,selection=forecast.select_on_development(development,candidates=candidates,horizon=CONFIG['T'],nonnegative=True)
    selection['seasonal_naive_scope']='excluded: internal origins shorter than its 96-slot lag'
    result=dict(config=CONFIG,raw_provenance=series.provenance,calendar=CALENDAR,inventory=inventory,
        windows=windows,forecast_config=asdict(fc),forecast_selection=selection,
        window_selection='chronological availability only; no future magnitude or outcome ranking',
        independent_clusters={p:len(w) for p,w in windows.items()})
    write(path,result);write(output/'data_inventory.json',dict(raw=result['raw_provenance'],calendar=CALENDAR,
        windows=inventory,forecast_selection=selection))
    return serial(result)

def specification(phase,index,n,source_mode):
    base=CONFIG['seed_bases'][phase];physical=base+100*index+10*n
    return dict(phase=phase,cluster_index=index,n=n,source_mode=source_mode,
        physical_seed=physical,seed=physical+(source_mode=='DER'))

def specifications(prepared,phase):
    return [specification(phase,i,n,s) for i in range(len(prepared['windows'][phase]))
        for n in CONFIG['terminal_counts'] for s in CONFIG['source_modes']]

def path_for(output,spec):
    return Path(output)/spec['phase']/('history_%03d_n%d_%s.json.gz'%(
        spec['cluster_index'],spec['n'],spec['source_mode']))

class Responses(FixedResponses):
    def __init__(self,bank,D,dm,fm,fitted,timestamps,config,fc):
        began=perf_counter();self.bank=tuple(bank);self.base_kw=config['base_kw']
        if not fitted.complete:raise ValueError('historical failure retained; no competitor may be dropped')
        self.states={c.name:dict(r=fitted.fits[c.name].r.copy(),x=fitted.fits[c.name].x.copy(),
            source_p=fitted.fits[c.name].source_p.copy(),source_q=fitted.fits[c.name].source_q.copy()) for c in bank}
        self.forecast=forecast.forecast_for(D,dm,self.states,T=config['T'],base_kw=self.base_kw,
            timestamps=timestamps,config=forecast.ForecastConfig(**fc))
        clips={}
        for name,row in self.forecast['source_by_candidate'].items():
            clips[name]={}
            for key in ('source_p','source_q'):
                before=np.asarray(row[key]);bounds=fitted.options[key+'_bounds']
                row[key]=np.clip(before,*bounds);clips[name][key]=int(np.count_nonzero(row[key]!=before))
        self.forecast['metadata']['source_domain_clips']=clips
        from ..g_precision.context import sigmas
        self.shape=(config['T'],D.n);self.sigma=sigmas(fm,*self.shape)
        self.voltage_bounds=np.asarray(fitted.options['voltage_bounds'])
        self.cache={};self.ac_calls=0;self.ac_states=0;self.derivative_calls=0
        self.zero=self.responses(np.zeros(self.shape),derivatives=True);self.seconds=perf_counter()-began

def _family(domains,tasks,baselines,config,extra=None):
    plans={};rows={};began=perf_counter()
    for name,domain in domains.items():
        parents={k:nested.lift_operation(domains['fixed_charge'],domain,x) for k,x in baselines.items()}
        if extra is not None:parents['same_domain_P-SQ']=extra[name]
        for p in nested.PARENTS[name]:parents['parent_'+p]=nested.lift_operation(domains[p],domain,plans[p])
        row=solver.solve_tasks(domain,tasks,parents,iterations=config['iterations'],segments=config['segments'],
            max_segments=config['max_segments'],solver_seconds=config['solver_seconds'],total_seconds=config['total_seconds_per_domain'],
            backend=config['backend'],relative_gap=config['relative_gap'],absolute_gap=config['absolute_gap'])
        if row['operation_x'] is None:raise RuntimeError('no feasible operation in '+name)
        plans[name]=np.asarray(row['operation_x']);rows[name]=row
    return plans,rows,perf_counter()-began

def _random_plan(domain,parents,seed,config):
    began=perf_counter()
    rng=np.random.default_rng(seed);arrays=domain.model.arrays();c=np.zeros(domain.model.n_variables)
    c[np.asarray(domain.indices.u).ravel()]=rng.normal(size=np.asarray(domain.indices.u).size)
    result=solver._backend(c,np.asarray(domain.model.integer),np.asarray(domain.model.lower),np.asarray(domain.model.upper),
        arrays['constraints'],seconds=config['solver_seconds'],relative_gap=config['relative_gap'],backend=config['backend'])
    info=dict(success=False,role='reference_only',seed=int(seed),status=int(result.status),
        message=str(result.message),seconds=perf_counter()-began)
    x=None if result.x is None else np.asarray(result.x)
    if x is not None:
        check=solver.operation_diagnostics(domain,x);info['raw_operation_validation']=check
        if not check['feasible']:
            x,repair=solver.repair_operation(domain,x,time_limit=.5);info['repair']=repair
    if x is None:x=np.asarray(next(iter(parents.values()))).copy();info['fallback']='feasible parent'
    else:info['success']=True
    return x,info

def select(observed,fit_options,config,fc,seed,report=lambda *x:None):
    """History-only algorithm. No complete case, source class or true labels."""
    began=perf_counter();D=observed['history'];dm=observed['history_meter'];fm=observed['future_meter']
    bank,pre=passive.propose(D,dm,config['passive'],seed=seed)
    report('passive',dict(candidates=len(bank),failures=len(pre['failures'])))
    history=refit.fit_history(bank,D,dm,fit_options,seed=seed+1)
    report('history_fit',dict(complete=history.complete,failures=history.failures))
    template=MobileServiceConfig(n_slots=config['T'],n_sites=D.n,dt_hours=config['dt_hours'],
        p_charge_max_kw=40./7,p_discharge_max_kw=40./7,allow_discharge=True,
        battery_capacity_kwh=60.,initial_soc=.4,terminal_energy_kwh=26.85,
        eta_charge=.95,eta_discharge=.95,initial_site=0,initial_connected=True,
        mobile=True,access=True,travel_slots=1,setup_slots=1,travel_energy_kwh=.2,
        max_moves=1,max_throughput_kwh=7.5,travel_cost=15.,setup_cost=15.,driver_budget=30.,
        cost_units='worker_minutes',min_arrival_throughput_kwh=.25)
    domains=nested.build_nested_domains(template);shape=np.asarray(domains['mobile_bidir'].indices.u).shape
    constant=np.zeros(shape);constant[:,0]=2.85/(.95*config['T']*config['dt_hours'])
    highlow=np.zeros(shape);highlow[:config['T']//2,0]=2*constant[0,0]
    sb=perf_counter();baselines={}
    for label,power in [('CC',constant),('HL',highlow)]:
        x=complete_curve(domains['fixed_charge'],power)
        if not domains['fixed_charge'].check(x)['feasible']:
            x,repair=solver.repair_operation(domains['fixed_charge'],x)
            if x is None:raise RuntimeError(label+' could not satisfy battery task: '+str(repair))
        baselines[label]=x
    base_seconds=perf_counter()-sb
    plans={name:_plan_record('fixed_charge',domains['fixed_charge'],x) for name,x in baselines.items()}
    zero=np.zeros(shape);plans['OBS']=dict(domain='passive',power=zero,action_key=nested.action_key(zero),primary=True)
    cost=dict(passive_pre_seconds=pre['seconds'],history_seconds=history.seconds,baseline_completion_seconds=base_seconds)
    sr=perf_counter();mobile_parents={k:nested.lift_operation(domains['fixed_charge'],domains['mobile_bidir'],x) for k,x in baselines.items()}
    random,random_log=_random_plan(domains['mobile_bidir'],mobile_parents,seed+2,config)
    plans['RAND']=_plan_record('mobile_bidir',domains['mobile_bidir'],random);cost['random_seconds']=perf_counter()-sr
    squares,rows,spent=_family(domains,[precision_task(fm,shape)],baselines,config)
    cost['P-SQ_seconds']=spent;families={'P-SQ':rows}
    plans['P-SQ']=_plan_record('mobile_bidir',domains['mobile_bidir'],squares['mobile_bidir'])
    plans['P-SQ/F±']=_plan_record('fixed_bidir',domains['fixed_bidir'],squares['fixed_bidir'])
    result=dict(plans=plans,passive_pre=pre,history=history.record(),domain_config=asdict(template),
        random=random_log,families=families,cost=cost,tasks={},nominal_AC={},failures={},
        selector_uses_truth=False,selector_uses_future_noise=False)
    try:
        if len(bank)<2:raise ValueError('only one candidate, no pairwise planning score')
        context=Responses(bank,D,dm,fm,history,observed['timestamps'],config,fc)
        cost['response_setup_seconds']=context.seconds
        for method,intercept in [('G-SQ',False),('G-AQ',True)]:
            tasks=build_tasks_from_responses(context.zero,context.sigma,include_intercept=intercept)
            family,logs,seconds=_family(domains,tasks,baselines,config,extra=squares)
            families[method]=logs;cost[method+'_seconds']=seconds;result['tasks'][method]=[t.record() for t in tasks]
            for suffix,domain in [('', 'mobile_bidir'),('/F±','fixed_bidir')]:
                name=method+suffix;plan=_plan_record(domain,domains[domain],family[domain])
                try:context.scores(plan['power'])
                except Exception as exc:
                    result['failures'][name+'_forecast_physical']=repr(exc)
                    plan=deepcopy(plans['P-SQ'+suffix]);plan['fallback_reason']='forecast physical failure'
                plans[name]=plan
            report(method,dict(seconds=seconds))
        for name,plan in plans.items():
            try:result['nominal_AC'][name]=context.scores(plan['power'])
            except Exception as exc:result['nominal_AC'][name]=dict(failure=repr(exc))
        result['context']=context.record()
    except Exception as exc:result['failures']['response_or_G_planning']=repr(exc)
    for method in ('G-SQ','G-AQ'):
        for suffix in ('','/F±'):
            if method+suffix not in plans:
                plans[method+suffix]=deepcopy(plans['P-SQ'+suffix]);plans[method+suffix]['fallback_reason']='retained P-SQ after G failure'
    for plan in plans.values():plan['primary']=True
    cost['selection_total_seconds']=perf_counter()-began
    return result,bank,history,domains

def _structures(bank):return [dict(name=c.name,n=c.n,supports=c.supports) for c in bank]

def evaluate(case,spec,observed,planning,bank,history,config,fit_options,seed,report):
    D=observed['history'];dm=observed['history_meter'];fm=observed['future_meter'];actions={}
    rng=np.random.default_rng(seed)
    white={k:rng.standard_normal((config['T'],D.n) if k in ('p','q','v') else config['T'])
           for k in ('p','q','v','master_p','master_q')}
    for method,plan in planning['plans'].items():
        key=plan['action_key']
        if key in actions:actions[key]['methods'].append(method);continue
        action=dict(methods=[method],power=plan['power'],primary=plan.get('primary',True),failures={})
        try:
            E,physical,truth_inputs=cases.actual_observation(case,np.asarray(plan['power']),white)
            action.update(observations=observations_record(E),actual_white_noise=white,truth_inputs_evaluation_only=truth_inputs,
                physical={k:float(physical[k]) for k in ('ac_equation_residual','power_balance_residual')})
            voltage=np.abs(np.asarray(physical['voltage']))
            action['physical']['voltage_range']=[float(voltage.min()),float(voltage.max())]
            lo,hi=fit_options['voltage_bounds']
            action['physical']['voltage_domain_satisfied']=bool(np.all((voltage>=lo)&(voltage<=hi)))
            if not action['physical']['voltage_domain_satisfied']:
                action['failures']['actual_voltage_domain']='realized physical execution outside declared voltage domain'
            sb=perf_counter();fit=refit.evaluate_refits(bank,D,dm,E,fm,history,fit_options,seed=seed+1)
            action['pre_refit']=fit;action['pre_refit_seconds']=perf_counter()-sb
            report('evaluated_'+method,dict(complete=fit['complete']))
            if config['post_candidate_update']:
                updated=lk.concat_observations(D,E);um=legacy.concat_models((D,E),(dm,fm))
                newbank,post=passive.propose(updated,um,config['passive'],seed=seed+2)
                action['post_passive']=post
                keys={tuple(tuple(s) for s in c.supports) for c in bank};extra=[]
                for c in newbank:
                    k=tuple(tuple(s) for s in c.supports)
                    if k not in keys and len(extra)<config['post_max_new_candidates']:
                        extra.append(c);keys.add(k)
                action['candidate_union']=_structures([*bank,*extra]);action['new_candidates']=_structures(extra)
                if extra:
                    nh=refit.fit_history(extra,D,dm,fit_options,seed=seed+3)
                    nf=refit.evaluate_refits(extra,D,dm,E,fm,nh,fit_options,seed=seed+4)
                    action['post_history']=nh.record();action['post_refit']=nf
                else:action['post_history']=None;action['post_refit']=None
            else:action['candidate_union']=_structures(bank)
            if config['clean_development_only'] and spec['phase']=='development' and spec['cluster_index']==0:
                clean,_,_=cases.actual_observation(case,np.asarray(plan['power']),white,clean=True)
                action['clean_development_refit']=refit.evaluate_refits(bank,D,dm,clean,fm,history,fit_options,seed=seed+5)
        except Exception as exc:action['failures']['exception']=dict(message=repr(exc),traceback=traceback.format_exc())
        actions[key]=compact_fits(action)
    return dict(actions=actions,true_supports_evaluation_only=case['truth']['candidate'].supports,
        true_lines_evaluation_only=dict(r=case['truth']['r'],x=case['truth']['x']),
        truth_used_for_planning=False,shared_noise_across_actions=True,
        actual_future='one realized real-data continuation; no future-regime input to selector')

def run(spec,output,prepared_path,*,freeze_policy=True):
    output=Path(output);prepared=read(prepared_path);config=deepcopy(CONFIG);config['forecast_config']=prepared['forecast_config']
    frozen=freeze(output,config,prepared_path) if freeze_policy else dict(identity='development-unfrozen',source_sha256=source_hashes())
    path=path_for(output,spec)
    if path.exists():raise FileExistsError(path)
    began=perf_counter();record=dict(spec=spec,config=config,freeze_identity=frozen['identity'],failures={})
    def report(stage,details=None):
        item=dict(spec=spec,stage=stage,seconds=perf_counter()-began,details=details)
        write(path.with_suffix('.progress.json'),item)
        print('%s c%d n%d %s %s %.1fs'%(spec['phase'],spec['cluster_index'],spec['n'],spec['source_mode'],stage,item['seconds']),flush=True)
    try:
        window=prepared['windows'][spec['phase']][spec['cluster_index']]
        case=cases.make_case(spec,window,config);observed=cases.planning_inputs(case);options=cases.fit_options(config)
        record.update(provenance=case['provenance'],fit_options=options,
            observed_history=observations_record(observed['history']),history_meter=meter_to_record(observed['history_meter']),
            future_meter=meter_to_record(observed['future_meter']),timestamps=observed['timestamps'],
            truth_evaluation_only=dict(supports=case['truth']['candidate'].supports,r=case['truth']['r'],x=case['truth']['x']))
        report('started')
        planning,bank,history,domains=select(observed,options,config,prepared['forecast_config'],
            cases._seed(spec['physical_seed'],100),report)
        record['planning']=compact_fits(planning);record['passive_pre']=planning['passive_pre'];record['candidate_structures']=_structures(bank)
        if spec['phase']=='evaluation' and spec['n']==6 and spec['source_mode']=='none' and spec['cluster_index'] in config['pure_charge_ablation_clusters']:
            for base in ('P-SQ','G-SQ'):
                for suffix,domain in [('/F+','fixed_charge'),('/M+','mobile_charge')]:
                    row=planning['families'].get(base,{}).get(domain)
                    if row and row.get('operation_x') is not None:
                        plan=_plan_record(domain,domains[domain],np.asarray(row['operation_x']));plan['primary']=False
                        planning['plans'][base+suffix]=plan
            record['planning']=compact_fits(planning)
        report('selection_sealed',dict(methods=list(planning['plans']),future_seen=False))
        record['evaluation']=evaluate(case,spec,observed,planning,bank,history,config,options,
            cases._seed(spec['physical_seed'],200),report)
        record['failures'].update(planning['failures'])
        record['status']='complete' if not record['failures'] and all(not a['failures'] for a in record['evaluation']['actions'].values()) else 'complete_with_retained_failures'
    except Exception as exc:
        record['status']='failed';record['failures']['exception']=dict(message=repr(exc),traceback=traceback.format_exc())
    record['seconds']=perf_counter()-began;record['source_unchanged']=source_hashes()==frozen['source_sha256']
    write(path,record);report('saved',dict(status=record['status'],path=str(path)))
    return dict(spec=spec,path=str(path),status=record['status'],seconds=record['seconds'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','case'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--phase',choices=['development','calibration','evaluation'],default='development')
    parser.add_argument('--cluster',type=int,default=0);parser.add_argument('--n',type=int,choices=[6,8],default=6)
    parser.add_argument('--source',choices=['none','DER'],default='none')
    parser.add_argument('--development-unfrozen',action='store_true')
    args=parser.parse_args()
    if args.development_unfrozen and args.phase!='development':parser.error('unfrozen applies only to development')
    if args.mode=='prepare':prepare(args.output);print('prepared',args.output/'prepared.json.gz',flush=True)
    else:print(serial(run(specification(args.phase,args.cluster,args.n,args.source),args.output,
        args.prepared or args.output/'prepared.json.gz',freeze_policy=not args.development_unfrozen)),flush=True)

if __name__=='__main__':main()
