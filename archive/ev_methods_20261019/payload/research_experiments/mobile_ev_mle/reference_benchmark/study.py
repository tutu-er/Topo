"""Native feeder benchmark: observations-only planning, then sealed execution."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import traceback
import numpy as np
from .. import likelihood as lk
from .. import ev_systematic_refit as legacy
from ..full_pipeline import passive
from ..information_context import meter_to_record
from ..gradient_price_study import observations_record
from . import cases, inference, selector
from .config import CONFIG
from .io import ROOT,read,write,freeze,serial,compact_fits,source_hashes


def specification(phase,index,network,source_mode,prepared):
    order=CONFIG['network_names'].index(network)
    physical=CONFIG['seed_bases'][phase]+1000*index+20*order
    return dict(phase=phase,cluster_index=int(index),network=network,
        n=len(prepared['networks'][network]['observation_bus_ids']),source_mode=source_mode,
        physical_seed=physical,seed=physical+int(source_mode!='none'))


def specifications(prepared,phase):
    return [specification(phase,i,name,s,prepared)
        for i in range(len(prepared['windows'][phase])) for name in CONFIG['network_names']
        for s in CONFIG['source_modes']]


def path_for(output,spec):
    return Path(output)/spec['phase']/('history_%03d_%s_%s.json.gz'%(
        spec['cluster_index'],spec['network'],spec['source_mode']))


def structures(bank):
    return [dict(name=c.name,n=c.n,supports=c.supports) for c in bank]


def evaluate(case,spec,observed,planning,bank,history,config,options,seed,report):
    D=observed['history'];dm=observed['history_meter'];fm=observed['future_meter']
    rng=np.random.default_rng(seed)
    white={k:rng.standard_normal((config['T'],D.n) if k in ('p','q','v') else config['T'])
        for k in ('p','q','v','master_p','master_q')}
    actions={}
    for method,plan in planning['plans'].items():
        key=plan['action_key']
        if key in actions:actions[key]['methods'].append(method);continue
        action=dict(methods=[method],power=plan['power'],primary=True,failures={})
        try:
            E,physical,latent=cases.actual_observation(case,np.asarray(plan['power']),white)
            voltage=np.abs(physical['voltage'])
            action.update(observations=observations_record(E),actual_white_noise=white,
                truth_inputs_evaluation_only=latent,physical=dict(
                    ac_equation_residual=float(physical['ac_equation_residual']),
                    power_balance_residual=float(physical['power_balance_residual']),
                    voltage_range=[float(voltage.min()),float(voltage.max())],
                    voltage_domain_satisfied=bool(np.all((voltage>=options['voltage_bounds'][0])&
                                                        (voltage<=options['voltage_bounds'][1]))),
                    service_voltage_09_11_satisfied=bool(np.all((voltage>=.9)&(voltage<=1.1)))))
            action['physical']['independent_reference_AC']=physical.get('reference_audit')
            if not action['physical']['voltage_domain_satisfied']:
                action['failures']['actual_voltage_domain']='realized voltage outside declared AC fitting domain'
            if not action['physical']['service_voltage_09_11_satisfied']:
                action['failures']['actual_service_voltage']='realized voltage outside declared [0.9,1.1] service band; fitting still retained'
            started=perf_counter()
            action['pre_refit']=inference.evaluate_refits(bank,D,dm,E,fm,history,options,seed=seed+1)
            action['pre_refit_seconds']=perf_counter()-started
            report('evaluated_'+method,dict(complete=action['pre_refit']['complete']))
            if config['post_candidate_update']:
                arrived=lk.concat_observations(D,E);am=legacy.concat_models((D,E),(dm,fm))
                pcfg=dict(config['passive'])
                pcfg['coefficient_bound']=2*max(np.max(options['r_bounds'][1]),np.max(options['x_bounds'][1]))
                newbank,post=passive.propose(arrived,am,pcfg,seed=seed+2)
                action['post_passive']=post
                seen={tuple(tuple(s) for s in c.supports) for c in bank};extra=[]
                for c in newbank:
                    k=tuple(tuple(s) for s in c.supports)
                    if k not in seen and len(extra)<config['post_max_new_candidates']:
                        extra.append(c);seen.add(k)
                action['candidate_union']=structures([*bank,*extra])
                action['new_candidates']=structures(extra)
                if extra:
                    nh=inference.fit_history(extra,D,dm,options,seed=seed+3)
                    action['post_history']=nh.record()
                    action['post_refit']=inference.evaluate_refits(extra,D,dm,E,fm,nh,options,seed=seed+4)
                else:action['post_history']=None;action['post_refit']=None
            else:action['candidate_union']=structures(bank)
        except Exception as exc:
            action['failures']['exception']=dict(message=repr(exc),traceback=traceback.format_exc())
        actions[key]=compact_fits(action)
    return dict(actions=actions,true_supports_evaluation_only=case['truth']['candidate'].supports,
        true_lines_evaluation_only=dict(r=case['truth']['r'],x=case['truth']['x']),
        truth_used_for_planning=False,shared_noise_across_actions=True,
        actual_future='published profile continuation; simulator only; no future access in selector')


def run(spec,output,prepared_path,*,freeze_policy=True,overrides=None):
    output=Path(output);prepared=read(prepared_path);config=deepcopy(CONFIG)
    if overrides:
        if freeze_policy:raise ValueError('ad hoc overrides restricted to development-unfrozen')
        config.update(overrides)
    if not freeze_policy and spec['phase']!='development':raise ValueError('only development may be unfrozen')
    sealed=freeze(output,config,prepared_path) if freeze_policy else dict(
        identity='development-unfrozen',source_sha256=source_hashes())
    path=path_for(output,spec)
    if path.exists():raise FileExistsError(path)
    began=perf_counter();record=dict(spec=spec,config=config,freeze_identity=sealed['identity'],failures={})
    def report(stage,details=None):
        write(path.with_suffix('.progress.json'),dict(spec=spec,stage=stage,
            seconds=perf_counter()-began,details=details))
        print('%s c%d %s %s %s %.1fs'%(spec['phase'],spec['cluster_index'],
            spec['network'],spec['source_mode'],stage,perf_counter()-began),flush=True)
    try:
        window=prepared['windows'][spec['phase']][spec['cluster_index']]
        case=cases.make_case(spec,window,prepared,config)
        observed=cases.planning_inputs(case);options=case['fit_options']
        config['base_kw']=float(case['base_kw']) # actual physical units retained per record
        config['base_kv']=float(case['provenance'].get('base_kv',case.get('base_kv',1.)))
        record.update(provenance=case['provenance'],fit_options=options,
            observed_history=observations_record(observed['history']),
            history_meter=meter_to_record(observed['history_meter']),future_meter=meter_to_record(observed['future_meter']),
            timestamps=observed['timestamps'],
            truth_evaluation_only=dict(supports=case['truth']['candidate'].supports,
                r=case['truth']['r'],x=case['truth']['x']))
        report('started')
        planning,bank,history,domains=selector.select(observed,options,config,
            prepared['forecast_configs'],seed=spec['physical_seed']+100,report=report)
        record['planning']=compact_fits(planning);record['passive_pre']=planning['passive_pre']
        record['candidate_structures']=structures(bank)
        report('selection_sealed',dict(methods=list(planning['plans']),future_seen=False))
        record['evaluation']=evaluate(case,spec,observed,planning,bank,history,config,options,
            spec['physical_seed']+200,report)
        record['failures'].update(planning['failures'])
        record['status']='complete' if not record['failures'] and all(not a['failures']
            for a in record['evaluation']['actions'].values()) else 'complete_with_retained_failures'
    except Exception as exc:
        record['status']='failed';record['failures']['exception']=dict(
            message=repr(exc),traceback=traceback.format_exc())
    record['seconds']=perf_counter()-began
    record['source_unchanged']=source_hashes()==sealed['source_sha256']
    write(path,record);report('saved',dict(status=record['status'],path=str(path)))
    return dict(spec=spec,path=str(path),status=record['status'],seconds=record['seconds'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/reference_v1')
    p.add_argument('--prepared',type=Path,default=ROOT/'outputs/inputs_v2/prepared.json.gz')
    p.add_argument('--phase',choices=['development','calibration','evaluation'],default='development')
    p.add_argument('--cluster',type=int,default=0);p.add_argument('--network',choices=CONFIG['network_names'],default='cigre_residential')
    p.add_argument('--source',choices=CONFIG['source_modes'],default='none')
    p.add_argument('--development-unfrozen',action='store_true')
    a=p.parse_args();prepared=read(a.prepared)
    print(serial(run(specification(a.phase,a.cluster,a.network,a.source,prepared),a.output,a.prepared,
        freeze_policy=not a.development_unfrozen)),flush=True)


if __name__=='__main__':main()
