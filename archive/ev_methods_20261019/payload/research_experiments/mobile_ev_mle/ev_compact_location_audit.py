"""Frozen-action electrical location ablation; selection is never rerun.

Reuses the stage-16 physical generator, paired measurement errors and joint
AC fitter. Compact config and inherited-operation ledgers are adapted here.
This process has no calibrated rejection rule and no equal-service claim.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy
from . import ev_original_location_ablation as old
from .ev_mobile_service_domain import MobileServiceConfig

HERE = Path(__file__).resolve().parent
METHODS = ('refined_v2g_mobile','capped_v2g_mobile')
MATH = 'ev_compact_location_audit_math.md'
SETTINGS = dict(version='compact_electrical_location_v1',methods=list(METHODS),destination=0,
    waveform='u_remap[t,0]=sum_j u_selected[t,j]; other sites zero',
    reactive_EV_power=0.,reselect=False,choose_using_future=False,
    errors='same saved future white draws, calibrated biases and meter sigmas',
    original_fits='reuse attained parent fits including failures',
    refit='all four candidates, shared H+N lines and fixed source interface',
    noisy_fit_seed_offset=802,noiseless_fit_seed_offset=1202,
    service='electrical isolation only; canceled drive changes terminal battery energy',
    root_meter='recompute from full AC; root power and losses are not held fixed',
    statistical_threshold=None,formal_rejection_calibration=False)


def compact_view(trial):
    """Shallow read-only compatibility view; never modify the parent object."""
    physical = trial.get('config',{}).get('physical')
    if not isinstance(physical,dict):
        raise ValueError('compact config.physical required')
    if trial.get('source_interface',1) != 1:
        raise ValueError('this saved simulator uses source interface 1')
    result = dict(trial,config=physical)
    if len(trial.get('proposal',{}).get('bank',[])) != 4:
        raise ValueError('all four saved topology candidates required')
    return result


def resource_audit(trial, method, power):
    """Replay signed battery energy without solving a new route problem."""
    config = trial['config']['physical'];device = config['device']
    record = trial['selection']['methods'].get(method) or {}
    operation = record.get('operation') or {}
    power = np.asarray(power,float);net = power.sum(axis=1);dt=float(config['dt_hours'])
    recorded = np.asarray(operation.get('u_kw',[]),float)
    match = recorded.shape==power.shape and np.isfinite(recorded).all() and np.max(abs(recorded-power),initial=0.)<=1e-7
    eta_c,eta_d=float(device['eta_charge']),float(device['eta_discharge'])
    increments=dt*(eta_c*np.maximum(net,0.)-np.maximum(-net,0.)/eta_d)
    energy=np.r_[device['initial_energy_kwh'],device['initial_energy_kwh']+np.cumsum(increments)]
    old_energy=operation.get('energy_kwh') or []
    old_final=float(old_energy[-1]) if old_energy else None
    drive=operation.get('travel_energy_kwh')
    excess=float(energy[-1]-device['terminal_energy_kwh'])
    actual_difference=float(energy[-1]-old_final) if old_final is not None else None
    command_device=(trial.get('operations_config') or {}).get('device') or {}
    power_limit=float(command_device.get('power_kw',trial['config'].get('command_power_kw',device['power_kw'])))
    defaults=MobileServiceConfig();lo=defaults.soc_min*device['capacity_kwh'];hi=defaults.soc_max*device['capacity_kwh']
    single_site=bool(np.max(np.count_nonzero(abs(power)>1e-7,axis=1),initial=0)<=1)
    power_ok=bool(np.max(abs(net),initial=0.)<=power_limit+1e-7)
    soc_ok=bool(np.min(energy)>=lo-1e-7 and np.max(energy)<=hi+1e-7)
    throughput=dt*float(np.abs(net).sum())
    throughput_ok=throughput<=float(device['throughput_kwh'])+1e-7
    terminal_ok=abs(excess)<=1e-7
    return dict(original_executed_operation=operation,original_curve_matches_operation=bool(match),
        original_operation_feasible=bool(operation.get('feasible') and match),
        original_selected_origin=record.get('selected_origin'),
        original_guard_accepted=(record.get('guard') or {}).get('accepted'),
        original_drive_kwh=drive,original_terminal_energy_kwh=old_final,
        original_worker_cost=operation.get('driver_cost'),original_worker_cost_units=operation.get('cost_units'),
        grid_net_kwh=dt*float(net.sum()),grid_throughput_kwh=throughput,single_site_per_slot=single_site,
        remap_no_drive_energy_kwh=energy,
        remap_hypothetical_terminal_without_driving_kwh=float(energy[-1]),
        remap_terminal_excess_over_original_target_kwh=excess,
        remap_minus_original_terminal_kwh=actual_difference,
        terminal_minus_drive_identity_error_kwh=None if actual_difference is None or drive is None else actual_difference-float(drive),
        remap_point_checks=dict(command_power=power_ok,SOC=soc_ok,throughput=throughput_ok,
            no_drive_initial_site_path=True,initial_site0_access_all_slots=True,
            original_terminal_service=terminal_ok,power_limit_kw=power_limit,SOC_bounds_kwh=[lo,hi]),
        remap_same_terminal_service_feasible=bool(power_ok and soc_ok and throughput_ok and terminal_ok),
        public_operation_scope='saved parent domain has initial connected site0 and access at site0 in every slot',
        equal_service_comparison=False,no_new_route_optimization=True)


def enrich_variant(variant,trial,bank):
    """All twelve cost differences and true-candidate impedance errors."""
    names=[c.name for c in bank];true=old.true_name(trial,bank)
    for kind in ('noisy','noiseless'):
        state=variant[kind];fits={f['name']:f for f in state.get('fits') or []}
        losses={name:float(f['loss']) for name,f in fits.items() if f.get('success') and f.get('loss') is not None}
        state['pairwise_half_SSE_differences']=[dict(a=a,b=b,
            value=.5*(losses[b]-losses[a]) if a in losses and b in losses else None)
            for a in names for b in names if a!=b]
        fit=fits.get(true) or {}
        for key in ('r','x'):
            estimate=fit.get(key);truth=trial['truth'][key]
            value=None
            if fit.get('success') and estimate is not None and np.asarray(estimate).shape==np.asarray(truth).shape:
                value=float(np.linalg.norm(np.asarray(estimate)-truth)/np.linalg.norm(truth))
            state['true_candidate_'+key.upper()+'_relative']=value
    return variant


def process_history(trial):
    began=perf_counter();view=compact_view(trial);bank=old.candidate_bank(view)
    history=old.observations_from_record(view['history'])
    hm,fm,jm=(old.meter_from_record(view['measurement_models'][key]) for key in ('hm','fm','jm'))
    hfits={r['name']:r for r in view['historical_fits']};rows={r['method']:r for r in view['rows']}
    cache={};counts=dict(forward_AC_calls=0,candidate_fit_calls=0,fit_AC_calls=0)
    originals={};conditions=[]
    for method in METHODS:
        row=rows.get(method)
        if row is not None and row.get('power_kw') is not None:
            result=old.variant_from_saved(view,row,bank,hm,fm)
            result['cache_origin']=dict(kind='saved_parent_action',method=method)
            originals[method]=enrich_variant(result,view,bank)
            cache.setdefault(old.curve_key(row['power_kw']),originals[method])
    for method in METHODS:
        original=originals.get(method)
        condition=dict(seed=trial['seed'],method=method,status='missing_original_action',original=original,
            remap=None,delta_M=None,delta_half_M=None,delta_noiseless_M=None,
            statistical_threshold_applied=False,calibrated_rejection_decision=None)
        if original is not None:
            power=np.asarray(original['power_kw'],float);remapped=old.remap_to_site0(power)
            stationary=bool(np.array_equal(power,remapped))
            if stationary:
                result={**original,'new_forward_AC_calls':0,
                    **{kind:{**original[kind],'cache_hit':True,'new_fit_calls':0,'new_fit_AC_calls':0}
                       for kind in ('noisy','noiseless')}}
            else:
                result=old.evaluate_remap(view,remapped,bank,history,hfits,fm,jm,cache,counts)
                result.setdefault('cache_origin',dict(kind='new_remapped_curve',first_method=method))
                if result.get('physical') is not None:
                    result['physical']['root_v']=np.asarray(view['truth']['root_v'])[view['config']['history']:]
            result=enrich_variant(result,view,bank)
            noise=old.meter_noise_audit(original,result,view,fm)
            scope=resource_audit(trial,method,power)
            complete=all(v[k]['success'] for v in (original,result) for k in ('noisy','noiseless')) and noise['passed']
            condition.update(remap=result,resource_scope=scope,meter_noise_audit=noise,
                stationary_exact_reuse=stationary,
                waveform_max_abs_difference_kw=float(np.max(abs(power.sum(axis=1)-remapped.sum(axis=1)))),
                grid_energy_difference_kwh=float(view['config']['dt_hours']*(power.sum()-remapped.sum())),
                grid_throughput_difference_kwh=float(view['config']['dt_hours']*(np.abs(power).sum()-np.abs(remapped).sum())),
                status='complete' if complete and scope['original_operation_feasible'] and scope['single_site_per_slot']
                    else 'fit_or_physical_or_noise_or_original_operation_failure')
            for kind,key in (('noisy','delta_M'),('noiseless','delta_noiseless_M')):
                left,right=original[kind]['M'],result[kind]['M']
                condition[key]=None if left is None or right is None else left-right
            condition['delta_half_M']=None if condition['delta_M'] is None else condition['delta_M']/2
        conditions.append(condition)
    shared={key:np.asarray(view['truth'][key])[view['config']['history']:] for key in ('p','q','source_p','source_q','root_v')}
    shared.update(r=view['truth']['r'],x=view['truth']['x'],family=view['truth']['family'],
        future_white={key:np.asarray(value)[view['config']['history']:] for key,value in view['truth']['white'].items()})
    actual_fit_calls=sum(c['remap'][k].get('new_fit_calls',0) for c in conditions if c['remap'] for k in ('noisy','noiseless'))
    counts['actual_returned_fit_calls']=actual_fit_calls
    return dict(seed=trial['seed'],phase=trial['phase'],family=view['truth']['family'],conditions=conditions,
        counts=counts,shared_future_inputs=shared,shared_future_input_sha256=old.digest(shared),
        parent_historical_failures=view['historical_failures'],parent_selection_failures=view['selection'].get('failures'),
        source_interface=1,new_selector_runs=0,statistical_threshold_applied=False,seconds=perf_counter()-began)


def source_paths(parent):
    root=HERE.parents[1].resolve();paths={}
    for name in parent['source_sha256']:
        path=((root/name) if name.startswith('rnj_wzzt_core/') else HERE/name).resolve()
        path.relative_to(root);paths[name]=path
    for name in (Path(old.__file__).name,Path(__file__).name,MATH):paths[name]=HERE/name
    return paths


def source_hashes(parent):
    return {name:old.file_digest(path) for name,path in source_paths(parent).items()}


def prepare_manifest(directory,output,*,parent_review=None):
    directory,output=Path(directory).resolve(),Path(output).resolve()
    if directory==output:raise ValueError('new output directory required')
    parent=old.read_json(directory/'freeze.json');phase=parent.get('phase');seeds=parent.get('seeds',[])
    expected=4 if phase=='development' else 12 if phase=='evaluation' else None
    if expected is None or len(seeds)!=expected or len(set(seeds))!=expected or seeds!=parent['config'][phase+'_seeds']:
        raise ValueError('complete predeclared 4H development or 12H evaluation required; calibration is not an ablation input')
    paths=[];identities=[];raw_hashes={}
    for seed in seeds:
        raw=directory/f'trial_{seed}.json.gz';slim=directory/f'trial_{seed}.slim.json.gz'
        path=raw if raw.is_file() else slim
        if not path.is_file():raise ValueError('all parent histories required before freezing')
        trial=old.read_json(path)
        if (trial.get('seed')!=seed or trial.get('phase')!=phase or trial.get('policy_id')!=parent.get('policy_id')
                or trial.get('config')!=parent['config'] or trial.get('source_sha256')!=parent['source_sha256']):
            raise ValueError('parent history/freeze differs')
        if path==slim:
            delivery=trial.get('delivery') or {};raw_name=Path(delivery.get('source_raw_path','')).name
            raw_sha=delivery.get('source_raw_sha256')
            if raw_name!=raw.name or not raw_sha:raise ValueError('slim must retain exact parent raw identity')
        else:raw_name,raw_sha=raw.name,old.file_digest(raw)
        raw_hashes[raw_name]=raw_sha
        paths.append(path);identities.append(dict(name=path.name,sha256=old.file_digest(path),raw_name=raw_name,raw_sha256=raw_sha))
    hashes=source_hashes(parent)
    if any(hashes[name]!=value for name,value in parent['source_sha256'].items()):
        raise ValueError('parent frozen scientific source changed')
    review=old.read_json(parent_review) if parent_review else None
    if phase=='evaluation' and review is None:raise ValueError('formal evaluation ablation requires an independent parent review')
    if review is not None and (not review.get('success') or review.get('policy_id')!=parent.get('policy_id')
            or set(review.get('seeds',[]))!=set(seeds) or review.get('trial_sha256')!=raw_hashes):
        raise ValueError('independent parent review must match exact histories and policy')
    freeze=dict(schema='ev_compact_location_freeze_v1',settings=SETTINGS,phase=phase,seeds=seeds,
        planned_histories=len(seeds),planned_conditions=len(seeds)*len(METHODS),parent_policy_id=parent['policy_id'],
        parent_freeze_sha256=old.file_digest(directory/'freeze.json'),parent_directory=str(directory),
        parent_trial_hashes=identities,parent_source_sha256=parent['source_sha256'],source_sha256=hashes,
        parent_review_sha256=None if parent_review is None else old.file_digest(parent_review),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        independent_unit='one parent H; each method has its own paired-location comparison',
        formal_calibration_histories=0,parameter_sha256=old.digest(dict(settings=SETTINGS,parent_config=parent['config'])))
    output.mkdir(parents=True,exist_ok=True);fp=output/'freeze.json'
    if fp.exists():
        if old.read_json(fp)!=old.serial(freeze):raise ValueError('ablation freeze differs; new output directory required')
    else:
        snapshot={}
        for name,path in source_paths(parent).items():
            data=path.read_bytes()
            if old.file_digest(path)!=hashes[name]:raise ValueError('source changed during snapshot')
            snapshot[name]=dict(sha256=hashes[name],bytes_base64=base64.b64encode(data).decode())
        old.write_json(output/'source_snapshot.json.gz',snapshot);old.write_json(fp,freeze)
    return parent,freeze,paths


def save_summary(output,freeze):
    output=Path(output)
    histories=[old.read_json(output/f'history_{s}.json.gz') for s in freeze['seeds'] if (output/f'history_{s}.json.gz').exists()]
    rows,costs=old.tabulate(histories)
    source={(c['seed'],c['method']):c for h in histories for c in h['conditions']}
    for row in rows:
        c=source[row['seed'],row['method']];scope=c.get('resource_scope') or {}
        row.update(stationary_exact_reuse=c.get('stationary_exact_reuse'),
            original_selected_origin=scope.get('original_selected_origin'),
            original_operation_feasible=scope.get('original_operation_feasible'),
            remap_same_terminal_service_feasible=scope.get('remap_same_terminal_service_feasible'),
            original_worker_minutes=scope.get('original_worker_cost') if scope.get('original_worker_cost_units')=='worker_minutes' else None,
            calibrated_rejection_decision=None,statistical_threshold_applied=False)
        for variant in ('original','remap'):
            for kind in ('noisy','noiseless'):
                fit=(c.get(variant) or {}).get(kind,{})
                for parameter in ('R','X'):
                    row[variant+'_'+kind+'_'+parameter+'_relative']=fit.get('true_candidate_'+parameter+'_relative')
    for filename,table in (('paired_locations.csv',rows),('candidate_channel_costs.csv',costs)):
        keys=list(dict.fromkeys(k for r in table for k in r))
        with (output/filename).open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(table)
    summary=dict(phase=freeze['phase'],planned_histories=freeze['planned_histories'],available_histories=len(histories),
        planned_conditions=freeze['planned_conditions'],available_conditions=len(rows),
        complete_conditions=sum(r['status']=='complete' for r in rows),status_counts=dict(Counter(r['status'] for r in rows)),
        methods={m:dict(history_denominator=len(histories),planned_histories=freeze['planned_histories'],
            complete_pairs=sum(r['method']==m and r['status']=='complete' for r in rows),
            stationary_pairs=sum(r['method']==m and bool(r.get('stationary_exact_reuse')) for r in rows)) for m in METHODS},
        counts={key:sum(h['counts'][key] for h in histories) for key in ('forward_AC_calls','candidate_fit_calls','fit_AC_calls','actual_returned_fit_calls')},
        rows=rows,independent_unit='one H; method-location conditions are paired measurements',
        formal_calibration_histories=0,statistical_threshold_applied=False,new_selector_runs=0,
        equal_service_comparison=False,parameter_sha256=freeze['parameter_sha256'])
    old.write_json(output/'summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--parent-review',type=Path)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    parent,freeze,paths=prepare_manifest(args.input,args.output,parent_review=args.parent_review)
    if not args.execute:
        print(json.dumps(dict(prepared=True,phase=freeze['phase'],planned_conditions=freeze['planned_conditions'],new_fits=0)));return
    for path,identity,seed in zip(paths,freeze['parent_trial_hashes'],freeze['seeds']):
        target=args.output/f'history_{seed}.json.gz'
        if target.exists():
            saved=old.read_json(target)
            if saved.get('parameter_sha256')!=freeze['parameter_sha256'] or saved.get('source_sha256')!=freeze['source_sha256']:
                raise ValueError('saved location history/freeze differs')
            continue
        if old.file_digest(path)!=identity['sha256'] or source_hashes(parent)!=freeze['source_sha256']:
            raise ValueError('source or parent input changed before location audit')
        result=process_history(old.read_json(path))
        result.update(parent_trial_sha256=identity['sha256'],source_sha256=freeze['source_sha256'],
            parameter_sha256=freeze['parameter_sha256'],parent_policy_id=freeze['parent_policy_id'])
        if old.file_digest(path)!=identity['sha256'] or source_hashes(parent)!=freeze['source_sha256']:
            result['invalid_source_or_input']=True;old.write_json(args.output/f'invalid_history_{seed}.json.gz',result)
            raise ValueError('source or parent changed; invalid evidence retained')
        old.write_json(target,result);summary=save_summary(args.output,freeze)
        old.write_json(args.output/'checkpoint.json',dict(available_histories=summary['available_histories'],
            planned_histories=freeze['planned_histories'],planned_conditions=freeze['planned_conditions'],
            available_conditions=summary['available_conditions'],counts=summary['counts'],research_complete=False))
        print(json.dumps(dict(seed=seed,seconds=result['seconds'],counts=result['counts'],
            statuses=[c['status'] for c in result['conditions']])),flush=True)


if __name__=='__main__':main()
