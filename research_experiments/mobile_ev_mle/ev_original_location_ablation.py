"""Fixed-waveform electrical-location ablation of the frozen 12-history study.

There is no selector or action optimization here. Every saved charge_mobile
and v2g_mobile action is remapped to site 0 with its total P(t) unchanged.
Original complete fits are reused; only genuinely new remapped meters are
jointly refitted. This electrical comparison does not assert equal battery
service after removing the original driving loss.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy
from . import likelihood as lk
from . import gradient_price_study as io
from .information_context import meter_from_record

HERE=Path(__file__).resolve().parent
METHODS=('charge_mobile','v2g_mobile')
CHANNELS=('p','q','v','master_p','master_q','source_p','source_q')
PROTOCOL='ev_original_location_ablation_protocol.md'
SETTINGS=dict(version='electrical_location_ablation_v1',methods=list(METHODS),destination=0,
    mapping='u_remap[t,0]=sum_j u_original[t,j], all other locations zero',
    EV_reactive_power=0.,all_histories_and_methods_required=True,reselect_actions=False,
    observation_error='same saved future standard-normal draw and saved future meter model',
    noisy_fit_seed_offset=802,noiseless_fit_seed_offset=1202,
    voltage_limits=[.9,1.1],cache='exact complete power arrays, with signed zero normalized',
    original_fits='reuse saved complete/failed records, do not repair original results',
    service_scope='same grid waveform and energy; electrical ablation, not equal terminal battery service')


def read_json(path):
    path=Path(path);raw=path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix=='.gz' else raw)


def serial(value):
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(serial(value),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    raw=gzip.compress(raw,mtime=0) if path.suffix=='.gz' else raw
    temporary=path.with_name(path.name+'.tmp');temporary.write_bytes(raw);os.replace(temporary,path)


def digest(value):
    return hashlib.sha256(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remap_to_site0(power):
    power=np.asarray(power,float)
    if power.ndim!=2 or power.shape[1]!=3 or not len(power) or not np.isfinite(power).all():
        raise ValueError('finite time by three-site saved action required')
    result=np.zeros_like(power);result[:,0]=power.sum(axis=1)
    return result


def curve_key(power):
    power=np.asarray(power,dtype='<f8').copy();power[power==0]=0.
    return hashlib.sha256(power.tobytes()).hexdigest()


def observation(physical,meter,white):
    values={}
    for key in CHANNELS:
        sigma=getattr(meter,key+'_sigma',None)
        if key not in physical or physical[key] is None or sigma is None:values[key]=None;continue
        values[key]=np.asarray(physical[key],float)+np.asarray(sigma)*np.asarray(white[key])
        if key in ('p','q','v'):values[key]+=np.asarray(getattr(meter,'bias_'+key))
    return lk.Observations(**values,root_v=np.asarray(physical['root_v']))


def observations_from_record(record):
    return lk.Observations(**{k:None if v is None else np.asarray(v,float) for k,v in record.items()})


def candidate_bank(trial):
    return [lk.RadialCandidate.from_supports(r['name'],3,r['supports']) for r in trial['proposal']['bank']]


def true_name(trial,bank):
    def family(supports):return tuple(sorted(tuple(sorted(s)) for s in supports if 1<len(s)<3))
    wanted=family(trial['truth']['family']);found=[c.name for c in bank if family(c.supports)==wanted]
    return found[0] if len(found)==1 else None


def fit_options(trial):
    config=trial['config'];bounds=config['asset_bounds'] if trial['domain']=='asset' else dict(r_bounds=(1e-6,.1),x_bounds=(1e-6,.1))
    return {**config['fit'],**bounds}


def fit_summary(records,failures,names,true):
    byname={r['name']:r for r in records};complete=(not failures and len(records)==len(names) and set(byname)==set(names)
        and all(r.get('success') and math.isfinite(float(r['loss'])) for r in records))
    result=dict(success=complete,required_candidates=len(names),returned_candidates=len(records),
        candidate_miss=true is None,M=None,half_M=None,correct_min_loss=None,nearest_wrong=None,winner=None)
    if complete and true is not None:
        losses={n:float(byname[n]['loss']) for n in names};wrong=min((n for n in names if n!=true),key=lambda n:(losses[n],n))
        winner=min(names,key=lambda n:(losses[n],n));M=losses[wrong]-losses[true]
        result.update(M=M,half_M=.5*M,correct_min_loss=losses[true]<=min(losses.values())+1e-8,
            nearest_wrong=wrong,winner=winner,losses=losses)
    return result


def channel_costs(records,names,future_observations,future_meter,H,historical_fits):
    """Preserve all four candidates; absent future fits have explicit None costs."""
    byname={r['name']:r for r in records};out=[]
    for name in names:
        fit=byname.get(name);entry=dict(candidate=name,success=bool(fit and fit.get('success')),
            joint_SSE=None,historical_SSE=None,historical_penalty=None,future_channel_SSE=None)
        if fit is not None and (fit.get('loss') is None or not math.isfinite(float(fit['loss']))):
            entry.update(success=False,reason='nonfinite_joint_cost');out.append(entry);continue
        if fit is not None and future_observations is not None:
            costs={}
            for key in CHANNELS:
                observed=getattr(future_observations,key)
                if observed is None:continue
                predicted=fit.get('predicted',{}).get(key)
                if predicted is None and key in ('source_p','source_q'):predicted=fit.get(key)
                if predicted is None:raise ValueError('saved prediction channel missing: '+name+':'+key)
                residual=(np.asarray(predicted,float)[H:]-observed)/np.asarray(getattr(future_meter,key+'_sigma'))
                costs[key]=float(np.sum(residual*residual))
            total=float(fit['loss']);historical=total-sum(costs.values());old=historical_fits.get(name)
            entry.update(joint_SSE=total,historical_SSE=historical,
                historical_penalty=None if old is None else historical-float(old['loss']),future_channel_SSE=costs)
        out.append(entry)
    return out


def resource_scope(trial,method,power):
    selection=trial['selection'];record=selection['methods'][method];accepted=(record.get('guard') or {}).get('accepted')
    operation=record.get('operation') if accepted else selection.get('constant_operation')
    device=trial['config']['device'];dt=trial['config']['dt_hours'];net=np.asarray(power,float).sum(axis=1)
    ideal_terminal=device['initial_energy_kwh']+dt*float(np.sum(device['eta_charge']*np.maximum(net,0.)
        -np.maximum(-net,0.)/device['eta_discharge']))
    return dict(original_executed_operation=operation,original_raw_planned_operation=record.get('operation'),
        original_guard_accepted=accepted,grid_net_kwh=dt*float(net.sum()),
        grid_throughput_kwh=dt*float(np.abs(net).sum()),same_grid_waveform=True,
        remap_hypothetical_terminal_without_driving_kwh=ideal_terminal,
        remap_terminal_excess_over_original_target_kwh=ideal_terminal-device['terminal_energy_kwh'],
        remap_has_no_operation_feasibility_or_equal_service_claim=True)


def meter_noise_audit(original,remap,trial,fm):
    H=trial['config']['history'];errors=[];records=[]
    for name,variant in (('original',original),('remap',remap)):
        physical=variant.get('physical') or {}
        for kind in ('noisy','noiseless'):
            obs=variant.get(kind,{}).get('observations') or {};channel_errors={}
            for key in CHANNELS:
                if obs.get(key) is None or physical.get(key) is None:continue
                expected=np.asarray(trial['truth']['white'][key])[H:] if kind=='noisy' else 0.
                bias=getattr(fm,'bias_'+key) if key in ('p','q','v') else 0.
                actual=(np.asarray(obs[key])-np.asarray(physical[key])-bias)/np.asarray(getattr(fm,key+'_sigma'))
                error=float(np.max(abs(actual-expected),initial=0.));errors.append(error);channel_errors[key]=error
            records.append(dict(variant=name,kind=kind,channel_max_abs_white_error=channel_errors,
                complete=all(key in channel_errors for key in ('p','q','v','master_p','master_q'))))
    complete=all(r['complete'] for r in records)
    return dict(complete=complete,passed=complete and max(errors,default=0.)<=1e-8,
        max_abs_white_error=max(errors,default=None),tolerance=1e-8,records=records)


def variant_from_saved(trial,row,bank,hm,fm):
    H=trial['config']['history'];names=[c.name for c in bank];true=true_name(trial,bank)
    old={r['name']:r for r in trial['historical_fits']};out={}
    for kind in ('noisy','noiseless'):
        saved=row if kind=='noisy' else row.get('noiseless_AC_refit') or {}
        fits=saved.get('fits',[]);fail=saved.get('failures',{})
        obsrecord=saved.get('observations') if kind=='noisy' else saved.get('future_mean_observations')
        obs=observations_from_record(obsrecord) if obsrecord else None
        summary=fit_summary(fits,fail,names,true)
        if obs is None:summary.update(success=False,M=None,half_M=None,correct_min_loss=None,nearest_wrong=None,winner=None)
        out[kind]=dict(**summary,fits=fits,failures=fail,observations=obsrecord,
            candidate_costs=channel_costs(fits,names,obs,fm,H,old),cache_hit=True,
            cache_source='frozen_original_method',new_fit_calls=0,new_fit_AC_calls=0,
            inherited_parent_cost=saved.get('cost'),status='reused_complete' if summary['success'] else 'reused_failure')
    out.update(power_kw=np.asarray(row['power_kw'],float),physical=row.get('physical'),
        new_forward_AC_calls=0,original_parent_status=row.get('status'),curve_sha256=curve_key(row['power_kw']))
    return out


def evaluate_remap(trial,power,bank,history,hfits,fm,jm,cache,counts):
    key=curve_key(power)
    if key in cache:
        previous=cache[key]
        return {**previous,'new_forward_AC_calls':0,'noisy':{**previous['noisy'],'cache_hit':True,'new_fit_calls':0,'new_fit_AC_calls':0},
            'noiseless':{**previous['noiseless'],'cache_hit':True,'new_fit_calls':0,'new_fit_AC_calls':0}}
    H=trial['config']['history'];truth=trial['truth'];config=trial['config'];names=[c.name for c in bank];true=true_name(trial,bank)
    result=dict(power_kw=np.asarray(power,float),curve_sha256=key,new_forward_AC_calls=1,physical=None)
    counts['forward_AC_calls']+=1
    try:
        candidate=lk.RadialCandidate.from_supports('saved_evaluation_truth',3,truth['family'])
        physical=lk.ac_predict(candidate,truth['r'],truth['x'],np.asarray(truth['p'])[H:]+np.asarray(power)/config['base_kw'],
            np.asarray(truth['q'])[H:],np.asarray(truth['source_p'])[H:],np.asarray(truth['source_q'])[H:],np.asarray(truth['root_v'])[H:])
        result['physical']={k:physical[k] for k in ('p','q','v','master_p','master_q','ac_equation_residual','power_balance_residual')}
        if np.min(physical['v'])<SETTINGS['voltage_limits'][0] or np.max(physical['v'])>SETTINGS['voltage_limits'][1]:
            raise ValueError('remapped physical voltage outside declared evaluation range')
        for kind,offset in (('noisy',SETTINGS['noisy_fit_seed_offset']),('noiseless',SETTINGS['noiseless_fit_seed_offset'])):
            white={k:np.asarray(v)[H:] if kind=='noisy' else np.zeros_like(np.asarray(v)[H:]) for k,v in truth['white'].items()}
            obs=observation(physical,fm,white);counts['candidate_fit_calls']+=len(bank)
            fitted,failed,cost=io.fit_all_candidates(bank,lk.concat_observations(history,obs),jm,
                initial=hfits,seed=trial['seeds']['policy']+offset,fit_options=fit_options(trial))
            counts['fit_AC_calls']+=cost['ac_calls'];records=[io.fit_record(c,fitted[c.name]) for c in bank if c.name in fitted]
            result[kind]=dict(**fit_summary(records,failed,names,true),fits=records,failures=failed,
                observations=io.observations_record(obs),candidate_costs=channel_costs(records,names,obs,fm,H,hfits),
                cache_hit=False,cache_source=None,new_fit_calls=cost['fit_calls'],new_fit_AC_calls=cost['ac_calls'],cost=cost)
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        result['error']=type(error).__name__+':'+str(error)
        for kind in ('noisy','noiseless'):
            if kind not in result:result[kind]=dict(**fit_summary([],{'__physical_or_fit__':result['error']},names,true),
                fits=[],failures={'__physical_or_fit__':result['error']},observations=None,
                candidate_costs=channel_costs([],names,None,fm,H,hfits),cache_hit=False,cache_source=None,
                new_fit_calls=0,new_fit_AC_calls=0,status='physical_or_fit_exception')
    cache[key]=result
    return result


def process_history(trial):
    """All two predeclared methods, regardless of truth or original performance."""
    began=perf_counter();bank=candidate_bank(trial);history=observations_from_record(trial['history'])
    hm,fm,jm=(meter_from_record(trial['measurement_models'][key]) for key in ('hm','fm','jm'))
    hfits={r['name']:r for r in trial['historical_fits']};saved={r['method']:r for r in trial['rows']}
    counts=dict(forward_AC_calls=0,candidate_fit_calls=0,fit_AC_calls=0);cache={};conditions=[]
    # Populate saved original actions before any new fits, allowing exact reuse
    # even if a remapped curve equals the other method's original curve.
    originals={}
    for method in METHODS:
        row=saved.get(method)
        if row is not None and row.get('power_kw') is not None:
            originals[method]=variant_from_saved(trial,row,bank,hm,fm);cache.setdefault(curve_key(row['power_kw']),originals[method])
    for method in METHODS:
        original=originals.get(method);condition=dict(seed=trial['seed'],method=method,status='missing_original_action',
            original=original,remap=None,delta_M=None,delta_half_M=None,delta_noiseless_M=None,
            independent_unit='one parent history; two methods and paired locations are not extra histories')
        if original is not None:
            power=np.asarray(original['power_kw'],float);remap=remap_to_site0(power)
            result=evaluate_remap(trial,remap,bank,history,hfits,fm,jm,cache,counts)
            noise_audit=meter_noise_audit(original,result,trial,fm)
            condition.update(remap=result,resource_scope=resource_scope(trial,method,power),
                waveform_max_abs_difference_kw=float(np.max(abs(power.sum(axis=1)-remap.sum(axis=1)))),
                grid_energy_difference_kwh=float(trial['config']['dt_hours']*(power.sum()-remap.sum())),
                grid_throughput_difference_kwh=float(trial['config']['dt_hours']*(np.abs(power).sum()-np.abs(remap).sum())),
                meter_noise_audit=noise_audit,
                status='complete' if original['noisy']['success'] and result['noisy']['success']
                    and original['noiseless']['success'] and result['noiseless']['success'] and noise_audit['passed'] else 'fit_or_physical_or_noise_failure')
            for kind,key in (('noisy','delta_M'),('noiseless','delta_noiseless_M')):
                left,right=original[kind]['M'],result[kind]['M']
                condition[key]=None if left is None or right is None else left-right
            condition['delta_half_M']=None if condition['delta_M'] is None else .5*condition['delta_M']
        conditions.append(condition)
    shared={key:np.asarray(trial['truth'][key])[trial['config']['history']:] for key in ('p','q','source_p','source_q','root_v')}
    shared.update(r=trial['truth']['r'],x=trial['truth']['x'],family=trial['truth']['family'],
        future_white={key:np.asarray(value)[trial['config']['history']:] for key,value in trial['truth']['white'].items()})
    return dict(seed=trial['seed'],family=trial['truth']['family'],conditions=conditions,counts=counts,
        shared_future_inputs=shared,shared_future_input_sha256=digest(shared),
        parent_historical_failures=trial['historical_failures'],parent_selection_failures=trial['selection']['failures'],
        seconds=perf_counter()-began,new_selector_runs=0)


def source_paths(parent):
    root=HERE.parents[1].resolve();paths={}
    for name in parent['source_sha256']:
        path=((root/name) if name.startswith('rnj_wzzt_core/') else HERE/name).resolve();path.relative_to(root);paths[name]=path
    for name in (Path(__file__).name,PROTOCOL):paths[name]=HERE/name
    return paths


def source_hashes(parent):return {name:file_digest(path) for name,path in source_paths(parent).items()}


def prepare_manifest(input_directory,output_directory,parent_review=None):
    input_directory=Path(input_directory).resolve();output_directory=Path(output_directory).resolve()
    if input_directory==output_directory:raise ValueError('new evidence directory required')
    parent=read_json(input_directory/'freeze.json');planned=parent['seeds']
    if parent.get('phase')!='evaluation' or len(planned)!=12 or len(set(planned))!=12 or planned!=parent['config']['evaluation_seeds']:
        raise ValueError('complete predeclared 12-history evaluation required')
    paths=[input_directory/f'trial_{seed}.json.gz' for seed in planned]
    if not all(path.is_file() for path in paths):raise ValueError('wait for all 12 parent histories before freezing the ablation')
    trial_hashes=[dict(name=p.name,sha256=file_digest(p)) for p in paths]
    for seed,path in zip(planned,paths):
        trial=read_json(path)
        if (trial['seed']!=seed or trial['config']!=parent['config'] or trial['source_sha256']!=parent['source_sha256']
                or trial.get('phase')!='evaluation' or trial.get('domain')!=parent.get('domain')):
            raise ValueError('parent trial/freeze identity differs: '+path.name)
    hashes=source_hashes(parent)
    if any(hashes[name]!=value for name,value in parent['source_sha256'].items()):raise ValueError('parent frozen scientific source changed')
    review=read_json(parent_review) if parent_review else None
    if review is not None:
        review_hashes={r['name']:r['sha256'] for r in review.get('trial_hashes',[])}
        audited_source=review.get('source_audit',{})
        reviewed_source={r['name']:r['sha256'] for r in audited_source.get('rows',[])}
        if (not review.get('success') or review.get('phase')!='evaluation' or review.get('domain')!=parent.get('domain')
                or review.get('audited_histories')!=12 or review.get('available_independent_histories')!=12 or review.get('missing_histories')
                or review_hashes!={r['name']:r['sha256'] for r in trial_hashes}
                or not audited_source.get('success') or not audited_source.get('current_source_matches')
                or reviewed_source!=parent['source_sha256']):
            raise ValueError('successful independent parent review of these exact 12 trial files required')
    freeze=dict(settings=SETTINGS,parent_directory=str(input_directory),parent_freeze_sha256=file_digest(input_directory/'freeze.json'),
        parent_source_sha256=parent['source_sha256'],source_sha256=hashes,parent_trial_hashes=trial_hashes,
        independent_parent_review_required_for_execution=True,seeds=planned,
        planned_conditions=24,condition_order=[dict(seed=s,method=m) for s in planned for m in METHODS],
        parameter_sha256=digest(dict(settings=SETTINGS,parent_config=parent['config'])),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        parent_strategy_unchanged=True,action_selection_uses_truth=False,formal_calibration_histories=0)
    output_directory.mkdir(parents=True,exist_ok=True);fp=output_directory/'freeze.json'
    if fp.exists():
        if read_json(fp)!=serial(freeze):raise ValueError('ablation freeze differs; use a new output directory')
    else:
        snapshot={}
        for name,path in source_paths(parent).items():
            raw=path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=hashes[name]:raise ValueError('source changed during ablation snapshot')
            snapshot[name]=dict(sha256=hashes[name],bytes_base64=base64.b64encode(raw).decode())
        write_json(output_directory/'source_snapshot.json.gz',snapshot);write_json(fp,freeze)
    if parent_review:
        write_json(output_directory/'parent_review_validation.json',dict(path=str(Path(parent_review).resolve()),
            sha256=file_digest(parent_review),success=True,covered_histories=12,parent_trial_hashes=trial_hashes))
    return parent,freeze,paths


def tabulate(results):
    rows=[];cost_rows=[]
    for history in results:
        family='star' if not history['family'] else 'cherry_'+''.join(map(str,history['family'][0]))
        for c in history['conditions']:
            scope=c.get('resource_scope',{});op=scope.get('original_executed_operation') or {}
            row=dict(seed=history['seed'],family=family,method=c['method'],status=c['status'],
                delta_M=c['delta_M'],delta_half_M=c['delta_half_M'],delta_noiseless_M=c['delta_noiseless_M'],
                grid_energy_difference_kwh=c.get('grid_energy_difference_kwh'),original_grid_kwh=scope.get('grid_net_kwh'),
                grid_throughput_difference_kwh=c.get('grid_throughput_difference_kwh'),
                same_noise_verified=(c.get('meter_noise_audit') or {}).get('passed'),
                original_throughput_kwh=scope.get('grid_throughput_kwh'),original_drive_kwh=op.get('travel_energy_kwh'),
                original_worker_minutes=op.get('driver_cost'),remap_terminal_excess_kwh=scope.get('remap_terminal_excess_over_original_target_kwh'))
            for variant in ('original','remap'):
                record=c.get(variant)
                for kind in ('noisy','noiseless'):
                    fit=(record or {}).get(kind,{})
                    for key in ('M','half_M','correct_min_loss','nearest_wrong','success','cache_hit'):
                        row[variant+'_'+kind+'_'+key]=fit.get(key)
                    for detail in fit.get('candidate_costs',[]):
                        item=dict(seed=history['seed'],family=family,method=c['method'],variant=variant,kind=kind,**detail)
                        channels=item.pop('future_channel_SSE') or {}
                        for channel in CHANNELS:item['future_'+channel+'_SSE']=channels.get(channel)
                        cost_rows.append(item)
            rows.append(row)
    return rows,cost_rows


def save_summary(output_directory,freeze):
    output_directory=Path(output_directory);results=[read_json(output_directory/f'history_{seed}.json.gz')
        for seed in freeze['seeds'] if (output_directory/f'history_{seed}.json.gz').is_file()]
    rows,cost_rows=tabulate(results);summary=dict(planned_histories=12,available_histories=len(results),planned_conditions=24,
        available_conditions=len(rows),complete_conditions=sum(r['status']=='complete' for r in rows),
        status_counts=dict(Counter(r['status'] for r in rows)),counts={key:sum(r['counts'][key] for r in results)
            for key in ('forward_AC_calls','candidate_fit_calls','fit_AC_calls')},rows=rows,
        statistical_unit='12 independent parent histories, four fixed topology strata with three histories each',
        scope=SETTINGS['service_scope'],formal_calibration_histories=0,parameter_sha256=freeze['parameter_sha256'])
    for name,data in (('paired_locations.csv',rows),('candidate_channel_costs.csv',cost_rows)):
        columns=list(dict.fromkeys(key for row in data for key in row))
        with (output_directory/name).open('w',encoding='utf-8-sig',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(data)
    write_json(output_directory/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--parent-review',type=Path);p.add_argument('--execute',action='store_true')
    args=p.parse_args();parent,freeze,paths=prepare_manifest(args.input,args.output,args.parent_review)
    if not args.execute:
        print(json.dumps(dict(prepared=True,new_AC_fits=0,planned_conditions=24,parameter_sha256=freeze['parameter_sha256'])));return
    if args.parent_review is None:raise ValueError('execute requires successful independent parent review and prior protocol review')
    for seed,path,identity in zip(freeze['seeds'],paths,freeze['parent_trial_hashes']):
        target=args.output/f'history_{seed}.json.gz'
        if target.exists():
            saved=read_json(target)
            if saved.get('parameter_sha256')!=freeze['parameter_sha256'] or saved.get('source_sha256')!=freeze['source_sha256']:
                raise ValueError('saved ablation history/freeze mismatch')
            continue
        if source_hashes(parent)!=freeze['source_sha256'] or file_digest(path)!=identity['sha256']:raise ValueError('frozen input or source changed before ablation')
        trial=read_json(path);result=process_history(trial)
        after=source_hashes(parent)
        if after!=freeze['source_sha256'] or file_digest(path)!=identity['sha256']:
            result.update(source_after=after,invalid_source=True);write_json(args.output/f'invalid_history_{seed}.json.gz',result)
            raise ValueError('source/input changed during ablation; invalid result retained')
        result.update(parameter_sha256=freeze['parameter_sha256'],source_sha256=freeze['source_sha256'],parent_trial_sha256=identity['sha256'])
        write_json(target,result);summary=save_summary(args.output,freeze)
        write_json(args.output/'checkpoint.json',dict(available_histories=summary['available_histories'],
            available_conditions=summary['available_conditions'],planned_conditions=24,research_complete=False,counts=summary['counts']))
        print(json.dumps(dict(seed=seed,seconds=result['seconds'],counts=result['counts'],statuses=[c['status'] for c in result['conditions']])) ,flush=True)


if __name__=='__main__':main()
