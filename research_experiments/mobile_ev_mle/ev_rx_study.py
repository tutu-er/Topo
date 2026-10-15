"""Physical RX envelope and nearest-competitor study with fixed trained lines."""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np
import scipy

from . import ev_dynamic_study as dynamic
from . import ev_fixed_train_study as old
from . import ev_fixed_train_design as legacy
from . import ev_original_study as base
from . import gradient_price_study as io
from . import likelihood as lk
from .candidates import propose
from .ev_dynamic_loads import fit_forecast
from .ev_fixed_train_review import review_fixed_fit
from .information_context import meter_to_record
from .ev_rx_design import select_controls
from .ev_rx_fit import fit_rx_inputs

HERE=Path(__file__).resolve().parent
CONFIG=deepcopy(dynamic.CONFIG)
CONFIG.update(version='physical_RX_nearest_fixed_lines_v1',training_domain='cone',
    ratio_bounds=[.083/.642,.080/.208],mean_policy_ablation=False,diagnostics=False,
    select_modes=['none','constant',*(prefix+'_'+domain for prefix in ('linear','square','mean','nearest')
        for domain in legacy.DOMAIN_FLAGS)],
    development_seeds=list(range(992101,992105)),calibration_seeds=list(range(992301,992337)),
    evaluation_seeds=list(range(992201,992213)),paired_training_domains=['box','cone'])


def source_hashes():
    values=dynamic.source_hashes()
    for name in ('ev_rx_math.md','ev_rx_protocol.md','ev_rx_fit.py','ev_rx_precision.py',
                 'ev_rx_design.py','ev_rx_study.py'):
        values[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return values


def freeze(directory,config):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    hashes=source_hashes();identity=io.identity(config,hashes)
    record=dict(config=config,source_sha256=hashes,policy_id=identity,base_commit='87ca36f',
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    path=directory/'freeze.json'
    if path.exists():
        saved=io.read_json(path)
        if saved['policy_id']!=identity:raise ValueError('freeze differs; preserve existing directory')
        return saved
    io.write_json(path,record)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/')
        else HERE/name).read_bytes()).decode() for name in hashes}
    io.write_json(directory/'source_snapshot.json.gz',dict(source_sha256=hashes,source_base64=snapshot),compressed=True)
    return record


def train_bank(bank,obs,meter,config,seed):
    if config['training_domain']=='box':
        return old.fit_bank(bank,obs,meter,options={**config['training'],**config['source']},seed=seed)
    if config['training_domain']!='cone':raise ValueError('box or cone training required')
    fits={};records=[];errors={};reviews=[];counter={};began=perf_counter()
    for index,candidate in enumerate(bank):
        try:
            fitted=fit_rx_inputs(candidate,obs,meter,ratio_bounds=config['ratio_bounds'],
                seed=seed+index,call_counter=counter,**config['training'],**config['source'])
            fits[candidate.name]=fitted;row=io.serial(io.fit_record(candidate,fitted));records.append(row)
            if not fitted.success:errors[candidate.name]='fit_nonconverged';continue
            reviews.append(dict(review_fixed_fit(candidate,obs,meter,row),candidate=candidate.name))
        except (ValueError,RuntimeError,AssertionError,FloatingPointError,np.linalg.LinAlgError) as exc:
            errors[candidate.name]=type(exc).__name__+':'+str(exc)
    return fits,dict(fits=records,failures=errors,independent_review=reviews,
        all_complete=len(fits)==len(bank) and not errors,cost=counter,seconds=perf_counter()-began)


def response_diagnostic(case,candidate,fit,forecast,power):
    """Offline only: common true inputs/source isolate line-response error."""
    start=case['future_start'];a=case['assets'];u=np.asarray(power)/base.BASE_KW
    p,q=case['p'][start:],case['q'][start:];sp,sq=case['source_p'][start:],case['source_q'][start:]
    root=case['root_v'][start:];trues=[];fitted=[]
    for inc in (np.zeros_like(u),u):
        trues.append(lk.ac_predict(candidate,a['r'],a['x'],p+inc,q,sp,sq,root))
        fitted.append(lk.ac_predict(candidate,fit.r,fit.x,p+inc,q,sp,sq,root))
    out=dict(scope='offline true-structure response check; true inputs/source shared; selector never receives it',channels={})
    for channel in ('v','master_p','master_q'):
        true_delta=trues[1][channel]-trues[0][channel]
        fit_delta=fitted[1][channel]-fitted[0][channel]
        error=fit_delta-true_delta
        out['channels'][channel]=dict(RMSE=float(np.sqrt(np.mean(error**2))),
            relative_L2=float(np.linalg.norm(error)/max(np.linalg.norm(true_delta),1e-14)),
            true_increment=true_delta,fitted_increment=fit_delta)
    return out


def _run_trial(seed,family_index,config):
    cfg=deepcopy(config);began=perf_counter();before=source_hashes()
    case=dynamic.make_case(seed,family_index,cfg);tr,tm=case['train'],case['tm'];T=cfg['horizon']
    bank,proposal=propose(tr,run_wzzt=True,wzzt_time_limit=.3)
    fitted,training=train_bank(bank,tr,tm,cfg,case['assets']['seeds']['policy'])
    true_name=next((b.name for b in bank if b.supports==case['assets']['candidate'].supports),None)
    record=dict(schema='ev_physical_RX_nearest_v1',seed=seed,family_index=family_index,config=cfg,
        policy_id=io.identity(cfg,before),source_sha256=before,training=io.observations_record(tr),
        training_meter=meter_to_record(tm),training_result=training,
        validation=io.observations_record(case['validation']),validation_meter=meter_to_record(case['vm']),
        future_meter=meter_to_record(case['fm']),proposal=proposal,
        candidates=[dict(name=b.name,supports=b.supports) for b in bank],true_name=true_name,
        candidate_miss=true_name is None,background=case['background'],seeds=case['assets']['seeds'],
        truth=dict(r=case['assets']['r'],x=case['assets']['x'],supports=case['assets']['candidate'].supports),methods=[])
    if not training['all_complete']:
        record.update(status='training_failure',seconds=perf_counter()-began);return io.serial(record)
    _,validated=old.fit_bank(bank,case['validation'],case['vm'],trained=fitted,
        options={**cfg['profile'],**cfg['source']},seed=seed+100)
    record['validation_result']=validated
    forecast=fit_forecast(tr,tm,fitted,horizon=T,gap=cfg['forecast_gap'],base_kw=base.BASE_KW,kind=cfg['forecast'])
    record['forecast']=forecast
    start=case['future_start']
    record['forecast_error']=dict(P_RMSE_kw=float(np.sqrt(np.mean((forecast['p']-case['p'][start:])**2)))*base.BASE_KW,
        Q_RMSE_kvar=float(np.sqrt(np.mean((forecast['q']-case['q'][start:])**2)))*base.BASE_KW,
        uses_future_only_for_offline_evaluation=True)
    choices,selection=select_controls(tr,bank,fitted,tm,case['fm'],forecast=forecast,seed=seed+200,
        fit_options={**cfg['profile'],**cfg['source']},exchange_options=cfg['exchange'],
        highlow=dynamic.highlow_curve(T,tr.n))
    record['selection']=selection
    cache={}
    for name,power in choices.items():
        row=dict(name=name,power_kw=power,status='selection_failure')
        if power is None:record['methods'].append(row);continue
        op=(dict(success=True,battery_service=False) if name=='none' else dynamic.operation_record(
            power,mobile='mobile' in name,discharge='v2g' in name))
        row['operation']=op
        if not op['success']:row['status']='operation_failure';record['methods'].append(row);continue
        key=legacy._control_key(power)
        if key not in cache:
            try:cache[key]=dynamic.actual_curve(case,bank,fitted,validated,power,cfg,seed=seed+500)
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                cache[key]=dict(status='physical_or_fit_failure',M=None,failure=type(exc).__name__+':'+str(exc))
        row.update(cache[key]);record['methods'].append(row)
    record['training_parameter_error']={};record['line_response_diagnostics']={}
    if true_name:
        fit=fitted[true_name];a=case['assets'];candidate=next(b for b in bank if b.name==true_name)
        record['training_parameter_error']=dict(R=float(np.linalg.norm(fit.r-a['r'])/np.linalg.norm(a['r'])),
            X=float(np.linalg.norm(fit.x-a['x'])/np.linalg.norm(a['x'])))
        for name in ('constant','highlow','square_v2g_fixed','nearest_v2g_fixed','nearest_v2g_mobile'):
            if choices.get(name) is not None:
                try:
                    record['line_response_diagnostics'][name]=dict(success=True,
                        **response_diagnostic(case,candidate,fit,forecast,choices[name]))
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
                    record['line_response_diagnostics'][name]=dict(success=False,
                        failure=type(exc).__name__+':'+str(exc))
    record['PQ_error_scale']={}
    for b in bank:
        try:record['PQ_error_scale'][b.name]=dict(success=True,**old.PQ_error_scale(b,fitted[b.name],tm))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as exc:
            record['PQ_error_scale'][b.name]=dict(success=False,failure=type(exc).__name__+':'+str(exc))
    record.update(status='complete' if validated['all_complete'] and all(r['status']=='complete' for r in record['methods']) else 'partial_failure',
        seconds=perf_counter()-began,unique_actual_curves=len(cache),validation_used_by_selector=False,
        future_noise_used_by_selector=False,one_trained_parameter_vector=True)
    return io.serial(record)


def run_trial(seed,family_index,config=None):
    cfg=deepcopy(CONFIG if config is None else config);before=source_hashes();began=perf_counter()
    with dynamic.count_physics_calls() as counters:
        try:record=_run_trial(seed,family_index,cfg)
        except Exception as exc:
            record=dict(schema='ev_physical_RX_nearest_v1',seed=seed,family_index=family_index,config=cfg,
                policy_id=io.identity(cfg,before),source_sha256=before,status='driver_failure',
                failure=type(exc).__name__+':'+str(exc),traceback=traceback.format_exc(),methods=[],seconds=perf_counter()-began)
    record.update(source_unchanged=before==source_hashes(),actual_solver_calls=counters)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--family-offset',type=int,default=0)
    p.add_argument('--training-domain',choices=('box','cone'),default='cone')
    args=p.parse_args();cfg=deepcopy(CONFIG);cfg['training_domain']=args.training_domain;frozen=freeze(args.output,cfg)
    for index,seed in enumerate(args.seeds):
        path=args.output/f'trial_{seed}.json.gz'
        if path.exists():raise ValueError('existing trial retained')
        record=run_trial(seed,(index+args.family_offset)%4,cfg)
        if frozen['policy_id']!=record['policy_id'] or not record['source_unchanged']:raise RuntimeError('policy changed')
        io.write_json(path,record,compressed=True)
        io.write_json(args.output/'checkpoint.json',dict(completed=index+1,planned=len(args.seeds),last_seed=seed))
        print(__import__('json').dumps(dict(seed=seed,status=record['status'],seconds=record['seconds'],
            M={r['name']:r.get('M') for r in record['methods']})),flush=True)
        if record['status']=='driver_failure':raise RuntimeError(record['failure'])


if __name__=='__main__':main()
