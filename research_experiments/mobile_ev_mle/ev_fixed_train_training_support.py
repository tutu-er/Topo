"""Paired training-information sensitivity with the original held-out paths.

Only a newly generated, naturally varying synthetic TRAINING sequence changes.
The four physical assets, four validation slots, six future backgrounds/noise,
meter precisions, line bounds and EV device stay as in the frozen weak8 study.
Rich8/rich32/rich96 are nested conditions on four paired histories, not twelve
or sixteen independent histories. No oracle parameter is passed to a selector.
"""
from __future__ import annotations

import argparse
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_fixed_train_study as study
from . import ev_original_study as base
from . import gradient_price_study as io
from . import likelihood as lk
from .candidates import propose
from .ev_fixed_train_design import select_fixed_controls
from .ev_fixed_train_fit import subset_model, subset_observations
from .information_context import meter_to_record

HERE=Path(__file__).resolve().parent
LENGTHS=(8,32,96)
RICH_CONFIG=dict(version='independent_natural_training_rich96_nested_v1',
    lengths=list(LENGTHS),maximum_sequence_length=96,seed_offset=104729,
    seed_modulus=2**32,P_mean_kw=2.4,P_independent_halfwidth_kw=.6,
    Q_mean_kvar=.8,Q_independent_halfwidth_kvar=.3,
    source_P_mean_kw=2.,source_P_halfwidth_kw=.25,
    source_Q_mean_kvar=.7,source_Q_halfwidth_kvar=.1,
    training_root_v=1.,
    source_interface=1,natural_load_sensitivity=True,training_EV_applied=False,
    physical_lengths_and_line_bounds_unchanged=True,
    validation_origin_indices=[8,9,10,11],future_origin_indices=[12,13,14,15,16,17],
    statistical_unit='one paired asset/history; four development units',
    weak8_role='reuse original development_v2; do not regenerate or overwrite',
    scope='synthetic training support and sample-size sensitivity; not field evidence')


def identity(value):
    return hashlib.sha256(json.dumps(io.serial(value),sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def source_hashes():
    result=study.source_hashes()
    result[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def condition_config(length,run_selector=True):
    if isinstance(length,bool) or int(length)!=length or int(length) not in LENGTHS:
        raise ValueError('training length must be one of 8, 32 or 96')
    config=deepcopy(study.CONFIG)
    config.update(version=RICH_CONFIG['version'],training_points=int(length),
        training_support=deepcopy(RICH_CONFIG),run_selector=bool(run_selector),
        validation_points=4,horizon=6,original_history_points=12,
        formal_calibration_count=0,statistical_threshold_applied=False,
        independent_development_units=4,conditions_are_paired_not_independent=True)
    return io.serial(config)


def _physical_record(physical):
    return {k:physical[k] for k in ('p','q','v','root_v','master_p','master_q',
        'ac_equation_residual','power_balance_residual')}


def make_training_inputs(seed,family_index,length):
    """Generate all rich96 first, then slice; do not alter the original holdout.

    Its RNGs are fresh streams from the original background/source/noise seed
    plus a predeclared fixed offset. Future values are not used by these RNGs.
    Truth enters only physical generation and offline error diagnostics here.
    """
    condition_config(length)
    if isinstance(family_index,bool) or int(family_index)!=family_index or not 0<=int(family_index)<4:
        raise ValueError('family_index must be 0, 1, 2 or 3')
    seed=int(seed);length=int(length);case=base.make_case(seed,base.FAMILIES[int(family_index)])
    count=RICH_CONFIG['maximum_sequence_length'];offset=RICH_CONFIG['seed_offset']
    seeds={k:int((int(case['seeds'][k])+offset)%RICH_CONFIG['seed_modulus'])
        for k in ('background','source','noise')}
    rng=np.random.default_rng(seeds['background'])
    p=(RICH_CONFIG['P_mean_kw']+rng.uniform(-RICH_CONFIG['P_independent_halfwidth_kw'],
        RICH_CONFIG['P_independent_halfwidth_kw'],(count,3)))/base.BASE_KW
    q=(RICH_CONFIG['Q_mean_kvar']+rng.uniform(-RICH_CONFIG['Q_independent_halfwidth_kvar'],
        RICH_CONFIG['Q_independent_halfwidth_kvar'],(count,3)))/base.BASE_KW
    sr=np.random.default_rng(seeds['source'])
    sp=(RICH_CONFIG['source_P_mean_kw']+sr.uniform(-RICH_CONFIG['source_P_halfwidth_kw'],
        RICH_CONFIG['source_P_halfwidth_kw'],count))/base.BASE_KW
    sq=(RICH_CONFIG['source_Q_mean_kvar']+sr.uniform(-RICH_CONFIG['source_Q_halfwidth_kvar'],
        RICH_CONFIG['source_Q_halfwidth_kvar'],count))/base.BASE_KW
    root=np.full(count,RICH_CONFIG['training_root_v'])
    nr=np.random.default_rng(seeds['noise'])
    white={k:nr.standard_normal((count,3) if k in ('p','q','v') else count) for k in base.CHANNELS}
    physical=lk.ac_predict(case['candidate'],case['r'],case['x'],p,q,sp,sq,root)
    full=base.observation(physical,case['hm'],white)
    train=subset_observations(full,slice(0,length))
    tm=subset_model(case['hm'],full,slice(0,length))
    validation=subset_observations(case['history'],slice(8,12))
    vm=subset_model(case['hm'],case['history'],slice(8,12))
    future_white={k:np.asarray(v)[12:].copy() for k,v in case['white'].items()}
    holdout=dict(validation=io.observations_record(validation),validation_meter=meter_to_record(vm),
        future_p=case['p'][12:],future_q=case['q'][12:],future_source_p=case['source_p'][12:],
        future_source_q=case['source_q'][12:],future_root_v=case['root_v'][12:],
        future_white=future_white,future_meter=meter_to_record(case['fm']),
        r=case['r'],x=case['x'],assets=case['assets'],device=study.CONFIG['physics']['device'])
    truth=dict(p=p[:length],q=q[:length],source_p=sp[:length],source_q=sq[:length],root_v=root[:length])
    generation=dict(config=deepcopy(RICH_CONFIG),rich_seeds=seeds,original_seeds=case['seeds'],
        rich96_physical_inputs_sha256=identity(dict(p=p,q=q,source_p=sp,source_q=sq,root_v=root)),
        rich96_white_sha256=identity(white),rich96_observations_sha256=identity(io.observations_record(full)),
        selected_prefix_observations_sha256=identity(io.observations_record(train)),
        paired_validation_future_asset_sha256=identity(holdout),
        training_length=length,training_prefix_indices=list(range(length)),
        validation_future_unchanged=True,training_EV_applied=False,
        generator_reads_future_for_training=False,truth_used_by_selector=False,
        generation_AC_calls=2,scope=RICH_CONFIG['scope'])
    return dict(case=case,training=train,training_meter=tm,validation=validation,validation_meter=vm,
        future_white=future_white,training_truth=truth,training_white={k:v[:length].copy() for k,v in white.items()},
        training_physical={k:np.asarray(v)[:length] if isinstance(v,np.ndarray) and v.ndim>0 else v
            for k,v in _physical_record(physical).items()},generation=generation)


def _input_summary(observations,meter):
    p=observations.p-lk._array(meter.bias_p,observations.p.shape,'bias_p')
    q=observations.q-lk._array(meter.bias_q,observations.q.shape,'bias_q')
    values=np.c_[p,q]*base.BASE_KW;centered=values-values.mean(axis=0)
    singular=np.linalg.svd(centered,compute_uv=False)
    return dict(columns=['P0_kW','P1_kW','P2_kW','Q0_kvar','Q1_kvar','Q2_kvar'],
        mean=values.mean(axis=0),sample_std=values.std(axis=0,ddof=1),
        minimum=values.min(axis=0),maximum=values.max(axis=0),
        centered_singular_values=singular,centered_rank=int(np.linalg.matrix_rank(centered)),
        sample_correlation=np.corrcoef(values,rowvar=False),
        scope='descriptive calibrated measured inputs; rank is not a global identifiability proof')


def _parameter_error(case,bank,fits,true_name):
    row=dict(true_name=true_name,candidate_miss=true_name is None,scope='offline true-candidate fit only')
    if true_name is None or true_name not in fits:
        return dict(row,R_relative=None,X_relative=None)
    fit=fits[true_name]
    if not fit.success:return dict(row,R_relative=None,X_relative=None,fit_failure=True)
    r=np.asarray(fit.r);x=np.asarray(fit.x)
    return dict(row,R_relative=float(np.linalg.norm(r-case['r'])/np.linalg.norm(case['r'])),
        X_relative=float(np.linalg.norm(x-case['x'])/np.linalg.norm(case['x'])),
        r_error=r-case['r'],x_error=x-case['x'],training_lines_fixed_in_following_stages=True)


def _operation(selection,name):
    if name=='constant':return selection.get('constant_operation')
    return selection.get('methods',{}).get(name,{}).get('operation')


def _cost_add(total,counter):
    for key in ('fit_calls','solver_calls','ac_calls'):
        total[key]=total.get(key,0)+int(counter.get(key,0))


def run_training_condition(seed,family_index,length,*,run_selector=True):
    """Run one rich-prefix condition on one fixed development asset/history.

    All fitting, proposal and planning use rich training only. Validation is
    held out; future paths and paired white errors are exactly the original
    frozen case's six rows. No previous score selects a training condition.
    """
    config=condition_config(length,run_selector);began=perf_counter();before=source_hashes()
    data=make_training_inputs(seed,family_index,length)
    case=data['case'];train=data['training'];tm=data['training_meter']
    validation=data['validation'];vm=data['validation_meter']
    bank,proposal=propose(train,run_wzzt=True,wzzt_time_limit=.3)
    opts={**config['training'],**config['source']}
    fitted,training_result=study.fit_bank(bank,train,tm,options=opts,seed=case['seeds']['policy'])
    true_name=next((b.name for b in bank if b.supports==case['candidate'].supports),None)
    record=dict(schema='ev_fixed_train_training_support_trial_v1',seed=int(seed),family_index=int(family_index),
        family=case['family'],condition='rich'+str(length),training_length=int(length),
        policy_id=io.identity(config,before),config=config,source_sha256=before,
        original_weak8_reference='evidence/ev_fixed_train/development_v2; saved record kept unchanged',
        training_indices=list(range(int(length))),training_index_space='new rich96 stream prefix',
        validation_indices=[8,9,10,11],validation_index_space='original 12-point case history',
        selector_inputs=['rich training meters','rich training fits','public future precision','unchanged device constraints'],
        training=io.observations_record(train),training_meter=meter_to_record(tm),
        validation=io.observations_record(validation),validation_meter=meter_to_record(vm),
        future_meter=meter_to_record(case['fm']),proposal=proposal,
        candidates=[dict(name=b.name,supports=b.supports) for b in bank],true_name=true_name,
        candidate_miss=true_name is None,training_result=training_result,
        training_parameter_error=_parameter_error(case,bank,fitted,true_name),
        training_input_summary=_input_summary(train,tm),training_generation=data['generation'],
        training_truth=data['training_truth'],training_white=data['training_white'],training_physical=data['training_physical'],
        truth=dict(supports=case['candidate'].supports,r=case['r'],x=case['x'],p=case['p'],q=case['q'],
            source_p=case['source_p'],source_q=case['source_q'],root_v=case['root_v'],assets=case['assets']),
        future_white=data['future_white'],seeds=case['seeds'],methods=[],
        statistical_unit='one paired asset/history',independent_development_units=4,
        number_of_training_conditions_is_not_independent_sample_size=True,
        statistically_calibrated=False,statistical_threshold_applied=False,
        field_evidence=False,validation_used_by_selector=False,truth_used_by_selector=False,future_noise_used_by_selector=False)
    total=dict(fit_calls=0,solver_calls=0,ac_calls=0);_cost_add(total,training_result['cost'])
    if not training_result['all_complete']:
        record.update(status='training_failure',fit_cost=total,seconds=perf_counter()-began,source_unchanged=before==source_hashes())
        return io.serial(record)
    _,validated=study.fit_bank(bank,validation,vm,options={**config['source_profile'],**config['source']},
        seed=case['seeds']['policy']+100,trained=fitted)
    _cost_add(total,validated['cost'])
    record['validation_result']=validated;record['validation_M']=study._margin(validated,true_name)
    record['PQ_error_scale']={b.name:study.PQ_error_scale(b,fitted[b.name],tm) for b in bank}
    if not run_selector:
        record.update(status='training_validation_only_complete' if validated['all_complete'] else 'validation_failure',
            selection=None,selector_executed=False,unique_actual_curves=0,fit_cost=total,
            seconds=perf_counter()-began,source_unchanged=before==source_hashes())
        return io.serial(record)
    # No case, validation, physical truth, future inputs or future white errors
    # enter this call. The future meter is public precision/calibration only.
    choices,selection=select_fixed_controls(train,bank,fitted,tm,case['fm'],seed=case['seeds']['policy']+200,
        fit_options={**config['source_profile'],**config['source']},exchange_options=config['exchange'],modes=config['methods'])
    record['selection']=selection;record['selector_executed']=True;cache={};future_forward_calls=0
    for name,power in choices.items():
        row=dict(name=name,status='selection_failure',power_kw=power,operation=_operation(selection,name))
        if power is None:
            record['methods'].append(row);continue
        try:
            power=np.asarray(power,float);key=np.round(power,10).astype('<f8').tobytes()
            hit=key in cache
            if not hit:
                P=case['p'][12:]+power/base.BASE_KW;Q=case['q'][12:]
                future_forward_calls+=1
                physical=lk.ac_predict(case['candidate'],case['r'],case['x'],P,Q,
                    case['source_p'][12:],case['source_q'][12:],case['root_v'][12:])
                actual=base.observation(physical,case['fm'],data['future_white'])
                clean=base.observation(physical,case['fm'],{k:np.zeros_like(v) for k,v in data['future_white'].items()})
                _,noisy=study.fit_bank(bank,actual,case['fm'],trained=fitted,
                    options={**config['source_profile'],**config['source']},seed=case['seeds']['policy']+500)
                _,noiseless=study.fit_bank(bank,clean,case['fm'],trained=fitted,
                    options={**config['source_profile'],**config['source']},seed=case['seeds']['policy']+600)
                _cost_add(total,noisy['cost']);_cost_add(total,noiseless['cost'])
                value=dict(observations=io.observations_record(actual),noiseless_observations=io.observations_record(clean),
                    physical=_physical_record(physical),noisy=noisy,noiseless=noiseless,
                    future_M=study._margin(noisy,true_name),noiseless_future_M=study._margin(noiseless,true_name),
                    combined_loss={},combined_noiseless_loss={})
                if validated['all_complete']:
                    losses={f['candidate']:f['loss'] for f in validated['fits']}
                    for field,part in (('combined_loss',noisy),('combined_noiseless_loss',noiseless)):
                        if part['all_complete']:value[field]={f['candidate']:losses[f['candidate']]+f['loss'] for f in part['fits']}
                cache[key]=value
            row.update(cache[key],actual_fit_cache_hit=hit,grid_net_kwh=float(config['physics']['dt_hours']*power.sum()),
                grid_throughput_kwh=float(config['physics']['dt_hours']*np.abs(power).sum()))
            row['status']='complete' if row['noisy']['all_complete'] and validated['all_complete'] else 'fit_failure'
            for label,field in (('M','combined_loss'),('noiseless_M','combined_noiseless_loss')):
                losses=row[field]
                row[label]=None if true_name not in losses else float(min(v for k,v in losses.items() if k!=true_name)-losses[true_name])
            row['delta_M_from_validation']=None if row['M'] is None or record['validation_M'] is None else row['M']-record['validation_M']
        except (ValueError,RuntimeError,AssertionError,FloatingPointError,np.linalg.LinAlgError) as exc:
            row.update(status='physical_or_fit_exception',failure=type(exc).__name__+':'+str(exc))
        record['methods'].append(row)
    record.update(status='complete' if validated['all_complete'] and all(m['status']=='complete' for m in record['methods']) else 'partial_failure',
        unique_actual_curves=len(cache),fit_cost=total,selector_cost=dict(
            fit_calls=selection.get('predicted_fit_calls'),ac_calls=selection.get('predicted_ac_calls'),
            scope='selector separately records predictions, derivative construction, QP/MILP and AC gates'),
        physical_forward_calls=future_forward_calls,generation_AC_calls=2,
        seconds=perf_counter()-began,source_unchanged=before==source_hashes(),
        combined_profile_identity='sum held-out validation and future output SSE with the same fixed rich-training R/X')
    return io.serial(record)


def freeze_directory(directory,length,*,run_selector=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    config=condition_config(length,run_selector);hashes=source_hashes()
    record=dict(config=config,source_sha256=hashes,policy_id=io.identity(config,hashes),
        source_count=len(hashes),purpose='new paired synthetic training-information development sensitivity',
        source_frozen_before_trials=True,conditions_are_paired_not_independent=True)
    path=directory/'freeze.json'
    if path.exists():
        if io.read_json(path)!=record:raise ValueError('existing freeze differs; preserve and choose new directory')
        return record
    io.write_json(path,record)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/')
        else (HERE/name)).read_bytes()).decode() for name in hashes}
    io.write_json(directory/'source_snapshot.json.gz',dict(source_sha256=hashes,source_base64=snapshot),compressed=True)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lengths',type=int,nargs='+',choices=LENGTHS,default=[8,32])
    parser.add_argument('--seeds',type=int,nargs='+',default=study.CONFIG['development_seeds'])
    parser.add_argument('--family-offset',type=int,default=0)
    parser.add_argument('--skip-selector',action='store_true')
    args=parser.parse_args()
    if len(set(args.lengths))!=len(args.lengths) or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('duplicate conditions/seeds are not independent samples')
    for length in args.lengths:
        directory=args.output/('rich'+str(length));frozen=freeze_directory(directory,length,run_selector=not args.skip_selector)
        for index,seed in enumerate(args.seeds):
            path=directory/f'trial_{seed}.json.gz'
            if path.exists():raise ValueError('existing raw trial retained: '+str(path))
            record=run_training_condition(seed,(index+args.family_offset)%4,length,run_selector=not args.skip_selector)
            if not record['source_unchanged'] or record['policy_id']!=frozen['policy_id']:
                raise RuntimeError('source/config drift; record not accepted into frozen condition')
            io.write_json(path,record,compressed=True)
            io.write_json(directory/'checkpoint.json',dict(completed=index+1,planned=len(args.seeds),last_seed=seed,
                training_length=length,paired_units=True,status='running' if index+1<len(args.seeds) else 'complete'))
            print(json.dumps(dict(seed=seed,training_length=length,status=record['status'],seconds=record['seconds'],
                validation_M=record.get('validation_M'),theta_error=record['training_parameter_error'],
                methods=[dict(name=m['name'],M=m.get('M'),status=m['status']) for m in record['methods']])),flush=True)


if __name__=='__main__':main()
