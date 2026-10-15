"""Train-only fixed-input AC models and held-out EV experiment.

Truth, validation and future errors are never passed to the selector.
Saved costs are attained local output fits, not global MLE certificates.
"""
from __future__ import annotations
import argparse, base64, gzip, hashlib, json, platform
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
import numpy as np
from . import likelihood as lk
from . import ev_original_study as base
from . import ev_compact_study as parent
from . import gradient_price_study as io
from .candidates import propose
from .information_context import meter_to_record
from .ev_fixed_train_fit import fit_fixed_inputs, subset_model, subset_observations
from .ev_fixed_train_design import select_fixed_controls, METHODS, DEFAULT_EXCHANGE
from .ev_fixed_train_review import review_fixed_fit

HERE=Path(__file__).resolve().parent
CONFIG=dict(version='fixed_training_PQ_input_v1',training_points=8,validation_points=4,
    horizon=6,methods=list(METHODS),candidate_proposal='training only; all proposed/repaired candidates',
    training=dict(multistart=4,max_nfev=350,tol=1e-10,**base.ASSET_BOUNDS),
    source=dict(source_p_bounds=[-.5,.5],source_q_bounds=[-.5,.5]),
    source_profile=dict(multistart=2,max_nfev=180,tol=1e-10),
    forecast='training calibrated PQ mean and candidate training source mean; one scenario',
    validation_used_by_selector=False,truth_used_by_selector=False,future_noise_used_by_selector=False,
    PQ_inputs_fixed=True,PQ_random_error_propagated=False,source_node=1,
    criterion='conditional V/rootP/rootQ profile SSE at exactly fixed training lines',
    exchange=dict(DEFAULT_EXCHANGE),family_order=[list(map(list,f)) for f in base.FAMILIES],
    development_seeds=list(range(990101,990105)),calibration_seeds=list(range(990301,990337)),
    evaluation_seeds=list(range(990201,990213)),formal_alpha=.1,
    physics=dict(base.CONFIG),prior_policy_id='033d4f3bd0077ed68216268adbd3259110f6c1b1bdb67993ad7ea44fa19ad42f')


def source_hashes():
    hashes=parent.source_hashes()
    for name in ('ev_fixed_train_fit.py','ev_fixed_train_design.py','ev_fixed_train_review.py',
                 'ev_fixed_train_study.py','ev_fixed_train_protocol.md','ev_fixed_train_math.md'):
        hashes[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    return hashes


def freeze_directory(directory, config=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    config=io.serial(CONFIG if config is None else config);hashes=source_hashes()
    record=dict(config=config,source_sha256=hashes,policy_id=io.identity(config,hashes),
        source_count=len(hashes),base_commit='1c71cb253be86c5207262cd51c49261427a41c78',
        execution_environment=dict(python=platform.python_version(),numpy=np.__version__))
    path=directory/'freeze.json'
    if path.exists():
        saved=io.read_json(path)
        if any(saved[k]!=record[k] for k in ('config','source_sha256','policy_id')):
            raise ValueError('existing freeze differs; keep it and choose a new directory')
        return saved
    io.write_json(path,record)
    snapshot={name:base64.b64encode(((HERE.parents[1]/name) if name.startswith('rnj_wzzt_core/')
        else (HERE/name)).read_bytes()).decode() for name in hashes}
    io.write_json(directory/'source_snapshot.json.gz',dict(source_sha256=hashes,source_base64=snapshot),compressed=True)
    return record


def fit_bank(bank, obs, meter, *, options, seed, trained=None):
    fits={};records=[];errors={};reviews=[];counter={};began=perf_counter()
    for index,candidate in enumerate(bank):
        name=candidate.name
        try:
            fixed={};initial=None
            if trained is not None:
                old=trained[name];fixed=dict(fixed_r=old.r,fixed_x=old.x)
                initial=SimpleNamespace(r=old.r,x=old.x,
                    source_p=np.full(obs.t,np.mean(old.source_p)),source_q=np.full(obs.t,np.mean(old.source_q)))
            fit=fit_fixed_inputs(candidate,obs,meter,initial=initial,seed=int(seed)+index,
                call_counter=counter,**fixed,**options)
            fits[name]=fit;row=io.serial(io.fit_record(candidate,fit));records.append(row)
            if not fit.success:
                errors[name]='fit_nonconverged';continue
            review=review_fixed_fit(candidate,obs,meter,row,
                trained_r=None if trained is None else trained[name].r,
                trained_x=None if trained is None else trained[name].x)
            reviews.append(dict(review,candidate=name))
        except (ValueError,RuntimeError,AssertionError,FloatingPointError,np.linalg.LinAlgError) as error:
            errors[name]=type(error).__name__+':'+str(error)
    return fits,dict(fits=records,failures=errors,independent_review=reviews,
        all_complete=len(fits)==len(bank) and not errors,cost=counter,seconds=perf_counter()-began)


def _margin(records,true_name):
    if not records['all_complete']:return None
    losses={row['candidate']:row['loss'] for row in records['fits']}
    if true_name not in losses:return None
    return float(min(v for k,v in losses.items() if k!=true_name)-losses[true_name])


def PQ_error_scale(candidate, fit, meter):
    """First-order input-error scale before/after source projection, not a CI."""
    d=lk._response_derivatives(candidate,fit.predicted,True)
    T,n=fit.p.shape;channels=('v','master_p','master_q');Hs=[];Hx=[]
    for name in channels:
        width=n if name=='v' else 1
        shape=(T,n) if name=='v' else (T,)
        sigma=lk._array(getattr(meter,name+'_sigma'),shape,name+'_sigma').reshape(T,width)
        jac=np.asarray(d[name]).reshape(T,width,-1)/sigma[:,:,None]
        Hx.append(jac[:,:,2*candidate.m:2*candidate.m+2*n])
        Hs.append(jac[:,:,-2:])
    Hx=np.concatenate(Hx,axis=1);Hs=np.concatenate(Hs,axis=1)
    sig=np.c_[lk._array(meter.p_sigma,(T,n),'p_sigma'),lk._array(meter.q_sigma,(T,n),'q_sigma')]
    raw=[];projected=[]
    for t in range(T):
        influence=Hx[t]*sig[t][None,:];P=np.eye(n+2)-Hs[t]@np.linalg.pinv(Hs[t])
        raw.append(float(np.linalg.norm(influence,2)**2))
        projected.append(float(np.linalg.norm(P@influence,2)**2))
    return dict(raw_max_variance_ratio=max(raw),source_profile_max_variance_ratio=max(projected),
        raw_by_time=raw,source_profile_by_time=projected,first_order_only=True,
        source_projection_unconstrained=True,measurement_covariance_changed=False)


def run_trial(seed, family_index, *, config=None):
    config=io.serial(CONFIG if config is None else config);began=perf_counter();before=source_hashes()
    case=base.make_case(int(seed),base.FAMILIES[int(family_index)])
    h=case['history'];cut=int(config['training_points'])
    train=subset_observations(h,slice(0,cut));validation=subset_observations(h,slice(cut,None))
    tm=subset_model(case['hm'],h,slice(0,cut));vm=subset_model(case['hm'],h,slice(cut,None))
    bank,proposal=propose(train,run_wzzt=True,wzzt_time_limit=.3)
    opts={**config['training'],**config['source']}
    fitted,training_result=fit_bank(bank,train,tm,options=opts,seed=case['seeds']['policy'])
    names=[b.name for b in bank]
    true_name=next((b.name for b in bank if b.supports==case['candidate'].supports),None)
    record=dict(schema='ev_fixed_training_trial_v1',seed=int(seed),family_index=int(family_index),
        family=case['family'],policy_id=io.identity(config,before),config=config,source_sha256=before,
        training_indices=list(range(cut)),validation_indices=list(range(cut,h.t)),
        selector_inputs=['training meters','training fits','public future meter precision','device constraints'],
        training=io.observations_record(train),validation=io.observations_record(validation),
        training_meter=meter_to_record(tm),validation_meter=meter_to_record(vm),future_meter=meter_to_record(case['fm']),
        proposal=proposal,candidates=[dict(name=b.name,supports=b.supports) for b in bank],
        true_name=true_name,candidate_miss=true_name is None,training_result=training_result,
        truth=dict(supports=case['candidate'].supports,r=case['r'],x=case['x'],p=case['p'],q=case['q'],
            source_p=case['source_p'],source_q=case['source_q'],root_v=case['root_v'],assets=case['assets']),
        future_white={k:v[h.t:] for k,v in case['white'].items()},seeds=case['seeds'],methods=[])
    if not training_result['all_complete']:
        record.update(status='training_failure',seconds=perf_counter()-began,source_unchanged=before==source_hashes())
        return io.serial(record)
    _,validated=fit_bank(bank,validation,vm,options={**config['source_profile'],**config['source']},
        seed=case['seeds']['policy']+100,trained=fitted)
    record['validation_result']=validated;record['validation_M']=_margin(validated,true_name)
    record['PQ_error_scale']={b.name:PQ_error_scale(b,fitted[b.name],tm) for b in bank}
    # The following call receives no validation, simulator case, truth or future noise.
    choices,selection=select_fixed_controls(train,bank,fitted,tm,case['fm'],seed=case['seeds']['policy']+200,
        fit_options={**config['source_profile'],**config['source']},exchange_options=config['exchange'],modes=config['methods'])
    record['selection']=selection;cache={}
    for name,power in choices.items():
        row=dict(name=name,status='selection_failure',power_kw=power)
        if power is None:
            record['methods'].append(row);continue
        key=np.round(np.asarray(power,float),10).astype('<f8').tobytes()
        if key not in cache:
            P=case['p'][h.t:]+np.asarray(power)/base.BASE_KW;Q=case['q'][h.t:]
            phy=lk.ac_predict(case['candidate'],case['r'],case['x'],P,Q,
                case['source_p'][h.t:],case['source_q'][h.t:],case['root_v'][h.t:])
            actual=base.observation(phy,case['fm'],record['future_white'])
            clean=base.observation(phy,case['fm'],{k:np.zeros_like(v) for k,v in record['future_white'].items()})
            noisy_fits,noisy=fit_bank(bank,actual,case['fm'],trained=fitted,
                options={**config['source_profile'],**config['source']},seed=case['seeds']['policy']+500)
            _,noiseless=fit_bank(bank,clean,case['fm'],trained=fitted,
                options={**config['source_profile'],**config['source']},seed=case['seeds']['policy']+600)
            cache[key]=dict(observations=io.observations_record(actual),noiseless_observations=io.observations_record(clean),
                physical={k:phy[k] for k in ('p','q','v','master_p','master_q','root_v')},
                noisy=noisy,noiseless=noiseless,future_M=_margin(noisy,true_name),
                noiseless_future_M=_margin(noiseless,true_name),combined_loss={},combined_noiseless_loss={})
            if noisy['all_complete'] and validated['all_complete']:
                val={f['candidate']:f['loss'] for f in validated['fits']}
                cache[key]['combined_loss']={f['candidate']:val[f['candidate']]+f['loss'] for f in noisy['fits']}
            if noiseless['all_complete'] and validated['all_complete']:
                val={f['candidate']:f['loss'] for f in validated['fits']}
                cache[key]['combined_noiseless_loss']={f['candidate']:val[f['candidate']]+f['loss'] for f in noiseless['fits']}
        data=cache[key];row.update(data,actual_fit_cache_hit=any(m.get('_key')==key.hex() for m in record['methods']))
        row['status']='complete' if data['noisy']['all_complete'] and validated['all_complete'] else 'fit_failure'
        for label,field in (('M','combined_loss'),('noiseless_M','combined_noiseless_loss')):
            losses=data[field]
            row[label]=None if true_name not in losses else float(min(v for k,v in losses.items() if k!=true_name)-losses[true_name])
        row['delta_M_from_validation']=None if row.get('M') is None or record['validation_M'] is None else row['M']-record['validation_M']
        row['_key']=key.hex();record['methods'].append(row)
    for row in record['methods']:row.pop('_key',None)
    record.update(status='complete' if validated['all_complete'] and all(m['status']=='complete' for m in record['methods']) else 'partial_failure',
        unique_actual_curves=len(cache),seconds=perf_counter()-began,source_unchanged=before==source_hashes(),
        validation_used_by_selector=False,combined_profile_identity='independent per-time source fits; sum validation and future SSE at same training R/X')
    return io.serial(record)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True);p.add_argument('--family-offset',type=int,default=0)
    args=p.parse_args();freeze=freeze_directory(args.output)
    for index,seed in enumerate(args.seeds):
        path=args.output/f'trial_{seed}.json.gz'
        if path.exists():raise ValueError('existing trial retained; choose another directory or new IDs')
        record=run_trial(seed,(index+args.family_offset)%4)
        if record['policy_id']!=freeze['policy_id'] or not record['source_unchanged']:raise RuntimeError('source/config drift')
        io.write_json(path,record,compressed=True)
        io.write_json(args.output/'checkpoint.json',dict(completed=index+1,planned=len(args.seeds),last_seed=seed))
        print(json.dumps(dict(seed=seed,status=record['status'],seconds=record['seconds'],
            methods=[dict(name=m['name'],M=m.get('M'),status=m['status']) for m in record['methods']])),flush=True)


if __name__=='__main__':main()
