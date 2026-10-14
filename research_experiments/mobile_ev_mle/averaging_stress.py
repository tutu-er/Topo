"""Fixed-policy paired diagnostic: true future V averaging error has rho=.25."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import numpy as np
import scipy
from . import gradient_price_study as old
from . import information_study as frozen
from .gradient_price_evidence import replay_fit, _close
from .information_context import meter_from_record, meter_to_record
from .information_replay import audit_source, _obs, _family, _truth_future
from .likelihood import RadialCandidate, concat_observations
from .simulation import observe, truth_metadata
from .single_site_milp import SingleSiteConstraints, SingleSitePlan, check_single_site_plan

HERE = Path(__file__).resolve().parent
NOMINAL_ROOT = HERE/'evidence/structured_information/v3_frozen'
OUTPUT = NOMINAL_ROOT/'stress_v_correlation'
IDS = tuple(range(950001,950017))
RHO = .25
THRESHOLD = 2.9617106149485366
CHANNELS = ('p','q','v','root_v','master_p','master_q','source_p','source_q')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def value_hash(value): return hashlib.sha256(json.dumps(old.serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def array_hash(value):
    a=np.asarray(value,dtype='<f8')
    return hashlib.sha256(json.dumps(a.shape).encode()+a.tobytes()).hexdigest()
def _plan(action):
    return SingleSitePlan(int(action['site']),np.asarray(action['p'],float),np.asarray(action['q'],float),np.asarray(action['sites'],int),np.asarray(action['soc'],float),action.get('solver',{}))
def _action_key(action):
    return tuple(np.asarray(action[k],dtype='<f8').tobytes() for k in ('p','q'))


def actual_generation_model(history_model,assumed_future,repeat_counts,rho=RHO):
    """Marginal mean of M equicorrelated raw errors; biases are unchanged."""
    m=np.asarray(repeat_counts,float)
    if m.shape!=(4,3) or not np.isfinite(m).all() or np.any(m<1) or not np.equal(m,np.floor(m)).all(): raise ValueError('positive integer 4x3 counts required')
    if not 0<=rho<=1: raise ValueError('rho outside [0,1]')
    raw=np.broadcast_to(np.asarray(history_model.v_sigma,float),(12,3))
    if not np.array_equal(raw,np.broadcast_to(raw[0],raw.shape)) or np.any(raw<=0): raise ValueError('positive time-constant raw sigma required')
    assumed=np.broadcast_to(np.asarray(assumed_future.v_sigma,float),m.shape)
    if not np.allclose(assumed,raw[0]/np.sqrt(m),rtol=1e-12,atol=1e-15): raise ValueError('nominal averaging rule differs')
    return replace(assumed_future,v_sigma=raw[0]*np.sqrt(rho+(1-rho)/m))


def decision(row,names,true_name,threshold=THRESHOLD):
    losses=row.get('losses',{})
    complete=row.get('status')=='complete' and set(losses)==set(names)
    gaps={n:float(losses[n])-min(losses.values()) for n in names} if complete else {}
    retained=[n for n in names if gaps[n]<=threshold] if complete else list(names)
    return dict(retained_candidates=retained,false_rejection=bool(true_name is not None and true_name not in retained),only_truth_retained=bool(true_name is not None and retained==[true_name]),unresolved=len(retained)>1,wrong_candidates_removed=sum(n!=true_name and n not in retained for n in names) if true_name is not None else None,gate_status='relative_gate_applied' if complete else 'fit_failure_retained_all',relative_gate_threshold=threshold)


def outcome(row):
    if row['status']!='complete': return 'failure'
    if row['false_rejection']: return 'false_rejection'
    if row['only_truth_retained']: return 'unique_correct'
    return 'unresolved'


def _nominal_summary(row,names,true_name):
    values={k:row.get(k) for k in ('status','winner','true_gap','signed_margin_full_SSE','losses','gaps')}
    values.update(decision(row,names,true_name));values['outcome']=outcome(values)
    return values


def reconstruct_nominal(trial,expected):
    """Exact deterministic frozen-path reconstruction before altered acquisition."""
    errors=[]
    for k in ('config','source_sha256','policy_id'):
        if trial[k]!=expected[k]: errors.append('frozen_'+k)
    if old.identity(trial['config'],trial['source_sha256'])!=trial['policy_id']: errors.append('policy_digest')
    if trial.get('selection_uses_truth') or trial.get('selection_uses_future_noise'): errors.append('nominal_selector_isolation')
    path,seeds,label,quality,permutation,white,amplitude=frozen.make_path(trial['seed'],trial['config'])
    reconstructed_truth=old.serial(truth_metadata(path))
    if reconstructed_truth!=trial['truth']: errors.append('truth_arrays_or_metadata')
    if seeds!=trial['seeds'] or label!=trial['case'] or quality!=trial['quality'] or permutation!=trial['quality_record']['row_permutation'] or amplitude!=trial['history_innovation_amplitude']: errors.append('original_path_seed_metadata')
    for channel,value in white.items(): _close(value,trial['white_evaluation_noise'][channel],'white_'+channel,errors,atol=0.,rtol=0.)
    h,_=observe(path,slice(0,path.history_count))
    for channel in CHANNELS:
        a,b=getattr(h,channel),trial['history_observations'][channel]
        if a is None or b is None:
            if a is not None or b is not None: errors.append('history_channel_'+channel)
        else: _close(a,b,'history_'+channel,errors,atol=0.,rtol=0.)
    fingerprint=old.fingerprint(h.p,h.q,h.v,h.master_p,h.master_q)
    if fingerprint!=trial['history_hash']: errors.append('history_fingerprint')
    assumed=meter_from_record(trial['measurement_models']['future'])
    records={r['method']:r for r in trial['records']}
    if tuple(records)!=frozen.METHODS or len(records)!=11 or len(trial['candidate_names'])!=4: errors.append('frozen_method_or_candidate_denominator')
    checks=0
    for method in frozen.METHODS:
        row=records.get(method)
        if row is None or row.get('action') is None:
            errors.append('missing_saved_action:'+method);continue
        try:
            future,_=frozen.future_observe(path,_plan(row['action']),assumed,white)
            if 'observations' not in row:
                if row['status'] not in ('operational_failure','physical_failure'): errors.append('missing_nominal_observations:'+method)
                continue
            for channel in CHANNELS:
                a,b=getattr(future,channel),row['observations'][channel]
                if a is None or b is None:
                    if a is not None or b is not None: errors.append(method+':nominal_'+channel)
                else:
                    _close(a,b,method+':nominal_'+channel,errors,atol=0.,rtol=0.);checks+=1
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error: errors.append(method+':nominal_acquisition:'+str(error))
    record=dict(success=not errors,errors=errors,history_hash_saved=trial['history_hash'],history_hash_reconstructed=fingerprint,saved_truth_sha256=value_hash(trial['truth']),reconstructed_truth_sha256=value_hash(reconstructed_truth),truth_array_hashes={k:array_hash(trial['truth'][k]) for k in ('r_pu','x_pu','p','q','source_p','source_q')},H_array_hashes={k:array_hash(getattr(h,k)) for k in CHANNELS if getattr(h,k) is not None},nominal_method_count=len(records),nominal_array_checks=checks,deterministic_make_path_without_truth_override=True,exact_arrays_required=True)
    return path,h,white,record


def _candidate_bank(trial):
    definitions={r['name']:r for r in trial['historical_fits']};candidates=[]
    for name in trial['candidate_names']:
        row=definitions[name];candidate=RadialCandidate.from_supports(name,3,_family(row['supports'],3))
        if [list(s) for s in candidate.supports]!=row['supports']: raise ValueError('saved candidate parameter ordering differs')
        candidates.append(candidate)
    return candidates


def run_trial(nominal,expected,source_audit,*,input_sha256):
    began=perf_counter();names,true_name=nominal['candidate_names'],nominal['true_name']
    result={k:nominal[k] for k in ('history_id','seed','case','quality','policy_id','config','source_sha256','history_observations','history_hash','historical_fits','historical_failures','truth','true_name','true_covered','candidate_names','seeds','white_evaluation_noise','device_constraints','quality_record')}
    result.update(schema='fixed_policy_voltage_averaging_stress_v1',nominal_trial_sha256=input_sha256,rho=RHO,threshold=THRESHOLD,assumed_measurement_models=nominal['measurement_models'],records=[],errors=[],same_saved_actions=True,selector_rerun=False,paired_reused_history=True,extra_independent_history_count=0,nominal_calibration_risk_transferred=False,global_MLE_certified=False)
    originals={r['method']:r for r in nominal['records']}
    try:
        path,history,white,checks=reconstruct_nominal(nominal,expected);result['nominal_reconstruction']=checks;result['errors'].extend(checks['errors'])
        if not source_audit['success']: result['errors'].extend(source_audit['errors'])
        hmodel=meter_from_record(nominal['measurement_models']['history']);nmodel=meter_from_record(nominal['measurement_models']['future']);joint_model=meter_from_record(nominal['measurement_models']['joint'])
        actual=actual_generation_model(hmodel,nmodel,nominal['quality_record']['v_repeat_counts']);result['true_generation_model']=meter_to_record(actual)
        if result['errors']: raise ValueError('nominal reconstruction or source check failed')
        candidates=_candidate_bank(nominal);initial={r['name']:r for r in nominal['historical_fits'] if r['success'] and r['loss'] is not None}
        constraints=SingleSiteConstraints(**nominal['device_constraints']);cache={}
        for method in frozen.METHODS:
            started=perf_counter();original=originals[method]
            row=dict(method=method,action=original.get('action'),selection=nominal['selection']['methods'][method],nominal=_nominal_summary(original,names,true_name),fits=[],losses={},fit_failures={},fit_calls=0,ac_calls=0,fit_seconds=0.,equivalent_action_fit_cache_hit=False,true_name=true_name,true_covered=nominal['true_covered'])
            try:
                if row['action'] is None: raise ValueError('saved action unavailable')
                plan=_plan(row['action'])
                validation=dict(feasible=bool(np.all(plan.p==0) and np.all(plan.q==0)),no_EV=True,soc_constant=bool(np.all(plan.soc==constraints.initial_soc))) if method=='none' else check_single_site_plan(plan,constraints)
                row['validation']=validation
                if not validation['feasible']: row['status']='operational_failure'
                else:
                    future,physical=frozen.future_observe(path,plan,actual,white);row['observations']=old.observations_record(future)
                    row['voltage_min'],row['voltage_max']=float(abs(physical['voltage']).min()),float(abs(physical['voltage']).max());row['operational_voltage_ok']=.9<=row['voltage_min'] and row['voltage_max']<=1.1
                    if not row['operational_voltage_ok']: row['status']='physical_failure'
                    else:
                        key=_action_key(row['action']);cached=key in cache;row['equivalent_action_fit_cache_hit']=cached
                        if not cached: cache[key]=old.fit_all_candidates(candidates,concat_observations(history,future),joint_model,initial=initial,seed=nominal['seeds']['policy']+200,fit_options=nominal['config']['evaluation_fit_options'])
                        fits,failed,cost=cache[key]
                        row.update(fits=[old.fit_record(c,fits[c.name]) for c in candidates if c.name in fits],losses={c.name:float(fits[c.name].loss) for c in candidates if c.name in fits and fits[c.name].success and np.isfinite(fits[c.name].loss)},fit_failures=failed,fit_calls=0 if cached else cost['fit_calls'],ac_calls=0 if cached else cost['ac_calls'],fit_seconds=0. if cached else cost['seconds'])
                        row['status']='complete' if len(row['losses'])==len(names) else 'joint_fit_failure'
                        if row['status']=='complete':
                            best=min(row['losses'].values());row.update(gaps={n:v-best for n,v in row['losses'].items()},winner=min(row['losses'],key=lambda n:(row['losses'][n],n)),true_gap=row['losses'][true_name]-best,signed_margin_full_SSE=min(v for n,v in row['losses'].items() if n!=true_name)-row['losses'][true_name])
            except (ValueError,RuntimeError,FloatingPointError,KeyError,TypeError,np.linalg.LinAlgError) as error: row.update(status='stress_acquisition_or_fit_failure',error=type(error).__name__+':'+str(error))
            row.update(decision(row,names,true_name));row['outcome']=outcome(row);row['seconds']=perf_counter()-started;result['records'].append(row)
        result['joint_unique_action_count']=len(cache)
    except (ValueError,RuntimeError,KeyError,TypeError,FloatingPointError,np.linalg.LinAlgError) as error:
        result['errors'].append(type(error).__name__+':'+str(error))
        for method in frozen.METHODS:
            original=originals.get(method,{'status':'missing_nominal_record'});row=dict(method=method,status='nominal_reconstruction_failure',action=original.get('action'),nominal=_nominal_summary(original,names,true_name),fits=[],losses={},fit_calls=0,ac_calls=0,fit_seconds=0.,equivalent_action_fit_cache_hit=False,error=';'.join(result['errors']))
            row.update(decision(row,names,true_name));row['outcome']=outcome(row);result['records'].append(row)
        result['joint_unique_action_count']=0
    result.update(complete=not result['errors'] and len(result['records'])==11 and all(r['status']=='complete' for r in result['records']),actual_joint_fit_calls=sum(r['fit_calls'] for r in result['records']),actual_joint_ac_calls=sum(r['ac_calls'] for r in result['records']),fit_cache_hits=sum(r['equivalent_action_fit_cache_hit'] for r in result['records']),any_method_false_rejection=any(r['false_rejection'] for r in result['records']),seconds=perf_counter()-began,runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__))
    return old.serial(result)


def replay_trial(stress,nominal):
    """Replay without optimizer/generator; actual generation and likelihood differ."""
    errors,methods=[],[];history=_obs(stress['history_observations'])
    hm=meter_from_record(stress['assumed_measurement_models']['history']);nm=meter_from_record(stress['assumed_measurement_models']['future']);jm=meter_from_record(stress['assumed_measurement_models']['joint'])
    actual=meter_from_record(stress['true_generation_model']) if 'true_generation_model' in stress else None
    for key in ('history_observations','truth','white_evaluation_noise'):
        if stress[key]!=nominal[key]: errors.append('nominal_changed:'+key)
    if stress['assumed_measurement_models']!=nominal['measurement_models']: errors.append('assumed_model_changed')
    if stress['rho']!=RHO or stress['threshold']!=THRESHOLD: errors.append('stress_protocol_constants')
    expected=actual_generation_model(hm,nm,stress['quality_record']['v_repeat_counts'])
    if actual is not None:
        ar,nr=meter_to_record(actual),meter_to_record(nm)
        if ar!=meter_to_record(expected): errors.append('true_generation_sigma_formula')
        for key in ar:
            if key!='v_sigma' and ar[key]!=nr[key]: errors.append('other_generation_changed:'+key)
    hfits=[replay_fit(row,history,hm) for row in stress['historical_fits']]
    for f in hfits: errors.extend('H:'+f['name']+':'+e for e in f['errors'])
    constraints=SingleSiteConstraints(**stress['device_constraints']);originals={r['method']:r for r in nominal['records']};seen={}
    for row in stress['records']:
        local,fit_replays=[],[];original=originals[row['method']]
        if row.get('action')!=original.get('action'): local.append('saved_action_changed')
        if row['nominal']!=_nominal_summary(original,stress['candidate_names'],stress['true_name']): local.append('nominal_paired_outcome')
        replay_decisions=decision(dict(status=row['status'],losses={}),stress['candidate_names'],stress['true_name'])
        operation=dict(feasible=False,recorded_failure=True)
        if row.get('action') is not None:
            plan=_plan(row['action'])
            operation=dict(feasible=bool(np.all(plan.p==0) and np.all(plan.q==0) and np.all(plan.soc==constraints.initial_soc)),no_EV=True) if row['method']=='none' else check_single_site_plan(plan,constraints)
            if 'observations' in row:
                if not operation['feasible']: local.append('device_or_SOC')
                physical,generated=_truth_future(dict(truth=stress['truth'],history_observations=stress['history_observations']),row['action'],actual,stress['white_evaluation_noise'])
                if physical['ac_equation_residual']>2e-10 or physical['power_balance_residual']>2e-10: local.append('actual_AC_balance')
                for channel,values in generated.items():
                    _close(values,row['observations'][channel],'actual_generation:'+channel,local,atol=2e-10,rtol=1e-10)
                    if channel!='v': _close(row['observations'][channel],original['observations'][channel],'unchanged_nominal:'+channel,local,atol=0.,rtol=0.)
                one=np.asarray(stress['quality_record']['v_repeat_counts'])==1
                _close(np.asarray(row['observations']['v'])[one],np.asarray(original['observations']['v'])[one],'M1_V_unchanged',local,atol=0.,rtol=0.)
                operational=bool(abs(physical['voltage']).min()>=.9 and abs(physical['voltage']).max()<=1.1)
                if row['operational_voltage_ok']!=operational: local.append('actual_voltage_flag')
                for key,value in (('voltage_min',float(abs(physical['voltage']).min())),('voltage_max',float(abs(physical['voltage']).max()))): _close(value,row[key],key,local,atol=2e-10,rtol=1e-10)
                joint=concat_observations(history,_obs(row['observations']));fit_replays=[replay_fit(fit,joint,jm) for fit in row['fits']]
                for fit in fit_replays: local.extend(fit['name']+':'+e for e in fit['errors'])
                key=_action_key(row['action']);cached=key in seen
                if row['fit_calls'] > 0 or row['equivalent_action_fit_cache_hit']:
                    if row['equivalent_action_fit_cache_hit']!=cached: local.append('cache_hit_flag')
                    if cached:
                        if row['fits']!=seen[key]['fits'] or row['observations']!=seen[key]['observations']: local.append('cached_fits_or_acquisition')
                        if row['fit_calls']!=0 or row['ac_calls']!=0: local.append('cache_charged_execution')
                    else:
                        seen[key]=row
                        if row['fit_calls']!=len(stress['candidate_names']): local.append('fit_calls_candidate_denominator')
                losses={f['name']:f['full_loss'] for f in fit_replays if f.get('replayed') and not f['recorded_failed_fit']}
                if row['status']=='complete' and set(losses)!=set(stress['candidate_names']): local.append('complete_missing_candidate_fit')
                for name,value in row['losses'].items(): _close(losses.get(name),value,'assumed_loss:'+name,local)
                replay_decisions=decision(dict(status=row['status'],losses=losses),stress['candidate_names'],stress['true_name'])
                if row['status']=='complete':
                    _close(losses[stress['true_name']]-min(losses.values()),row['true_gap'],'true_gap',local)
                    _close(min(v for n,v in losses.items() if n!=stress['true_name'])-losses[stress['true_name']],row['signed_margin_full_SSE'],'signed_margin',local)
        for key,value in replay_decisions.items():
            if row[key]!=value: local.append('decision:'+key)
        if row['outcome']!=outcome(row): local.append('outcome')
        errors.extend(row['method']+':'+e for e in local)
        methods.append(dict(method=row['method'],success=not local,errors=local,operation=operation,recorded_status=row['status'],actual_generation_replayed='observations' in row,assumed_joint_fit_replays=fit_replays,decisions=replay_decisions))
    if len(stress['records'])!=11 or {r['method'] for r in stress['records']}!=set(frozen.METHODS): errors.append('method_denominator')
    if stress['actual_joint_fit_calls']!=sum(r['fit_calls'] for r in stress['records']) or stress['actual_joint_ac_calls']!=sum(r['ac_calls'] for r in stress['records']): errors.append('executed_cost_totals')
    if stress['joint_unique_action_count']!=len(seen): errors.append('unique_action_count')
    fits=hfits+[f for m in methods for f in m['assumed_joint_fit_replays']]
    return dict(history_id=stress['history_id'],success=not errors,errors=errors,method_count=len(methods),H_fit_replays=hfits,methods=methods,recorded_failures=sum(r['status']!='complete' for r in stress['records']),fit_record_replay_count=len(fits),attained_state_replay_count=sum(f.get('replayed',False) for f in fits),unique_action_count=len(seen),replay_uses_true_generation_model=True,replay_uses_assumed_joint_likelihood=True)


def summarize(trials):
    methods={}
    for method in frozen.METHODS:
        rows=[next(r for r in t['records'] if r['method']==method) for t in trials]
        transitions=Counter(r['nominal']['outcome']+' -> '+r['outcome'] for r in rows)
        margins=[r['signed_margin_full_SSE']-r['nominal']['signed_margin_full_SSE'] for r in rows if r['status']=='complete' and r['nominal']['status']=='complete']
        methods[method]=dict(all_paired_history_count=len(rows),complete_count=sum(r['status']=='complete' for r in rows),failure_count=sum(r['status']!='complete' for r in rows),unique_correct=sum(r['only_truth_retained'] for r in rows),false_rejections=sum(r['false_rejection'] for r in rows),unresolved_including_failures=sum(r['unresolved'] for r in rows),outcome_counts=dict(Counter(r['outcome'] for r in rows)),nominal_outcome_counts=dict(Counter(r['nominal']['outcome'] for r in rows)),paired_transition_counts=dict(transitions),paired_complete_margin_count=len(margins),mean_stress_minus_nominal_margin=float(np.mean(margins)) if margins else None,actual_joint_fit_calls=sum(r['fit_calls'] for r in rows),actual_joint_ac_calls=sum(r['ac_calls'] for r in rows),fit_cache_hit_count=sum(r['equivalent_action_fit_cache_hit'] for r in rows))
    return dict(schema='fixed_policy_voltage_averaging_stress_summary_v1',rho=RHO,fixed_nominal_threshold=THRESHOLD,planned_history_ids=[str(i) for i in IDS],retained_history_ids=[t['history_id'] for t in trials],paired_reused_history_count=len(trials),extra_independent_history_count=0,method_count=11,complete_history_count=sum(t['complete'] for t in trials),failure_history_count=sum(not t['complete'] for t in trials),any_method_false_rejection_history_count=sum(t['any_method_false_rejection'] for t in trials),nominal_any_method_false_rejection_history_count=sum(any(r['nominal']['false_rejection'] for r in t['records']) for t in trials),method_summary=methods,actual_joint_fit_calls=sum(t['actual_joint_fit_calls'] for t in trials),actual_joint_ac_calls=sum(t['actual_joint_ac_calls'] for t in trials),unique_action_count=sum(t['joint_unique_action_count'] for t in trials),fit_cache_hits=sum(t['fit_cache_hits'] for t in trials),logical_method_action_count=11*len(trials),logical_candidate_fit_count=44*len(trials),nominal_calibration_risk_transferred=False,diagnostic_scope='paired model misspecification on fixed first16 of nominal80; no superiority claim',actual_voltage_variance_ratio_to_assumed='1+rho*(M-1)',ratio_by_M={'1':1.,'5':2.,'9':3.},total_seconds=sum(t['seconds'] for t in trials))


def run_batch(nominal_root=NOMINAL_ROOT,output=OUTPUT,*,replay_only=False):
    nominal_root,output=Path(nominal_root),Path(output);expected=old.read_json(nominal_root/'freeze.json');cal=old.read_json(nominal_root/'calibration/thresholds.json')
    if cal['relative']['threshold']!=THRESHOLD or cal['policy_id']!=expected['policy_id'] or cal['relative']['vacuous']: raise ValueError('frozen nominal threshold/policy differs')
    source=audit_source(expected['source_sha256'],snapshot=nominal_root/'science_source_snapshot.json.gz')
    config=dict(schema='voltage_averaging_stress_protocol_v1',rho=RHO,threshold=THRESHOLD,history_ids=[str(i) for i in IDS],nominal_policy_id=expected['policy_id'],frozen_source_sha256=expected['source_sha256'],nominal_root=str(nominal_root.resolve()),stress_source_sha256=sha(HERE/'averaging_stress.py'),protocol_sha256=sha(HERE/'averaging_stress_protocol.md'),only_generation_change='future random V sigma',selector_rerun=False,extra_independent_history_count=0)
    if (output/'config.json').exists() and old.read_json(output/'config.json')!=config: raise ValueError('stress source/config changed; use fresh output')
    if not replay_only: old.write_json(output/'config.json',config)
    trials,audits,inputs=[],[],[]
    for seed in IDS:
        ip=nominal_root/'evaluation'/f'trial_{seed}.json.gz';nominal=old.read_json(ip);path=output/'raw'/f'trial_{seed}.json.gz'
        if path.exists():
            stress=old.read_json(path)
            if stress['nominal_trial_sha256']!=sha(ip): raise ValueError('nominal input changed')
        elif replay_only: raise ValueError('missing stress record:'+str(seed))
        else:
            stress=run_trial(nominal,expected,source,input_sha256=sha(ip));old.write_json(path,stress,compressed=True)
        try:
            audit=replay_trial(stress,nominal)
        except Exception as error:
            # Audit exceptions are retained as failures for this planned history.
            audit=dict(history_id=str(seed),success=False,errors=['replay_exception:'+type(error).__name__+':'+str(error)],method_count=len(stress['records']),recorded_failures=sum(r['status']!='complete' for r in stress['records']),fit_record_replay_count=0,attained_state_replay_count=0,unique_action_count=stress['joint_unique_action_count'],methods=[])
        rp=output/'replay'/f'trial_{seed}.json.gz';old.write_json(rp,audit,compressed=True)
        trials.append(stress);audits.append(audit);inputs.append(dict(history_id=str(seed),input_sha256=sha(ip),raw_sha256=sha(path),replay_sha256=sha(rp)))
        old.write_json(output/'progress.json',dict(completed_history_count=len(trials),planned_history_count=16,last_history_id=str(seed),source_sha256=config['stress_source_sha256'],replay_success_count=sum(a['success'] for a in audits)))
        print(json.dumps(dict(history_id=str(seed),completed=len(trials),planned=16,complete=stress['complete'],replay_success=audit['success'],unique_actions=stress['joint_unique_action_count'],actual_fit_calls=stress['actual_joint_fit_calls'],seconds=stress['seconds'])),flush=True)
    summary=summarize(trials);old.write_json(output/'summary.json',summary);old.write_json(output/'source_audit.json',source)
    old.write_json(output/'replay_summary.json',dict(success=source['success'] and all(a['success'] for a in audits),paired_history_count=len(audits),failed_replay_count=sum(not a['success'] for a in audits),source_audit_success=source['success'],all_method_acquisition_count=sum(sum(m['actual_generation_replayed'] for m in a['methods']) for a in audits),fit_record_replay_count=sum(a['fit_record_replay_count'] for a in audits),attained_state_replay_count=sum(a['attained_state_replay_count'] for a in audits),recorded_method_failure_count=sum(a['recorded_failures'] for a in audits),results=[{k:a[k] for k in ('history_id','success','errors','method_count','recorded_failures','fit_record_replay_count','unique_action_count')} for a in audits],generation_model='true_generation_model',fitted_likelihood='assumed_measurement_models.joint',global_MLE_certified=False))
    old.write_json(output/'manifest.json',dict(config=config,inputs=inputs,raw_records_kept_local=True,compact_files=['config.json','summary.json','source_audit.json','replay_summary.json','manifest.json'],summary_sha256=sha(output/'summary.json'),replay_summary_sha256=sha(output/'replay_summary.json')))
    return trials,summary,audits


def main():
    p=argparse.ArgumentParser();p.add_argument('--nominal-root',type=Path,default=NOMINAL_ROOT);p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--replay-only',action='store_true');a=p.parse_args();run_batch(a.nominal_root,a.output,replay_only=a.replay_only)


if __name__=='__main__': main()
