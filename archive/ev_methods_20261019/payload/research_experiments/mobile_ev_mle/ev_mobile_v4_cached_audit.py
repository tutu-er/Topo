"""Read-only physical audit of completed offline target/refit caches.

Adapts attribution/gap histories to the existing independent branch-current
replay. No optimizer, production AC predictor, simulator, or selector is called.
Original target/candidate slots and deduplicated retained states are separate
denominators. Failed fits remain recorded outcomes rather than disappearing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from . import ev_mobile_v2_review as physical
from . import ev_mobile_v4_audit as audit
from . import gradient_price_study as io
from .likelihood import RadialCandidate


HERE=Path(__file__).resolve().parent
STATES=tuple(''.join(map(str,bits)) for bits in product((0,1),repeat=3))
FACTORS=('lines','background','source')


def fingerprint(value):
    return sha256(json.dumps(io.serial(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _cfg(meta):
    options=meta['fit_options']
    return dict(training={name:options[name] for name in ('r_bounds','x_bounds','voltage_bounds')},
        source={name:options[name] for name in ('source_p_bounds','source_q_bounds')},
        ratio_bounds=options['ratio_bounds'],source_interface=meta['source_interface'])


def _bank(meta):
    return {row['name']:RadialCandidate.from_supports(row['name'],row['n'],row['supports'])
            for row in meta['candidate_bank']}


def _target_request(meta,target):
    return fingerprint(dict(candidate=meta['true_name'],parameters=target['parameters'],
        power_kw=target['power_kw'],future_meter=meta['future_meter'],base_kw=meta['base_kw']))


def _fit_request(meta,target):
    return fingerprint(dict(observed=meta['observed'],observed_meter=meta['observed_meter'],
        target=target['observations'],future_meter=meta['future_meter'],trained=meta['trained'],
        options=meta['fit_options'],seed=meta['fit_seed'],candidate_bank=meta['candidate_bank']))


def _expected_parameters(meta,regime,state):
    predicted,truth=meta['predicted'],meta['true_by_regime'][regime]
    line=truth if state[0]=='1' else predicted
    load=truth if state[1]=='1' else predicted
    source=truth if state[2]=='1' else predicted
    return dict(r=line['r'],x=line['x'],p=load['p'],q=load['q'],
                sp=source['sp'],sq=source['sq'],root=predicted['root'])


def _audit_cached_fit(meta,target,cached,key,metrics,replay,attempt,absent):
    """The same measured-P/Q saved-state audit for clean and noisy targets."""
    bank,cfg=_bank(meta),_cfg(meta);outcome=cached['outcome'];actual_losses={}
    blocks=(('observed',meta['observed'],meta['observed_meter']),('future',target['observations'],meta['future_meter']))
    for name,candidate in bank.items():
        comparison=outcome.get('comparisons',{}).get(name)
        if comparison is None:absent(f'cached fit/{key}/{name}',str(outcome.get('failures',{}).get(name)));continue
        metrics['unique_failed_joint_candidates']+=not bool(comparison.get('joint',{}).get('success'))
        metrics['unique_failed_fixed_source_candidates']+=not bool(comparison.get('fixed_source_success'))
        for kind in ('fixed','joint'):
            saved=comparison.get(kind)
            if saved is None or saved.get('state') is None:absent(f'{kind} state/{key}/{name}');continue
            def fit_work(row=saved,c=candidate,mode=kind):
                audit._score_arithmetic(row,blocks,c.n)
                if mode=='fixed':
                    for line in ('r','x'):physical.close(row['state'][line],meta['trained'][c.name][line],'fixed theta_D '+line,atol=0.,rtol=0.)
                    for channel in ('source_p','source_q'):
                        physical.close(np.asarray(row['state'][channel])[:len(meta['observed']['p'])],meta['trained'][c.name][channel],'fixed observed source '+channel,atol=0.,rtol=0.)
                    physical.close(row['blocks']['observed']['loss'],meta['trained'][c.name]['loss'],'fixed D loss once')
                actual=replay(f'{mode}/{key}/{c.name}',c,row,row['state'],blocks,cfg)
                metrics['unique_'+mode+'_states']+=1
                if mode=='joint':
                    actual_losses[c.name]=actual['loss']
                    physical.close(outcome['losses'][c.name],actual['loss'],'cached profile loss')
                    fixed_score=comparison['fixed']
                    physical.require(row['loss']<=fixed_score['loss']+1e-8*(1+abs(fixed_score['loss'])),'joint attainment worse than fixed witness')
                    physical.close(row['feasible_witness']['loss'],fixed_score['loss'],'same fixed feasible witness')
            attempt(f'{kind} cached state/{key}/{name}',fit_work)
    if outcome.get('complete') and set(actual_losses)==set(bank):
        truth=meta['true_name']
        return min(v for k,v in actual_losses.items() if k!=truth)-actual_losses[truth]
    return None


def audit_directory(directory,output):
    began=perf_counter();directory,output=Path(directory),Path(output)
    if output.exists():raise FileExistsError(output)
    frozen=physical.read(directory/'freeze.json')
    indices=list(frozen['indices']) if 'indices' in frozen else [frozen['index']]
    regimes=list(frozen['regimes']) if 'regimes' in frozen else [frozen['regime']]
    methods=list(frozen['methods'])
    expected_slots=len(indices)*len(regimes)*len(methods)*len(STATES)
    expected_candidates=4*expected_slots
    errors,missing,failures,states,slot_rows=[],[],[],[],[]
    metrics=dict(saved_states_replayed=0,AC_block_replays=0,max_whitened_residual_error=0.,
        max_block_SSE_error=0.,max_total_SSE_error=0.,max_AC_equation=0.,max_power_balance=0.,
        training_states=0,target_states=0,unique_fixed_states=0,unique_joint_states=0,
        completed_slot_files=0,slot_candidate_denominator=expected_candidates,
        recorded_failed_joint_candidate_slots=0,recorded_failed_fixed_source_slots=0,
        unique_failed_joint_candidates=0,unique_failed_fixed_source_candidates=0,
        slot_margin_checks=0,EV_once_checks=0,max_noiseless_target_SSE=0.)
    target_cache,fit_cache,target_context,fit_context={},{},{},{}

    def attempt(label,work):
        try:return work()
        except audit.EXCEPTIONS as exc:
            errors.append(dict(check=label,reason=type(exc).__name__+': '+str(exc)));return None

    def absent(label,reason='saved evidence unavailable'):
        missing.append(dict(check=label,reason=reason))

    def replay(label,candidate,saved,state,blocks,cfg):
        actual=physical.replay_state(candidate,state,blocks,cfg)
        physical._compare_score(saved,actual,label,metrics,selected=bool(saved.get('success',saved.get('usable',True))))
        metrics['max_total_SSE_error']=max(metrics['max_total_SSE_error'],abs(saved['loss']-actual['loss']))
        states.append(dict(check=label,saved_loss=saved['loss'],replay_loss=actual['loss'],
            line_domain_feasible=actual['line_domain_feasible'],source_box_feasible=actual['source_box_feasible'],
            physical_voltage_feasible=actual['physical_voltage_feasible'],recorded_fit_success=saved.get('success')))
        return actual

    histories={}
    for index in indices:
        path=directory/f'history_{index:03d}'/'history.json.gz'
        if not path.exists():absent('history/'+str(index));continue
        meta=physical.read(path);histories[index]=meta;bank=_bank(meta);cfg=_cfg(meta)
        attempt('history identity/'+str(index),lambda:physical.require(
            meta['spec']['index']==index and set(meta['candidate_names'])==set(bank)
            and len(bank)==4 and meta['source_interface']==1,'history/candidate/source identity'))
        for name,candidate in bank.items():
            saved=meta['trained'].get(name)
            if saved is None:absent(f'training/{index}/{name}');continue
            def training_work(c=candidate,row=saved):
                physical.require(tuple(map(tuple,row['supports']))==c.supports,'training edge ordering')
                actual=replay(f'training/{index}/{c.name}',c,row,row,
                    (('observed',meta['observed'],meta['observed_meter']),),cfg)
                audit._training_predictions(row,meta['observed'],meta['observed_meter'],actual)
                metrics['training_states']+=1
            attempt(f'training/{index}/{name}',training_work)
        for regime in regimes:
            for method in methods:
                for state in STATES:
                    label=f'{index}/{regime}/{method}/{state}'
                    slot_path=path.parent/regime/(method+'_'+state+'.json')
                    if not slot_path.exists():absent('slot/'+label);continue
                    slot=physical.read(slot_path);metrics['completed_slot_files']+=1
                    def link_work(row=slot):
                        physical.require((row['index'],row['regime'],row['method'],row['state'])==(index,regime,method,state),'slot identity')
                        physical.require(row['factors']==dict(zip(FACTORS,map(int,state))),'slot factor flags')
                        request=row['target_request'];target_path=directory/'targets'/(request+'.json.gz')
                        physical.require(sha256(target_path.read_bytes()).hexdigest()==row['target_source']['sha256'],'target file hash')
                        if request not in target_cache:target_cache[request]=physical.read(target_path)
                        target=target_cache[request]
                        physical.require(target.get('success') is True,'target generation failed')
                        physical.require(target['target_request']==request==_target_request(meta,target),'target request identity')
                        physical.require(fingerprint(target['parameters'])==fingerprint(_expected_parameters(meta,regime,state)),'slot factor parameters differ from stored history')
                        physical.close(target['power_kw'],meta['powers'][method],'retained method powers',atol=0.,rtol=0.)
                        physical.require(target['true_reference']==meta['true_name'],'target reference changed')
                        target_context.setdefault(request,(meta,target))
                        identity=row['fit_identity'];fit_path=directory/'fit_cache'/(identity+'.json.gz')
                        physical.require(sha256(fit_path.read_bytes()).hexdigest()==row['fit_source']['sha256'],'fit cache file hash')
                        if identity not in fit_cache:fit_cache[identity]=physical.read(fit_path)
                        fitted=fit_cache[identity]
                        physical.require(identity==fitted['fit_identity']==_fit_request(meta,target),'same D/target/options/seed fit identity')
                        physical.require(fitted['fit_seed']==meta['fit_seed'],'fit seed identity')
                        fit_context.setdefault(identity,(meta,target,fitted))
                        outcome=fitted['outcome'];comparison=outcome.get('comparisons',{})
                        complete=set(comparison)==set(bank) and all(comparison[k]['joint'].get('success') for k in bank)
                        physical.require(bool(outcome['complete'])==complete==bool(row['complete']),'slot/cache completeness flags')
                        succeeded=sum(bool(comparison.get(k,{}).get('joint',{}).get('success')) for k in bank)
                        seen=(set(comparison)|set(outcome.get('failures',{})))&set(bank)
                        expected=dict(expected=4,attempted=len(seen),success=succeeded,failed=len(seen)-succeeded,not_run=4-len(seen))
                        physical.require(row['candidate_slots']==expected,'candidate slot denominators')
                        metrics['recorded_failed_joint_candidate_slots']+=4-succeeded
                        metrics['recorded_failed_fixed_source_slots']+=sum(not comparison.get(k,{}).get('fixed_source_success',False) for k in bank)
                        if complete:
                            losses={k:comparison[k]['joint']['loss'] for k in bank};truth=meta['true_name']
                            M=min(v for k,v in losses.items() if k!=truth)-losses[truth]
                            physical.close(row['M'],M,'slot full-SSE margin')
                            physical.close(row['halfM'],.5*M,'slot half-SSE margin')
                            physical.require(row['status']=='complete','successful slot status')
                            metrics['slot_margin_checks']+=1
                        else:
                            physical.require(row['M'] is None and row['halfM'] is None,'failed slot has no complete margin')
                            failures.append(dict(kind='slot_fit_failure',slot=label,failures=outcome.get('failures',{})))
                        slot_rows.append(dict(slot=label,target_request=request,fit_identity=identity,complete=complete,candidate_slots=expected))
                    attempt('slot links/'+label,link_work)

    for key,(meta,target) in target_context.items():
        def target_work():
            n=len(meta['observed']['p'][0]);parameters=target['parameters'];obs=target['observations'];fm=meta['future_meter']
            T=len(obs['p']);shape=(T,n)
            physical.require(target['physical'].get('future_white_noise') is False,'only saved noiseless target audit is supported here')
            physical.close(np.asarray(obs['p'])-physical._array(fm['bias_p'],shape),
                np.asarray(parameters['p'])+np.asarray(target['power_kw'])/meta['base_kw'],'target EV P exactly once',atol=2e-15,rtol=2e-13)
            physical.close(np.asarray(obs['q'])-physical._array(fm['bias_q'],shape),parameters['q'],'target fixed Q',atol=2e-15,rtol=2e-13)
            physical.close(obs['root_v'],parameters['root'],'target root',atol=0.,rtol=0.)
            state=dict(r=parameters['r'],x=parameters['x'],source_p=parameters['sp'],source_q=parameters['sq'])
            candidate=_bank(meta)[meta['true_name']]
            actual=physical.replay_state(candidate,state,(('future',obs,fm),),_cfg(meta))
            physical.require(actual['line_domain_feasible'] and actual['source_box_feasible'] and actual['physical_voltage_feasible'],'target original physical domain')
            physical.require(actual['loss']<1e-10,'saved noiseless target differs from its parameter/input AC response')
            metrics['target_states']+=1;metrics['EV_once_checks']+=1
            metrics['saved_states_replayed']+=1;metrics['AC_block_replays']+=1
            metrics['max_noiseless_target_SSE']=max(metrics['max_noiseless_target_SSE'],actual['loss'])
            metrics['max_AC_equation']=max(metrics['max_AC_equation'],actual['max_AC_equation'])
            metrics['max_power_balance']=max(metrics['max_power_balance'],actual['max_power_balance'])
        attempt('target physics/'+key,target_work)

    for key,(meta,target,cached) in fit_context.items():
        M=_audit_cached_fit(meta,target,cached,key,metrics,replay,attempt,absent)
        if M is not None:
            for slot in slot_rows:
                if slot['fit_identity']==key:
                    slot['replayed_M']=M
    physical_targets={path.name.split('.')[0] for path in (directory/'targets').glob('*.json.gz')}
    physical_caches={path.name.split('.')[0] for path in (directory/'fit_cache').glob('*.json.gz')}
    attempt('target cache coverage',lambda:physical.require(set(target_context)==physical_targets,'unreferenced or missing target cache'))
    attempt('fit cache coverage',lambda:physical.require(set(fit_context)==physical_caches,'unreferenced or missing fit cache'))
    if metrics['completed_slot_files']!=expected_slots:absent('target slot coverage',f"{metrics['completed_slot_files']}/{expected_slots}")
    source_status={}
    for name,expected in frozen.get('source_sha256',{}).items():
        path=HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name
        source_status[name]=path.is_file() and sha256(path.read_bytes()).hexdigest()==expected
    result=dict(success=not errors and not missing and all(source_status.values()),
        available_checks_passed=not errors,complete_saved_state_coverage=not missing,
        input_directory=str(directory.resolve()),policy_id=frozen.get('policy_id'),
        expected_target_slots=expected_slots,expected_joint_candidate_slots=expected_candidates,
        completed_target_slots=metrics['completed_slot_files'],
        unique_target_cache_files=len(physical_targets),unique_fit_cache_files=len(physical_caches),
        unique_joint_candidate_denominator=4*len(physical_caches),
        unique_fixed_candidate_denominator=4*len(physical_caches),
        metrics=metrics,errors=errors,missing=missing,recorded_failures=failures,states=states,slots=slot_rows,
        scientific_source_count=len(source_status),all_scientific_sources_unchanged=all(source_status.values()),
        source_hash_checks=source_status,seconds=perf_counter()-began,
        optimizer_calls=0,production_AC_calls=0,new_simulation_calls=0,action_selection_calls=0,
        scope='all retained observed-D fits, noiseless targets, fixed and joint fitted states; slot/cache identities and SSE margins',
        guarantee='independent saved-state physical/likelihood consistency; no global optimum or statistical calibration certificate',
        freeze_sha256=sha256((directory/'freeze.json').read_bytes()).hexdigest(),
        auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    physical.write(output,io.serial(result));return result


NOISE_CHANNELS=('p','q','v','master_p','master_q')


def check_noise_relation(clean,noisy,meter,white):
    """Known offsets and EV are already in clean; only sigma*white is added."""
    physical.require(set(white)==set(NOISE_CHANNELS),'five paired white-noise channels required')
    maximum=0.;input_error={}
    for channel in NOISE_CHANNELS:
        mean,value=np.asarray(clean[channel],float),np.asarray(noisy[channel],float)
        eps=np.asarray(white[channel],float)
        physical.require(mean.shape==value.shape==eps.shape,'noise channel dimensions '+channel)
        sigma=physical._array(meter[channel+'_sigma'],mean.shape)
        physical.require(np.all(sigma>0.) and np.isfinite(eps).all(),'positive noise sigma and finite white '+channel)
        expected=mean+sigma*eps
        physical.close(value,expected,'clean + sigma * white, known bias/EV once: '+channel,atol=3e-15,rtol=2e-13)
        maximum=max(maximum,float(np.max(abs(value-expected))))
        if channel in ('p','q'):input_error[channel]=float(np.sqrt(np.mean((sigma*eps)**2)))
    for channel in ('root_v','source_p','source_q'):
        a,b=clean.get(channel),noisy.get(channel)
        if a is None or b is None:physical.require(a is None and b is None,'unperturbed optional channel '+channel)
        else:physical.close(a,b,'unperturbed channel '+channel,atol=0.,rtol=0.)
    return dict(max_relation_error=maximum,measured_input_noise_RMS_pu=input_error,
        channel_count=5,known_bias_added_again=False,EV_added_again=False)


def audit_noise_directory(directory,output):
    """Audit a fully saved conditional-noise diagnostic, without sampling noise.

    True physical P/Q is used only for the two clean target checks. All saved
    fitted states are replayed conditional on their noisy measured P/Q inputs.
    No zero-residual requirement is imposed on noisy targets or fitted states.
    """
    began=perf_counter();directory,output=Path(directory),Path(output)
    if output.exists():raise FileExistsError(output)
    frozen=physical.read(directory/'freeze.json');inputs=physical.read(directory/'inputs.json.gz')
    seeds,methods=list(frozen['seeds']),list(frozen['methods'])
    expected_slots=len(seeds)*len(methods);expected_candidates=4*expected_slots
    expected_paths=[directory/'replicates'/str(seed)/(method+'.json') for seed in seeds for method in methods]
    physical.require(all(path.is_file() for path in expected_paths),'wait for every declared noise slot before final audit')
    physical.require(frozen['target_slots']==expected_slots and frozen['candidate_slots']==expected_candidates,'frozen noise denominators')
    physical.require(fingerprint(inputs)==frozen['inputs_sha256'],'fixed inputs identity')
    meta=inputs['meta'];bank,cfg=_bank(meta),_cfg(meta)
    physical.require(set(meta['candidate_names'])==set(bank) and len(bank)==4 and meta['source_interface']==1,'fixed noise history/candidate/source')
    errors,missing,failures,states,slot_rows=[],[],[],[],[]
    metrics=dict(saved_states_replayed=0,AC_block_replays=0,max_whitened_residual_error=0.,
        max_block_SSE_error=0.,max_total_SSE_error=0.,max_AC_equation=0.,max_power_balance=0.,
        training_states=0,clean_target_states=0,unique_fixed_states=0,unique_joint_states=0,
        completed_slot_files=0,slot_candidate_denominator=expected_candidates,
        recorded_failed_joint_candidate_slots=0,recorded_failed_fixed_source_slots=0,
        unique_failed_joint_candidates=0,unique_failed_fixed_source_candidates=0,
        slot_margin_checks=0,EV_once_clean_checks=0,noise_channel_relation_checks=0,
        paired_white_seed_checks=0,max_noise_relation_error=0.,max_clean_target_SSE=0.)
    fit_context={};used_targets=set();white_sources={}
    def attempt(label,work):
        try:return work()
        except audit.EXCEPTIONS as exc:
            errors.append(dict(check=label,reason=type(exc).__name__+': '+str(exc)));return None
    def absent(label,reason='saved evidence unavailable'):missing.append(dict(check=label,reason=reason))
    def replay(label,candidate,saved,state,blocks,configuration):
        actual=physical.replay_state(candidate,state,blocks,configuration)
        physical._compare_score(saved,actual,label,metrics,selected=bool(saved.get('success',saved.get('usable',True))))
        metrics['max_total_SSE_error']=max(metrics['max_total_SSE_error'],abs(saved['loss']-actual['loss']))
        states.append(dict(check=label,saved_loss=saved['loss'],replay_loss=actual['loss'],
            line_domain_feasible=actual['line_domain_feasible'],source_box_feasible=actual['source_box_feasible'],
            physical_voltage_feasible=actual['physical_voltage_feasible'],recorded_fit_success=saved.get('success')))
        return actual
    for name,candidate in bank.items():
        def training_work(c=candidate):
            row=meta['trained'][c.name]
            physical.require(tuple(map(tuple,row['supports']))==c.supports,'training edge ordering')
            checked=replay('training/'+c.name,c,row,row,(('observed',meta['observed'],meta['observed_meter']),),cfg)
            audit._training_predictions(row,meta['observed'],meta['observed_meter'],checked)
            metrics['training_states']+=1
        attempt('training/'+name,training_work)
    true=meta['true_by_regime'][frozen['regime']]
    for method in methods:
        def clean_work(m=method):
            obs=inputs['clean'][m];T,n=np.asarray(obs['p']).shape;meter=meta['future_meter']
            physical.close(np.asarray(obs['p'])-physical._array(meter['bias_p'],(T,n)),
                np.asarray(true['p'])+np.asarray(meta['powers'][m])/meta['base_kw'],'clean true physical P + EV once',atol=2e-15,rtol=2e-13)
            physical.close(np.asarray(obs['q'])-physical._array(meter['bias_q'],(T,n)),true['q'],'clean true physical Q',atol=2e-15,rtol=2e-13)
            state=dict(r=true['r'],x=true['x'],source_p=true['sp'],source_q=true['sq'])
            checked=physical.replay_state(bank[meta['true_name']],state,(('future',obs,meter),),cfg)
            physical.require(checked['line_domain_feasible'] and checked['source_box_feasible'] and checked['physical_voltage_feasible'],'clean target physical domain')
            physical.require(checked['loss']<1e-10,'clean target differs from its true physical AC mean')
            metrics['clean_target_states']+=1;metrics['EV_once_clean_checks']+=1
            metrics['saved_states_replayed']+=1;metrics['AC_block_replays']+=1
            metrics['max_clean_target_SSE']=max(metrics['max_clean_target_SSE'],checked['loss'])
            metrics['max_AC_equation']=max(metrics['max_AC_equation'],checked['max_AC_equation'])
            metrics['max_power_balance']=max(metrics['max_power_balance'],checked['max_power_balance'])
        attempt('clean target/'+method,clean_work)
    for seed in seeds:
        noise_path=directory/'white_noise'/(str(seed)+'.json.gz');white_record=physical.read(noise_path)
        white_identity=fingerprint(white_record['white']);white_sha=sha256(noise_path.read_bytes()).hexdigest()
        attempt('white identity/'+str(seed),lambda:physical.require(white_record['seed']==seed and
            white_record['white_sha256']==white_identity,'saved white seed/hash identity'))
        references=[]
        for method in methods:
            slot_path=directory/'replicates'/str(seed)/(method+'.json');row=physical.read(slot_path)
            metrics['completed_slot_files']+=1
            def slot_work(m=method,slot=row):
                physical.require((slot['seed'],slot['method'],slot['history_index'],slot['regime'])==
                    (seed,m,frozen['history_index'],frozen['regime']),'noise slot identity')
                physical.require(slot['white_sha256']==white_identity,'both actions use identical saved white')
                target_path=directory/'targets'/str(seed)/(m+'.json.gz')
                physical.require(sha256(target_path.read_bytes()).hexdigest()==slot['target_source']['sha256'],'noisy target file hash')
                target=physical.read(target_path)
                physical.require(target.get('success') is True,'noisy target generation failed')
                physical.require(target['white_source']['sha256']==white_sha,'shared white file identity')
                physical.require(target['clean_observation_sha256']==fingerprint(inputs['clean'][m]) and
                    target['history_sha256']==fingerprint(meta['observed']),'fixed clean/history identity')
                relation=check_noise_relation(inputs['clean'][m],target['observations'],meta['future_meter'],white_record['white'])
                metrics['noise_channel_relation_checks']+=relation['channel_count']
                metrics['max_noise_relation_error']=max(metrics['max_noise_relation_error'],relation['max_relation_error'])
                references.append(target['white_source']['sha256']);used_targets.add((seed,m))
                identity=slot['fit_identity'];fit_path=directory/'fit_cache'/(identity+'.json.gz')
                physical.require(sha256(fit_path.read_bytes()).hexdigest()==slot['fit_source']['sha256'],'noise fit cache file hash')
                cached=physical.read(fit_path)
                physical.require(cached['fit_identity']==identity==_fit_request(meta,target),'same measured-PQ D/target fit identity')
                physical.require(cached.get('fit_seed',meta['fit_seed'])==meta['fit_seed'],'same saved fit seed')
                fit_context.setdefault(identity,(meta,target,cached))
                outcome=cached['outcome'];comparison=outcome.get('comparisons',{})
                succeeded=sum(bool(comparison.get(k,{}).get('joint',{}).get('success')) for k in bank)
                complete=set(comparison)==set(bank) and succeeded==4
                physical.require(bool(outcome['complete'])==bool(slot['complete'])==complete,'noise completeness flags')
                seen=(set(comparison)|set(outcome.get('failures',{})))&set(bank)
                expected=dict(expected=4,attempted=len(seen),success=succeeded,failed=len(seen)-succeeded,not_run=4-len(seen))
                physical.require(slot['candidate_slots']==expected,'noise candidate slot denominators')
                metrics['recorded_failed_joint_candidate_slots']+=4-succeeded
                metrics['recorded_failed_fixed_source_slots']+=sum(not comparison.get(k,{}).get('fixed_source_success',False) for k in bank)
                if complete:
                    losses={k:comparison[k]['joint']['loss'] for k in bank};truth_name=meta['true_name']
                    M=min(value for name,value in losses.items() if name!=truth_name)-losses[truth_name]
                    physical.close(slot['M'],M,'noise full-SSE margin');physical.close(slot['halfM'],.5*M,'noise half-SSE margin')
                    physical.require(slot['status']=='complete','successful noise slot status');metrics['slot_margin_checks']+=1
                else:
                    physical.require(slot['M'] is None and slot['halfM'] is None,'failed noise slot has no complete margin')
                    failures.append(dict(seed=seed,method=m,kind='fit_failure',failures=outcome.get('failures',{})))
                slot_rows.append(dict(seed=seed,method=m,fit_identity=identity,complete=complete,
                    candidate_slots=expected,measured_input_noise_RMS_pu=relation['measured_input_noise_RMS_pu']))
            attempt(f'noise slot/{seed}/{method}',slot_work)
        if len(references)==len(methods) and len(set(references))==1:
            metrics['paired_white_seed_checks']+=1;white_sources[seed]=references[0]
        else:absent('paired white/'+str(seed),'one saved white file must be shared by both actions')
    for key,(metadata,target,cached) in fit_context.items():
        M=_audit_cached_fit(metadata,target,cached,key,metrics,replay,attempt,absent)
        if M is not None:
            for row in slot_rows:
                if row['fit_identity']==key:row['replayed_M']=M
    caches={path.name.split('.')[0] for path in (directory/'fit_cache').glob('*.json.gz')}
    attempt('noise cache coverage',lambda:physical.require(set(fit_context)==caches,'unreferenced or missing noise fit cache'))
    attempt('noise target coverage',lambda:physical.require(len(used_targets)==expected_slots,'noisy target coverage'))
    source_status={}
    for name,expected in frozen['source_sha256'].items():
        path=HERE.parents[1]/name if name.startswith('rnj_wzzt_core/') else HERE/name
        source_status[name]=path.is_file() and sha256(path.read_bytes()).hexdigest()==expected
    result=dict(success=not errors and not missing and all(source_status.values()),available_checks_passed=not errors,
        complete_saved_state_coverage=not missing,input_directory=str(directory.resolve()),policy_id=frozen['policy_id'],
        independent_history_denominator=1,conditional_noise_seed_denominator=len(seeds),
        expected_target_slots=expected_slots,expected_joint_candidate_slots=expected_candidates,
        completed_target_slots=metrics['completed_slot_files'],unique_fit_cache_files=len(caches),
        unique_joint_candidate_denominator=4*len(caches),unique_fixed_candidate_denominator=4*len(caches),
        metrics=metrics,errors=errors,missing=missing,recorded_failures=failures,states=states,slots=slot_rows,
        paired_white_sources=white_sources,scientific_source_count=len(source_status),
        all_scientific_sources_unchanged=all(source_status.values()),source_hash_checks=source_status,
        optimizer_calls=0,production_AC_calls=0,new_simulation_calls=0,action_selection_calls=0,
        true_input_scope='true physical P/Q/source only verify clean fixed-action means; no source/location change',
        fitted_input_scope='saved fits condition on noisy measured P/Q after the same known offsets; no P/Q-noise propagation added',
        noisy_target_zero_residual_required=False,
        scope='saved clean means, clean+sigma*white relations for five channels, same-seed common noise, all fixed/joint fitted states',
        guarantee='saved-state physical and likelihood consistency; conditional repeats are one history, no global optimum or calibration certificate',
        freeze_sha256=sha256((directory/'freeze.json').read_bytes()).hexdigest(),
        inputs_file_sha256=sha256((directory/'inputs.json.gz').read_bytes()).hexdigest(),
        auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),seconds=perf_counter()-began)
    physical.write(output,io.serial(result));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    frozen=physical.read(Path(args.directory)/'freeze.json')
    noise=frozen.get('schema')=='mobile_v4_development_conditional_noise_freeze_v1'
    result=(audit_noise_directory if noise else audit_directory)(args.directory,args.output)
    print(json.dumps({key:result[key] for key in ('success','expected_target_slots','expected_joint_candidate_slots',
        'completed_target_slots','unique_fit_cache_files','metrics','errors','missing')}))


if __name__=='__main__':main()
