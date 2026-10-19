"""Replay saved dynamic-EV records without fitting, planning or production AC.

Every numerical fit check uses the separate branch-current/voltage sweep.
Missing or failed experiment records remain in the denominator. Forecast
reconstruction needs the current load/forecast module to match its frozen
bytes; arbitrary code from a snapshot is never executed. Calibration and
false-rejection rates are outside this arithmetic replay.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from dataclasses import fields
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import ev_dynamic_loads as loads
from .ev_fixed_train_review import review_fixed_fit
from .ev_original_review import forward_physics
from .likelihood import MeasurementModel, Observations, RadialCandidate

HERE = Path(__file__).resolve().parent
CHANNELS = ('p', 'q', 'v', 'master_p', 'master_q')
OBS_CHANNELS = (*CHANNELS, 'root_v', 'source_p', 'source_q')


def _serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [_serial(v) for v in value]
    return value


def _identity(value):
    return hashlib.sha256(json.dumps(_serial(value), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _read(path):
    data = Path(path).read_bytes()
    return json.loads(gzip.decompress(data) if str(path).endswith('.gz') else data)


def _require(ok, message):
    if not ok: raise ValueError(message)


def _close(actual, expected, message, *, atol=2e-7, rtol=5e-8):
    a, b = np.asarray(actual, float), np.asarray(expected, float)
    _require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        and np.allclose(a, b, atol=atol, rtol=rtol), message)
    return float(np.max(abs(a - b), initial=0.))


def _obs(row):
    return Observations(**{k: row.get(k) for k in OBS_CHANNELS})


def _meter(row):
    expected = {f.name for f in fields(MeasurementModel)}
    _require(set(row) == expected, 'measurement-model schema differs')
    return MeasurementModel(**{k: np.asarray(v, float) if isinstance(v, list) else v for k, v in row.items()})


def _candidate(name, supports, n):
    candidate = RadialCandidate.from_supports(name, n,
        tuple(tuple(s) for s in supports if 1 < len(s) < n))
    _require(candidate.supports == tuple(tuple(s) for s in supports), 'candidate edge/support order changed')
    return candidate


def _observation(physical, meter):
    values = {k: np.asarray(physical[k]) + (np.asarray(getattr(meter, 'bias_' + k))
        if k in ('p', 'q', 'v') else 0.) for k in CHANNELS}
    return Observations(**values, root_v=np.asarray(physical['root_v']))


def _source_audit(freeze, snapshot):
    hashes = freeze['source_sha256']
    _require(snapshot['source_sha256'] == hashes, 'snapshot/freeze source hashes differ')
    _require(set(snapshot['source_base64']) == set(hashes), 'snapshot source coverage differs')
    rows = []
    for name, expected in hashes.items():
        payload = base64.b64decode(snapshot['source_base64'][name], validate=True)
        _require(hashlib.sha256(payload).hexdigest() == expected, 'snapshot byte hash: ' + name)
        current_path = ((HERE.parents[1] / name) if name.startswith('rnj_wzzt_core/') else HERE / name).resolve()
        current_path.relative_to(HERE.parents[1].resolve())
        current = hashlib.sha256(current_path.read_bytes()).hexdigest() if current_path.is_file() else None
        rows.append(dict(name=name, frozen_sha256=expected, current_sha256=current,
                         current_matches=current == expected))
    _require(freeze['policy_id'] == _identity(dict(config=freeze['config'], source_sha256=hashes)),
             'freeze policy hash differs')
    return dict(source_count=len(rows), rows=rows,
        current_source_matches=all(r['current_matches'] for r in rows),
        frozen_bytes_verified=True, snapshot_code_executed=False)


def _operation(row, cfg):
    u = np.asarray(row['power_kw'], float)
    _require(u.ndim == 2 and np.isfinite(u).all(), 'finite time-by-site control required')
    op = row['operation']; T, n = u.shape
    if row['name'] == 'none':
        _close(u, np.zeros_like(u), 'no-EV control is nonzero', atol=1e-12, rtol=0.)
        _require(op.get('battery_service') is False, 'no-EV battery-service exception missing')
        return dict(battery_service=False, no_EV=True)
    _require(op.get('success') is True, 'saved operation failed')
    dt, ec, ed = cfg['dt_hours'], cfg['eta_charge'], cfg['eta_discharge']
    C = float(dt * np.maximum(u, 0.).sum()); D = float(dt * np.maximum(-u, 0.).sum())
    _close(op['charging_kwh'], C, 'charging energy')
    _close(op['discharging_kwh'], D, 'discharging energy')
    _close(op['grid_energy_kwh'], C - D, 'net grid energy')
    _close(op['throughput_kwh'], C + D, 'throughput energy')
    _close(op['charging_efficiency'], ec, 'charge efficiency')
    _close(op['discharging_efficiency'], ed, 'discharge efficiency')
    energy = np.asarray(op['energy_kwh'], float)
    _require(energy.shape == (T + 1,), 'SOC path shape')
    _close(energy[0], cfg['initial_energy_kwh'], 'initial battery energy')
    _close(energy[-1], cfg['terminal_energy_kwh'], 'terminal battery service')
    _require(np.all(energy >= .1 * cfg['capacity_kwh'] - 2e-7)
        and np.all(energy <= .9 * cfg['capacity_kwh'] + 2e-7), 'SOC operating bounds')
    _require(np.max(abs(u), initial=0.) <= cfg['command_power_kw'] + 2e-7, 'command power limit')
    _require(np.all(np.sum(abs(u) > 1e-7, axis=1) <= 1), 'simultaneous sites')
    if 'v2g' not in row['name'] and row['name'] != 'timing_reverse_fixed':
        _require(np.min(u, initial=0.) >= -1e-7, 'discharge in charge-only comparison')
    _require(C + D <= cfg['throughput_kwh'] + 2e-7, 'throughput limit')
    drive_by_slot = ec * dt * np.maximum(u, 0.).sum(axis=1) - dt * np.maximum(-u, 0.).sum(axis=1) / ed - np.diff(energy)
    _require(np.min(drive_by_slot, initial=0.) >= -2e-7, 'unexplained battery-energy increase')
    _close(drive_by_slot.sum(), op['driving_kwh'], 'driving battery-energy balance')
    moves = int(op['moves'])
    _require(moves == op['moves'] and 0 <= moves <= cfg['max_moves'], 'move count')
    _close(op['driving_kwh'], moves * cfg['drive_kwh'], 'driving energy per move')
    _close(op['worker_minutes'], moves * (cfg['travel_worker_minutes'] + cfg['setup_worker_minutes']), 'worker budget accounting')
    _require(op['worker_minutes'] <= cfg['driver_budget_minutes'] + 2e-7, 'worker limit')
    _close(op['power_variance'], np.var(u.sum(axis=1)), 'power variance')
    # This reviewer covers the declared one-move/all-sites-open device only.
    _require(cfg['travel_slots'] == cfg['setup_slots'] == 1 and cfg['max_moves'] == 1,
             'route audit supports the frozen one-travel/one-setup-slot device')
    if moves == 0:
        _require(np.max(abs(u[:, 1:]), initial=0.) <= 1e-7, 'zero-move action leaves initial site')
        _close(drive_by_slot, np.zeros(T), 'zero-move energy', atol=2e-7, rtol=0.)
    else:
        _require('mobile' in row['name'], 'move in fixed-site comparison')
        travel = np.flatnonzero(drive_by_slot > cfg['drive_kwh'] / 2.)
        _require(len(travel) == 1, 'travel slot not recoverable from energy path')
        t = int(travel[0]); _require(t + 2 < T, 'no time for arrival service')
        _close(drive_by_slot[t], cfg['drive_kwh'], 'travel-slot energy')
        _require(np.max(abs(u[t:t+2]), initial=0.) <= 1e-7, 'charging during travel/setup')
        _require(np.max(abs(u[:t, 1:]), initial=0.) <= 1e-7, 'leaves initial site before travel')
        sites = np.flatnonzero(np.sum(abs(u[t+2:]), axis=0) > 1e-7)
        _require(len(sites) == 1 and sites[0] != 0, 'one destination after travel')
        _require(dt * np.sum(abs(u[t+2:, sites[0]])) >= cfg['min_arrival_throughput_kwh'] - 2e-7, 'arrival throughput')
    return dict(battery_service=True, charging_kwh=C, discharging_kwh=D,
        grid_energy_kwh=C-D, throughput_kwh=C+D, driving_kwh=float(drive_by_slot.sum()),
        moves=moves, route_checked_without_solver=True)


def review_trial(trial, *, freeze=None, snapshot=None):
    """Review one parsed record; failures become labelled checks, not omissions."""
    checks = []; fit_audits = []; methods = []; failures = []
    counts = Counter(fit_records_seen=0, fit_states_verified=0, failed_fit_records=0,
                     missing_fit_records=0, planning_profiles_verified=0, planning_profiles_unavailable=0,
                     price_pairs_verified=0,price_witnesses_unavailable=0,price_replay_AC_batches=0)
    def check(name, function):
        try:
            result = function(); checks.append(dict(name=name, status='passed', result=result)); return result
        except (ValueError, KeyError, TypeError, IndexError, AttributeError, AssertionError,
                FloatingPointError, np.linalg.LinAlgError) as error:
            checks.append(dict(name=name, status='failed', message=type(error).__name__+': '+str(error))); return None
    def missing(name, reason): checks.append(dict(name=name, status='unavailable', message=str(reason)))
    def provenance():
        _require(trial.get('source_unchanged') is True, 'source unchanged flag missing/false')
        _require(freeze is not None and snapshot is not None, 'freeze/snapshot missing')
        result = _source_audit(freeze, snapshot)
        _require(trial['source_sha256'] == freeze['source_sha256'], 'trial/freeze source hashes differ')
        _require(trial['config'] == freeze['config'] and trial['policy_id'] == freeze['policy_id'], 'trial/freeze configuration differs')
        return result
    source = check('source_and_policy', provenance)
    def finish():
        summary = Counter(c['status'] for c in checks)
        return _serial(dict(schema='ev_dynamic_readonly_review_v1', seed=trial.get('seed'),
            trial_status=trial.get('status'), success=bool(checks) and not summary['failed'] and not summary['unavailable'],
            checks=checks, check_counts=dict(summary), counts=dict(counts), methods=methods,
            fit_audits=fit_audits, experiment_failures=failures, no_new_fit=True, no_selector=True,
            no_production_AC_predictor=True, calibrated_rejection_statistics_computed=False,
            statistical_unit='one asset/history; paired methods do not add independent histories'))
    required = ('training','training_meter','validation','validation_meter','future_meter',
                'training_result','background','candidates','truth','seeds')
    absent = [k for k in required if k not in trial]
    if absent:
        missing('trial_data', 'missing '+', '.join(absent)); return finish()
    prepared = check('record_schema', lambda: _prepare(trial))
    if prepared is None:return finish()
    cfg, training, validation, tm, vm, fm, bank, truth, H, V, T, n, bkw = prepared
    checks[-1]['result']=dict(training_points=H,validation_points=V,horizon=T,
        terminals=n,candidates=len(bank),base_kw=bkw)
    trained = {row.get('name',row.get('candidate')): row for row in trial['training_result'].get('fits', [])}
    def fits(label, result, obs, meter, fixed):
        if result is None: missing(label, 'fit result absent'); return {}
        rows = result.get('fits', []); seen = [r.get('name',r.get('candidate')) for r in rows]
        errors = result.get('failures', {})
        check(label+'/coverage', lambda: _fit_coverage(bank,seen,errors,result.get('all_complete')))
        output = {}
        for name in set(bank)-set(seen):
            counts['missing_fit_records'] += 1; missing(label+'/'+name, errors.get(name,'fit state absent'))
        for index, row in enumerate(rows):
            counts['fit_records_seen'] += 1; name = row.get('name',row.get('candidate'))
            def one():
                _require(name in bank, 'unknown candidate')
                kwargs = {} if not fixed else dict(trained_r=trained[name]['r'], trained_x=trained[name]['x'])
                report = review_fixed_fit(bank[name],obs,meter,row,**kwargs)
                _check_source_box(row,cfg); return report
            result_one = check(label+'/'+str(name)+'/'+str(index), one)
            if result_one is not None:
                counts['fit_states_verified'] += 1
                fit_audits.append(dict(block=label,**result_one))
                if row.get('success') is True: output[name] = result_one['full_SSE']
            if row.get('success') is not True:
                counts['failed_fit_records'] += 1; missing(label+'/'+str(name)+'/fit_success', row.get('message','unsuccessful fit'))
        if errors: failures.append(dict(block=label, failures=errors))
        return output
    train_cost = fits('training',trial['training_result'],training,tm,False)
    val_cost = fits('validation',trial.get('validation_result'),validation,vm,True)
    load_match = bool(source and any(r['name']=='ev_dynamic_loads.py' and r['current_matches'] for r in source['rows']))
    if load_match:
        check('ordinary_process_and_continuous_prefix', lambda: _background(trial,training,validation,tm,vm,H,V,T,n,bkw))
        for key,kind in (('forecast',cfg['forecast']),('mean_forecast','mean')):
            if key not in trial:
                if key=='forecast' or cfg.get('mean_policy_ablation'):missing(key,'forecast absent')
                continue
            def forecast_check(key=key,kind=kind):
                fit_objects={name:SimpleNamespace(**{k:row[k] for k in ('source_p','source_q','success')}) for name,row in trained.items()}
                rebuilt=loads.fit_forecast(training,tm,fit_objects,horizon=T,gap=cfg['forecast_gap'],base_kw=bkw,kind=kind)
                _compare_tree(trial[key],rebuilt,key)
                return dict(reconstructed_from_training_only=True,gap=cfg['forecast_gap'],kind=kind)
            check(key+'/training_only_reconstruction',forecast_check)
    else: missing('forecast_and_generator_reconstruction','current forecast module does not match frozen bytes')
    device = trial.get('selection',{}).get('configuration')
    if device is None:missing('device_configuration','selector device configuration absent')
    white = None; numerical = {}; operations = {}
    rng=np.random.default_rng(trial['seeds']['noise']);total=H+V+T
    seeded_white={k:rng.standard_normal((total,n) if k in ('p','q','v') else total)[H+V:] for k in CHANNELS}
    check('method_coverage',lambda:_method_coverage(trial))
    for row in trial.get('methods', []):
        name = row.get('name','<missing>'); item=dict(name=name,status=row.get('status')); methods.append(item)
        if row.get('power_kw') is None or row.get('status')!='complete':
            failures.append(dict(method=name,status=row.get('status'),failure=row.get('failure')))
            missing('method/'+name, 'experiment method unavailable: '+str(row.get('status')))
            continue
        power = np.asarray(row['power_kw'],float)
        if device is not None:
            op=check('operation/'+name,lambda row=row:_operation(row,device))
            if op is not None:operations[name]=op
        def physical_check():
            background=trial['background'];p=np.asarray(background['p_kw'])[H+V:]/bkw+power/bkw
            q=np.asarray(background['q_kvar'])[H+V:]/bkw
            phy=forward_physics(truth,np.asarray(trial['truth']['r']),np.asarray(trial['truth']['x']),p,q,
                np.asarray(background['source_p_kw'])[H+V:]/bkw,np.asarray(background['source_q_kvar'])[H+V:]/bkw,
                np.asarray(background['root_v'])[H+V:])
            maximum=max(_close(row['physical'][k],phy[k],'future physical '+k,atol=2e-10,rtol=0.) for k in (*CHANNELS,'root_v'))
            return dict(max_response_error=maximum,EV_added_exactly_once=True,
                AC_equation_residual=phy['equation_residual'],AC_balance_residual=phy['balance_residual'])
        check('physical/'+name,physical_check)
        try: obs=_obs(row['observations']); clean=_observation(row['physical'],fm)
        except (ValueError,KeyError,TypeError) as error:missing('observations/'+name,error);continue
        def noise_check():
            w={k:(np.asarray(getattr(obs,k))-np.asarray(row['physical'][k])-
                (np.asarray(getattr(fm,'bias_'+k)) if k in ('p','q','v') else 0.))/np.asarray(getattr(fm,k+'_sigma')) for k in CHANNELS}
            _close(obs.root_v,np.ones(T),'future known root',atol=0.,rtol=0.)
            if white is not None:
                for k in CHANNELS:_close(w[k],white[k],'paired future noise '+k,atol=2e-9,rtol=0.)
            for k in CHANNELS:_close(w[k],seeded_white[k],'saved future noise seed '+k,atol=2e-9,rtol=0.)
            return w
        w=check('paired_noise/'+name,noise_check)
        if w is not None and white is None:white=w
        noisy=fits('future/'+name,row.get('noisy'),obs,fm,True)
        clean_cost=fits('clean_future/'+name,row.get('noiseless'),clean,fm,True)
        def margins():
            _require(set(noisy)==set(bank) and set(val_cost)==set(bank),'incomplete noisy/validation fit costs')
            losses={k:noisy[k]+val_cost[k] for k in bank}
            _compare_tree(row['loss_by_candidate'],losses,'combined loss')
            M=_margin(losses,trial.get('true_name'))
            _optional_close(row.get('M'),M,'actual M')
            cleanM=None
            if set(clean_cost)==set(bank):cleanM=_margin({k:clean_cost[k]+val_cost[k] for k in bank},trial.get('true_name'))
            _optional_close(row.get('clean_future_M',row.get('noiseless_M')),cleanM,'clean-future M')
            return dict(M=M,clean_future_M=cleanM,clean_future_scope='same noisy validation retained')
        m=check('margins/'+name,margins)
        if m is not None:numerical[name]=m;item.update(m)
    for label,entry in trial.get('contrasts',{}).items():
        def contrast():
            left,right=entry['left'],entry['right'];a,b=numerical.get(left),numerical.get(right)
            valid=bool(a and b and a['M'] is not None and b['M'] is not None)
            _require(entry['valid']==valid,'contrast validity flag')
            delta=a['M']-b['M'] if valid else None
            _optional_close(entry.get('delta_M'),delta,'contrast delta M')
            values=dict(left=left,right=right,delta_M=delta,valid=valid)
            if left in operations and right in operations:
                oa,ob=operations[left],operations[right]
                for key in ('grid_energy_kwh','throughput_kwh','charging_kwh','discharging_kwh'):
                    d=oa[key]-ob[key];values['delta_'+key]=d;values['matched_'+key]=abs(d)<=2e-7
                if label in ('timing','amplitude'):
                    _close(oa['grid_energy_kwh'],ob['grid_energy_kwh'],'contrast net energy')
                    _close(oa['throughput_kwh'],ob['throughput_kwh'],'contrast throughput')
            return values
        check('contrast/'+label,contrast)
    for label in ('selection','mean_selection'):
        if label not in trial:
            if label=='selection' or cfg.get('mean_policy_ablation'):missing(label,'selection record absent')
            continue
        selection=trial[label]
        check(label+'/information_flags',lambda selection=selection:_information_flags(selection))
        forecast=trial.get('forecast' if label=='selection' else 'mean_forecast')
        if forecast is not None:
            check(label+'/planning_replay',lambda selection=selection,forecast=forecast,label=label:
                _planning_profiles(check,missing,selection,forecast,bank,trained,fm,bkw,counts,label,cfg))
    if 'price_diagnostics' in trial:
        for label,diagnostic in trial['price_diagnostics'].items():
            price=diagnostic.get('complete_AC')
            if not price or not price.get('success'):
                missing('price/'+label,'complete AC price diagnostic failed or absent');continue
            check('price/'+label,lambda price=price,label=label:
                _price_witnesses(price,trial['forecast'],bank,trained,fm,bkw,cfg,
                    check,missing,counts,'price/'+label))
    elif cfg.get('diagnostics'):
        missing('price_diagnostics','price diagnostics were requested but not saved')
    check('fixed_root_source_and_metadata',lambda:_metadata(trial,cfg,training,validation,train_cost,bank))
    if 'actual_solver_calls' in trial:
        check('saved_call_accounting',lambda:_call_accounting(trial['actual_solver_calls']))
    else:missing('saved_call_accounting','actual production/replay counters absent')
    if trial.get('status')!='complete':missing('trial_completion',trial.get('status'))
    return finish()


def _prepare(trial):
    cfg=trial['config'];H,V,T=(int(cfg[k]) for k in ('training_points','validation_points','horizon'))
    _require(cfg['forecast_gap']==V,'forecast gap differs from validation interval')
    training,validation=_obs(trial['training']),_obs(trial['validation']);n=training.n
    _require(training.t==H and validation.t==V,'training/validation lengths')
    bank={r['name']:_candidate(r['name'],r['supports'],n) for r in trial['candidates']}
    _require(len(bank)==len(trial['candidates']) and len(bank)>=2,'candidate coverage/uniqueness')
    truth=_candidate('offline_truth',trial['truth']['supports'],n)
    true_name=next((k for k,v in bank.items() if v.supports==truth.supports),None)
    _require(true_name==trial.get('true_name') and trial['candidate_miss']==(true_name is None),'candidate miss metadata')
    return cfg,training,validation,_meter(trial['training_meter']),_meter(trial['validation_meter']),_meter(trial['future_meter']),bank,truth,H,V,T,n,float(trial.get('selection',{}).get('configuration',{}).get('base_kw',20.))


def _fit_coverage(bank,names,errors,complete):
    _require(len(names)==len(set(names)) and set(names)<=set(bank),'fit names/duplicates')
    _require(set(names)|set(errors)==set(bank),'failed candidates missing from denominator')
    if complete:_require(set(names)==set(bank) and not errors,'complete fit block lacks a candidate')
    return dict(expected=len(bank),saved=len(names),failure_names=sorted(errors))


def _method_coverage(trial):
    rows=trial.get('methods',[]);names=[r['name'] for r in rows]
    aliases=dict(quality_fixed='quality_charge_fixed',quality_mobile='quality_v2g_mobile',
        **{k:'design_'+k for k in ('charge_fixed','v2g_fixed','charge_mobile','v2g_mobile')})
    expected={aliases.get(k,k) for k in trial['config']['select_modes']}|{'highlow'}
    if trial['config'].get('mean_policy_ablation'):expected.add('design_mean_v2g_mobile')
    byname={r['name']:r for r in rows}
    if trial['config'].get('diagnostics'):
        if byname.get('design_charge_fixed',{}).get('power_kw') is not None:expected.add('amplitude_half_fixed')
        if byname.get('design_v2g_fixed',{}).get('power_kw') is not None:expected.add('timing_reverse_fixed')
    _require(len(names)==len(set(names)) and set(names)==expected,'missing/duplicated method denominator')
    return dict(expected=len(expected),saved=len(names))


def _check_source_box(row,cfg):
    for key in ('source_p','source_q'):
        lo,hi=cfg['source'][key+'_bounds'];v=np.asarray(row[key],float)
        _require(np.all(v>=lo-1e-10) and np.all(v<=hi+1e-10),'source box '+key)


def _background(trial,training,validation,tm,vm,H,V,T,n,bkw):
    background=trial['background'];total=H+V+T
    regenerated=loads.generate_background(trial['seeds']['background'],total,n)
    _compare_tree(background,regenerated,'background regeneration')
    prefix=loads.generate_background(trial['seeds']['background'],H,n)
    for key in ('p_kw','q_kvar','pf','source_p_kw','source_q_kvar','root_v'):
        _close(prefix[key],np.asarray(background[key])[:H],'continuous stationary prefix '+key,atol=0.,rtol=0.)
    p,q,pf=(np.asarray(background[k]) for k in ('p_kw','q_kvar','pf'))
    _require(p.shape==q.shape==pf.shape==(total,n),'background shape')
    _require(np.all(p>0.) and np.all((pf>=.94)&(pf<=.99)),'ordinary-load positivity/PF')
    _close(q,p*np.tan(np.arccos(pf)),'Q/P/PF relation',atol=1e-12,rtol=0.)
    rng=np.random.default_rng(trial['seeds']['noise'])
    white={k:rng.standard_normal((total,n) if k in ('p','q','v') else total) for k in CHANNELS}
    truth=_candidate('offline_truth',trial['truth']['supports'],n)
    physical=forward_physics(truth,trial['truth']['r'],trial['truth']['x'],p/bkw,q/bkw,
        np.asarray(background['source_p_kw'])/bkw,np.asarray(background['source_q_kvar'])/bkw,background['root_v'])
    for obs,meter,part in ((training,tm,slice(0,H)),(validation,vm,slice(H,H+V))):
        for key in CHANNELS:
            expected=np.asarray(physical[key])[part]+np.asarray(getattr(meter,key+'_sigma'))*white[key][part]
            if key in ('p','q','v'):expected+=np.asarray(getattr(meter,'bias_'+key))
            _close(getattr(obs,key),expected,'continuous observations '+key,atol=2e-10,rtol=0.)
    return dict(total=total,train=H,validation=V,future=T,source_interface=1,
        trajectory_regenerated=True,Q_from_P_and_PF=True,exact_training_prefix=True)


def _compare_tree(saved,rebuilt,label):
    if isinstance(rebuilt,dict):
        _require(isinstance(saved,dict) and set(saved)==set(rebuilt),label+' keys')
        for key in rebuilt:_compare_tree(saved[key],rebuilt[key],label+'/'+key)
    elif isinstance(rebuilt,(np.ndarray,list,tuple)):
        if isinstance(rebuilt,np.ndarray) or (len(rebuilt) and isinstance(rebuilt[0],(int,float,np.number,list))):
            _close(saved,rebuilt,label,atol=2e-11,rtol=2e-10)
        else:_require(saved==_serial(rebuilt),label+' values')
    elif isinstance(rebuilt,(int,float,np.number)) and not isinstance(rebuilt,(bool,np.bool_)):
        _close(saved,rebuilt,label,atol=2e-11,rtol=2e-10)
    else:_require(saved==rebuilt,label+' value')


def _margin(losses,true_name):
    return None if true_name not in losses else min(v for k,v in losses.items() if k!=true_name)-losses[true_name]


def _optional_close(actual,expected,label):
    if expected is None:_require(actual is None,label+' should be unavailable')
    else:_close(actual,expected,label)


def _information_flags(selection):
    for key in ('uses_validation','uses_truth','uses_future_noise','random_used_by_primary'):
        _require(selection.get(key) is False,'selector information flag '+key)
    _require(selection.get('line_parameters_unchanged') is True,'selector changed training R/X')
    return dict(training_only_declared=True,lines_unchanged=True,producer_call_path_not_reexecuted=True)


def _metadata(trial,cfg,training,validation,train_cost,bank):
    _require(cfg['source_interface']==1 and cfg['PQ_inputs_fixed'] is True,'source/input policy')
    _require(trial.get('validation_used_by_selector') is False and trial.get('future_noise_used_by_selector') is False,'trial information flags')
    _require(trial.get('one_trained_parameter_vector') is True,'one-training-vector flag')
    _close(training.root_v,np.ones(training.t),'training root',atol=0.,rtol=0.)
    _close(validation.root_v,np.ones(validation.t),'validation root',atol=0.,rtol=0.)
    if trial.get('true_name') is not None and set(train_cost)==set(bank):
        trained=next(r for r in trial['training_result']['fits'] if r['name']==trial['true_name'])
        for key,label in (('r','R'),('x','X')):
            true=np.asarray(trial['truth'][key]);estimate=np.asarray(trained[key])
            error=np.linalg.norm(estimate-true)/np.linalg.norm(true)
            _close(trial['training_parameter_error'][label],error,'training parameter error '+label)
    return dict(source_interface=1,fixed_root=True,calibrated_test_not_claimed=True)


def _call_accounting(values):
    keys=('AC_batches','AC_snapshots','AC_failures','QP_calls',
        'independent_replay_AC_batches','independent_replay_AC_snapshots','independent_replay_AC_failures')
    for key in keys:
        value=values[key]
        _require(type(value) is int and value>=0,'nonnegative integer counter '+key)
    _require(values['AC_failures']<=values['AC_batches'],'production AC failure count')
    _require(values['independent_replay_AC_failures']<=values['independent_replay_AC_batches'],'replay AC failure count')
    return dict(saved=values,total_AC_batches=values['AC_batches']+values['independent_replay_AC_batches'],
        total_AC_snapshots=values['AC_snapshots']+values['independent_replay_AC_snapshots'],
        scope='saved-counter arithmetic only; execution was not repeated')


def _price_witnesses(price,forecast,bank,trained,meter,bkw,cfg,check,missing,counts,label):
    """Audit two price coordinates at frozen source witnesses, without refits.

    The finite difference is of the residual vector, followed by r_u.T @ r.
    It checks the recorded partial/envelope expression at the recorded state;
    it does not establish stationarity or global optimality of the source fit.
    """
    rows=price.get('pairs',[]);T,n=np.asarray(forecast['p']).shape
    coordinates=list(dict.fromkeys(((0,0),(T-1,n-1))));step=1e-3
    required={'name','a','b','weight','source_p','source_q','source_solver','half_SSE','gradient_per_kw'}
    absent_control=price.get('power_kw') is None
    if absent_control:
        counts['price_witnesses_unavailable']+=len(rows)
        missing(label+'/legacy_witness','legacy price diagnostic has no saved control/source witness association')
        return dict(complete=False,legacy=True)
    u=np.asarray(price['power_kw'],float)
    _require(u.shape==(T,n) and np.isfinite(u).all(),'price control shape/finite values')
    expected={(a,b) for a in bank for b in bank if a!=b}
    present=[(r.get('a'),r.get('b')) for r in rows if required<=set(r)]
    if all(required<=set(r) for r in rows):
        _require(len(present)==len(set(present)) and set(present)==expected,'price directed-pair denominator')
    else:
        # Preserve missing legacy records; never silently reduce the task mean.
        _require(len(rows)==len(expected),'legacy price directed-pair denominator')
    gradients=[];weights=[];costs=[];FD=[];pair_names=[]
    for entry in rows:
        if not required<=set(entry):
            counts['price_witnesses_unavailable']+=1
            missing(label+'/'+entry.get('name','?')+'/legacy_witness',
                'legacy price lacks '+', '.join(sorted(required-set(entry))))
            continue
        def one(entry=entry):
            a,b=entry['a'],entry['b'];state=entry['source_solver']
            _require(a in bank and b in bank and a!=b,'price candidate names')
            _require(entry['name']==a+'__to__'+b,'price task name')
            _require(state.get('success') is True,'price source solver failed')
            _close(state['r'],trained[b]['r'],'price R changed',atol=0.,rtol=0.)
            _close(state['x'],trained[b]['x'],'price X changed',atol=0.,rtol=0.)
            for key in ('source_p','source_q'):
                _require(np.asarray(entry[key]).shape==(T,),'price source witness shape '+key)
            _check_source_box(entry,cfg)
            weight=float(entry['weight'])
            _close(weight,1./len(expected),'price task weight',atol=1e-14,rtol=0.)
            def residual(control):
                totalp=np.asarray(forecast['p'])+control/bkw;totalq=np.asarray(forecast['q'])
                ref_source=forecast['source_by_candidate'][a]
                ref=forward_physics(bank[a],trained[a]['r'],trained[a]['x'],totalp,totalq,
                    ref_source['source_p'],ref_source['source_q'],forecast['root'])
                comp=forward_physics(bank[b],trained[b]['r'],trained[b]['x'],totalp,totalq,
                    entry['source_p'],entry['source_q'],forecast['root'])
                counts['price_replay_AC_batches']+=2
                return np.concatenate([((np.asarray(comp[k])-np.asarray(ref[k]))/
                    np.broadcast_to(getattr(meter,k+'_sigma'),np.asarray(ref[k]).shape)).ravel()
                    for k in ('v','master_p','master_q')])
            r=residual(u);cost=float(r@r)/2
            _close(entry['half_SSE'],cost,'price half SSE')
            _close(state['loss'],2*cost,'price source solver SSE')
            normalizer=0.
            for key in ('v','master_p','master_q'):
                shape=(T,n) if key=='v' else (T,)
                sigma=np.broadcast_to(getattr(meter,key+'_sigma'),shape)
                normalizer+=float(np.log(2*np.pi*sigma**2).sum())
            _close(state['loglik'],-.5*(2*cost+normalizer),'price source solver loglik')
            gradient=np.asarray(entry['gradient_per_kw'],float)
            _require(gradient.shape==(T,n) and np.isfinite(gradient).all(),'price gradient shape')
            fd=[];saved=[]
            for t,j in coordinates:
                plus=u.copy();minus=u.copy();plus[t,j]+=step;minus[t,j]-=step
                derivative=(residual(plus)-residual(minus))/(2*step)
                fd.append(float(derivative@r));saved.append(float(gradient[t,j]))
            error=_close(saved,fd,'price fixed-witness finite difference',atol=3e-5,rtol=2e-5)
            gradients.append(gradient);weights.append(weight);costs.append(cost);FD.append(fd);pair_names.append(entry['name'])
            return dict(half_SSE=cost,coordinates=[list(c) for c in coordinates],
                finite_difference_per_kw=fd,saved_per_kw=saved,max_abs_gradient_error=error,
                source_witness_held_fixed=True,training_R_X_held_fixed=True)
        if check(label+'/'+entry['name'],one) is not None:counts['price_pairs_verified']+=1
    if len(gradients)!=len(expected):
        missing(label+'/mean_gradient','not every price witness is available and verified')
        return dict(complete=False,verified_pairs=len(gradients),expected_pairs=len(expected),legacy=counts['price_witnesses_unavailable']>0)
    mean=np.einsum('k,ktj->tj',np.asarray(weights),np.asarray(gradients))
    _close(price['gradient_per_kw'],mean,'mean price from task weights',atol=2e-10,rtol=2e-10)
    _close(price['mean_half_SSE'],np.asarray(weights)@np.asarray(costs),'mean price-diagnostic distance')
    meanFD=np.asarray(weights)@np.asarray(FD)
    _close([mean[t,j] for t,j in coordinates],meanFD,'mean fixed-witness price finite difference',atol=3e-5,rtol=2e-5)
    _close(price['temporal_spread_by_site'],np.ptp(mean,axis=0),'price temporal spread')
    return dict(complete=True,pairs=len(pair_names),coordinates=[list(c) for c in coordinates],step_kw=step,
        mean_finite_difference_per_kw=meanFD.tolist(),no_new_source_fit=True,
        source_stationarity_or_global_optimality_certified=False,
        scope='two residual finite-difference coordinates per saved source witness; all mean-gradient entries summed arithmetically')


def _planning_profiles(check,missing,selection,forecast,bank,trained,meter,bkw,counts,label,cfg):
    known={}
    def verify(prediction,power,where):
        digest=_identity(prediction)
        if digest in known:return
        known[digest]=True;u=np.asarray(power,float);T,n=u.shape
        for entry in prediction.get('pairs',[]):
            if not entry.get('success'):
                missing(where+'/'+entry.get('name','?'),'failed planning source fit');continue
            def profile(entry=entry):
                a,b=entry['a'],entry['b'];line=trained[a]
                totalp=np.asarray(forecast['p'])+u/bkw;totalq=np.asarray(forecast['q'])
                source=forecast['source_by_candidate'][a]
                phy=forward_physics(bank[a],line['r'],line['x'],totalp,totalq,source['source_p'],source['source_q'],forecast['root'])
                obs=_observation(phy,meter);state=entry['solver']
                _close(state['r'],trained[b]['r'],'planning R unchanged',atol=0.,rtol=0.)
                _close(state['x'],trained[b]['x'],'planning X unchanged',atol=0.,rtol=0.)
                _check_source_box(entry,cfg)
                competitor=forward_physics(bank[b],state['r'],state['x'],obs.p-np.asarray(meter.bias_p),
                    obs.q-np.asarray(meter.bias_q),entry['source_p'],entry['source_q'],obs.root_v)
                cost=0.;normalizer=0.
                for k in ('v','master_p','master_q'):
                    y=np.asarray(getattr(obs,k));mu=np.asarray(competitor[k])+(np.asarray(meter.bias_v) if k=='v' else 0.)
                    sigma=np.broadcast_to(getattr(meter,k+'_sigma'),y.shape)
                    cost+=float(np.sum(((mu-y)/sigma)**2));normalizer+=float(np.log(2*np.pi*sigma**2).sum())
                _close(entry['joint'],cost,'planning profile SSE');_close(entry['added'],cost/2,'planning profile half SSE')
                _close(state['loss'],cost,'planning solver SSE');_close(state['loglik'],-.5*(cost+normalizer),'planning solver loglik')
                _require(entry['source_interface']==1 and entry['line_parameters_fixed'] is True,'planning source/line flags')
                _require(entry['EV_in_total_PQ_once'] is True,'planning EV flag')
                return dict(full_SSE=cost,source_interface=1,training_lines_unchanged=True)
            if check(where+'/'+entry['name'],profile) is not None:counts['planning_profiles_verified']+=1
    configuration=selection.get('configuration',{});T=np.asarray(forecast['p']).shape[0];n=np.asarray(forecast['p']).shape[1]
    for name,prediction in selection.get('predicted_AC',{}).items():
        if name=='constant':
            power=np.zeros((T,n));power[:,0]=(configuration['terminal_energy_kwh']-configuration['initial_energy_kwh'])/(configuration['eta_charge']*configuration['dt_hours']*T)
        else:
            record=selection.get('methods',{}).get(name,{})
            if record.get('selected_origin')=='MILP':power=record.get('raw',{}).get('u')
            else:power=next((r['power_kw'] for r in record.get('initial_candidates',[]) if r['origin']==record.get('selected_origin')),None)
            if power is None:missing(label+'/selected/'+name,'selected control unavailable');continue
            power=np.asarray(power,float).reshape(T,n)
        verify(prediction,power,label+'/selected/'+name)
    for name,record in selection.get('methods',{}).items():
        raw,prediction=record.get('raw',{}),record.get('raw_prediction')
        if prediction is not None and raw.get('u') is not None:
            verify(prediction,np.asarray(raw['u']).reshape(T,n),label+'/proposal/'+name)
    for index,prediction in enumerate(selection.get('all_prediction_records',[])):
        if _identity(prediction) not in known:
            if prediction.get('power_kw') is not None:
                verify(prediction,np.asarray(prediction['power_kw']).reshape(T,n),label+'/cache/'+str(index))
            else:
                amount=len(prediction.get('pairs',[]));counts['planning_profiles_unavailable']+=amount
                missing(label+'/cache/'+str(index),str(amount)+' planning profiles have no saved control association')


def review_directory(directory, output=None):
    directory=Path(directory);source_errors=[]
    try:freeze=_read(directory/'freeze.json');snapshot=_read(directory/'source_snapshot.json.gz')
    except (OSError,ValueError,KeyError) as error:freeze=snapshot=None;source_errors.append(str(error))
    trials=[]
    for path in sorted(directory.glob('trial_*.json.gz')):
        try:
            result=review_trial(_read(path),freeze=freeze,snapshot=snapshot)
            result.update(input_path=str(path.resolve()),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        except (OSError,ValueError,TypeError,KeyError) as error:
            result=dict(input_path=str(path.resolve()),success=False,review_exception=str(error))
        trials.append(result)
    result=dict(schema='ev_dynamic_directory_review_v1',directory=str(directory.resolve()),
        success=bool(trials) and not source_errors and all(t['success'] for t in trials),
        trial_count=len(trials),verified_trial_count=sum(t['success'] for t in trials),
        source_errors=source_errors,trials=trials,calibrated_rejection_statistics_computed=False,
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if not trials:result['pending']='no saved trials; nothing marked passed'
    if output is not None:
        output=Path(output)
        if output.suffix not in ('.json','.gz'):output=output/'review.json'
        _require(output.resolve() not in {p.resolve() for p in directory.glob('trial_*.json.gz')}
            and output.resolve() not in {(directory/'freeze.json').resolve(),(directory/'source_snapshot.json.gz').resolve()},'review must not overwrite evidence')
        output.parent.mkdir(parents=True,exist_ok=True)
        payload=(json.dumps(_serial(result),ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
        output.write_bytes(gzip.compress(payload,mtime=0) if output.suffix=='.gz' else payload)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=review_directory(args.directory,args.output)
    print(json.dumps({k:result[k] for k in ('success','trial_count','verified_trial_count','source_errors')}))
    raise SystemExit(0 if result['success'] else 1)


if __name__=='__main__':main()
