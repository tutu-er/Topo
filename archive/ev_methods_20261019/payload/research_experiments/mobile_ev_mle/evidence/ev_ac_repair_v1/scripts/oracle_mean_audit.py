"""Oracle mean-error diagnostics of saved EV plans, never a selector input.

Only saved records are replayed through AC. Single-factor substitutions include
interactions and are not additive causal percentages. A missing historical
posterior remains missing; no fitted-center replacement is invented.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import numpy as np

from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle.candidates import RadialCandidate

CHANNELS=('p','q','v','master_p','master_q')


def digest(path):
    h=sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def whiten(phy,meter):
    result=[]
    for key in CHANNELS:
        value=np.asarray(phy[key],float);shape=value.shape
        if key in ('p','q','v'):value=value+np.broadcast_to(meter['bias_'+key],shape)
        result.append((value/np.broadcast_to(meter[key+'_sigma'],shape)).ravel())
    return np.concatenate(result)


def layout(record):
    n=len(record['observed_training']['p'][0]);T=record['config']['horizon']
    base=record['provenance']['base_kw'];voltage_base=1000.*record['provenance']['base_kv']
    result={};start=0
    for key in CHANNELS:
        shape=(T,n) if key in ('p','q','v') else (T,)
        count=int(np.prod(shape));result[key]=dict(indices=np.arange(start,start+count),
            sigma=np.broadcast_to(record['future_meter'][key+'_sigma'],shape).ravel(),
            physical_scale=voltage_base if key=='v' else base,
            physical_unit='V_LL' if key=='v' else 'kvar' if key in ('q','master_q') else 'kW')
        start+=count
    return result,n,T


def metrics(delta,parts):
    delta=np.asarray(delta,float);result={}
    for key,part in parts.items():
        d=delta[part['indices']];physical=d*part['sigma']*part['physical_scale']
        result[key]=dict(RMS_whitened=float(np.sqrt(np.mean(d*d))),
            RMS_physical=float(np.sqrt(np.mean(physical*physical))),unit=part['physical_unit'])
    idx=np.concatenate([parts[k]['indices'] for k in ('v','master_p','master_q')])
    result['electrical_whitened_RMS']=float(np.sqrt(np.mean(delta[idx]**2)))
    result['all_meter_whitened_RMS']=float(np.sqrt(np.mean(delta**2)))
    return result


def squared_losses(delta,parts):
    metric=metrics(delta,parts)
    return {key:dict(MSE_whitened=value['RMS_whitened']**2,
        MSE_physical=value['RMS_physical']**2,unit=value['unit']+'^2')
        for key,value in metric.items() if isinstance(value,dict)}


def audit_history(path,cost=None):
    began=perf_counter();h0=digest(path)
    cost=dict(AC_calls=0,AC_states=0,AC_seconds=0.) if cost is None else cost
    with gzip.open(path,'rt',encoding='utf-8') as stream:record=json.load(stream)
    result=dict(path=str(path.resolve()),raw_sha256=h0,spec=record.get('spec'),
        status=record.get('status'),saved_failures=record.get('failures',{}),available=False,
        preserved_in_denominator=True,actions=[],response_differences=[],issues=[],audit_cost=cost)
    names=record.get('candidate_names',[]);name=record.get('true_name')
    prediction=record.get('prediction') or {};updates=prediction.get('candidate_updates') or {}
    centers=record.get('posterior_centers') or {}
    if name not in updates or name not in centers:
        result.update(unavailable_reason='true-candidate historical posterior unavailable',seconds=perf_counter()-began,
            raw_unchanged=digest(path)==h0)
        return result
    parts,n,T=layout(record);meter=record['future_meter'];base=record['provenance']['base_kw']
    structure=next(x for x in record['candidate_structures'] if x['name']==name)
    candidate=RadialCandidate.from_supports(name,n,[s for s in structure['supports'] if 1<len(s)<n])
    predicted=np.asarray(updates[name]['future_latent_mean'],float)
    center=centers[name];truth_lines=record['true_lines_evaluation_only'];truth_index=names.index(name)
    if record['true_index']!=truth_index:result['issues'].append('true candidate index mismatch')
    cost=result['audit_cost']
    def response(lines,z,power,root):
        start=perf_counter();cost['AC_calls']+=1;cost['AC_states']+=T
        try:
            phy=lk.ac_predict(candidate,lines['r'],lines['x'],z[:,:n]+power/base,
                z[:,n:2*n],z[:,-2],z[:,-1],root)
        finally:cost['AC_seconds']+=perf_counter()-start
        return whiten(phy,meter),dict(ac_equation_residual=float(phy['ac_equation_residual']),
            power_balance_residual=float(phy['power_balance_residual']))
    replay_cache={}
    for key,action in record.get('actions',{}).items():
        power=np.asarray(action['power'],float);banks=action['propagation'].get('banks',{})
        if 'G00' not in banks:
            result['actions'].append(dict(action_key=key,methods=action['methods'],available=False,
                reason='G00 unavailable'));continue
        mu0=np.asarray(banks['G00']['means'])[0,truth_index]
        mu1=np.asarray(banks['G11']['means'])[0,truth_index] if 'G11' in banks else None
        action_result=dict(action_key=key,methods=action['methods'],available=True,
            propagation_status=action['propagation']['status'],regimes=[])
        for regime,g in action['regimes'].items():
            clean=np.asarray(g['clean'],float);truth=g['truth_inputs_evaluation_only']
            true_z=np.c_[truth['p'],truth['q'],truth['source_p'],truth['source_q']]
            z_pq=predicted.copy();z_pq[:,:2*n]=true_z[:,:2*n]
            z_source=predicted.copy();z_source[:,-2:]=true_z[:,-2:]
            variants=dict(original=(center,predicted),line_truth_only=(truth_lines,predicted),
                PQ_truth_only=(center,z_pq),source_truth_only=(center,z_source),
                state_truth_only=(center,true_z),all_truth=(truth_lines,true_z))
            outputs={};diagnostics={};errors={}
            for label,(lines,z) in variants.items():
                value,diagnostic=response(lines,z,power,np.asarray(truth['root_v']))
                outputs[label]=value;diagnostics[label]=diagnostic;errors[label]=metrics(value-clean,parts)
            center_error=float(np.max(abs(outputs['original']-mu0)))
            truth_error=float(np.max(abs(outputs['all_truth']-clean)))
            if center_error>2e-8:result['issues'].append(dict(action=key,regime=regime,issue='center differs from saved G00',max_error=center_error))
            if truth_error>2e-8:result['issues'].append(dict(action=key,regime=regime,issue='all-truth replay differs',max_error=truth_error))
            errors['G00_actual_mean_bias']=metrics(mu0-clean,parts)
            response_means=dict(outputs,G00=mu0)
            if mu1 is not None:
                errors['G11_minus_G00_correction']=metrics(mu1-mu0,parts)
                errors['G11_actual_mean_bias']=metrics(mu1-clean,parts)
                response_means['G11']=mu1
            replay_cache[(key,regime)]=dict(means=response_means,clean=clean,
                background=true_z,root=np.asarray(truth['root_v']))
            # Pointwise non-additivity of line and state response corrections.
            interaction=outputs['all_truth']-outputs['line_truth_only']-outputs['state_truth_only']+outputs['original']
            losses={label:squared_losses(outputs[label]-clean,parts) for label in ('original','line_truth_only','state_truth_only','all_truth')}
            signed={}
            for channel in CHANNELS:
                signed[channel]={field:losses['original'][channel][field]-losses['line_truth_only'][channel][field]
                    -losses['state_truth_only'][channel][field]+losses['all_truth'][channel][field]
                    for field in ('MSE_whitened','MSE_physical')}
                signed[channel]['physical_unit']=losses['original'][channel]['unit']
            action_result['regimes'].append(dict(regime=regime,errors=errors,
                line_state_response_interaction=metrics(interaction,parts),
                anchored_line_state_error_MSE_interaction=signed,
                AC_diagnostics=diagnostics,G00_center_replay_max_error=center_error,
                all_truth_replay_max_error=truth_error))
        result['actions'].append(action_result)
    # Every difference below uses the same saved future background/source and
    # the same true topology. Only the already executed EV plan changes.
    baselines=[key for key,a in record.get('actions',{}).items() if 'constant' in a['methods']]
    if len(baselines)!=1:result['issues'].append('exactly one saved constant action required')
    else:
        baseline=baselines[0];baseline_power=np.asarray(record['actions'][baseline]['power'])
        for key,action in record['actions'].items():
            delta_zero=np.array_equal(np.asarray(action['power']),baseline_power)
            for regime in action['regimes']:
                if (key,regime) not in replay_cache or (baseline,regime) not in replay_cache:continue
                here,reference=replay_cache[(key,regime)],replay_cache[(baseline,regime)]
                same_background=np.array_equal(here['background'],reference['background']) and np.array_equal(here['root'],reference['root'])
                if not same_background:
                    result['issues'].append(dict(action=key,regime=regime,issue='action comparison background changed'));continue
                actual_delta=here['clean']-reference['clean'];signal=metrics(actual_delta,parts);variants={}
                for variant in sorted(set(here['means'])&set(reference['means'])):
                    predicted_delta=here['means'][variant]-reference['means'][variant]
                    error=metrics(predicted_delta-actual_delta,parts)
                    relative={channel:(None if signal[channel]['RMS_physical']==0. else
                        error[channel]['RMS_physical']/signal[channel]['RMS_physical']) for channel in CHANNELS}
                    variants[variant]=dict(error=error,relative_RMS=relative,
                        predicted_signal=metrics(predicted_delta,parts))
                result['response_differences'].append(dict(action_key=key,methods=action['methods'],
                    baseline_key=baseline,regime=regime,delta_zero=bool(delta_zero),
                    counts_as_new_action_comparison=not delta_zero,shared_future_background=True,
                    actual_signal=signal,variants=variants,
                    covariance_or_coverage_estimated=False))
    result.update(available=any(a['available'] for a in result['actions']),
        true_candidate=name,raw_unchanged=digest(path)==h0,seconds=perf_counter()-began)
    if not result['raw_unchanged']:result['issues'].append('raw changed during audit')
    return result


def aggregate(histories):
    """Each history receives one weight; regimes/noise never increase n."""
    cells={}
    for h in histories:
        within={}
        for action in h['actions']:
            if not action.get('available'):continue
            for method in action['methods']:
                for regime in action['regimes']:
                    for variant,mm in regime['errors'].items():
                        for channel in CHANNELS:
                            key=(method,regime['regime'],variant,channel)
                            within.setdefault(key,[]).append(mm[channel])
        for key,values in within.items():
            cells.setdefault(key,[]).append(dict(physical=np.mean([v['RMS_physical'] for v in values]),
                whitened=np.mean([v['RMS_whitened'] for v in values]),unit=values[0]['unit']))
    return [dict(method=k[0],regime=k[1],variant=k[2],channel=k[3],histories=len(v),
        RMS_physical_mean=float(np.mean([z['physical'] for z in v])),
        RMS_whitened_mean=float(np.mean([z['whitened'] for z in v])),unit=v[0]['unit'])
        for k,v in sorted(cells.items())]


def aggregate_responses(histories):
    cells={}
    for history in histories:
        for row in history.get('response_differences',[]):
            if not row['counts_as_new_action_comparison']:continue
            for method in row['methods']:
                for variant,detail in row['variants'].items():
                    for channel in CHANNELS:
                        key=(method,row['regime'],variant,channel)
                        cells.setdefault(key,[]).append(dict(error=detail['error'][channel],
                            signal=row['actual_signal'][channel],relative=detail['relative_RMS'][channel]))
    return [dict(method=k[0],regime=k[1],variant=k[2],channel=k[3],histories=len(rows),
        error_RMS_physical_mean=float(np.mean([r['error']['RMS_physical'] for r in rows])),
        error_RMS_whitened_mean=float(np.mean([r['error']['RMS_whitened'] for r in rows])),
        actual_signal_RMS_physical_mean=float(np.mean([r['signal']['RMS_physical'] for r in rows])),
        relative_RMS_mean=(None if all(r['relative'] is None for r in rows) else
            float(np.mean([r['relative'] for r in rows if r['relative'] is not None]))),
        unit=rows[0]['error']['unit']) for k,rows in sorted(cells.items())]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--phase',choices=('development','calibration','evaluation'))
    parser.add_argument('--expected-histories',type=int);parser.add_argument('--limit',type=int)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();root=Path(args.root).resolve();output=Path(args.output).resolve()
    if output.exists() and not args.overwrite:raise FileExistsError(str(output))
    paths=sorted((root/args.phase if args.phase else root).glob('**/history_*.json.gz'))
    if args.limit is not None:paths=paths[:args.limit]
    if not paths:raise FileNotFoundError('no saved histories')
    if args.phase=='evaluation' and (args.expected_histories is None or len(paths)!=args.expected_histories):
        raise RuntimeError('evaluation oracle audit requires the complete expected saved batch')
    began=perf_counter();rows=[]
    for path in paths:
        counter=dict(AC_calls=0,AC_states=0,AC_seconds=0.)
        try:row=audit_history(path,counter)
        except Exception as exc:row=dict(path=str(path),available=False,preserved_in_denominator=True,
            audit_error=type(exc).__name__+': '+str(exc),raw_sha256=digest(path),actions=[],
            issues=[str(exc)],audit_cost=counter)
        rows.append(row);print('ORACLE_AUDITED',path.name,'available',row['available'],flush=True)
    result=dict(root=str(root),script_sha256=digest(__file__),scope='saved-record oracle mean diagnosis only; truth is not supplied to selector or estimator',
        warning='Single-factor replacements interact; changes are not additive causal percentages.',
        histories=rows,aggregate=aggregate(rows),response_aggregate=aggregate_responses(rows),
        summary=dict(histories=len(rows),available=sum(r['available'] for r in rows),
            unavailable=sum(not r['available'] for r in rows),status_counts=dict(Counter(r.get('status','audit_exception') for r in rows)),
            histories_with_issues=sum(bool(r['issues']) for r in rows),
            expected_histories=args.expected_histories,missing_raw_histories=None if args.expected_histories is None else max(0,args.expected_histories-len(rows)),
            audit_AC_calls=sum(r['audit_cost']['AC_calls'] for r in rows),audit_AC_states=sum(r['audit_cost']['AC_states'] for r in rows),
            audit_AC_seconds=sum(r['audit_cost']['AC_seconds'] for r in rows),audit_seconds=perf_counter()-began))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',output,json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
