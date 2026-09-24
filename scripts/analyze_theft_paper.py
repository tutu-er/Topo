"""Summarize the frozen study without selecting methods or altering thresholds."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import beta
ROOT=Path(__file__).resolve().parents[1]/'theft_wzzt'
OUT=ROOT/'outputs/theft_paper'
METHODS=['stable_positive_lp','fixed_stable','fixed_free','balance_only','voltage_gain']

def rate(k,n):
    return dict(k=int(k),n=int(n),rate=k/n if n else None,
        ci95=[float(beta.ppf(.025,k,n-k+1)) if k else 0.,float(beta.ppf(.975,k+1,n-k)) if k<n else 1.] if n else [None,None])

def score(row,method):
    if method in ('balance_only','voltage_gain'):return row[method]
    return next(r['gain'] for r in row['results'] if r['method']==method)

def result(row,method='stable_positive_lp'):
    return next(r for r in row['results'] if r['method']==method)

def cutoff(values,alpha=.05):
    k=int(np.ceil((len(values)+1)*(1-alpha)))
    if k>len(values):raise ValueError('Insufficient calibration')
    return float(np.sort(values)[k-1])

def alarm(value,values):
    return (1+np.count_nonzero(np.asarray(values)>=value-1e-7))/(len(values)+1)<=.05

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true');args=parser.parse_args()
    protocol=json.loads((OUT/'frozen_protocol.json').read_text())
    allrows=[json.loads(p.read_text()) for p in (OUT/'rows').glob('*.json')]
    errors=[r for r in allrows if 'error' in r];rows=[r for r in allrows if 'error' not in r]
    if not args.partial and (len(allrows)!=len(protocol['jobs']) or errors):raise RuntimeError('Study incomplete or failed rows exist')
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['case'],r['condition'],r['role'])].append(r)
    summary=dict(status='partial' if args.partial else 'complete',expected=len(protocol['jobs']),completed=len(rows),errors=errors,groups=[],localization=[],grid=[],timing={},diagnostics={})
    calibration={}
    for case,cond,_ in sorted(groups):
        key=(case,cond)
        if key in calibration:continue
        cal=groups.get((case,cond,'null_cal'),[])
        if not cal:continue
        calibration[key]={m:[score(r,m) for r in cal] for m in METHODS}
        null=groups.get((case,cond,'null_test'),[])
        for method in METHODS:
            values=calibration[key][method]
            if not null:continue
            nominal=[score(r,method) for r in groups[(case,'nominal','null_cal')]]
            summary['groups'].append(dict(case=case,condition=cond,method=method,calibration_n=len(values),
                null_fpr=rate(sum(alarm(score(r,method),values) for r in null),len(null)),
                transferred_nominal_fpr=rate(sum(alarm(score(r,method),nominal) for r in null),len(null))))
    for (case,cond,role),grid in sorted(groups.items()):
        if role!='grid_test' or (case,cond) not in calibration:continue
        for bus,amp in sorted({(r['bus'],r['amp_kw']) for r in grid}):
            cell=[r for r in grid if r['bus']==bus and r['amp_kw']==amp]
            for method in METHODS:
                alarms=[alarm(score(r,method),calibration[(case,cond)][method]) for r in cell]
                point=None;temporal=None
                if method in ('stable_positive_lp','fixed_stable'):
                    point=rate(sum(result(r,method)['point_correct'] for r in cell),len(cell))
                if method not in ('balance_only','voltage_gain'):
                    temporal=float(np.mean([result(r,method)['active_time_correct'] for r in cell]))
                summary['grid'].append(dict(case=case,condition=cond,bus=bus,amp_kw=amp,method=method,detection=rate(sum(alarms),len(cell)),point=point,active_time_accuracy=temporal))
    for case in ['paper15','soumalas11','flynn16']:
        cal=groups.get((case,'nominal','event_cal'),[]);test=groups.get((case,'nominal','event_test'),[])
        if len(cal)!=199 or not test:continue
        radius=cutoff([r['amplitude_max_abs_error_pu'] for r in cal])
        for method in ['stable_positive_lp','fixed_stable']:
            gaps=[result(r,method)['truth_profile_gap'] for r in cal]
            if any(v is None for v in gaps):raise RuntimeError('True region missing from candidate domain')
            tau=cutoff(gaps)
            covered=[];sizes=[];reported=[];widths=[]
            for r in test:
                fit=result(r,method);minimum=fit['profile_candidate_min']
                region=[p['location'] for p in fit['profile'] if p['loss']-minimum<=tau+1e-7]
                covered.append(r['truth_region'] in region);sizes.append(len(region))
                detected=alarm(score(r,method),calibration[(case,'nominal')][method])
                reported.append(detected and covered[-1])
                widths.append(float(np.mean(np.asarray(r['amplitude_input_pu'])+radius-np.maximum(np.asarray(r['amplitude_input_pu'])-radius,0)))*r['base_mva']*1000)
            summary['localization'].append(dict(case=case,method=method,n=len(test),location_cutoff=tau,
                detection=rate(sum(alarm(score(r,method),calibration[(case,'nominal')][method]) for r in test),len(test)),
                point=rate(sum(result(r,method)['point_correct'] for r in test),len(test)),
                region_coverage=rate(sum(covered),len(test)),detection_and_coverage=rate(sum(reported),len(test)),
                region_size_mean=float(np.mean(sizes)),region_size_max=max(sizes),region_size_median=float(np.median(sizes)),
                candidate_count=len(result(test[0],method)['profile']),amplitude_radius_kw=radius*test[0]['base_mva']*1000,
                amplitude_window_coverage=rate(sum(r['amplitude_max_abs_error_pu']<=radius+1e-12 for r in test),len(test)),
                amplitude_interval_mean_width_kw=float(np.mean(widths)),
                amplitude_active_relative_mae=float(np.mean([r['amplitude_active_relative_mae'] for r in test])),
                old_loss_only_envelope_active_coverage=float(np.mean([r['old_loss_only_envelope_active_coverage'] for r in test]))))
            if method=='stable_positive_lp' and not args.partial:
                calrows=groups[(case,'nominal','null_cal')]
                signatures={r['protocol_signature'] for r in calrows+cal+test}
                if len(signatures)!=1:raise RuntimeError('Inconsistent nominal signatures')
                artifact=dict(case=case,condition='nominal',protocol_signature=signatures.pop(),null_gains=calibration[(case,'nominal')][method],
                    method=method,set_alpha=.05,location_cutoff=tau,amplitude_radius_pu=radius,
                    inference_source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['paper_study/lp_profile.py','detect_simple.py','detect_window.py']},
                    null_calibration_replicates=sorted(r['replicate'] for r in calrows),labelled_calibration_replicates=sorted(r['replicate'] for r in cal),
                    scope='Nominal simulation law only. Region and amplitude guarantees separately marginal; no conditional-on-alarm guarantee.')
                (OUT/f'calibration_{case}.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    for method in ['stable_positive_lp','fixed_stable','fixed_free']:
        seconds=[result(r,method)['seconds'] for r in rows]
        summary['timing'][method]=dict(median=float(np.median(seconds)),p95=float(np.quantile(seconds,.95)),max=max(seconds),total=sum(seconds))
    ds=[]
    for r in rows:
        fit=result(r);ds += [fit['h0_diagnostics']]+[p['diagnostics'] for p in fit['profile']]
    summary['diagnostics']={k:max(d[k] for d in ds) for k in ['objective_error','scaled_constraint_violation','primal_dual_gap']}
    summary['diagnostics']['lp_count']=len(ds)
    summary['dataset_seconds_total']=sum(r['seconds'] for r in rows)
    name='partial_summary.json' if args.partial else 'summary.json'
    (OUT/name).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['status','expected','completed','errors','timing','diagnostics']},indent=2))
if __name__=='__main__':main()
