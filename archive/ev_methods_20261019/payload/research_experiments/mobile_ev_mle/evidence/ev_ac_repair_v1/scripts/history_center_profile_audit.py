"""Extract saved fixed-Bayes-line historical source profiles; no new fits/AC.

Compares the SAME direct-measured-PQ conditional historical SSE.  This is not
the latent-PQ Bayesian posterior objective or a calibrated confidence test.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
import numpy as np


def read(path):
    p=Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz'
                      else p.read_text(encoding='utf-8'))


def statistics(values):
    a=np.asarray(values,float)
    if not len(a):return dict(count=0)
    return dict(count=len(a),minimum=float(a.min()),median=float(np.median(a)),
                mean=float(a.mean()),maximum=float(a.max()))


def oracle_summary(oracle):
    result={}
    for history in (oracle or {}).get('histories',[]):
        by_method={};full={}
        for action in history.get('actions',[]):
            for method in action.get('methods',[]):
                for regime in action.get('regimes',[]):
                    errors=regime.get('errors',{})
                    parts={}
                    for label in ('G00_actual_mean_bias','G11_actual_mean_bias',
                                  'line_truth_only','G11_minus_G00_correction'):
                        value=errors.get(label,{}).get('v',{}).get('RMS_physical')
                        if value is not None:parts[label]=value
                    if parts:full.setdefault(method,[]).append(parts)
        for row in history.get('response_differences',[]):
            if row.get('delta_zero'):continue
            for method in row.get('methods',[]):
                values={}
                for label,variant in row.get('variants',{}).items():
                    value=variant.get('error',{}).get('v',{}).get('RMS_physical')
                    relative=variant.get('relative_RMS',{}).get('v')
                    if value is not None:values[label]=dict(error_RMS_V=value,relative_RMS=relative)
                by_method.setdefault(method,[]).append(dict(regime=row['regime'],variants=values,
                    actual_signal_RMS_V=row.get('actual_signal',{}).get('v',{}).get('RMS_physical')))
        result[history['spec']['index']]=dict(available=history.get('available'),
            full_voltage_errors=full,response_relative_to_constant=by_method,
            input_raw_sha256=history.get('raw_sha256'),
            scope='saved oracle diagnostic; true replacements never enter selector or fits')
    return result


def inspect_history(record,raw_hash,oracle):
    names=record.get('candidate_names',[]);saved=record.get('refit_history',{}).get('candidates',{})
    rows=[]
    for name in names:
        reference=saved.get(name,{});profiles=[];failures=[]
        for key,action in record.get('actions',{}).items():
            for regime,actual in action.get('regimes',{}).items():
                for label,fit in [*(('noise_'+str(i),x) for i,x in enumerate(actual.get('refits',[]))),
                                  ('clean',actual.get('clean_refit',{}))]:
                    candidate=fit.get('comparisons',{}).get(name,{})
                    for start in candidate.get('starts',[]):
                        if start.get('label')!='bayes':continue
                        source=start.get('source_fits',{}).get('history',{})
                        block=start.get('witness',{}).get('blocks',{}).get('history')
                        if source.get('success') and block is not None:
                            profiles.append(dict(action_key=key,regime=regime,noise=label,
                                SSE=float(block['loss']),channel_SSE=block['channel_loss'],
                                source_profile_SSE=source.get('loss'),
                                source_status=source.get('status'),
                                source_nfev=source.get('nfev'),
                                source_optimality=source.get('optimality'),
                                source_physically_feasible=source.get('diagnostics',{}).get('physical_voltage_feasible'),
                                r=start.get('witness_state',{}).get('r'),
                                x=start.get('witness_state',{}).get('x')))
                        else:
                            failures.append(dict(action_key=key,regime=regime,noise=label,
                                start_error=start.get('error'),source_success=source.get('success')))
        row=dict(candidate=name,is_true_candidate=name==record.get('true_name'),
            available=False,reference_success=reference.get('success'),
            cached_history_local_reference_SSE=reference.get('loss'),
            repeated_profile_count=len(profiles),profile_failures=failures)
        if profiles and reference.get('success'):
            losses=[p['SSE'] for p in profiles];deltas=[x-reference['loss'] for x in losses]
            base=profiles[0];channels=reference['blocks']['history']['channel_loss']
            repeat_spread=float(max(losses)-min(losses))
            row.update(available=True,Bayes_center_history_source_profile_SSE=base['SSE'],
                delta_SSE=base['SSE']-reference['loss'],
                delta_conditional_loglik=-.5*(base['SSE']-reference['loss']),
                SSE_ratio=base['SSE']/max(reference['loss'],1e-30),
                channel_delta_SSE={k:base['channel_SSE'][k]-channels[k] for k in channels},
                channel_reference_SSE=channels,channel_Bayes_profile_SSE=base['channel_SSE'],
                duplicate_D_profile_SSE_spread=repeat_spread,
                duplicate_D_profiles_consistent=repeat_spread<=2e-7*(1+abs(base['SSE'])),
                saved_profile_and_source_solver_loss_error=max(
                    abs(p['SSE']-p['source_profile_SSE']) for p in profiles),
                all_source_profiles_success=True,
                all_source_profiles_physical=all(p['source_physically_feasible'] for p in profiles),
                first_profile=base,
                source_profile_provenance=[{k:v for k,v in p.items() if k not in ('r','x')}
                                           for p in profiles])
        rows.append(row)
    result=dict(spec=record['spec'],status=record.get('status'),true_name=record.get('true_name'),
        source_failures=record.get('failures',{}),candidate_count=len(names),candidates=rows)
    truth=next((r for r in rows if r['is_true_candidate']),None)
    result['true_candidate_profile']=truth
    if oracle is not None:
        result['oracle_saved_response']=oracle
        result['oracle_matches_same_raw_history']=oracle.get('input_raw_sha256')==raw_hash
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--phase',default='development');parser.add_argument('--oracle')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    began=perf_counter();script_sha=sha256(Path(__file__).read_bytes()).hexdigest()
    paths=sorted((Path(args.root)/args.phase).glob('history_*.json.gz'))
    oracle_file=None if args.oracle is None else Path(args.oracle)
    oracle=oracle_summary(None if oracle_file is None else read(oracle_file))
    histories=[];inputs=[]
    for path in paths:
        data=path.read_bytes();digest=sha256(data).hexdigest();r=read(path)
        histories.append(inspect_history(r,digest,oracle.get(r['spec']['index'])))
        inputs.append(dict(path=str(path.resolve()),sha256=digest,bytes=len(data)))
    available=[c for h in histories for c in h['candidates'] if c['available']]
    truth=[h['true_candidate_profile'] for h in histories
           if h['true_candidate_profile'] is not None and h['true_candidate_profile']['available']]
    summary=dict(histories=len(histories),histories_with_true_profile=len(truth),
        candidate_slots=sum(h['candidate_count'] for h in histories),
        available_candidate_profiles=len(available),
        unavailable_candidate_profiles=sum(h['candidate_count'] for h in histories)-len(available),
        delta_SSE_all_candidates=statistics([c['delta_SSE'] for c in available]),
        delta_SSE_true_candidates=statistics([c['delta_SSE'] for c in truth]),
        SSE_ratio_true_candidates=statistics([c['SSE_ratio'] for c in truth]),
        negative_delta_candidates=sum(c['delta_SSE'] < -1e-6 for c in available),
        descriptive_delta_counts={str(t):sum(c['delta_SSE']>t for c in truth) for t in (1,4,10,100)},
        threshold_scope='descriptive SSE cutoffs only; not calibrated tests or posterior regions',
        duplicate_D_profiles_all_consistent=all(c['duplicate_D_profiles_consistent'] for c in available),
        all_saved_source_profiles_physically_feasible=all(c['all_source_profiles_physical'] for c in available))
    result=dict(scope='saved conditional historical source-profile extraction; no fitting, AC solving or case generation',
        formula='Delta=S_D(theta_Bayes,argmin_s S_D)-S_D(theta_local,argmin_theta,s S_D); reported local attained values',
        probability_scope='P/Q readings fixed; conditional voltage/master Gaussian SSE; differs from latent-PQ Bayesian integral',
        interpretation=['positive Delta quantifies loss of historical conditional fit at the Bayes line center',
            'source and voltage feasibility distinguish physical legality from worse statistical fit',
            'large Delta does not by itself exclude the center under the different latent-PQ posterior model',
            'near-zero Delta plus poor EV response would support a historical compensation explanation',
            'negative Delta flags a poorer cached local history optimum, not a contradiction of exact nesting',
            'repeated historical profiles across plans/regimes are deduplicated to one history/candidate'],
        summary=summary,histories=histories,input_files=inputs,
        oracle_file=None if oracle_file is None else dict(path=str(oracle_file.resolve()),sha256=sha256(oracle_file.read_bytes()).hexdigest()),
        new_optimizations=0,new_AC_calls=0,new_scientific_samples=0,
        raw_and_frozen_sources_modified=False,audit_script_sha256=script_sha,
        audit_seconds=perf_counter()-began)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
