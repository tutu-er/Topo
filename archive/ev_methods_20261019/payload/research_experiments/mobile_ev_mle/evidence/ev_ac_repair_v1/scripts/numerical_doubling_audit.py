"""Saved-history propagation quadrature audit; no fit or plan is changed."""
from __future__ import annotations
import argparse
from hashlib import sha256
import gzip
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import numpy as np
from scipy.special import logsumexp
from research_experiments.mobile_ev_mle import likelihood as lk
from research_experiments.mobile_ev_mle import ev_ac_repair_propagation as propagation
from research_experiments.mobile_ev_mle import ev_bayes_response_ac as old
from research_experiments.mobile_ev_mle.candidates import RadialCandidate


def serialize(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def sha(path):return sha256(Path(path).read_bytes()).hexdigest()


def classify(bank,values):
    values=np.atleast_2d(values);scores=[]
    for mean,cov,lp in zip(np.asarray(bank['means'])[0],np.asarray(bank['covariances'])[0],bank['log_prior']):
        L=np.linalg.cholesky(cov);z=np.linalg.solve(L,(values-mean).T)
        scores.append(lp-.5*(np.sum(z*z,axis=0)+2*np.log(np.diag(L)).sum()+len(mean)*np.log(2*np.pi)))
    scores=np.asarray(scores).T;posterior=np.exp(scores-logsumexp(scores,axis=1)[:,None])
    return dict(choices=posterior.argmax(1).tolist(),confidence=posterior.max(1).tolist())


def reconstruct(record):
    update=record['prediction'];rows=[]
    for s in record['candidate_structures']:
        name=s['name'];c=RadialCandidate.from_supports(name,s['n'],[x for x in s['supports'] if 1<len(x)<s['n']])
        center=record['posterior_centers'][name]
        rows.append(dict(candidate=c,r=np.asarray(center['r']),x=np.asarray(center['x']),
            line_sha256=old._line_id(center['r'],center['x']),line_dimension=center['line_dimension'],
            joint_mean=np.asarray(center['joint_mean']),joint_covariance=np.asarray(center['joint_covariance']),
            latent_mean=np.asarray(update['candidate_updates'][name]['future_latent_mean'])))
    return dict(rows=rows,n=rows[0]['candidate'].n,horizon=record['config']['horizon'],
        base_kw=record['provenance']['base_kw'],root_v=np.ones(record['config']['horizon']),
        future_meter=lk.MeasurementModel(**record['future_meter']),options=update['options'],
        log_prior_all_history=np.asarray(update['log_prior']),
        log_prior_evidence_only=np.asarray(update['log_prior_evidence_only'])),update


def run(path,draws):
    began=perf_counter();digest=sha(path)
    with gzip.open(path,'rt',encoding='utf-8') as stream:r=json.load(stream)
    key=r['plans']['SQMD']['action_key'];action=r['actions'][key];power=np.asarray(action['power'])
    if 'G11' not in action['propagation'].get('banks',{}):
        return dict(path=str(path),available=False,reason='saved G11 unavailable',raw_sha256=digest)
    context,update=reconstruct(r)
    reference_started=perf_counter();reference=old.forecast_bank(context,power)
    reference_seconds=perf_counter()-reference_started
    reference_difference=max(float(np.max(abs(np.asarray(reference['bank'][field])-np.asarray(action['propagation']['banks']['G00'][field])))) for field in ('means','covariances','log_prior'))
    seed=int(np.random.SeedSequence([r['spec']['seed'],6200,20261017]).generate_state(1)[0])
    cache={};result=propagation.forecast_moments(context,update,power,seed=seed,draws=draws,replicates=4,posterior_cache=cache,reference=reference)
    rows=[]
    for oldrow,newrow in zip(action['propagation']['rows'],result['rows']):
        row=dict(candidate=oldrow['candidate'],old_status=oldrow['status'],new_status=newrow['status'])
        if 'mean' in oldrow and 'mean' in newrow:
            mu0,mu1=np.asarray(oldrow['mean']),np.asarray(newrow['mean']);C0,C1=np.asarray(oldrow['covariance']),np.asarray(newrow['covariance'])
            D=C1-C0;mean_diff=mu1-mu0
            rowscale=np.sqrt(np.maximum(np.diag(C0),1e-30));total_cov_scale=max(np.linalg.norm(C0),1e-30)
            propagated_cov_scale=max(np.linalg.norm(C0-np.eye(len(mu0))),1e-30)
            row.update(mean_whitened_max_change=float(np.max(abs(mean_diff))),
                mean_whitened_RMS_change=float(np.sqrt(np.mean(mean_diff**2))),
                mean_predictive_SD_scaled_max_change=float(np.max(abs(mean_diff)/rowscale)),
                covariance_relative_fro_change=float(np.linalg.norm(D)/total_cov_scale),
                propagated_covariance_relative_fro_change=float(np.linalg.norm(D)/propagated_cov_scale),
                covariance_max_abs_change=float(np.max(abs(D))),
                old_mean_MCSE_whitened_max=oldrow['mean_MCSE_whitened_max'],
                new_mean_MCSE_whitened_max=newrow['mean_MCSE_whitened_max'],
                old_covariance_MCSE_relative_fro=oldrow['covariance_MCSE_relative_fro'],
                new_covariance_MCSE_relative_fro=newrow['covariance_MCSE_relative_fro'],
                old_covariance_fallback=oldrow['covariance_roundoff']['ordinary_AC_sample_covariance_fallback'],
                new_covariance_fallback=newrow['covariance_roundoff']['ordinary_AC_sample_covariance_fallback'])
        rows.append(row)
    classifications=[]
    for regime,actual in action['regimes'].items():
        values=np.vstack((actual['clean'],actual['values']))
        for model in ('G00','G10','G01','G11'):
            if model not in result['banks']:continue
            before=classify(action['propagation']['banks'][model],values);after=classify(result['banks'][model],values)
            classifications.append(dict(regime=regime,model=model,old=before,new=after,
                changed=sum(a!=b for a,b in zip(before['choices'],after['choices'])),
                true_index=r['true_index'],first_row_is_clean=True))
    return dict(path=str(path.resolve()),spec=r['spec'],raw_sha256=digest,raw_unchanged=sha(path)==digest,
        available=True,action_key=key,methods=action['methods'],old_draws=r['config']['draws'],new_draws=draws,
        replicates=4,reference_max_abs_difference=reference_difference,new_status=result['status'],
        comparisons=rows,classifications=classifications,new_propagation=result,
        integration={k:{kk:vv for kk,vv in v.items() if kk!='samples'} for k,v in cache.items()},
        audit_cost=dict(reference_AC_states=len(context['rows'])*context['horizon'],
            reference_AC_batches=len(context['rows']),reference_seconds=reference_seconds,
            propagation_AC_states=result['diagnostics']['AC_states'],
            propagation_AC_batches=len(context['rows']),
            GHK_points=sum(v['diagnostics']['integration']['points'] for v in cache.values()),
            GHK_and_conditional_sampling_seconds=sum(v['seconds'] for v in cache.values()),
            propagation_seconds=result['seconds'],seconds=perf_counter()-began))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--indices',default='0,7');p.add_argument('--draws',default=512,type=int)
    p.add_argument('--overwrite',action='store_true');args=p.parse_args();output=Path(args.output)
    if output.exists() and not args.overwrite:raise FileExistsError(str(output))
    began=perf_counter();records=[]
    for index in map(int,args.indices.split(',')):
        path=Path(args.root)/'development'/('history_%03d.json.gz'%index)
        records.append(run(path,args.draws));print('DOUBLED',index,records[-1].get('new_status'),flush=True)
    result=dict(scope='saved approximate historical posterior, same fixed SQMD plan; diagnostic only, no main parameter change',
        script_sha256=sha(__file__),histories=records,summary=dict(histories=len(records),
            classifications_changed=sum(x['changed'] for r in records for x in r.get('classifications',[])),
            all_raw_unchanged=all(r.get('raw_unchanged',False) for r in records),
            extra_AC_states=sum(r.get('audit_cost',{}).get('reference_AC_states',0)+r.get('audit_cost',{}).get('propagation_AC_states',0) for r in records),
            seconds=perf_counter()-began))
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,default=serialize),encoding='utf-8')
    print('SAVED',output,json.dumps(result['summary']),flush=True)


if __name__=='__main__':main()
