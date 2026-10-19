"""Rebuild saved local Gaussian history posterior; compare full-AC likelihood.

No optimization, no generated case, no new statistical samples. Only saved
centers are reevaluated by the original AC/Jacobian, with saved-Jacobian hashes
and saved posterior marginals checked before interpreting likelihood changes.
"""
from __future__ import annotations
import argparse
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
import numpy as np
from scipy.linalg import cho_factor,cho_solve,block_diag


def read(path):
    p=Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz'
                      else p.read_text(encoding='utf-8'))


def check_close(a,b,*,atol=2e-8,rtol=2e-8):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return dict(success=bool(a.shape==b.shape and np.allclose(a,b,atol=atol,rtol=rtol)),
        maximum_absolute_error=float(np.max(abs(a-b),initial=0)) if a.shape==b.shape else None,
        atol=atol,rtol=rtol)


def condition(mean,cov,y,offset,A):
    cross=cov@A.T;S=np.eye(len(y))+A@cross;S=(S+S.T)/2
    factor=cho_factor(S,lower=True);alpha=cho_solve(factor,y-offset-A@mean)
    m=mean+cross@alpha;C=cov-cross@cho_solve(factor,cross.T)
    return m,(C+C.T)/2


def channel_blocks(vector,T,n):
    blocks={};start=0
    for name in ('p','q','v','master_p','master_q'):
        width=T*n if name in ('p','q','v') else T
        blocks[name]=np.asarray(vector)[start:start+width];start+=width
    if start!=len(vector):raise ValueError('invalid channel vector width')
    return blocks


def summarize_residuals(y,affine,actual,T,n,meter):
    r0=np.asarray(affine)-y;r1=np.asarray(actual)-y
    b0,b1=channel_blocks(r0,T,n),channel_blocks(r1,T,n)
    d=channel_blocks(np.asarray(actual)-affine,T,n)
    channels={}
    for name in b0:
        S0=float(b0[name]@b0[name]);S1=float(b1[name]@b1[name])
        sigma=np.broadcast_to(np.asarray(getattr(meter,name+'_sigma')), (T,n) if name in ('p','q','v') else (T,)).ravel()
        channels[name]=dict(affine_SSE=S0,full_AC_SSE=S1,delta_SSE=S1-S0,
            delta_pointwise_loglik=-.5*(S1-S0),
            AC_minus_affine_RMS_whitened=float(np.sqrt(np.mean(d[name]**2))),
            AC_minus_affine_RMS_pu=float(np.sqrt(np.mean((d[name]*sigma)**2))))
    S0=float(r0@r0);S1=float(r1@r1)
    return dict(affine_SSE=S0,full_AC_SSE=S1,delta_SSE=S1-S0,
        delta_pointwise_loglik=-.5*(S1-S0),channels=channels,
        original_whitened_observed=y,affine_whitened_mean=affine,
        full_AC_whitened_mean=actual)


def inspect_candidate(record,desc,source):
    from research_experiments.mobile_ev_mle import likelihood as lk
    from research_experiments.mobile_ev_mle import ev_bayes_response_ac as old
    from research_experiments.mobile_ev_mle.symmetric_ac_local import mean_and_jacobian,whiten_observations
    began=perf_counter();name=desc['name'];prediction=record['prediction'];updates=prediction['candidate_updates']
    u=updates[name];prior=u['line_prior'];d=len(prior['mean']);n=desc['n'];width=2*n+2
    H=len(record['observed_training']['p']);V=len(record['observed_evidence']['p']);F=record['config']['horizon']
    training=lk.Observations(**record['observed_training']);evidence=lk.Observations(**record['observed_evidence'])
    tm=lk.MeasurementModel(**record['train_meter']);vm=lk.MeasurementModel(**record['evidence_meter'])
    candidate=lk.RadialCandidate.from_supports(name,n,desc['supports'])
    center=u['training_center'];r0=np.asarray(center['r']);x0=np.asarray(center['x']);eta0=np.asarray(center['eta'])
    z0=np.asarray(center['latent'])
    response=mean_and_jacobian(candidate,r0,x0,z0[:,:n],z0[:,n:2*n],
        z0[:,-2],z0[:,-1],training.root_v,tm,with_jacobian=True)
    JL=response['line_jacobian']@old._eta_jacobian(r0,x0);Jz=response['state_jacobian']
    audits=dict(training_JL_hash=sha256(JL.astype('<f8').tobytes()).hexdigest()==u['training_JL_sha256'],
        training_Jz_hash=sha256(Jz.astype('<f8').tobytes()).hexdigest()==u['training_Jz_sha256'])
    _,K=old.process_kernel(prediction['process_hyperparameters'],H+V+F,
                           start_time=prediction['start_time'])
    m0=np.r_[prior['mean'],np.zeros((H+V+F)*width)]
    C0=block_diag(np.asarray(prior['cov']),K)
    AT=np.zeros((len(response['mean']),len(m0)));AT[:,:d]=JL;AT[:,d:d+H*width]=Jz
    offsetT=response['mean']-JL@eta0-Jz@z0.ravel()
    audits['original_training_prior_observation_mean']=check_close(
        offsetT+AT@m0,u['training_observation_mean'])
    yT=whiten_observations(training,tm);yV=whiten_observations(evidence,vm)
    mT,CT=condition(m0,C0,yT,offsetT,AT)
    ef=np.r_[np.arange(d),d+np.arange(H*width,(H+V+F)*width)]
    audits['T_saved_eta_EF_mean']=check_close(mT[ef],u['T_untruncated_EF_mean'])
    audits['T_saved_eta_EF_covariance']=check_close(CT[np.ix_(ef,ef)],u['T_untruncated_EF_covariance'])
    JV=np.asarray(u['evidence_affine_jacobian']);offsetV=np.asarray(u['evidence_affine_offset'])
    AV=np.zeros((len(yV),len(m0)));AV[:,:d]=JV[:,:d];AV[:,d+H*width:]=JV[:,d:]
    mTV,CTV=condition(mT,CT,yV,offsetV,AV)
    future=np.r_[np.arange(d),d+np.arange((H+V)*width,(H+V+F)*width)]
    audits['TV_saved_untruncated_eta_F_mean']=check_close(mTV[future],u['TV_untruncated_future']['mean'])
    audits['TV_saved_untruncated_eta_F_covariance']=check_close(CTV[np.ix_(future,future)],u['TV_untruncated_future']['cov'])
    lm=np.asarray(u['TV_line_integration']['mean']);LC=np.asarray(u['TV_line_integration']['cov'])
    inverse=old._inverse_psd(CTV[:d,:d]);gain=CTV[:,:d]@inverse
    bounded_mean=mTV+gain@(lm-mTV[:d]);bounded_mean[:d]=lm
    Csub=CTV[np.ix_(future,future)];gsub=CTV[np.ix_(future,np.arange(d))]@inverse
    bounded_cov=Csub+gsub@(LC-CTV[:d,:d])@gsub.T
    audits['TV_saved_bounded_eta_F_mean']=check_close(bounded_mean[future],u['TV_posterior_future']['mean'])
    audits['TV_saved_bounded_eta_F_covariance']=check_close(bounded_cov,u['TV_posterior_future']['cov'])
    good=all(v if isinstance(v,bool) else v['success'] for v in audits.values())
    result=dict(candidate=name,is_true_candidate=name==record['true_name'],
        reconstruction_success=good,reconstruction_checks=audits,
        matrix_dimensions=dict(variables=len(m0),training_observations=len(yT),evidence_observations=len(yV)),
        computations=dict(new_optimizations=0,new_scientific_samples=0,
            saved_center_AC_Jacobian_evaluations=1,pure_Gaussian_updates=2))
    if not good:
        result.update(status='reconstruction_mismatch',seconds=perf_counter()-began);return result
    eta=bounded_mean[:d].reshape(-1,2);r=np.exp(eta[:,0]);x=np.exp(eta.sum(1))
    latent=bounded_mean[d:].reshape(H+V+F,width)
    exacts=[]
    for label,obs,meter,rows,A,offset,y in (
        ('training',training,tm,slice(0,H),AT,offsetT,yT),
        ('evidence',evidence,vm,slice(H,H+V),AV,offsetV,yV)):
        z=latent[rows];actual=mean_and_jacobian(candidate,r,x,z[:,:n],z[:,n:2*n],z[:,-2],z[:,-1],
            obs.root_v,meter,with_jacobian=False)
        affine=offset+A@bounded_mean
        s=summarize_residuals(y,affine,actual['mean'],obs.t,n,meter)
        source_bounds=prediction['options'];physical=actual['physical'];voltage=abs(physical['voltage'])
        s['physical']=dict(ac_equation_residual=physical['ac_equation_residual'],
            power_balance_residual=physical['power_balance_residual'],
            min_voltage=float(voltage.min()),max_voltage=float(voltage.max()),
            source_p_in_domain=bool(np.all(z[:,-2]>=source_bounds['source_p_bounds'][0]) and
                                    np.all(z[:,-2]<=source_bounds['source_p_bounds'][1])),
            source_q_in_domain=bool(np.all(z[:,-1]>=source_bounds['source_q_bounds'][0]) and
                                    np.all(z[:,-1]<=source_bounds['source_q_bounds'][1])))
        result[label]=s;exacts.append(s)
    prior_eta=np.asarray(prior['mean']);invline=old._inverse_psd(np.asarray(prior['cov']))
    prior_line=.5*float((bounded_mean[:d]-prior_eta)@invline@(bounded_mean[:d]-prior_eta))
    history_latent=latent[:H+V].ravel();KD=K[:len(history_latent),:len(history_latent)]
    prior_process=.5*float(history_latent@cho_solve(cho_factor(KD,lower=True),history_latent))
    result.update(status='ok',r=r,x=x,posterior_history_latent_mean=latent[:H+V],
        saved_approximate_posterior_mean_not_MAP=True,
        history=dict(affine_SSE=sum(e['affine_SSE'] for e in exacts),
            full_AC_SSE=sum(e['full_AC_SSE'] for e in exacts),
            delta_SSE=sum(e['delta_SSE'] for e in exacts),
            delta_pointwise_loglik=sum(e['delta_pointwise_loglik'] for e in exacts),
            line_prior_penalty=prior_line,process_prior_penalty=prior_process,
            affine_negative_log_joint_up_to_constants=.5*sum(e['affine_SSE'] for e in exacts)+prior_line+prior_process,
            full_AC_negative_log_joint_up_to_constants=.5*sum(e['full_AC_SSE'] for e in exacts)+prior_line+prior_process),
        computations=dict(new_optimizations=0,new_scientific_samples=0,
            saved_center_AC_Jacobian_evaluations=1,history_posterior_mean_AC_evaluations=2,
            pure_Gaussian_updates=2),seconds=perf_counter()-began)
    return result


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',default='development')
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    sys.path.insert(0,str(Path(args.source).resolve().parents[1]))
    began=perf_counter();histories=[];inputs=[]
    for path in sorted((Path(args.root)/args.phase).glob('history_*.json.gz')):
        data=path.read_bytes();r=read(path);row=dict(spec=r['spec'],status=r.get('status'),
            saved_failures=r.get('failures',{}),candidates=[],available=False)
        if not r.get('prediction') or 'candidate_updates' not in r['prediction']:
            row['unavailable_reason']='saved historical posterior missing'
        else:
            for desc in r['candidate_structures']:
                try:result=inspect_candidate(r,desc,Path(args.source))
                except Exception as exc:result=dict(candidate=desc['name'],is_true_candidate=desc['name']==r['true_name'],
                    status='audit_exception',failure=type(exc).__name__+': '+str(exc),reconstruction_success=False)
                row['candidates'].append(result)
                print('HISTORY_AFFINE_AUDIT',r['spec']['index'],desc['name'],result['status'],
                      result.get('seconds'),flush=True)
            row['available']=all(c['status']=='ok' for c in row['candidates'])
        histories.append(row);inputs.append(dict(path=str(path.resolve()),sha256=sha256(data).hexdigest()))
    successful=[c for h in histories for c in h['candidates'] if c['status']=='ok']
    truth=[c for c in successful if c['is_true_candidate']]
    result=dict(scope='same saved empirical local-Gaussian history posterior; pointwise exact versus original affine AC measurement likelihood',
        warning='posterior means are not MAP; pointwise likelihood differences are not integrated topology evidence or calibrated exclusion',
        histories=histories,input_files=inputs,
        summary=dict(histories=len(histories),available_histories=sum(h['available'] for h in histories),
            expected_candidate_slots=sum(len(h.get('candidates',[])) or 4 for h in histories),
            candidate_reconstructions=len(successful),true_candidate_reconstructions=len(truth),
            true_candidate_affine_SSE=[c['history']['affine_SSE'] for c in truth],
            true_candidate_full_AC_SSE=[c['history']['full_AC_SSE'] for c in truth],
            true_candidate_delta_SSE=[c['history']['delta_SSE'] for c in truth]),
        new_optimizations=0,new_statistical_samples=0,
        audit_AC_Jacobian_calls=sum(c.get('computations',{}).get('saved_center_AC_Jacobian_evaluations',0) for c in successful),
        audit_AC_point_calls=sum(c.get('computations',{}).get('history_posterior_mean_AC_evaluations',0) for c in successful),
        audit_script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),audit_seconds=perf_counter()-began,
        raw_and_frozen_sources_modified=False)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(serial(result),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__=='__main__':main()
