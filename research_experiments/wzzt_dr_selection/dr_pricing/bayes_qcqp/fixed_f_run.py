"""Fixed-F posterior bounds: plan with existing solver, freeze, then sample.

Full time/node covariance is fixed input. No topology fitting or actual future
is imported. This driver differs from the original expected-evidence mode.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
from scipy.special import expit, log_expit, logit, logsumexp, ndtr
from scipy.linalg import eigh

if __package__:
    from .run import load_inputs,scalar_summary,_guard
    from .fixed_f import build_fixed_f_full,thresholds,guarantees
    from .solver import solve
    from ..solve import reference,write_arrays,write_json
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    from dr_pricing.bayes_qcqp.run import load_inputs,scalar_summary,_guard
    from dr_pricing.bayes_qcqp.fixed_f import build_fixed_f_full,thresholds,guarantees
    from dr_pricing.bayes_qcqp.solver import solve
    from dr_pricing.solve import reference,write_arrays,write_json


def load(case,settings_path):
    problem,settings,old_arrays,_,_,prior,_=load_inputs(case,settings_path)
    covariance_path=Path(settings_path).resolve().parent/settings['fixed_f_arrays_file']
    with np.load(covariance_path,allow_pickle=False) as z:
        fixed={key:z[key].copy() for key in z.files}
    sd=old_arrays['likelihood_sd']
    base=(problem.base['C']-problem.base['F'])/sd
    effect=np.broadcast_to(problem.effect['C']-problem.effect['F'],
        (problem.T,problem.N,problem.device.R))/sd[:,:,None]
    if fixed['covariance'].shape!=(base.size,base.size):
        raise ValueError('Full fixed covariance must cover every slot and node')
    if not np.isfinite(fixed['covariance']).all():
        raise ValueError('Finite fixed covariance required')
    if 'ar_spatial_factor' in fixed:
        factor=fixed['ar_spatial_factor'];ar=float(fixed['time_ar'])
        if factor.ndim!=3 or factor.shape[:2]!=base.shape or not 0<=ar<1:
            raise ValueError('AR factor differs from fixed observation shape')
        time=ar**np.abs(np.arange(problem.T)[:,None]-np.arange(problem.T)[None,:])
        rebuilt=np.einsum('tnc,smc,ts->tnsm',factor,factor,time).reshape(base.size,base.size)
        if not np.allclose(rebuilt,fixed['covariance'],atol=1e-9,rtol=1e-10):
            raise ValueError('Sampling factor and frozen full covariance differ')
    initial=prior+settings['history_log_bayes_factor']
    return problem,settings,fixed,base,effect,initial,covariance_path


def sources():
    folder=Path(__file__).resolve().parent
    return [folder/name for name in ('fixed_f.py','fixed_f_run.py','core.py','run.py','solver.py','FIXED_F_PROTOCOL.json')]+[folder.parent/'problem.py']


def _plan(case,settings_path,mode,out):
    out=Path(out).resolve()
    if (out/'request.json').exists():raise ValueError('Existing attempt preserved')
    p,s,fixed,base,effect,initial,cov_path=load(case,settings_path)
    target=thresholds(s['mean_posterior_target'],s['posterior_threshold'],s['event_probability_target'],initial)
    if mode not in target:raise ValueError('mode must be mean or probability')
    src=sources()
    request=dict(mode=mode,case=reference(p.case_path),case_arrays=reference(p.arrays_path),
        settings=reference(settings_path),bayes_arrays=reference(Path(settings_path).parent/s['arrays_file']),
        fixed_f_arrays=reference(cov_path),sources={str(x):reference(x)['sha256'] for x in src},
        initial_log_odds=initial,threshold=target[mode],covariance_is_price_independent=True,
        all_time_node_correlations_retained=True,no_actual_future_or_truth_access=True,
        python=platform.python_version(),numpy=np.__version__)
    write_json(out/'request.json',request)
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for i,path in enumerate(src):z.write(path,f'{i:02d}_{path.name}')
    began=time.perf_counter()
    evidence=build_fixed_f_full(base,effect,fixed['covariance'])
    write_arrays(out/'quadratic.npz',Q=evidence.Q,linear=evidence.linear,constant=evidence.constant)
    result=solve(p,evidence,target[mode],out/'solve',**s['solver'])
    summary=dict(mode=mode,qualified=bool(result['qualified']),initial_log_odds=initial,
        q_star=s['mean_posterior_target'],eta=s['posterior_threshold'],beta=s['event_probability_target'],
        threshold=target[mode],solver=scalar_summary(result),seconds=time.perf_counter()-began,
        covariance_shape=list(fixed['covariance'].shape),covariance_sha256=reference(cov_path)['sha256'],
        statistical_scope='conditional on fine frozen Gaussian candidate, sufficient guarantee only',
        fixed_F_scope=s['fixed_f_scope'])
    if result['qualified']:
        score=evidence.value(result['mean_u'])
        summary.update(fee=result['objective'],T=score,
            guarantees=guarantees(score,initial,s['posterior_threshold']),
            nominal_device=p.audit_delivery(result['mean_u']))
        write_arrays(out/'quote.npz',x=result['x'],mean_u=result['mean_u'],unit_reward=result['unit_reward'])
        write_json(out/'policy_freeze.json',dict(status='frozen',quote=reference(out/'quote.npz'),
            request=reference(out/'request.json'),no_actual_future_or_truth_access=True))
    write_json(out/'summary.json',summary)
    write_json(out/'complete.json',dict(status='complete',qualified=summary['qualified'],summary=reference(out/'summary.json')))
    return summary


def plan(case,settings_path,mode,out):
    return _guard(out,lambda:_plan(case,settings_path,mode,out))


def frozen_quote(folder,p,settings_path,cov_path):
    folder=Path(folder).resolve();freeze=json.loads((folder/'policy_freeze.json').read_text(encoding='utf-8'))
    if freeze['status']!='frozen':raise ValueError('Unfrozen quote')
    quote_path=Path(freeze['quote']['path'])
    if reference(quote_path)!=freeze['quote']:raise ValueError('Quote changed')
    req_path=Path(freeze['request']['path'])
    if reference(req_path)!=freeze['request']:raise ValueError('Frozen request changed')
    req=json.loads(req_path.read_text(encoding='utf-8'))
    for key,path in (('case',p.case_path),('case_arrays',p.arrays_path),('settings',settings_path),('fixed_f_arrays',cov_path)):
        if reference(path)!=req[key]:raise ValueError('Frozen input changed: '+key)
    settings=json.loads(Path(settings_path).read_text(encoding='utf-8'))
    if reference(Path(settings_path).parent/settings['arrays_file'])!=req['bayes_arrays']:
        raise ValueError('Frozen Bayes arrays changed')
    for path,digest in req['sources'].items():
        if reference(path)['sha256']!=digest:raise ValueError('Frozen source changed: '+path)
    with np.load(quote_path,allow_pickle=False) as z:
        x,u=z['x'].copy(),z['mean_u'].copy()
    if not np.allclose(p.device.nominal(x),u,atol=1e-9,rtol=0):raise ValueError('Quote replay differs')
    return u,reference(quote_path)


def _sample(case,settings_path,folders,out,count,seed):
    out=Path(out)
    if (out/'request.json').exists():raise ValueError('Existing attempt preserved')
    if count<2:raise ValueError('At least two independent paths required')
    p,s,fixed,base,effect,initial,cov_path=load(case,settings_path)
    contracts={};refs={}
    for name,folder in folders.items():
        contracts[name],refs[name]=frozen_quote(folder,p,settings_path,cov_path)
    contracts['natural_zero']=np.zeros((p.device.R,p.T))
    means={name:base+np.einsum('tnr,rt->tn',effect,u) for name,u in contracts.items()}
    cut=float(logit(s['posterior_threshold']))
    request=dict(count=count,seed=seed,quotes=refs,case=reference(p.case_path),settings=reference(settings_path),
        covariance=reference(cov_path),source=reference(__file__),shared_random_paths=True,
        scope='fixed background Gaussian power error plus shared voltage error; deterministic nominal DR, no clipping or AC')
    write_json(out/'request.json',request)
    rng=np.random.default_rng(seed);vals={name:[] for name in means};sexp={name:[] for name in means}
    if 'ar_spatial_factor' in fixed:
        factor=fixed['ar_spatial_factor'];ar=float(fixed['time_ar'])
        if factor.shape[:2]!=base.shape or not 0<=ar<1:raise ValueError('Invalid full AR sampling factor')
        eigfactor=None
    else:
        eig,vec=eigh(fixed['covariance']);scale=max(1.,abs(eig).max())
        if eig.min() < -1e-10*scale:raise ValueError('Invalid sampling covariance')
        keep=eig>1e-12*scale;eigfactor=vec[:,keep]*np.sqrt(eig[keep])
    began=time.perf_counter()
    for first in range(0,count,128):
        batch=min(128,count-first)
        if eigfactor is None:
            innovations=rng.standard_normal((batch,p.T,factor.shape[2]))
            for t in range(1,p.T):
                innovations[:,t]=ar*innovations[:,t-1]+np.sqrt(1-ar*ar)*innovations[:,t]
            power_noise=np.einsum('tnc,btc->btn',factor,innovations)
        else:
            power_noise=(rng.standard_normal((batch,eigfactor.shape[1]))@eigfactor.T).reshape(batch,p.T,p.N)
        voltage_noise=rng.standard_normal((batch,p.T,p.N))
        for name,mean in means.items():
            z=mean[None]+power_noise;S=np.sum(z*z,axis=(1,2))
            odds=initial+.5*S+np.sum(z*voltage_noise,axis=(1,2))
            vals[name].append(odds);sexp[name].append(S)
    metrics={}
    for name in means:
        odds=np.concatenate(vals[name]);S=np.concatenate(sexp[name]);q=expit(odds)
        root=np.sqrt(S)
        conditional=np.where(S>0,ndtr(np.divide(initial+.5*S-cut,root,
            out=np.zeros_like(root),where=S>0)),float(initial>=cut))
        metrics[name]=dict(count=count,mean_posterior_F=float(q.mean()),
            mean_posterior_standard_error=float(q.std(ddof=1)/np.sqrt(count)),
            log_mean_one_minus_posterior=float(logsumexp(log_expit(-odds))-np.log(count)),
            posterior_event_successes=int(np.sum(odds>=cut)),posterior_event_fraction=float(np.mean(odds>=cut)),
            voltage_integrated_event_estimate=float(conditional.mean()),
            mean_S=float(S.mean()),mean_log_odds=float(odds.mean()))
        write_arrays(out/f'{name}_private.npz',log_odds=odds,S=S)
    summary=dict(status='complete',unique_random_paths=count,policy_outputs=count*len(means),
        independent_event_days=0,synthetic_working_model=True,seed=seed,seconds=time.perf_counter()-began,
        expected_targets=dict(mean=s['mean_posterior_target'],posterior_eta=s['posterior_threshold'],event=s['event_probability_target']),
        metrics=metrics,no_actual_AC_or_truth_read=True)
    write_json(out/'summary.json',summary);write_json(out/'complete.json',dict(status='complete'))
    return summary


def sample(case,settings_path,folders,out,count=16384,seed=2026111501):
    return _guard(out,lambda:_sample(case,settings_path,folders,out,count,seed))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','sample'])
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['mean','probability'])
    parser.add_argument('--plan',action='append',default=[],help='Frozen policy label=directory')
    args=parser.parse_args()
    result=(plan(args.case,args.settings,args.mode,args.out) if args.action=='plan' else
            sample(args.case,args.settings,dict(item.split('=',1) for item in args.plan),args.out))
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
