"""Paired 3/7-day ordinary-load local AC-gradient study, with sealed inference."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter
import traceback
import sys

import numpy as np
import scipy

from .. import likelihood as lk
from ..local_rx_identifiability.cases import make_case, _rng
from ..local_rx_identifiability.experiment import read, write
from .solver import fit
from .diagnostics import diagnose


ROOT = Path(__file__).resolve().parents[3]


def specifications(pilot=False):
    if pilot:
        return [dict(network='paper15', seed=32991, N=288, boundary=b) for b in ('U','R')]
    return [dict(network=network,seed=seed,N=n,boundary=b)
            for network in ('paper15','paper11') for seed in range(33001,33005)
            for n in (288,672) for b in ('U','R','K')]


def key(spec):
    return f"{spec['network']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_sources():
    previous=read(ROOT/'research_experiments/mobile_ev_mle/local_rx_identifiability/evidence/source_manifest.json')
    names=set(previous)
    names.update('research_experiments/mobile_ev_mle/local_rx_three_day/'+name
                 for name in ('experiment.py','solver.py','diagnostics.py'))
    return {name:sha(ROOT/name) for name in sorted(names)}


def observed_case(spec):
    """Simulator makes truth; only the explicitly separated payload enters fit."""
    case=make_case(spec['network'],'ordinary',spec['seed'],total=768,boundary='U')
    n=spec['N']
    obs={k:np.asarray(case['observed'][k])[:n] for k in ('p','q','v')}
    sigma={k:np.asarray(case['meter']['sigma_'+k]) for k in ('p','q','v')}
    root_known=None;root_meter=None
    if spec['boundary']=='K':
        root_known=np.asarray(case['truth']['root_v'])[:n]
    elif spec['boundary']=='R':
        noisy=np.asarray(case['truth']['root_v'])+_rng(spec['seed'],spec['network'],14).normal(size=768)*.0005
        root_meter=(noisy[:n],.0005)
    payload=dict(spec=spec,observed=obs,sigma=sigma,domain=case['domain'],
                 candidates=[c.metadata() for c in case['candidates']],
                 root_known=root_known,root_meter=root_meter,
                 model='full_raw_PQV_EIV_AC_with_optional_noisy_root_meter',
                 truth_used=False,interval_minutes=15,
                 information_rule='R adds a noisy meter; K adds ideal boundary; neither is same-budget U',
                 generator=case['provenance'])
    # Old generator metadata describes its original 288/384 use. Override only
    # this run's descriptive metadata; leave the frozen generator untouched.
    payload['generator']=dict(payload['generator'],training_prefixes=[288,672],
        holdout_start=672,holdout_stop=768,study='ordinary_three_and_seven_day')
    return payload,case


def _relative(a,b):
    return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/np.linalg.norm(b))


def evaluate_sealed(case,payload,inference):
    """Truth is evaluated only after the observed inference file is saved."""
    rt,xt=np.asarray(case['truth']['r']),np.asarray(case['truth']['x'])
    bank={c.name:c for c in case['candidates']};sigma=payload['sigma']
    truth_physical=lk.ac_predict(bank['star'],rt,xt,case['truth']['p'],case['truth']['q'],
                                 root_v=case['truth']['root_v'])
    true_junction=abs(truth_physical['voltage'][:,0])
    records={}
    for name,fitted in inference['fits'].items():
        if 'r' not in fitted:
            records[name]=dict(qualified=False,status=fitted.get('status'));continue
        response=lk.ac_predict(bank[name],fitted['r'],fitted['x'],case['truth']['p'],case['truth']['q'],
                               root_v=case['truth']['root_v'])['v']
        true=np.asarray(case['truth']['v'])
        error=(response-true)/np.asarray(sigma['v'])
        local_r,local_x=np.array(fitted['r']),np.array(fitted['x'])
        local_r[0]=0.;local_x[0]=0.
        local_response=lk.ac_predict(bank[name],local_r,local_x,case['truth']['p'],case['truth']['q'],
                                    root_v=true_junction)['v']
        local_error=(local_response-true)/np.asarray(sigma['v'])
        # Orthogonal contrasts: two independent coordinates when sigmas equal.
        # Here used only as response-error diagnostics, not new likelihood.
        d=np.array([[1,-1,0],[1,1,-2]],float)/np.sqrt([[2],[6]])
        contrast=(local_response**2-true**2)@d.T
        standard=(2*true*np.asarray(sigma['v']))
        covariance=np.einsum('ij,tj,kj->tik',d,standard**2,d)
        contrast_sse=np.einsum('ti,tij,tj->t',contrast,np.linalg.inv(covariance),contrast)
        n=payload['spec']['N'];hold=slice(672,768)
        entry=dict(qualified=fitted['qualified'],
                   true_PQ_true_root_counterfactual=payload['spec']['boundary']=='U',
                   true_PQ_voltage_RMSE_sigma_train=float(np.sqrt(np.mean(error[:n]**2))),
                   true_PQ_voltage_RMSE_sigma_holdout=float(np.sqrt(np.mean(error[hold]**2))),
                   local_true_PQ_true_junction_RMSE_sigma_holdout=float(np.sqrt(np.mean(local_error[hold]**2))),
                   local_response_is_evaluation_only=True,
                   contrast_RMSE_sigma_holdout=float(np.sqrt(np.mean(contrast_sse[hold])/2)))
        if name=='star':
            r,x=np.asarray(fitted['r']),np.asarray(fitted['x'])
            p=np.asarray(payload['observed']['p'])@bank[name].M
            q=np.asarray(payload['observed']['q'])@bank[name].M
            kappa=np.sum(p*q,axis=0)/np.sum(p*p,axis=0)
            entry.update(r_total_relative=_relative(r,rt),x_total_relative=_relative(x,xt),
                         r_leaf_relative=_relative(r[1:],rt[1:]),x_leaf_relative=_relative(x[1:],xt[1:]),
                         r_leaf_each=(abs(r[1:]-rt[1:])/rt[1:]),
                         x_leaf_each=(abs(x[1:]-xt[1:])/xt[1:]),
                         beta_leaf_relative=_relative((r+kappa*x)[1:],(rt+kappa*xt)[1:]),
                         truth_information_not_used_to_fit=True)
        records[name]=entry
    return dict(truth_read_after_inference_seal=True,true_candidate=case['truth']['candidate_name'],
                candidates=records)


def run_one(spec,directory):
    started=perf_counter();folder=Path(directory);name=key(spec)
    if (folder/'results'/f'{name}.json.gz').exists():
        raise FileExistsError('refusing to overwrite '+name)
    try:
        payload,case=observed_case(spec)
        inference=dict(payload,fits={},diagnostics={})
        for index,candidate in enumerate(case['candidates']):
            fitted=fit(candidate,payload['observed'],payload['sigma'],payload['domain'],
                       root_known=payload['root_known'],root_meter=payload['root_meter'],
                       multistart=4,max_iter=200,seed=spec['seed']+index)
            inference['fits'][candidate.name]=fitted
            if 'r' in fitted:
                inference['diagnostics'][candidate.name]=diagnose(candidate,payload['observed'],
                    payload['sigma'],payload['domain'],fitted,
                    root_known=payload['root_known'],root_meter=payload['root_meter'])
        inference_path=folder/'inference'/f'{name}.json.gz'
        write(inference_path,inference)
        # Only now may the evaluation module read physical true parameters/PQ.
        evaluation=evaluate_sealed(case,payload,inference)
        write(folder/'truth'/f'{name}.json.gz',case['truth'])
        record=dict(spec=spec,status='completed',inference_sha256=sha(inference_path),
                    evaluation=evaluation,seconds=perf_counter()-started,
                    qualified=sum(f['qualified'] for f in inference['fits'].values()))
    except Exception as error:
        record=dict(spec=spec,status='failed',error=repr(error),traceback=traceback.format_exc(),
                    seconds=perf_counter()-started,qualified=0)
    write(folder/'results'/f'{name}.json.gz',record)
    return {k:record.get(k) for k in ('spec','status','qualified','seconds','error')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args();directory=Path(args.output)
    if directory.exists():raise FileExistsError('new output directory required')
    directory.mkdir(parents=True)
    config=dict(tasks=specifications(args.pilot),workers=args.workers,pilot=args.pilot,
        independent_random_families=1 if args.pilot else 8,
        pairing='N and U/R/K paired; ordinary only',multistart=4,max_iter=200,
        source_manifest=scientific_sources(),protocol_sha256=sha(Path(__file__).with_name('protocol.md')),
        environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                         platform=platform.platform()))
    write(directory/'configuration.json',config)
    started=perf_counter();completed=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_one,s,str(directory)):s for s in config['tasks']}
        for future in as_completed(jobs):
            result=future.result();completed.append(result);print(json.dumps(result),flush=True)
            write(directory/'progress.json',dict(completed=completed,expected=len(config['tasks']),
                                                elapsed_seconds=perf_counter()-started))
    write(directory/'completed.json',dict(completed=completed,expected=len(config['tasks']),
                                           elapsed_seconds=perf_counter()-started))


if __name__=='__main__':main()
