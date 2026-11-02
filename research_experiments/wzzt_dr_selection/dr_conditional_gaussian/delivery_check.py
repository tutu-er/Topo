"""One pre-frozen amplitude derating diagnostic, not a new excitation optimizer.

Along the existing returned direction, choose maximal scale with 95% all-day
component-bound probability under independent execution multipliers. Then reuse
the original frozen scorer and a new independent random stream. It does not
claim the separate 95% topology-candidate exclusion requirement is met.
"""
from __future__ import annotations
import argparse, math, shutil, zipfile
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
import runner


def log_delivery_probability(u, cv, lower, upper):
    u,cv,lower,upper=np.broadcast_arrays(np.asarray(u,float),cv,lower,upper)
    sd=cv*np.abs(u);active=sd>0
    if np.any((~active)&((u<lower)|(u>upper))):return -math.inf
    if not np.any(active):return 0.
    low=(lower[active]-u[active])/sd[active]
    high=(upper[active]-u[active])/sd[active]
    probability=ndtr(high)-ndtr(low)
    if np.any(probability<=0):return -math.inf
    return float(np.log(probability).sum())


def freeze(parent, output):
    parent=Path(parent);output=Path(output)
    if output.exists():raise RuntimeError('delivery diagnostic already exists; do not overwrite')
    protocol=runner.read(parent/'protocol.json')
    problem_ref=runner.read(parent/'problem_frozen.json')['input']
    p=runner.load_pickle(runner.checked(problem_ref))
    old=runner.read(parent/'price_freeze.json')
    if not (parent/'ended.json').exists():raise RuntimeError('bounded main pilot not ended')
    plan=np.asarray(old['policies']['continuous']['mean_u'])
    f=lambda scale: log_delivery_probability(scale*plan,p['cv'],p['lower'],p['upper'])-math.log(.95)
    alpha=1. if f(1.)>=0 else float(brentq(f,0.,1.,xtol=1e-13))
    while f(alpha)<0:alpha=float(np.nextafter(alpha,0.))
    u=alpha*plan;amplitude=float(np.sqrt(np.mean(u*u)))
    pattern=np.where((np.arange(96)//24)%2==0,1.,-1.)
    single=np.zeros(p['shape']);single[0]=np.sqrt(2)*amplitude
    plans={'zero':np.zeros(p['shape']),'continuous':u,
        'rotation':amplitude*np.stack([pattern,-pattern]),'single':single}
    policies={}
    for name,a in plans.items():
        if np.any(a<p['lower']-1e-12) or np.any(a>p['upper']+1e-12):raise ValueError('mean-infeasible comparison '+name)
        physical=math.exp(log_delivery_probability(a,p['cv'],p['lower'],p['upper']))
        policies[name]=dict(mean_u=a.tolist(),planned_cost=float(p['dt']*np.sum(a*a/p['elasticities'])),
            role=('natural no reward' if name=='zero' else 'amplitude diagnostic, not reoptimized or excitation-qualified'),
            analytic_component_bounds_probability=physical,delivery95_feasible=bool(physical>=.95-1e-12))
    output.mkdir(parents=True,exist_ok=False)
    new=dict(protocol)
    new.update(id='conditional_gaussian_delivery_diagnostic_v1',created_utc=runner.now(),
        parent_protocol=runner.ref(parent/'protocol.json'),parent_price=runner.ref(parent/'price_freeze.json'),
        parent_ended=runner.ref(parent/'ended.json'),evaluation_seed=2026110204,
        source_sha256={str(s.resolve()):runner.sha(s) for s in runner.source_files()},
        optimizer=dict(method='analytic maximal radial derating; no new price optimization',alpha=alpha,component_probability=.95),
        policy_selection='all four prelocked policies reported, no choice on new evaluation',
        qualification_samples=0,main_excitation_requirement_unchanged=.95,
        interpretation='follow-up execution-capability diagnostic; component safety95 is separate from excitation95')
    runner.write(output/'protocol.json',new)
    shutil.copyfile(parent/'problem.pkl',output/'problem.pkl')
    runner.write(output/'problem_frozen.json',dict(status='frozen',input=runner.ref(output/'problem.pkl'),parent_problem=problem_ref))
    with zipfile.ZipFile(output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for i,s in enumerate(runner.source_files()):z.write(s,f'{i:02d}_{s.name}')
    runner.write(output/'planning_results.json',dict(status='analytic_derating',alpha=alpha,physical_target=.95,
        new_fit_calls=0,new_optimization_calls=0,excitation_target_not_certified=True,parent_price=runner.ref(parent/'price_freeze.json')))
    runner.write(output/'price_freeze.json',dict(status='frozen',created_utc=runner.now(),policies=policies,
        numerically_qualified=False,probability_target=.95,independent_evaluation_stream_not_opened=True,
        planning_results=runner.ref(output/'planning_results.json'),phase='execution amplitude diagnostic'))
    runner.write(output/'run_lock.json',dict(pid=None,status='prices_frozen',updated_utc=runner.now()))
    print('delivery plan frozen',alpha,{k:(v['planned_cost'],v['analytic_component_bounds_probability']) for k,v in policies.items()},flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--parent',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();freeze(args.parent,args.output)
