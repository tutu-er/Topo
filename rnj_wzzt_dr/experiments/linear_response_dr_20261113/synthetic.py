"""A deliberately informative analytic linear-DR test, separate from real fits."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from time import perf_counter
import argparse
import numpy as np
from scipy.special import expit
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,sha,utc
from ..expected_logodds_admm_20261113.admm import run as admm
from ..expected_logodds_admm_20261113.evidence import Evidence


class ShapeDevice:
    def __init__(self,profile,maximum):
        self.profile=np.asarray(profile);self.maximum=maximum;self.requests=np.array([0.])
    def solve(self,quadratic,linear,rho,deadline):
        norm=float(self.profile@self.profile)
        value=float(np.clip(rho*(linear@self.profile)/(2/self.maximum**2+rho*quadratic*norm),0,self.maximum))
        self.requests[:]=value
        return value*self.profile,dict(optimal=True,no_incumbent=False)


def make():
    # H_t=I (two star branches); H_j=.98 I+.01 11^T has a positive shared
    # branch and equal row sums. Under balanced natural loads their responses
    # are identical. No real fitted number or production candidate is changed.
    delta=.01;H=np.array([np.eye(2),.98*np.eye(2)+.01*np.ones((2,2))])
    T=96;profile=np.sin(2*np.pi*(np.arange(T)+.5)/T)
    sigma=.003;effect=np.array([[[2*delta],[-2*delta]]])/np.sqrt(2*sigma*sigma)
    coefficient=float(np.sum((effect[0,:,0,None]*profile[None])**2))
    return H,profile,sigma,effect,coefficient


def run(out):
    require(not out.exists(),'Keep synthetic experiments separate');out.mkdir(parents=True)
    H,profile,sigma,effect,k=make();levels=[float(np.log(2)),2.,4.,8.]
    io.write(out/'protocol.json',dict(created_utc=utc(),source=io.ref(__file__),synthetic=True,
        construction='two valid positive branch response matrices, equal row sums, .01 shared-branch alternative, balanced background P=[1,1]',
        modification_of_real_fit=False,measurement_sigma=sigma,amplitude_bound=.1,
        horizon=96,gamma_levels=levels,fee='amplitude squared, synthetic contract unit',
        history='equal likelihood under balanced inputs; uniform prior remains 1/2',
        model_paths=65536,seed=2026111304,ADMM_seconds_per_level=2.,iterations=800,
        known_global_optimum='gain=k*a^2, a>=0, fee=a^2, so minimum fee=gamma/k',
        selection='All four preset levels retained. No AC or real-case claim.'))
    plans=[];policies=[]
    for gamma in levels:
        e=Evidence(np.zeros((1,96,2)),effect,np.ones(1),gamma,buffer=0.)
        device=ShapeDevice(profile,.1)
        value=admm(e,[device],np.zeros((1,96)),lambda u:float((u[0]@profile/(profile@profile)/.1)**2),
            seconds=2.,iterations=800,rho=10.)
        best=value.pop('best');trace=value.pop('trace');a=None if best is None else float(best['requests'][0][0])
        if a is not None:
            direct=float(np.sum(((np.column_stack((a*profile,-a*profile)))@(H[0]-H[1]).T)**2)/(2*sigma*sigma))
            require(direct>=gamma-1e-8,'Synthetic retained policy violates gain')
            require(a*a>=gamma/k-1e-10,'Synthetic retained fee undercuts valid analytic bound')
            policies.append((gamma,a,direct))
        plans.append(dict(gamma=gamma,amplitude=a,analytic_minimum_fee=gamma/k,retained_fee=None if a is None else a*a,
            attained_gain=None if a is None else direct,status=value['status'],iterations=value['iterations'],
            fee_error=None if a is None else a*a-gamma/k,ADMM_global_proof=False,
            independent_analytic_global_optimum_verified=bool(a is not None and abs(a*a-gamma/k)<1e-6)))
        io.write(out/f'trace_{gamma:.6g}.json',trace)
    io.write(out/'plans.json',plans);io.write(out/'policies_frozen.json',dict(created_utc=utc(),plans_sha256=sha(out/'plans.json')))
    rng=np.random.default_rng(2026111304);noise=rng.normal(size=(65536,96,2));rows=[]
    # All policies frozen before this common independent error sample.
    for gamma,a,gain in [(0.,0.,0.)]+policies:
        power=np.tile([1.,1.],(96,1))+np.column_stack((a*profile,-a*profile))
        difference=power@(H[0]-H[1]).T
        L=gain+np.einsum('stn,tn->s',noise,difference)/sigma
        posterior=expit(L)
        rows.append(dict(gamma=gamma,analytic_gain=gain,mean_gain=float(L.mean()),gain_standard_error=float(L.std(ddof=1)/np.sqrt(len(L))),
            mean_posterior=float(posterior.mean()),pass90=float(np.mean(posterior>=.9)),fee=a*a,
            minimum_cost_analytic=gamma/k,paths=len(L)))
    io.write(out/'results.json',rows);io.write(out/'complete.json',dict(status='passed',levels=len(levels),all_qualified=len(policies)==4,new_AC_paths=0))
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);run(p.parse_args().output.resolve())
