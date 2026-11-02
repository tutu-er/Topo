"""Existing L1 conditional integration composed with causal resource delivery."""
from __future__ import annotations
import sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
sys.path.insert(0,str(REPO/'research_experiments/wzzt_dr_selection/dr_conditional_gaussian'))
from conditional_integral import conditional_probability
from problem import integration_direction


class Probability:
    def __init__(self,packet,devices,samples,base_seed,response_seed):
        self.packet=packet;self.devices=devices;self.samples=samples
        self.multiplier=devices.draw(response_seed,samples)
        direction,comp=integration_direction(packet['mu_C'],packet['mu_F'],packet['L_C'],packet['L_F'],1.03)
        rng=np.random.default_rng(base_seed);z=rng.normal(size=(samples,comp.shape[1]))@comp.T
        self.a0=np.concatenate([packet['mu_C'][None]+z@packet['L_C'].T,packet['mu_F'][None]+z@packet['L_F'].T],axis=1)
        self.b=np.r_[packet['L_C']@direction,packet['L_F']@direction]
        self.m=len(packet['mu_C']);self.w=np.r_[np.ones(self.m),-1.03*np.ones(self.m)]/self.m
        self.last_x=None;self.evaluations=0

    def evaluate(self,x):
        x=np.asarray(x).reshape(self.devices.R,24)
        if self.last_x is not None and np.array_equal(x,self.last_x):return self.last
        d=self.devices;u,nom_cache=d.nominal(x,cache=True)
        delivered,exec_cache=d.project(u[None]*self.multiplier,cache=True)
        ec=np.einsum('nr,srt->stn',self.packet['effect_C'],delivered).reshape(self.samples,-1)
        ef=np.einsum('nr,srt->stn',self.packet['effect_F'],delivered).reshape(self.samples,-1)
        intercepts=self.a0+np.concatenate([ec,ef],axis=1)
        vals=[];gdel=[];invalid=[]
        for i in range(self.samples):
            out=conditional_probability(intercepts[i],self.b,self.w,self.packet['threshold'])
            vals.append(out['probability'])
            if not out['gradient_valid']:
                invalid.append(i);gdel.append(np.zeros_like(u));continue
            g=out['coeff_grad_a']
            gc=g[:self.m].reshape(96,-1)@self.packet['effect_C']
            gf=g[self.m:].reshape(96,-1)@self.packet['effect_F']
            gdel.append((gc+gf).T)
        gnom=np.mean(d.vjp(np.asarray(gdel),exec_cache)*self.multiplier,axis=0)
        gx=d.hourly_vjp(gnom,nom_cache)
        fee,fee_grad=d.fees(u,gradient=True)
        self.last=dict(probability=float(np.mean(vals)),gradient=gx.ravel(),invalid=invalid,cost=fee,
            cost_gradient=d.hourly_vjp(fee_grad,nom_cache).ravel(),mean_u=u,
            outer_standard_error=float(np.std(vals,ddof=1)/np.sqrt(len(vals))) if len(vals)>1 else 0.)
        self.last_x=x.copy();self.evaluations+=1
        return self.last


def solve(problem,seed,maxiter=24,seconds=180):
    """Two bounded SLSQP starts; exact integrated probability is the constraint.

    Returned status is local; failure to reach target does not prove infeasibility.
    """
    R=problem.devices.R;shape=(R,24);start_time=time.perf_counter()
    bounds=[(-1.,0. if problem.devices.kind[r]==1 else 1.) for r in range(R) for _ in range(24)]
    zero=np.zeros(shape);rz=problem.evaluate(zero)
    rng=np.random.default_rng(seed)
    # Smooth small nonzero start avoids a controller kink at an all-zero path.
    start=np.clip(.03*rng.normal(size=shape),-1,1)
    start[problem.devices.kind==1]=-np.abs(start[problem.devices.kind==1])
    records=[];traces=[]
    def keep(x):
        r=problem.evaluate(x)
        records.append(dict(x=np.asarray(x).reshape(shape).copy(),probability=r['probability'],cost=r['cost'],mean_u=r['mean_u'].copy(),invalid=r['invalid']))
        return r
    keep(zero)
    class WallLimit(Exception):pass
    for label,x0 in [('small_random',start),('negative_small_random',-start)]:
        x0=x0.copy();x0[problem.devices.kind==1]=-np.abs(x0[problem.devices.kind==1])
        begun=time.perf_counter();trace=[]
        def evaluation(x):
            if time.perf_counter()-begun>seconds:raise WallLimit()
            r=problem.evaluate(x)
            if r['invalid']:raise ArithmeticError('conditional derivative unresolved')
            return r
        def callback(x):
            r=keep(x);trace.append(dict(probability=r['probability'],cost=r['cost'],elapsed=time.perf_counter()-begun))
        try:
            result=minimize(lambda x:evaluation(x)['cost'],x0.ravel(),jac=lambda x:evaluation(x)['cost_gradient'],
                method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=lambda x:evaluation(x)['probability']-.95,
                    jac=lambda x:evaluation(x)['gradient'])],callback=callback,options=dict(maxiter=maxiter,ftol=1e-7,disp=False))
            keep(result.x);status=str(result.message);success=bool(result.success)
        except WallLimit:status='time_limit';success=False
        except ArithmeticError as e:status=str(e);success=False
        traces.append(dict(start=label,status=status,solver_success=success,trace=trace,elapsed=time.perf_counter()-begun))
    eligible=[r for r in records if r['probability']>=.95-1e-7 and not r['invalid']]
    best=min(eligible,key=lambda r:r['cost']) if eligible else max(records,key=lambda r:(r['probability'],-r['cost']))
    return dict(best=best,traces=traces,qualified=bool(eligible),evaluations=problem.evaluations,
        elapsed=time.perf_counter()-start_time,global_optimum=False,zero_probability=rz['probability'])
