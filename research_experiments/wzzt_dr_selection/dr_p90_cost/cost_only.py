"""One bounded cost stage using the existing controlled probability oracle.

No identifier, response law or conditional integration is reimplemented here.
"""
import time
import numpy as np
from scipy.optimize import minimize

def solve(prob,starts,target=.90,maxiter=40,seconds=180):
    if not 0<target<1:raise ValueError('target must be in (0,1)')
    d=prob.devices;shape=(d.R,24);records=[];started=time.perf_counter()
    def keep(x,label):
        q=prob.evaluate(x)
        row=dict(x=np.asarray(x).reshape(shape).copy(),mean_u=q['mean_u'].copy(),
            probability=q['probability'],cost=q['cost'],invalid=list(q['invalid']),label=label)
        records.append(row);return row
    for name,x in starts:keep(x,name)
    feasible=lambda:[r for r in records if r['probability']>=target-1e-7 and not r['invalid']]
    eligible=feasible();trace=[];status='no_feasible_start';success=False
    if eligible:
        initial=min(eligible,key=lambda r:r['cost'])['x'].ravel()
        inds=np.flatnonzero((d.scale.reshape(d.R,24,4).max(-1)>0).ravel())
        bounds=[(-1.,0. if d.kind[j//24]==1 else 1.) for j in inds]
        def expand(v):
            x=initial.copy();x[inds]=v;return x
        class WallLimit(Exception):pass
        def ev(v):
            if time.perf_counter()-started>seconds:raise WallLimit()
            q=prob.evaluate(expand(v))
            if q['invalid']:raise ArithmeticError('conditional derivative unresolved')
            return q
        def callback(v):
            r=keep(expand(v),'iterate');trace.append(dict(probability=r['probability'],cost=r['cost']))
        try:
            fit=minimize(lambda v:ev(v)['cost'],initial[inds],jac=lambda v:ev(v)['cost_gradient'][inds],
                bounds=bounds,method='SLSQP',constraints=[dict(type='ineq',
                fun=lambda v:ev(v)['probability']-target,jac=lambda v:ev(v)['gradient'][inds])],
                callback=callback,options=dict(maxiter=maxiter,ftol=1e-8))
            callback(fit.x);status=str(fit.message);success=bool(fit.success)
        except WallLimit:status='time_limit'
        except (ArithmeticError,FloatingPointError) as error:status=repr(error)
    eligible=feasible()
    best=min(eligible,key=lambda r:r['cost']) if eligible else max(records,key=lambda r:(r['probability'],-r['cost']))
    return dict(best=best,qualified=bool(eligible),target=target,status=status,solver_success=success,
        global_optimal=False,infeasibility_proven=False,records=records,trace=trace,
        seconds=time.perf_counter()-started,probability_evaluations=prob.evaluations)
