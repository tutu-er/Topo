"""Analytic scalar profiles and a two-slot MILP audit, not an AC selector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def quadratic_profile(power):
    """min_s .5*((p-s)^2+s^2) = p^2/4; optimizer s=p/2."""
    return np.asarray(power, float)**2/4.


def box_profile(power):
    """min_(0<=s<=1) .5*(p-s)^2; derivative is 1-Lipschitz."""
    p=np.asarray(power,float)
    return .5*(p-np.clip(p,0.,1.))**2


def chord_bounds(function, lower, upper, power, curvature_bound):
    """Conditional bounds; the caller supplies a justified curvature constant."""
    p=np.asarray(power,float);a=float(lower);b=float(upper);K=float(curvature_bound)
    if not np.isfinite([a,b,K]).all() or a>=b or K<0 or np.any(p<a) or np.any(p>b):
        raise ValueError("ordered interval, contained powers and nonnegative finite curvature required")
    interpolant=(b-p)/(b-a)*float(function(a))+(p-a)/(b-a)*float(function(b))
    epsilon=K*(b-a)**2/8.
    return interpolant-epsilon,interpolant+epsilon,interpolant,epsilon


def two_slot_milp(knots, *, side):
    """Audit max d(p1)+d(p2), p1+p2=2, 0<=p<=2, with adjacent weights.

    Ground truth is 1 at (0,2)/(2,0). Its proof is convexity on a line segment.
    The MILP is only a tiny bound-direction test, not a network EV policy.
    """
    grid=np.asarray(knots,float)
    if grid.ndim!=1 or len(grid)<2 or grid[0]!=0 or grid[-1]!=2 or np.any(np.diff(grid)<=0):
        raise ValueError("strictly ordered grid from 0 to 2 required")
    if side not in ("lower","upper"):raise ValueError("side must be lower or upper")
    L,S=len(grid),len(grid)-1;block=L+S;count=2*block
    lower=np.zeros(count);upper=np.ones(count);integrality=np.zeros(count,int)
    coefficients=np.zeros(count);rows=[];lo=[];hi=[]
    epsilon=.5*np.diff(grid)**2/8.  # exact d''=.5
    values=quadratic_profile(grid)
    for time in range(2):
        offset=time*block;integrality[offset+L:offset+block]=1
        coefficients[offset:offset+L]=-values
        coefficients[offset+L:offset+block]=epsilon if side=="lower" else -epsilon
        row=np.zeros(count);row[offset:offset+L]=1.;rows.append(row);lo.append(1.);hi.append(1.)
        row=np.zeros(count);row[offset+L:offset+block]=1.;rows.append(row);lo.append(1.);hi.append(1.)
        for knot in range(L):
            row=np.zeros(count);row[offset+knot]=1.
            if knot>0:row[offset+L+knot-1]-=1.
            if knot<S:row[offset+L+knot]-=1.
            rows.append(row);lo.append(-np.inf);hi.append(0.)
    row=np.zeros(count)
    for time in range(2):row[time*block:time*block+L]=grid
    rows.append(row);lo.append(2.);hi.append(2.)
    result=milp(coefficients,integrality=integrality,bounds=Bounds(lower,upper),
        constraints=LinearConstraint(np.asarray(rows),np.asarray(lo),np.asarray(hi)),
        options=dict(time_limit=5.,mip_rel_gap=1e-10))
    if result.x is None:raise RuntimeError("toy MILP returned no feasible incumbent: "+str(result.message))
    powers=np.array([grid@result.x[time*block:time*block+L] for time in range(2)])
    return dict(side=side,status=int(result.status),success=bool(result.success),
        powers=powers.tolist(),lambda_weights=[result.x[time*block:time*block+L].tolist() for time in range(2)],
        segment_binaries=[result.x[time*block+L:(time+1)*block].tolist() for time in range(2)],
        model_incumbent=float(-result.fun),maximization_model_best_bound=float(-result.mip_dual_bound),
        exact_physical_score=float(quadratic_profile(powers).sum()),analytic_maximum=1.,constant_score=.5,
        sum_interval_error=float(sum(epsilon@result.x[time*block+L:(time+1)*block] for time in range(2))),
        task="two analytic scalar profiles, two powers, one energy equality; no AC or routing",
        bound_scope="lower-model incumbent lower bound / upper-model global bound upper bound on this analytic toy")


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();cases=[]
    for label,grid in (("coarse",[0.,1.,2.]),("fine",[0.,.5,1.,1.5,2.])):
        lower=two_slot_milp(grid,side="lower");upper=two_slot_milp(grid,side="upper")
        cases.append(dict(grid=label,knots=grid,lower=lower,upper=upper,
            certified_toy_gap=upper["maximization_model_best_bound"]-lower["model_incumbent"]))
    result=dict(experiment="analytic-only scalar interpolation bound audit",cases=cases,
        AC_calls=0,EV_selector_calls=0,network_data_read=False,actual_AC_MILP_implemented=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print("analytic cases="+str(len(cases))+" AC_calls=0 actual_AC_MILP_implemented=False")


if __name__=="__main__":main()
