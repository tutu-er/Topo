"""Identical strict/paired-error-band selection for baseline and exit libraries."""
from __future__ import annotations
import numpy as np
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.rnj_count.engine import count


def library(bank,points,validation,*,resolution=.16):
    rows=[];cal=bank.calibration
    for index,point in enumerate(points):
        family=[s for s,r,x in zip(point['supports'],point['r'],point['x']) if len(s)==1 or
            cal['a']*r+cal['b']*x+1e-10>=resolution*cal['median_root_depth']]
        fitted=bank.fit(family)
        R,X=core.build_matrices_from_atoms(bank.data.n,fitted['supports'],fitted['r'],fitted['x'])
        per_time=np.concatenate([np.mean(abs(y-p@R.T-q@X.T),axis=1) for p,q,y in
            zip(validation.p,validation.q,validation.target)])
        # Archives are three independent scenarios of 96 validation times.
        if len(per_time)%12:raise ValueError('validation times must form complete 12-time blocks')
        blocks=per_time.reshape(-1,12).mean(axis=1)
        rows.append(dict(point=fitted,index=index,validation=float(per_time.mean()),blocks=blocks.tolist(),internal=count(fitted,bank.data.n)))
    return rows


def choose(rows,rule):
    best=min(rows,key=lambda r:(r['validation'],r['internal'],r['index']))
    if rule=='strict':
        eligible=[r for r in rows if r['validation']<=best['validation']+1e-12]
    elif rule=='paired':
        eligible=[]
        for r in rows:
            d=np.asarray(r['blocks'])-np.asarray(best['blocks'])
            se=float(np.std(d,ddof=1)/np.sqrt(len(d))) if len(d)>1 else 0.
            if r['validation']-best['validation']<=max(.001*best['validation'],se)+1e-12:
                eligible.append(r)
    else:raise ValueError(rule)
    selected=min(eligible,key=lambda r:(r['internal'],r['validation'],r['index']))
    return {**selected,'rule':rule,'eligible_count':len(eligible),'best_validation':best['validation']}
