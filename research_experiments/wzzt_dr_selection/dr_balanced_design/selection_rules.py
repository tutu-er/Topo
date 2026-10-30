"""Observation-score-only diagnostic selectors; no truth/fit/simulation input."""
from copy import deepcopy
import numpy as np

RULES=('original_3pct_contract','original_3pct_raw','strict_raw','strict_raw_or_contract')

def matrices(point,n=3):
    R=np.zeros((n,n));X=np.zeros((n,n))
    for s,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
        ix=np.ix_(s,s);R[ix]+=float(r);X[ix]+=float(x)
    return R,X

def score(point,observations):
    P,Q,Y=observations;R,X=matrices(point,P.shape[1])
    return float(np.mean(np.abs(P@R.T+Q@X.T-Y)))

def contract(point,cal):
    threshold=.16*float(cal['median_root_depth'])
    keep=[i for i,(s,r,x) in enumerate(zip(point['supports'],point['r'],point['x'],strict=True))
          if len(s)==1 or float(cal['a'])*r+float(cal['b'])*x+1e-10>=threshold]
    out=deepcopy(point)
    for field in ('supports','r','x'):out[field]=[point[field][i] for i in keep]
    return out

def effective(point):
    return {tuple(s) for s,r,x in zip(point['supports'],point['r'],point['x'],strict=True) if max(r,x)>1e-9}

def choose(scored_points,rule):
    """Each entry supplies raw/contract points and freshly computed val losses.

    Strict ties are the predeclared absolute 1e-12 numerical band. They prefer
    declared support count, original point index, then raw before contraction.
    No true structure or structural error can be passed to this function.
    """
    if rule not in RULES:raise ValueError('unknown selector')
    if not scored_points:raise ValueError('no saved points')
    raw=[dict(point=v['raw'],validation_mae=v['raw_val'],source_index=i,variant='raw') for i,v in enumerate(scored_points)]
    if rule in ('original_3pct_contract','original_3pct_raw'):
        best=min(v['validation_mae'] for v in raw)
        eligible=[v for v in raw if v['validation_mae']<=1.03*best+1e-12]
        selected=min(eligible,key=lambda v:(len(v['point']['supports']),v['validation_mae'],v['source_index']))
        i=selected['source_index']
        if rule=='original_3pct_contract':
            return dict(point=scored_points[i]['contract'],validation_mae=scored_points[i]['contract_val'],
                        source_index=i,variant='contract',selection_validation_mae=selected['validation_mae'],
                        eligible_count=len(eligible),numerical_tie=1e-12)
        return dict(selected,selection_validation_mae=selected['validation_mae'],eligible_count=len(eligible),numerical_tie=1e-12)
    pool=raw
    if rule=='strict_raw_or_contract':
        pool=raw+[dict(point=v['contract'],validation_mae=v['contract_val'],source_index=i,variant='contract') for i,v in enumerate(scored_points)]
    best=min(v['validation_mae'] for v in pool)
    eligible=[v for v in pool if v['validation_mae']<=best+1e-12]
    selected=min(eligible,key=lambda v:(len(v['point']['supports']),v['source_index'],int(v['variant']=='contract')))
    return dict(selected,selection_validation_mae=selected['validation_mae'],eligible_count=len(eligible),numerical_tie=1e-12)
