"""Compare already-fitted raw/contracted outputs; never fit or re-contract."""
from __future__ import annotations
import numpy as np
from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import contract,measure


def clean_point(point,n):
    supports=[list(s) for s in point['supports']]
    r=np.asarray(point['r'],dtype=float);x=np.asarray(point['x'],dtype=float)
    if r.shape!=(len(supports),) or x.shape!=r.shape:raise ValueError('unaligned point coefficients')
    if not np.isfinite(r).all() or not np.isfinite(x).all() or (r<0).any() or (x<0).any():
        raise ValueError('nonnegative finite fitted coefficients required')
    if len({tuple(s) for s in supports})!=len(supports):raise ValueError('duplicate supports')
    for s in supports:
        if not s or any(not isinstance(i,(int,np.integer)) or i<0 or i>=n for i in s) or len(set(s))!=len(s):
            raise ValueError('invalid support indices')
    return dict(supports=supports,r=r.tolist(),x=x.tolist())


def choose_output(raw_point,training,validation,calibration):
    """Use the original validation rule on two explicitly scored outputs.

    Inputs are normalized observed P/Q/V/root tuples. Truth, AC, LP and model
    generation are absent. Equality ties use the original candidate index.
    """
    n=training[0].shape[1]
    if not validation[0].columns.equals(training[0].columns):raise ValueError('terminal labels mismatch')
    raw=clean_point(raw_point,n)
    contracted=contract(raw,calibration)
    points=[measure(raw,training,validation),measure(contracted,training,validation)]
    for name,p in zip(('raw','contract'),points):p['output_variant']=name
    best=min(p['validation_mae'] for p in points);threshold=1.03*best+1e-12
    eligible=[i for i,p in enumerate(points) if p['validation_mae']<=threshold]
    chosen=min(eligible,key=lambda i:(len(points[i]['supports']),points[i]['validation_mae'],i))
    # Return the chosen point directly. The old select() contracts at this step.
    return dict(status='complete',point=points[chosen],candidate_points=points,chosen_index=chosen,
        eligible_indices=eligible,best_validation=best,eligibility_threshold=threshold,
        rule='3%+1e-12; minimum stored support count, then validation L1, then index; no post-selection contraction',
        fitted=False,AC_called=False,truth_input=False)
