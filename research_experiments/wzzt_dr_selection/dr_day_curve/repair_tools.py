"""Numerical energy-equality restoration, with explicit small-change limits.

This never clips a stochastic response or changes its law. A polished curve
must still pass the original resource, target and settlement replay.
"""
from __future__ import annotations
import numpy as np
from .day_model import canonical_prices,resource_audit,scenario_summary


def polish_curve(problem,scenarios,result,max_correction_kw=1.e-7):
    a=np.asarray(result['mean_u'],float);b=a.copy()
    lam=np.asarray(problem['elasticities']);d=np.asarray(problem['factors'])
    eta=1+np.abs(d).sum(axis=1)
    lo=np.maximum(np.asarray(problem['power_lower'])/eta[:,None],-lam[:,None]*np.asarray(problem['price_caps'])[:,None])
    hi=np.minimum(np.asarray(problem['power_upper'])/eta[:,None],lam[:,None]*np.asarray(problem['price_caps'])[:,None])
    records=[]
    for j in range(len(a)):
        residual=float(b[j].sum());available=np.asarray(problem['available'][j],bool)
        if abs(residual)>max_correction_kw:raise ValueError('raw daily energy residual is too large for numerical restoration')
        if residual!=0.:
            room=np.where(available,b[j]-lo[j] if residual>0 else hi[j]-b[j],-np.inf)
            k=int(np.argmax(room))
            if room[k]<abs(residual)-1.e-12:raise ValueError('no available power headroom for equality restoration')
            b[j,k]-=residual
        records.append(dict(resource=j,raw_power_sum_kw=residual,polished_power_sum_kw=float(b[j].sum())))
    maximum=float(np.max(np.abs(b-a),initial=0.))
    if maximum>max_correction_kw:raise ValueError('numerical curve change exceeds predeclared limit')
    audit=resource_audit(problem,b);summary=scenario_summary(problem,scenarios,b)
    if not audit['feasible']:raise ValueError('equality-restored curve fails original resource/fee check')
    before=scenario_summary(problem,scenarios,a)
    result=dict(result);result.update(mean_u=b.tolist(),prices=canonical_prices(b,lam).tolist(),
        training=summary,resource_audit=audit,
        numerical_polish=dict(maximum_change_kw=maximum,limit_kw=max_correction_kw,rows=records,
            raw_training_passed=before['passed'],polished_training_passed=summary['passed'],
            raw_expected_payment=resource_audit(problem,a)['expected_payment'],
            polished_expected_payment=audit['expected_payment'],
            scope='restore floating-point zero-energy equality only; recheck original constraints and exact target'))
    if result['status'] in ('target_feasible_cost_incumbent','zero_cost_optimal') and summary['passed']<summary['required']:
        result['status']='target_numerically_unresolved_after_polish'
    return result
