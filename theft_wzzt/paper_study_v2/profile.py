"""Finite LP profiles with one or two global impedance multipliers.
This explicitly restricted drift family cannot represent arbitrary edge changes.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,eye,hstack,vstack
from theft_wzzt.theft.theft_model import predict
from theft_wzzt.theft.credibility import evaluate


def fit_profile(tree,data,weights,*,two_scales=True):
    started=perf_counter();data.validate(len(tree.observed))
    z,c=tree.incidence();r0,x0=weights
    e=len(z);n=data.y.size;active=(data.amplitude>0).astype(float)
    y=data.y.ravel()/data.voltage_scale
    number=2 if two_scales else 1
    lows=np.r_[np.full(number,.5),np.zeros(n)]
    highs=np.r_[np.full(number,2.),np.full(n,np.inf)]
    rows=[];best=None;h0=None;best_label=None
    for label in [None,*tree.candidates]:
        if label is None:downstream=np.zeros((len(data.p),e));u=np.zeros(len(data.p))
        else:downstream=active[:,None]*c[:,tree.candidates.index(label)];u=active
        fp=data.p@z.T+data.amplitude[:,None]*downstream
        fq=data.q@z.T+data.extra_q[:,None]*downstream
        ppart=(fp*r0)@z;qpart=(fq*x0)@z
        design=np.stack([ppart.ravel(),qpart.ravel()],axis=1) if two_scales else (ppart+qpart).reshape(-1,1)
        design=csr_matrix(design/data.voltage_scale)
        a_ub=vstack((hstack((design,-eye(n))),hstack((-design,-eye(n)))),format='csr')
        b_ub=np.r_[y,-y];costs=np.r_[np.zeros(number),np.ones(n)]
        opt=linprog(costs,A_ub=a_ub,b_ub=b_ub,bounds=list(zip(lows,highs)),method='highs')
        if not opt.success:raise RuntimeError(opt.message)
        factors=opt.x[:number];r=r0*factors[0];x=x0*factors[-1]
        selection=np.zeros((len(data.p),len(tree.candidates)),int)
        if label is not None:selection[:,tree.candidates.index(label)]=active.astype(int)
        score=evaluate(data,predict(tree,data,r,x,selection),data.amplitude*u)
        dual=float(b_ub@opt.ineqlin.marginals+lows@opt.lower.marginals+highs[:number]@opt.upper.marginals[:number])
        diagnostics=dict(status=int(opt.status),objective_error=abs(score['voltage_loss']-opt.fun),
            scaled_constraint_violation=float(max(0,np.max(a_ub@opt.x-b_ub))),primal_dual_gap=abs(float(opt.fun)-dual))
        if max(diagnostics['objective_error'],diagnostics['primal_dual_gap'])>2e-5 or diagnostics['scaled_constraint_violation']>2e-6:raise RuntimeError(str(diagnostics))
        fit=dict(**score,diagnostics=diagnostics,factors=factors.tolist())
        if h0 is None:h0=fit;best=fit
        else:rows.append(dict(location=label,loss=fit['loss'],diagnostics=diagnostics))
        if fit['loss']<best['loss']-1e-8:best=fit;best_label=label
    return dict(method='global_rx2' if two_scales else 'global_rx1',h0_loss=h0['loss'],h1_loss=best['loss'],
        gain=h0['loss']-best['loss'],gain_voltage=h0['voltage_loss']-best['voltage_loss'],gain_balance=h0['balance_loss']-best['balance_loss'],
        best_location=best_label,locations=[best_label if a>0 else None for a in data.amplitude],profile=rows,
        h0_factors=h0['factors'],h1_factors=best['factors'],h0_diagnostics=h0['diagnostics'],h1_diagnostics=best['diagnostics'],
        seconds=perf_counter()-started,method_certificate='global optimum over H0 and all fixed-location LPs within the declared 1/2-scale impedance family')
