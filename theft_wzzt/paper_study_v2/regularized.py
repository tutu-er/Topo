"""L1 shrinkage of per-edge R/X toward an estimated common scale.
Same regularization and bounds in H0 and every candidate. Objective contains
voltage, balance, and prior terms; their gain components are reported separately.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix,eye,hstack,vstack
from theft_wzzt.theft.theft_model import predict
from theft_wzzt.theft.credibility import evaluate


def regularized_profile(tree,data,weights,bounds,*,strength=10.):
    start=perf_counter();data.validate(len(tree.observed))
    if strength<0:raise ValueError('Nonnegative regularization required')
    z,c=tree.incidence();r0,x0=weights;e=len(z);n=data.y.size
    w0=np.r_[r0,x0];scale=np.tile(np.maximum(np.maximum(r0,x0),1e-6),2)
    lo,hi=[np.broadcast_to(np.asarray(v,float),(e,)) for v in bounds]
    # [r,x,rho,u_voltage,v_deviation]
    lows=np.r_[lo,lo,.5,np.zeros(n+2*e)]
    highs=np.r_[hi,hi,2.,np.full(n+2*e,np.inf)]
    identity=eye(2*e)
    prior=hstack((identity,csr_matrix(-w0[:,None]),csr_matrix((2*e,n)),-eye(2*e)))
    prior_minus=hstack((-identity,csr_matrix(w0[:,None]),csr_matrix((2*e,n)),-eye(2*e)))
    costs=np.r_[np.zeros(2*e+1),np.ones(n),strength/scale]
    rows=[];best=None;h0=None;best_label=None
    y=data.y.ravel()/data.voltage_scale
    for label in [None,*tree.candidates]:
        active=(data.amplitude>0).astype(float) if label is not None else np.zeros(len(data.p))
        downstream=active[:,None]*c[:,tree.candidates.index(label)] if label is not None else np.zeros((len(data.p),e))
        fp=data.p@z.T+data.amplitude[:,None]*downstream;fq=data.q@z.T+data.extra_q[:,None]*downstream
        ar=(fp[:,None,:]*z.T[None,:,:]).reshape(-1,e);ax=(fq[:,None,:]*z.T[None,:,:]).reshape(-1,e)
        design=csr_matrix(np.hstack((ar,ax))/data.voltage_scale)
        a_ub=vstack((hstack((design,csr_matrix((n,1)),-eye(n),csr_matrix((n,2*e)))),hstack((-design,csr_matrix((n,1)),-eye(n),csr_matrix((n,2*e)))),prior,prior_minus),format='csr')
        b_ub=np.r_[y,-y,np.zeros(4*e)]
        opt=linprog(costs,A_ub=a_ub,b_ub=b_ub,bounds=list(zip(lows,highs)),method='highs')
        if not opt.success:raise RuntimeError(opt.message)
        r,x,rho=opt.x[:e],opt.x[e:2*e],float(opt.x[2*e])
        selection=np.zeros((len(data.p),len(tree.candidates)),int)
        if label is not None:selection[:,tree.candidates.index(label)]=active.astype(int)
        score=evaluate(data,predict(tree,data,r,x,selection),data.amplitude*active)
        penalty=float(strength*np.sum(np.abs(np.r_[r,x]-rho*w0)/scale))
        score['loss']+=penalty;score['penalty']=penalty
        finite=2*e+1
        dual=float(b_ub@opt.ineqlin.marginals+lows@opt.lower.marginals+highs[:finite]@opt.upper.marginals[:finite])
        diagnostics=dict(status=int(opt.status),objective_error=abs(score['voltage_loss']+penalty-opt.fun),scaled_constraint_violation=float(max(0,np.max(a_ub@opt.x-b_ub))),primal_dual_gap=abs(float(opt.fun)-dual))
        if diagnostics['objective_error']>2e-5 or diagnostics['scaled_constraint_violation']>2e-6 or diagnostics['primal_dual_gap']>2e-5:raise RuntimeError(str(diagnostics))
        fit=dict(**score,diagnostics=diagnostics,rho=rho)
        if h0 is None:h0=fit;best=fit
        else:rows.append(dict(location=label,loss=fit['loss'],diagnostics=diagnostics))
        if fit['loss']<best['loss']-1e-8:best=fit;best_label=label
    return dict(method='shrinkage',strength=strength,h0_loss=h0['loss'],h1_loss=best['loss'],gain=h0['loss']-best['loss'],
        gain_voltage=h0['voltage_loss']-best['voltage_loss'],gain_balance=h0['balance_loss']-best['balance_loss'],gain_penalty=h0['penalty']-best['penalty'],
        best_location=best_label,locations=[best_label if a>0 else None for a in data.amplitude],profile=rows,
        h0_rho=h0['rho'],h1_rho=best['rho'],h0_diagnostics=h0['diagnostics'],h1_diagnostics=best['diagnostics'],seconds=perf_counter()-start,
        method_certificate='global minimum over H0 and all fixed-location LPs including the same shrinkage prior')
