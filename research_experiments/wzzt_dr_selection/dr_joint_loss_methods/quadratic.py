"""Second Hermite order + deterministic characteristic-function inversion.

The quadratic is an L2 projection, not an exact rewrite of paired L1.
A separate remainder variance accounts for the approximation in a sufficient bound.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad,IntegrationWarning
from scipy.special import ndtr
from scipy.stats import ncx2,norm
import warnings
from kernel import abs_mean

def quadratic_projection(mu,cov,w):
    sd=np.sqrt(np.maximum(np.diag(cov),0));a=np.divide(mu,sd,out=np.zeros_like(mu),where=sd>0)
    mean=float(w@abs_mean(mu,sd))
    ev,U=eigh(cov,check_finite=False);keep=ev>max(1e-10,ev.max()*1e-12)
    B=U[:,keep]*np.sqrt(ev[keep])
    t=np.divide(w*np.exp(-a*a/2)/np.sqrt(2*np.pi),sd,out=np.zeros_like(sd),where=sd>0)
    linear=w*(2*ndtr(a)-1)
    A=B.T@(t[:,None]*B);b=B.T@linear
    lam,V=eigh(A,check_finite=False)
    beta=V.T@b
    var1=float(linear@(cov@linear));var2=float(2*t@((cov*cov)@t))
    return dict(mean=mean,constant=mean-float(lam.sum()),lam=lam,beta=beta,
        variance_full=var1+var2,variance_factored=float(2*(lam@lam)+beta@beta),
        dropped_covariance_trace=float(np.maximum(ev[~keep],0).sum()))

def quadratic_sf(lam,beta,constant,h,tol=1e-7):
    lam=np.asarray(lam,float);beta=np.asarray(beta,float)
    v=float(2*lam@lam+beta@beta)
    if v<1e-24:return dict(p=float(constant+lam.sum()>h),error=0.,status='deterministic')
    scale=np.sqrt(v);la=lam/scale;be=beta/scale;x=(h-constant)/scale
    nonzero=np.abs(la)>1e-13
    if not nonzero.any():return dict(p=float(norm.sf(x/np.linalg.norm(be))),error=0.,status='normal')
    if np.all(nonzero) and np.max(la)-np.min(la)<1e-12:
        val=la[0];shift=np.sum(be*be/(4*val));cut=(x+shift)/val
        p=ncx2.sf(cut,len(la),float(np.sum((be/(2*val))**2))) if val>0 else ncx2.cdf(cut,len(la),float(np.sum((be/(2*val))**2)))
        return dict(p=float(p),error=0.,status='noncentral_chi2')
    sorted_l=np.sort(np.abs(la[nonzero]))[::-1];k=np.arange(1,len(sorted_l)+1)
    base=-.5*np.cumsum(np.log(2*sorted_l))+np.log(2/k)-np.log(np.pi)
    def tail(T):
        linear=-.5*np.sum(be*be*T*T/(1+4*la*la*T*T))
        return float(np.exp(np.min(base-k*np.log(T)/2)+linear))
    T=2.
    while tail(T)>tol/2 and T<4096:T*=2
    bound=tail(T)
    def integrand(t):
        if t==0:return float(la.sum()-x)
        den=1-2j*t*la
        logcf=-.5*np.log(den).sum()-.5*t*t*np.sum(be*be/den)-1j*t*x
        return float(np.imag(np.exp(logcf))/t)
    edges=np.r_[0.,2.**np.arange(0,int(np.log2(T))+1)]
    total=0.;error=0.;messages=[]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always',IntegrationWarning)
        for left,right in zip(edges[:-1],edges[1:]):
            value,err=quad(integrand,left,right,epsabs=tol/(4*len(edges)),epsrel=tol/4,limit=300)
            total+=value;error+=err
        messages=[str(v.message) for v in caught]
    p=.5+total/np.pi
    return dict(p=float(np.clip(p,0,1)),error=float(error/np.pi+bound),tail_bound=bound,
        interval_upper=T,status='converged' if not messages and bound<=tol/2 and error/np.pi<tol else 'numerically_unresolved',
        messages=messages)

def evaluate(mu,cov,w,h,var_upper):
    q=quadratic_projection(mu,cov,w)
    ordinary=quadratic_sf(q['lam'],q['beta'],q['constant'],h)
    rem=max(0.,var_upper-q['variance_full'])
    rows=[]
    for alpha in [.01,.025,.04]:
        eta=np.sqrt(rem*(1-alpha)/alpha)
        r=quadratic_sf(q['lam'],q['beta'],q['constant'],h+eta)
        rows.append(dict(remainder_risk=alpha,eta=eta,
            probability_lower=max(0.,r['p']-r['error']-alpha),quadratic_evaluation=r))
    return dict(mean=q['mean'],quadratic_variance=q['variance_full'],variance_numerical_error=abs(q['variance_full']-q['variance_factored']),
        dropped_covariance_trace=q['dropped_covariance_trace'],remainder_variance_upper=rem,
        approximate_probability=ordinary,corrected_bounds=rows,
        best_corrected_lower=max(v['probability_lower'] for v in rows),
        label='quadratic projection is approximate; remainder-corrected bound is analytically sufficient under the Gaussian law')
