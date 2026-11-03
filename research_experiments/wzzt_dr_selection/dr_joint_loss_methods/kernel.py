"""Deterministic methods for the SAME frozen paired-L1 loss event.

No draws, topology fitting, or physical simulation. Affine Gaussian law only.
Floating-point bounds are analytic sufficient bounds, not interval certificates.
"""
from __future__ import annotations
import math,time
import numpy as np
from scipy.special import ndtr,ndtri,gammainc,gammaincc,logsumexp
from scipy.linalg import eigh
from scipy.optimize import milp,Bounds,LinearConstraint,minimize_scalar
from scipy.sparse import lil_matrix
from scipy.stats import qmc

def abs_mean(mu,sd):
    mu,sd=np.broadcast_arrays(np.asarray(mu,float),np.asarray(sd,float))
    a=np.divide(mu,sd,out=np.zeros_like(mu),where=sd>0)
    out=sd*np.sqrt(2/np.pi)*np.exp(-a*a/2)+mu*(2*ndtr(a)-1)
    return np.where(sd>0,out,np.abs(mu))

def gaussian_packet(packet,dev,u,cfg):
    """Interleaved time,C nodes,F nodes. All four original correlation factors."""
    T=96;n=len(dev.nodes);M=T*n
    effect=np.concatenate([packet['effect_C'],packet['effect_F']],axis=0)
    mu0=np.concatenate([packet['mu_C'].reshape(T,n),packet['mu_F'].reshape(T,n)],axis=1).ravel()
    delta=np.einsum('ir,rt->ti',effect,u).ravel()
    background=np.concatenate([packet['L_C'].reshape(T,n,-1),packet['L_F'].reshape(T,n,-1)],axis=1).reshape(2*M,-1)
    coeff=np.einsum('ir,rt->tir',effect,u*dev.relative_sd).reshape(2*M,-1)
    w0,w1,w2,w3=cfg['response']['main_variance_weights']
    phi=cfg['response']['main_ar1_phi']
    times=np.repeat(np.arange(T),2*n)
    ar=phi**np.abs(times[:,None]-times[None,:])
    common=coeff.sum(1)
    nodes=np.stack([coeff[:,dev.node==v].sum(1) for v in np.unique(dev.node)],axis=1)
    cov=background@background.T
    cov+=w0*np.outer(common,common)+w1*(nodes@nodes.T)
    cov+=(w2*np.outer(common,common)+w3*(coeff@coeff.T))*ar
    weights=np.tile(np.r_[np.ones(n),-1.03*np.ones(n)],T)/M
    C=np.arange(2*M).reshape(T,2*n)[:,:n].ravel()
    F=np.arange(2*M).reshape(T,2*n)[:,n:].ravel()
    return mu0+delta,cov,weights,mu0,delta,C,F

def hermite_moments(mu,cov,w,orders=(32,64,128)):
    """PSD expansion gives variance lower and analytic remainder upper.

    c_n=E[|mu+sd Z| He_n(Z)]/sqrt(n!).
    Var(D)=sum_n (w*c_n)' Corr**n (w*c_n).
    Tail <= t' abs(Corr)**(K+1) t, t_i=|w_i| sqrt(tail variance_i).
    """
    mu=np.asarray(mu);w=np.asarray(w);sd=np.sqrt(np.maximum(np.diag(cov),0))
    mean=abs_mean(mu,sd);a=np.divide(mu,sd,out=np.zeros_like(mu),where=sd>0)
    variance=np.maximum(mu*mu+sd*sd-mean*mean,0)
    variance=np.where(np.abs(a)>8,sd*sd,variance)
    den=np.outer(sd,sd)
    corr=np.divide(cov,den,out=np.zeros_like(cov),where=den>0)
    corr=np.clip((corr+corr.T)/2,-1.,1.)
    power=corr.copy();sumdiag=np.zeros_like(mu);partial=0.;rows=[]
    hprev=np.exp(-a*a/2)/np.sqrt(2*np.pi);hprevprev=None
    for order in range(1,max(orders)+1):
        if order==1:c=sd*(2*ndtr(a)-1)
        else:
            m=order-2
            if m==0:hm=hprev
            elif m==1:
                hm=-a*hprev;hprevprev,hprev=hprev,hm
            else:
                hm=(-a*hprev-np.sqrt(m-1)*hprevprev)/np.sqrt(m)
                hprevprev,hprev=hprev,hm
            c=2*sd*hm/np.sqrt((m+1)*(m+2))
        vc=w*c;term=float(vc@(power@vc))
        if term < -1e-6:raise ArithmeticError('negative chaos variance')
        partial+=max(0.,term);sumdiag+=c*c
        if order in orders:
            tail=np.maximum(variance-sumdiag,0.)
            t=np.abs(w)*np.sqrt(tail)
            tail_bound=float(t@(np.abs(power*corr)@t))
            rows.append(dict(order=order,variance_lower=partial,variance_upper=partial+max(0.,tail_bound),
                remainder_upper=max(0.,tail_bound),diagonal_overshoot=float(np.max(sumdiag-variance))))
        power*=corr
    return dict(mean=float(w@mean),rows=rows,
        marginal_independence_variance=float((w*w)@variance),method='Hermite with analytic remainder bound')

def chi_mass(lo,hi,df):
    lo=np.asarray(lo);hi=np.asarray(hi)
    a=gammainc(df/2,lo*lo/2);b=gammainc(df/2,hi*hi/2)
    return np.maximum(0,np.where(a>.5,gammaincc(df/2,lo*lo/2)-gammaincc(df/2,hi*hi/2),b-a))

def ray_probability(a,b,w,h,df):
    """Exact piecewise linear threshold intervals on radial coordinate >=0."""
    a=np.asarray(a);b=np.asarray(b);w=np.asarray(w)
    nz=b!=0
    roots=-a[nz]/b[nz];jumps=2*w[nz]*np.abs(b[nz])
    use=roots>0;roots=roots[use];jumps=jumps[use]
    order=np.argsort(roots);roots=roots[order];jumps=jumps[order]
    if len(roots):
        unique,idx=np.unique(roots,return_index=True)
        jumps=np.add.reduceat(jumps,idx);roots=unique
    s0=float(w@(np.where(a==0,np.sign(b),np.sign(a))*b))
    c0=float(w@np.abs(a)-h)
    slopes=np.r_[s0,s0+np.cumsum(jumps)]
    intercepts=np.r_[c0,c0-np.cumsum(jumps*roots)]
    lo=np.r_[0.,roots];hi=np.r_[roots,np.inf]
    pos=slopes>0;neg=slopes<0;flat=slopes==0
    cutoff=np.divide(-intercepts,slopes,out=np.zeros_like(slopes),where=~flat)
    left=np.where(pos,np.maximum(lo,cutoff),lo)
    right=np.where(neg,np.minimum(hi,cutoff),hi)
    valid=(right>left)&(~flat|(intercepts>0))
    return float(chi_mass(np.maximum(left[valid],0),right[valid],df).sum())

def spherical_probability(mu,cov,w,h,power=11):
    t=time.perf_counter()
    ev,vec=eigh(cov,check_finite=False,driver='evr')
    if ev.min() < -1e-8*max(1.,ev.max()):raise ArithmeticError('covariance is not PSD')
    tol=max(ev.max()*1e-12,1e-10)
    keep=ev>tol
    dropped=np.maximum(ev[~keep],0)
    dropdiag=(vec[:,~keep]**2)@dropped
    perturb=float(np.sqrt(2/np.pi)*(np.abs(w)@np.sqrt(np.maximum(dropdiag,0))))
    if not keep.any():
        return dict(rank=0,probabilities=[float(w@np.abs(mu)>h)],grid_spread=0.,expected_discard_error=perturb)
    factor=vec[:,keep][:,::-1]*np.sqrt(ev[keep][::-1])
    rank=factor.shape[1];N=2**power;records=[]
    for grid in range(2):
        if rank==1:
            directions=np.array([[-1.],[1.]])
        elif rank==2:
            angle=2*np.pi*(np.arange(N)+(0.5 if grid==0 else .25))/N
            directions=np.stack([np.cos(angle),np.sin(angle)],axis=1)
        else:
            pts=qmc.Sobol(rank,scramble=False).random_base2(power)
            shift=np.mod((np.arange(rank)+1)*(.4142135623730951 if grid==0 else .7320508075688772),1)
            pts=np.mod(pts+shift,1.)
            z=ndtri(np.clip(pts,2**-40,1-2**-40))
            directions=z/np.linalg.norm(z,axis=1)[:,None]
        values=[]
        for first in range(0,len(directions),64):
            B=directions[first:first+64]@factor.T
            values.extend(ray_probability(mu,b,w,h,rank) for b in B)
        for count in sorted(set([len(values)//2,len(values)])):
            # rank 2 half-grid prefix is a HALF CIRCLE, not valid quadrature!
            if rank<=2 and count<len(values):continue
            records.append(dict(grid=grid,directions=count,p=float(np.mean(values[:count]))))
    vals=[r['p'] for r in records]
    return dict(rank=rank,probabilities=records,grid_spread=max(vals)-min(vals),
        probability_estimate=float(np.mean([r['p'] for r in records if r['directions']==max(q['directions'] for q in records)])),
        expected_discard_error=perturb,discarded_trace=float(dropped.sum()),
        negative_eigenvalue=float(ev.min()),seconds=time.perf_counter()-t,
        numerical_only=True,certificate=False,random_draws=0,
        boundary='deterministic angular cubature, not an exact probability or SOCP')

def bvn_cdf(a,b,rho,order=64):
    """Deterministic Gauss-Legendre integration of Plackett's identity."""
    a,b,rho=np.broadcast_arrays(a,b,rho);rho=np.clip(rho,-1,1)
    nodes,weights=np.polynomial.legendre.leggauss(order)
    angle=np.arcsin(rho)
    result=ndtr(a)*ndtr(b)
    acc=np.zeros_like(result)
    for z,w in zip(nodes,weights):
        theta=angle*(z+1)/2;s=np.sin(theta);c2=np.maximum(np.cos(theta)**2,1e-300)
        exponent=-(a*a-2*a*b*s+b*b)/(2*c2)
        acc+=w*np.exp(np.minimum(exponent,0))
    result+=angle*acc/(4*np.pi)
    result=np.where(rho>=1-1e-14,ndtr(np.minimum(a,b)),result)
    result=np.where(rho<=-1+1e-14,np.maximum(0,ndtr(a)-ndtr(-b)),result)
    return np.clip(result,0,1)

def paired_logmgf(mu,cov,C,F,lam,order=64):
    """log E exp(-lambda (|C|-1.03|F|)), each pair integrated jointly."""
    mc=mu[C];mf=mu[F];vc=np.diag(cov)[C];vf=np.diag(cov)[F];cc=cov[C,F]
    sc=np.sqrt(vc);sf=np.sqrt(vf)
    if np.any(sc==0)|np.any(sf==0):raise ValueError('degenerate marginal requires separate evaluation')
    rho=np.clip(cc/(sc*sf),-1,1);terms=[]
    for s in [-1.,1.]:
        for t in [-1.,1.]:
            tc=-lam*s;tf=lam*1.03*t
            nc=mc+vc*tc+cc*tf;nf=mf+cc*tc+vf*tf
            prob=bvn_cdf(s*nc/sc,t*nf/sf,s*t*rho,order)
            exponent=tc*mc+tf*mf+.5*(tc*tc*vc+2*tc*tf*cc+tf*tf*vf)
            terms.append(exponent+np.log(np.maximum(prob,1e-300)))
    return logsumexp(np.stack(terms),axis=0)

def holder_tail(mu,cov,C,F,h):
    """Hölder bound valid under arbitrary cross-pair dependence.

    E exp(-t*mean(f)) <= product(E exp(-t*f_i))**(1/M).
    This may be weak, but never silently assumes independence.
    """
    scale=max(1.,float(np.sqrt(np.diag(cov)).mean()))
    def value(x,order=64):
        lam=x/scale
        return float(lam*h+paired_logmgf(mu,cov,C,F,lam,order).mean())
    fit=minimize_scalar(value,bounds=(0.,2.),method='bounded',options=dict(xatol=1e-5,maxiter=32))
    x=float(fit.x) if fit.fun<0 else 0.
    q64=value(x,64);q128=value(x,128)
    err=abs(q64-q128)
    return dict(lambda_micro=x/scale,log_failure_upper=q128,
        failure_upper=min(1.,float(np.exp(min(0.,q128)))),
        lower_success=max(0.,1-float(np.exp(min(0.,q128)))),
        quadrature_difference=err,quadrature_unresolved=err>1e-5,
        search_success=bool(fit.success),tilt_search_upper=2./scale,
        independence_assumed=False,interval_arithmetic_certificate=False)

def ray_enumerate(mu,delta,w,target):
    cut=np.r_[0.,1.,-mu[delta!=0]/delta[delta!=0]]
    cut=np.unique(cut[(cut>=0)&(cut<=1)])
    y=np.array([w@np.abs(mu+t*delta)-target for t in cut])
    if y[0]>=0:return 0.
    for j in range(len(cut)-1):
        if y[j+1]>=0:
            return float(cut[j]+(-y[j])*(cut[j+1]-cut[j])/(y[j+1]-y[j]))
    return None

def nominal_milp(mu,delta,w,h,seconds=15):
    """Exact L1 only on the prespecified [0,1] ray, no future signs."""
    target=h+1e-6
    low=np.minimum(mu,mu+delta);high=np.maximum(mu,mu+delta)
    unstable=(low<0)&(high>0)
    ids=np.flatnonzero(unstable);K=len(ids)
    stable=~unstable;s=np.where(low[stable]>=0,1.,-1.)
    c0=float((w[stable]*s)@mu[stable]);c1=float((w[stable]*s)@delta[stable])
    oracle=ray_enumerate(mu,delta,w,target)
    if K==0:
        got=oracle
        return dict(status='exact_affine',alpha=got,enum_alpha=oracle,unstable_signs=0,solver_calls=0)
    nv=1+2*K;A=lil_matrix((4*K+1,nv));lb=np.full(4*K+1,-np.inf);ub=np.full(4*K+1,np.inf)
    for j,i in enumerate(ids):
        y=1+j;z=1+K+j
        # y>=r; y>=-r; y<=r+2Mminus(1-z); y<=-r+2Mplus*z
        A[4*j,0]=-delta[i];A[4*j,y]=1;lb[4*j]=mu[i]
        A[4*j+1,0]=delta[i];A[4*j+1,y]=1;lb[4*j+1]=-mu[i]
        mm=-low[i];mp=high[i]
        A[4*j+2,0]=-delta[i];A[4*j+2,y]=1;A[4*j+2,z]=2*mm;ub[4*j+2]=mu[i]+2*mm
        A[4*j+3,0]=delta[i];A[4*j+3,y]=1;A[4*j+3,z]=-2*mp;ub[4*j+3]=-mu[i]
    A[-1,0]=c1
    for j,i in enumerate(ids):A[-1,1+j]=w[i]
    # Keep the strict micro-unit margin above HiGHS' absolute row tolerance.
    A[-1,:]*=1e4
    lb[-1]=1e4*(target-c0)
    costs=np.zeros(nv);costs[0]=1
    lower=np.zeros(nv);upper=np.r_[1.,np.maximum(-low[ids],high[ids]),np.ones(K)]
    integ=np.r_[np.zeros(1+K),np.ones(K)]
    fit=milp(costs,integrality=integ,bounds=Bounds(lower,upper),constraints=LinearConstraint(A.tocsr(),lb,ub),
        options=dict(time_limit=seconds,mip_rel_gap=1e-8))
    alpha=None if fit.x is None else float(fit.x[0])
    if alpha is not None and w@np.abs(mu+alpha*delta)<target-1e-7:raise ArithmeticError('MILP original L1 fails')
    if fit.status==0 and (oracle is None or abs(alpha-oracle)>1e-5):raise ArithmeticError(f'MILP vs exact ray enumeration: alpha={alpha}, oracle={oracle}, target={target}, x={fit.x}')
    return dict(status=int(fit.status),message=fit.message,alpha=alpha,enum_alpha=oracle,unstable_signs=K,
        solver_calls=1,gap=None if getattr(fit,'mip_gap',None) is None else float(fit.mip_gap),
        dual_bound=None if getattr(fit,'mip_dual_bound',None) is None else float(fit.mip_dual_bound))
