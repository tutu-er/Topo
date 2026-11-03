"""Analytic screen of the signed-L1 SOCP sufficient condition. No draws."""
from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr, ndtri
from scipy.optimize import brentq


def phi(z):
    return np.exp(-0.5*np.asarray(z)**2)/np.sqrt(2*np.pi)


def fixed_sign(mu):
    return np.where(np.asarray(mu) >= 0., 1., -1.)


def ar_cov(t, coefficient):
    return coefficient**np.abs(np.arange(t)[:, None]-np.arange(t)[None, :])


def variance_of_sum(coeff, nodes, weights, ar):
    """Var(sum_rt coeff_rt Z_rt), retaining all declared shared factors."""
    coeff=np.asarray(coeff, float);nodes=np.asarray(nodes)
    w0,w1,w2,w3=weights
    bytime=coeff.sum(axis=0)
    node_sums=[coeff[nodes == n].sum() for n in np.unique(nodes)]
    return float(w0*coeff.sum()**2 + w1*np.dot(node_sums,node_sums)
                 + w2*(bytime@ar@bytime) + w3*np.einsum('rt,ts,rs->',coeff,ar,coeff))


def residual_variances(effect, uncertainty, background, nodes, weights):
    """Marginal residual variances, (slot,node), using same-time factors."""
    w0,w1,w2,w3=weights
    coeff=np.einsum('ir,rt->tir',effect,uncertainty)
    v=(w0+w2)*coeff.sum(axis=2)**2+w3*(coeff**2).sum(axis=2)
    for node in np.unique(nodes):
        v+=w1*coeff[:,:,nodes == node].sum(axis=2)**2
    return v+(background**2).sum(axis=1).reshape(v.shape)


def expected_hinge(signed_mean, sd):
    """E[(-X)+] for Gaussian X, including deterministic coordinates."""
    a=np.asarray(signed_mean,float);sd=np.asarray(sd,float)
    result=np.maximum(-a,0.)
    active=sd>0
    z=a[active]/sd[active]
    result[active]=sd[active]*phi(z)-a[active]*ndtr(-z)
    return result


def correction(signed_mean, sd, risk):
    """Smallest v>=0 for the fixed per-coordinate risk allocations."""
    signed_mean,sd,risk=map(lambda a:np.asarray(a,float),(signed_mean,sd,risk))
    if np.any((risk<=0)|(risk>=.5)):
        raise ValueError('strictly positive risks below one half required')
    return np.maximum(0.,-ndtri(risk)*sd-signed_mean)


def optimal_correction_risks(signed_mean, sd, budget):
    """Fixed-price convex resource allocation, solved by scalar KKT equation.

    This diagnostic optimizes Bonferroni row allocation, not prices. It is not
    an assertion that variable-risk price optimization is a single SOCP.
    """
    a=np.asarray(signed_mean,float);s=np.asarray(sd,float)
    if not (0<budget<.5):raise ValueError('risk budget')
    active=s>0;alpha=np.zeros_like(s);v=np.maximum(0.,-a)
    if not np.any(active):return alpha,v
    sa=s[active];ma=a[active]
    q=ndtr(-ma/sa)
    if q.sum()<=budget:
        alpha[active]=q;v[active]=0.
        return alpha,v
    logbase=np.log(sa)+.5*np.log(2*np.pi)
    def allocation(loglambda):
        z=np.sqrt(np.maximum(0.,2*(loglambda-logbase)))
        return np.minimum(q,ndtr(-z))
    lo=float(logbase.max());hi=lo+2.
    while allocation(hi).sum()>budget:hi+=2.
    root=brentq(lambda z:float(allocation(z).sum()-budget),lo,hi,xtol=1e-13)
    aa=allocation(root)
    # Protect the risk budget from upward last-bit roundoff.
    aa*=min(1.,budget/aa.sum())
    alpha[active]=aa
    z=-ndtri(np.maximum(aa,np.finfo(float).tiny))
    v[active]=np.maximum(0.,z*sa-ma)
    return alpha,v


def fixed_quote_metrics(packet, dev, u, cfg, signs, allocation):
    """Affine BEFORE execution projection; same frozen nominal u and prices."""
    u=np.asarray(u,float);T,n=96,len(dev.nodes);M=T*n;kappa=1.03
    w=cfg['response']['main_variance_weights'];ar=ar_cov(T,cfg['response']['main_ar1_phi'])
    uncertainty=u*dev.relative_sd
    mu={};sd={};sgn={}
    for k in ['C','F']:
        mu[k]=(packet['mu_'+k].reshape(T,n)+np.einsum('ir,rt->ti',packet['effect_'+k],u)).ravel()
        vv=residual_variances(packet['effect_'+k],uncertainty,packet['L_'+k],dev.node,w)
        if np.any(vv < -1e-10):raise ArithmeticError('negative variance')
        sd[k]=np.sqrt(np.maximum(vv.ravel(),0.))
        sgn[k]=fixed_sign(packet['mu_'+k] if signs=='historical' else mu[k])
    gm=float((sgn['C']@mu['C']-kappa*sgn['F']@mu['F'])/M)
    gb=(sgn['C']@packet['L_C']-kappa*sgn['F']@packet['L_F'])/M
    coeff=((sgn['C'].reshape(T,n)@packet['effect_C']-kappa*sgn['F'].reshape(T,n)@packet['effect_F']).T/M)*uncertainty
    gv=float(gb@gb)+variance_of_sum(coeff,dev.node,w,ar);gs=math.sqrt(max(0.,gv))
    alpha0=.04;flip_budget=.01
    alpha=np.full(M,flip_budget/M)
    if allocation=='uniform':v=correction(sgn['F']*mu['F'],sd['F'],alpha)
    elif allocation=='optimal_fixed_quote':alpha,v=optimal_correction_risks(sgn['F']*mu['F'],sd['F'],flip_budget)
    else:raise ValueError(allocation)
    main_penalty=float(-ndtri(alpha0)*gs)
    flip_penalty=float(2*kappa*v.sum()/M)
    h=float(packet['threshold']);delta=1e-6
    lower=gm-main_penalty-flip_penalty
    ec=expected_hinge(sgn['C']*mu['C'],sd['C'])
    ef=expected_hinge(sgn['F']*mu['F'],sd['F'])
    ec_term=float(2*ec.sum()/M);ef_term=float(2*kappa*ef.sum()/M)
    abs_means={k:np.abs(mu[k])+2*expected_hinge(np.abs(mu[k]),sd[k]) for k in ['C','F']}
    exact_mean=float((abs_means['C'].sum()-kappa*abs_means['F'].sum())/M)
    identity_error=abs(exact_mean-(gm+ec_term-ef_term))
    if identity_error>1e-8:raise ArithmeticError('folded normal identity failed')
    used_risk=float(alpha.sum())
    if used_risk>flip_budget+1e-12:raise ArithmeticError('risk allocation exceeds budget')
    nz=sd['F']>0
    tail=np.zeros(M);tail[nz]=ndtr(-(v[nz]+sgn['F'][nz]*mu['F'][nz])/sd['F'][nz])
    tail[~nz]=((-sgn['F'][~nz]*mu['F'][~nz])>v[~nz]+1e-10).astype(float)
    risk_excess=float(np.max(tail-alpha,initial=0.))
    if risk_excess>1e-11:raise ArithmeticError('row Gaussian risk failed')
    return dict(signs=signs,allocation=allocation,rows=M,threshold_micro=h,
        mean_G_micro=gm,sd_G_micro=gs,main_penalty_micro=main_penalty,
        flip_penalty_micro=flip_penalty,certified_lower_micro=lower,
        certificate_margin_micro=lower-h-delta,certificate_pass=bool(lower>=h+delta),
        plain_gaussian_q05_micro=gm+ndtri(.05)*gs,
        plain_gaussian_not_a_certificate_for_D=True,exact_affine_mean_D_micro=exact_mean,
        omitted_C_expected_flip_micro=ec_term,F_expected_flip_micro=ef_term,
        mean_identity_error=identity_error,row_risk_max_excess=risk_excess,
        alpha0=alpha0,flip_risk_budget=flip_budget,used_flip_risk=used_risk,
        positive_correction_count=int(np.count_nonzero(v>1e-12)),
        expected_F_flip_fraction=float(np.mean(ndtr(np.divide(-sgn['F']*mu['F'],sd['F'],out=np.full(M,-np.inf),where=sd['F']>0)))),
        scope='95-percent sufficient bound for affine pre-controller working model only')


def device_screen(dev,u,cfg):
    """Marginal power tails plus exact shift/EV terminal variance. No draws."""
    u=np.asarray(u,float);sigma=np.abs(u*dev.relative_sd)
    active=sigma>0;p=np.zeros_like(u)
    p[active]=ndtr((dev.delta_lower_kw[active]-u[active])/sigma[active])+ndtr((u[active]-dev.delta_upper_kw[active])/sigma[active])
    p[~active]=((u[~active]<dev.delta_lower_kw[~active]-1e-10)|(u[~active]>dev.delta_upper_kw[~active]+1e-10)).astype(float)
    w=cfg['response']['main_variance_weights'];ar=ar_cov(96,cfg['response']['main_ar1_phi'])
    terminals=[]
    for r in range(dev.R):
        if dev.kind[r] not in [0,2]:continue
        for start,end in ([(0,96)] if dev.kind[r]==0 else [(0,32),(72,96)]):
            v=np.zeros(96);v[start:end]=dev.dt*u[r,start:end]*dev.relative_sd[r,start:end]
            var=(w[0]+w[1])*v.sum()**2+(w[2]+w[3])*(v@ar@v)
            terminals.append(dict(resource=r,kind=int(dev.kind[r]),start=start,end=end,
                mean_kwh=float(dev.dt*u[r,start:end].sum()),sd_kwh=float(np.sqrt(max(0.,var))),
                nondegenerate=bool(var>0.),activated=bool(np.any(u[r,start:end]!=0.))))
    return dict(max_marginal_power_violation_probability=float(p.max()),
        marginal_cells_above_5_percent=int(np.count_nonzero(p>.05)),
        marginal_power_union_upper=float(min(1.,p.sum())),
        any_power_violation_probability_lower=float(p.max()),
        shift_EV_terminal_checks=len(terminals),
        shift_EV_stochastic_terminal_equalities=int(sum(t['nondegenerate'] for t in terminals)),
        max_terminal_sd_kwh=max((t['sd_kwh'] for t in terminals),default=0.),
        all_terminal_equalities_hold_almost_surely=bool(all(not t['nondegenerate'] and abs(t['mean_kwh'])<1e-8 for t in terminals)),
        battery_terminal_not_certified=True,private_terminals=terminals)
