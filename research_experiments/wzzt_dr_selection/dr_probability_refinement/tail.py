"""Exact conditional positive shortfall of the original signed L1 difference.

This is an initialization aid, not a substitute for the final event probability.
It reuses the existing event integrator and adds Gaussian truncated moments.
"""
import math
import numpy as np
from conditional_integral import conditional_probability, _normal_mass


def density(x):
    return np.exp(-.5*np.asarray(x)**2)/np.sqrt(2*np.pi)


def shortfall(a,b,w,t):
    """E[(t-sum w|a+b Z|)+], d/da, d/dt for fixed b,w.

At atoms at t or identically zero residuals a selected subgradient is returned;
the original exact event check, with its derivative diagnostics, remains final.
"""
    a,b,w=map(lambda q:np.asarray(q,float),(a,b,w))
    out=conditional_probability(a,b,-w,-t,coefficient_gradients=False)
    ints=np.asarray(out.intervals,float).reshape(-1,2)
    if not len(ints):return dict(value=0.,gradient=np.zeros_like(a),dt=0.)
    left,right=ints.T
    masses=np.array([_normal_mass(l,r) for l,r in ints])
    moments=density(left)-density(right)
    pm=np.r_[0.,np.cumsum(masses)];pz=np.r_[0.,np.cumsum(moments)]
    mass=float(pm[-1]);moment=float(pz[-1])
    variable=b!=0;idx=np.flatnonzero(variable);knots=-a[idx]/b[idx]
    full=np.searchsorted(right,knots,side='right')
    fm=pm[full].copy();fz=pz[full].copy()
    partial=(full<len(left))
    j=np.flatnonzero(partial)
    j=j[knots[j]>left[full[j]]]
    if len(j):
        fm[j]+=np.array([_normal_mass(left[full[k]],knots[k]) for k in j])
        fz[j]+=density(left[full[j]])-density(knots[j])
    sign_integral=np.sign(a)*mass
    sign_z=np.sign(a)*moment
    sign_integral[idx]=np.sign(b[idx])*(mass-2*fm)
    sign_z[idx]=np.sign(b[idx])*(moment-2*fz)
    absolute_integral=a*sign_integral+b*sign_z
    value=math.fsum([float(t*mass)]+(-w*absolute_integral).tolist())
    tolerance=1e-10*(1+abs(t*mass)+np.sum(np.abs(w*absolute_integral)))
    if value < -tolerance:raise FloatingPointError('negative shortfall beyond roundoff')
    return dict(value=max(0.,value),gradient=-w*sign_integral,dt=mass)


class TailStart:
    """CVaR of h-D under the fixed outer integration; initialization only."""
    def __init__(self,probability):
        self.p=probability;self.scale=max(1.,abs(probability.packet['threshold']));self.last_x=None

    def evaluate(self,v):
        v=np.asarray(v)
        if self.last_x is not None and np.array_equal(v,self.last_x):return self.last
        p=self.p;d=p.devices;x=v[:-1].reshape(d.R,24);alpha=v[-1]*self.scale
        u,nc=d.nominal(x,cache=True);delivered,ec=d.project(u[None]*p.multiplier,cache=True)
        c=np.einsum('nr,srt->stn',p.packet['effect_C'],delivered).reshape(p.samples,-1)
        f=np.einsum('nr,srt->stn',p.packet['effect_F'],delivered).reshape(p.samples,-1)
        a=p.a0+np.concatenate([c,f],axis=1)
        outs=[shortfall(row,p.b,p.w,p.packet['threshold']-alpha) for row in a]
        g=20*np.array([r['gradient'] for r in outs])/self.scale
        gc=np.einsum('stn,nr->srt',g[:,:p.m].reshape(p.samples,96,-1),p.packet['effect_C'])
        gf=np.einsum('stn,nr->srt',g[:,p.m:].reshape(p.samples,96,-1),p.packet['effect_F'])
        gnom=np.mean(d.vjp(gc+gf,ec)*p.multiplier,axis=0)
        gx=d.hourly_vjp(gnom,nc).ravel()
        ga=1-20*np.mean([r['dt'] for r in outs])
        value=(alpha+20*np.mean([r['value'] for r in outs]))/self.scale
        self.last=dict(value=float(value),gradient=np.r_[gx,ga])
        self.last_x=v.copy();return self.last
