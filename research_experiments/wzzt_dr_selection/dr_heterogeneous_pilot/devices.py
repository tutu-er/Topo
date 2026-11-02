"""Causal state-feasible delivery for the frozen heterogeneous resource library.

The actuator uses only the current request, current energy, and declared
future power/availability. Future random errors are never inspected.
"""
from __future__ import annotations
import numpy as np


class Devices:
    def __init__(self, case, library, profiles):
        self.nodes=case['nodes'];self.cfg=library
        self.labels=[n['node_label'] for n in self.nodes]
        data=[profiles(n,library) for n in self.nodes]
        self.R=4*len(data);self.T=96;self.dt=.25
        self.kind=np.tile(np.arange(4),len(data))
        self.node=np.repeat(np.arange(len(data)),4)
        self.ids=[f'{n["node_label"]}_{k}' for n in self.nodes for k in ['shift','curtail','ev','battery']]
        self.cap=np.array([n[k] for n in self.nodes for k in ['shiftable_kw','curtailable_kw','ev_charger_kw','battery_power_kw']])
        for name in ['baseline_kw','delta_lower_kw','delta_upper_kw','cost_full_activation','relative_sd']:
            setattr(self,name,np.concatenate([d[name] for d in data]))
        self.qratio=np.concatenate([d['q_over_p'] for d in data])
        self.scale=np.maximum(-self.delta_lower_kw,self.delta_upper_kw)
        self.eta=np.where(self.kind==3,np.sqrt(library['battery']['roundtrip_efficiency']),1.)
        self.energy_low=np.where(self.kind==3,-.3*2.5*self.cap,np.where(self.kind<2,-2*self.cap,-np.inf))
        self.energy_high=np.where(self.kind==3,.3*2.5*self.cap,np.where(self.kind==0,2*self.cap,np.where(self.kind==1,0.,np.inf)))
        self.future_lo=np.zeros((self.R,self.T));self.future_hi=self.future_lo.copy()
        self.reset=np.zeros((self.R,self.T),bool)
        for r in range(self.R):
            segments=[(0,32),(72,96)] if self.kind[r]==2 else [(0,96)]
            for start,end in segments:
                self.reset[r,start]=True
                low=self.delta_lower_kw[r,start:end]
                high=self.delta_upper_kw[r,start:end]
                elo=np.where(low<0,low/self.eta[r],low*self.eta[r])*self.dt
                ehi=np.where(high<0,high/self.eta[r],high*self.eta[r])*self.dt
                self.future_lo[r,start:end]=np.cumsum(elo[::-1])[::-1]-elo
                self.future_hi[r,start:end]=np.cumsum(ehi[::-1])[::-1]-ehi
        # Curtailment has no repayment equality; its daily energy bound remains.
        curtail=self.kind==1
        self.future_lo[curtail]=-np.inf;self.future_hi[curtail]=np.inf
        self.next_lo=np.maximum(self.energy_low[:,None],-self.future_hi)
        self.next_hi=np.minimum(self.energy_high[:,None],-self.future_lo)

    def _inverse(self,energy_rate):
        return np.where(energy_rate>=0,energy_rate/self.eta,energy_rate*self.eta)

    def project(self,raw,cache=False):
        """raw (...,resource,slot), all decisions in physical kW."""
        raw=np.asarray(raw,float)
        if raw.shape[-2:]!=(self.R,self.T) or not np.isfinite(raw).all():
            raise ValueError('finite resource-by-96 requests required')
        shape=raw.shape;v=raw.reshape(-1,self.R,self.T);e=np.zeros(v.shape[:2])
        out=np.zeros_like(v);aa=np.zeros_like(v);bb=np.zeros_like(v);kk=np.zeros_like(v)
        for t in range(self.T):
            e=np.where(self.reset[:,t],0.,e)
            ell=self._inverse((self.next_lo[:,t]-e)/self.dt)
            ehi=self._inverse((self.next_hi[:,t]-e)/self.dt)
            low=np.maximum(self.delta_lower_kw[:,t],ell)
            high=np.minimum(self.delta_upper_kw[:,t],ehi)
            if np.any(low>high+1e-8):raise ArithmeticError('state left viability interval')
            high=np.maximum(high,low)
            p=np.minimum(np.maximum(v[:,:,t],low),high)
            slope=np.where(p>=0,self.eta,1/self.eta)
            direct=(v[:,:,t]>low)&(v[:,:,t]<high)
            # Choose the current face at a kink; line search uses exact map.
            direct |= (v[:,:,t]==0)&(low<high)&(v[:,:,t]>=low)&(v[:,:,t]<=high)
            energy_face=((v[:,:,t]<=low)&(ell>=self.delta_lower_kw[:,t]))|((v[:,:,t]>=high)&(ehi<=self.delta_upper_kw[:,t]))
            aa[:,:,t]=np.where(energy_face,0.,direct)
            bb[:,:,t]=np.where(energy_face,-1/(self.dt*slope),0.)
            kk[:,:,t]=self.dt*slope
            out[:,:,t]=p;e+=self.dt*slope*p
        if cache:return out.reshape(shape),(aa,bb,kk,shape)
        return out.reshape(shape)

    def vjp(self,gradient,cache):
        aa,bb,kk,shape=cache;g=np.asarray(gradient).reshape(aa.shape)
        previous=np.zeros(aa.shape[:2]);out=np.zeros_like(aa)
        for t in range(self.T-1,-1,-1):
            combined=g[:,:,t]+previous*kk[:,:,t]
            out[:,:,t]=combined*aa[:,:,t]
            previous=previous+combined*bb[:,:,t]
            previous=np.where(self.reset[:,t],0.,previous)
        return out.reshape(shape)

    def nominal(self,x,cache=False):
        x=np.asarray(x).reshape(self.R,24)
        raw=np.repeat(x,4,axis=1)*self.scale
        return self.project(raw,cache=cache)

    def hourly_vjp(self,g,cache):
        return (self.vjp(g,cache)*self.scale).reshape(self.R,24,4).sum(-1)

    def fees(self,u,gradient=False):
        u=np.asarray(u);a=np.where(u>=0,self.delta_upper_kw,-self.delta_lower_kw)
        coeff=np.divide(self.dt*self.cost_full_activation,a,out=np.zeros_like(a),where=a>0)
        if np.any((a==0)&(np.abs(u)>1e-8)):raise ValueError('zero headroom used')
        fee=float(np.sum(coeff*u*u))
        return (fee,2*coeff*u) if gradient else fee

    def draw(self,seed,samples,stress=False):
        rng=np.random.default_rng(seed);S=samples;N=len(self.nodes);R=self.R;T=self.T
        key='stress' if stress else 'main';w=self.cfg['response'][key+'_variance_weights'];phi=self.cfg['response'][key+'_ar1_phi']
        common=rng.normal(size=(S,1,1));node=rng.normal(size=(S,N,1))[:,self.node]
        def ar(x):
            for t in range(1,T):x[...,t]=phi*x[...,t-1]+np.sqrt(1-phi**2)*x[...,t]
            return x
        ct=ar(rng.normal(size=(S,1,T)));rt=ar(rng.normal(size=(S,R,T)))
        z=np.sqrt(w[0])*common+np.sqrt(w[1])*node+np.sqrt(w[2])*ct+np.sqrt(w[3])*rt
        return (.9 if stress else 1.)+self.relative_sd[None]*z

    def audit(self,u):
        u=np.asarray(u,float)
        lower=float(np.max(np.maximum(self.delta_lower_kw-u,0)))
        upper=float(np.max(np.maximum(u-self.delta_upper_kw,0)))
        e=np.zeros(self.R);bound=0.;deadline=0.;energy=np.zeros_like(u)
        for t in range(96):
            e=np.where(self.reset[:,t],0.,e)
            e+=self.dt*np.where(u[:,t]>=0,self.eta,1/self.eta)*u[:,t]
            energy[:,t]=e
            bound=max(bound,float(np.max(np.maximum(self.energy_low-e,0))),float(np.max(np.maximum(e-self.energy_high,0))))
            end=(self.kind!=1)&((t==95)|((self.kind==2)&(t==31)))
            deadline=max(deadline,float(np.max(np.abs(e[end]),initial=0.)))
        return dict(power_violation_kw=max(lower,upper),energy_violation_kwh=bound,deadline_violation_kwh=deadline,
            passed=bool(max(lower,upper,bound,deadline)<1e-7),absolute_DR_kwh=float(self.dt*np.abs(u).sum()),
            net_DR_kwh=float(self.dt*u.sum()),by_type_abs_kwh={str(k):float(self.dt*np.abs(u[self.kind==k]).sum()) for k in range(4)},
            max_sum_abs_kw=float(np.abs(u).sum(0).max()))

    def aggregate(self,u):
        p=np.zeros((96,len(self.nodes)));q=p.copy()
        for r,n in enumerate(self.node):p[:,n]+=u[r];q[:,n]+=self.qratio[r]*u[r]
        return p,q
