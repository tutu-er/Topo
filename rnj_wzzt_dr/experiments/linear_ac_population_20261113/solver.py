"""Centralized QCQP; exact single-competitor dual first, native joint solve if needed.

No energy, power, clipping, availability, direction or integer device constraints.
The strictly positive quadratic fee bounds any better-than-incumbent solution.
"""
from time import perf_counter
import numpy as np
from scipy.optimize import brentq


class Model:
    def __init__(self,H,target,competitors,scale,sigma2,qratio=.35):
        self.H=np.asarray(H,float);self.target=int(target);self.competitors=list(competitors)
        self.scale=np.asarray(scale,float);self.sigma2=float(sigma2);self.qratio=qratio
        n=self.H.shape[1];self.n=n;self.shape=(24,n);self.dimension=24*n
        D=self.H[target][None]-self.H[self.competitors]
        self.D=D;normal=np.sqrt(2*sigma2)
        self.M=(D[:,:,:n]+qratio*D[:,:,n:])*self.scale[None,None]/normal
        self.Q=4*np.einsum('kni,knj->kij',self.M,self.M)
        self.spectral=[]
        for M in self.M:
            _,s,Vh=np.linalg.svd(M,full_matrices=False)
            self.spectral.append((4*s*s,Vh.T))

    def prepare(self,forecast):
        self.forecast=np.asarray(forecast,float)
        self.b=np.einsum('tc,knc->ktn',self.forecast,self.D)/np.sqrt(2*self.sigma2)
        self.a=np.einsum('kni,khtn->khi',self.M,self.b.reshape(len(self.M),24,4,self.n))
        return self

    def shift(self,x):
        p=np.repeat(np.asarray(x).reshape(self.shape),4,axis=0)*self.scale
        return np.column_stack((p,self.qratio*p))

    def gains(self,x):
        x=np.asarray(x).reshape(self.shape)
        effect=np.einsum('kni,hi->khn',self.M,x)
        return np.sum(2*self.b*np.repeat(effect,4,axis=1)+np.repeat(effect*effect,4,axis=1),axis=(1,2))

    def direct_gains(self,x):
        shift=self.shift(x);answer=[]
        for difference in self.D:
            b=self.forecast@difference.T;d=shift@difference.T
            answer.append(float(np.sum(2*b*d+d*d)/(2*self.sigma2)))
        return np.asarray(answer)

    def norm_cost(self,x):return float(np.sum(np.asarray(x)**2))


def single(model,j,gamma):
    """min ||x||² s.t. x'Qx+2a'x>=gamma, via a scalar dual root.

    For lambda < 1/lambda_max(Q), the Lagrangian is strictly convex.
    x=lambda*(I-lambda Q)^-1*a; matching the gain gives a primal/dual pair.
    The top-eigenspace boundary case is completed by one quadratic equation.
    Returned lower bounds use the ORIGINAL gamma, not the small feasible buffer.
    """
    q,V=model.spectral[j];a=model.a[j];c=a@V;maximum=float(q.max())
    if maximum==0:return dict(feasible=False,reason='exactly_zero_control_effect')
    required=gamma*(1+2e-8)+1e-12
    def at(s):return (s/maximum)*c/(1-s*q/maximum)[None]
    def gain(z):return float(np.sum(z*z*q[None]+2*c*z))
    high=1-1e-10
    if gain(at(high))>=required:
        s=brentq(lambda v:gain(at(v))-required,0.,high,xtol=np.nextafter(0.,1.),rtol=1e-12,maxiter=160)
        z=at(s)
    else:
        s=high;z=at(s);gap=required-gain(z)
        i=int(np.argmax(q));h=0
        linear=q[i]*z[h,i]+c[h,i];sign=1. if linear>=0 else -1.
        t=2*gap/(2*abs(linear)+np.sqrt(4*linear*linear+4*q[i]*gap))
        z[h,i]+=sign*t
    x=z@V.T;lam=s/maximum
    lower=lam*gamma-lam*lam*float(np.sum(c*c/(1-lam*q)[None]))
    actual=float(model.gains(x)[j]);cost=model.norm_cost(x)
    if actual<gamma-1e-8:raise ArithmeticError('Single-pair analytic point fails independent gain')
    if lower>cost+1e-7*max(1.,cost):raise ArithmeticError('Single-pair dual exceeds primal')
    return dict(feasible=True,x=x,cost=cost,lower_bound=max(0.,lower),gain=actual,
                gap=max(0.,cost-lower)/max(cost,1e-30))


def along(model,direction,gamma):
    """Exact positive scale along a fixed direction; used only as a warm start."""
    d=np.asarray(direction).reshape(model.shape)
    linear=2*np.einsum('khi,hi->k',model.a,d)
    response=np.einsum('kni,hi->khn',model.M,d)
    quadratic=4*np.sum(response*response,axis=(1,2));scales=[]
    for q,l in zip(quadratic,linear,strict=True):
        if q==0:
            if l<=0:return None
            scales.append(gamma/l)
        else:
            root=np.sqrt(l*l+4*q*gamma)
            scales.append(2*gamma/(root+l) if l>=0 else (root-l)/(2*q))
    return d*max(scales)*(1+2e-8)


def solve(model,gamma,*,seconds=.15,env=None):
    began=perf_counter();answers=[single(model,j,gamma) for j in range(len(model.M))]
    if any(not a['feasible'] for a in answers):
        return dict(qualified=False,status='uncontrollable_competitor',seconds=perf_counter()-began,
                    lower_bound=None,cost=None,global_optimality_verified=False)
    lower=max(a['lower_bound'] for a in answers)
    good=[a['x'] for a in answers if np.min(model.gains(a['x']))>=gamma-1e-8]
    native=None;source='analytic_pair_verified_for_all';seed=None
    if good:
        x=min(good,key=model.norm_cost)
    else:
        directions=[a['x'] for a in answers]
        generic=np.sin(np.arange(model.dimension).reshape(model.shape)*np.sqrt(2)+.3)
        directions.extend([generic,-generic])
        points=[along(model,d,gamma) for d in directions];points=[p for p in points if p is not None]
        if not points:return dict(qualified=False,status='no_feasible_warm_direction',seconds=perf_counter()-began,
                                  lower_bound=lower/model.dimension,cost=None,global_optimality_verified=False)
        x=min(points,key=model.norm_cost);seed=model.norm_cost(x);source='feasible_direction_seed'
    cost=model.norm_cost(x)
    gap=max(0.,cost-lower)/max(cost,1e-30)
    if gap>1e-5 and perf_counter()-began<seconds:
        import gurobipy as gp
        owns=env is None
        if owns:
            env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
        try:
            with gp.Model('linear_DR_joint_QCQP',env=env) as m:
                for k,v in dict(OutputFlag=0,Threads=1,NonConvex=2,MIPGap=.01,FeasibilityTol=1e-8,NumericFocus=1).items():m.setParam(k,v)
                # Any cheaper point lies in this box. These are computational
                # bounds derived from a feasible fee, not device power limits.
                limit=np.sqrt(cost)*(1+1e-8)
                v=m.addMVar(model.shape,lb=-limit,ub=limit);v.Start=x
                for j in range(len(model.M)):
                    quadratic=sum(v[h]@model.Q[j]@v[h] for h in range(24))
                    m.addConstr(quadratic+2*(model.a[j]*v).sum()>=gamma)
                m.setObjective((v*v).sum());m.Params.TimeLimit=max(.001,seconds-(perf_counter()-began));m.optimize()
                bound=float(m.ObjBound);native_gap=float(m.MIPGap) if m.SolCount else None
                native=dict(status=int(m.Status),solutions=int(m.SolCount),bound=bound if np.isfinite(bound) else None,
                            gap=native_gap if native_gap is not None and np.isfinite(native_gap) else None)
                if m.SolCount:
                    candidate=np.asarray(v.X)
                    if min(model.gains(candidate))>=gamma-1e-8 and model.norm_cost(candidate)<cost:
                        x=candidate;cost=model.norm_cost(x);source='native_QCQP_incumbent'
                if native['bound'] is not None and native['bound']<=cost+1e-7*max(1.,cost):lower=max(lower,native['bound'])
                else:native['bound_withdrawn']=True
        finally:
            if owns:env.dispose()
    gains=model.gains(x);direct=model.direct_gains(x)
    if np.max(abs(gains-direct))>1e-6*max(1.,float(max(abs(gains)))):raise ArithmeticError('Full H gain audit differs')
    if min(direct)<gamma-1e-8:raise ArithmeticError('QCQP quote fails original gain')
    gap=max(0.,cost-lower)/max(cost,1e-30)
    return dict(qualified=True,x=x,status='qualified',source=source,seconds=perf_counter()-began,
        native=native,cost=cost/model.dimension,lower_bound=lower/model.dimension,relative_gap=gap,
        global_optimality_verified=bool(gap<=1e-5),gains=direct,
        initialization_cost=None if seed is None else seed/model.dimension,
        competitor_count=len(model.M),no_ADMM=True)
