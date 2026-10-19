"""Root-aware response profiling and convex QPs on an identical line domain."""
from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from ..local_response_score.scoring import contrast_matrix,_profile_h,_polish_qp
from .model import line_map,project,linear_support,scales


def prepare(candidate,payload):
    obs=payload["observed"];sig=scales(payload);n=candidate.n;vpos=payload["voltage_positions"]
    var=4*sig["v"]**2+2*sig["v"]**4
    if payload["root_meter"] is None:
        d=contrast_matrix(n);y=(obs["v"]**2-sig["v"]**2)@d.T
        cov=np.einsum('ai,ti,bi->tab',d,var,d);direction=-d
    else:
        b,bs=payload["root_meter"];direction=np.eye(n)[vpos]
        y=(np.asarray(b)**2-bs**2)[:,None]-(obs["v"][:,vpos]**2-sig["v"][:,vpos]**2)
        cov=np.zeros((len(y),len(vpos),len(vpos)))+(4*bs**2+2*bs**4)
        for j,k in enumerate(vpos):cov[:,j,j]+=var[:,k]
    g=2*np.einsum('ae,je->aje',direction@candidate.M,candidate.M);zero=np.zeros_like(g)
    basis=np.concatenate((np.concatenate((g,zero),axis=1),np.concatenate((zero,g),axis=1)),axis=2)
    transform,_=line_map(candidate,payload);basis=basis@transform
    return dict(x=np.c_[obs["p"],obs["q"]],y=y,sigma_x=np.c_[sig["p"],sig["q"]],
                cov_y=cov,n=n,count=len(y),basis=basis,transform=transform)


def constraint(width):
    d=width//2
    return np.c_[np.eye(d),np.eye(d)]


_QP_ENV = None


def solve_quadratic(hessian,linear,constant,initial,*,certify=True):
    width=len(linear);a=constraint(width);scale=max(1.,float(np.linalg.norm(hessian,2)))
    def fg(z):return (.5*z@hessian@z+linear@z+constant)/scale,(hessian@z+linear)/scale
    result=minimize(fg,project(initial),jac=True,method="SLSQP",bounds=Bounds(0,1),
        constraints=LinearConstraint(a,-np.inf,1.),options=dict(maxiter=600,ftol=1e-13))
    point=project(result.x)
    polished,ok=_polish_qp(point,hessian,linear,a,np.ones(width//2),np.zeros(width),np.ones(width),1.)
    if ok and fg(polished)[0]<=fg(point)[0]+1e-12:point=project(polished)
    fallback=False
    gradient=hessian@point+linear
    if certify and gradient@point-linear_support(gradient)>1e-6:
        # The project already uses Gurobi. Weakly identified full-tree QPs can
        # satisfy SLSQP's change test before their verifiable gap is small.
        # A convex QP fallback repairs the solve, without changing its model.
        import gurobipy as gp
        global _QP_ENV
        if _QP_ENV is None:
            _QP_ENV=gp.Env(empty=True);_QP_ENV.setParam('OutputFlag',0);_QP_ENV.start()
        with gp.Model(env=_QP_ENV) as model:
            model.Params.Threads=1;model.Params.OptimalityTol=1e-9
            model.Params.FeasibilityTol=1e-9;model.Params.BarConvTol=1e-12
            model.Params.NonConvex=0
            z=model.addMVar(width,lb=0,ub=1)
            model.addConstr(a@z<=1)
            model.setObjective(.5*z@hessian@z+linear@z+constant)
            model.optimize()
            if model.Status==gp.GRB.OPTIMAL:
                point=project(z.X);fallback=True
                polished,ok=_polish_qp(point,hessian,linear,a,np.ones(width//2),np.zeros(width),np.ones(width),1.)
                if ok and fg(polished)[0]<=fg(point)[0]+1e-12:point=project(polished)
    gradient=hessian@point+linear;value=.5*point@hessian@point+linear@point+constant
    lower=float(value+linear_support(gradient)-gradient@point)
    return point,dict(value=float(value),lower_bound=lower,absolute_gap=max(0.,float(value-lower)),
        success=bool(result.success),iterations=int(result.nit),message=str(result.message),convex_solver_fallback=fallback)


def quadratic(data,covariance,initial):
    design=np.einsum('aje,tj->tae',data['basis'],data['x'])
    invd=np.linalg.solve(covariance,design);invy=np.linalg.solve(covariance,data['y'][...,None])[...,0]
    hess=2*np.einsum('tae,taf->ef',design,invd);linear=-2*np.einsum('tae,ta->e',design,invy)
    const=float(np.sum(data['y']*invy))
    point,info=solve_quadratic(hess,linear,const,initial)
    residual=data['y']-np.einsum('tae,e->ta',design,point)
    exact=float(np.sum(residual*np.linalg.solve(covariance,residual[...,None])[...,0]))
    info.update(value=exact,absolute_gap=max(0.,exact-info['lower_bound']),quadratic_recompute_error=abs(exact-(.5*point@hess@point+linear@point+const)))
    info['qualified']=info['absolute_gap']<=max(1e-5,1e-7*max(1.,exact))
    return point,info


def fit_qp(candidate,payload):
    started=perf_counter();data=prepare(candidate,payload);width=data['basis'].shape[2]
    initial,initinfo=quadratic(data,data['cov_y'],np.zeros(width))
    h=np.einsum('aje,e->aj',data['basis'],initial)
    covariance=data['cov_y']+np.einsum('ai,ti,bi->tab',h,data['sigma_x']**2,h)
    point,info=quadratic(data,covariance,initial);r,x=np.split(data['transform']@point,2)
    return dict(info,theta=point,r=r,x=x,loss=info['value'],J=info['value']/2,
        covariance=covariance,initial_theta=initial,initial_qp=initinfo,
        seconds=perf_counter()-started,ac_calls=0,truth_used=False,AC_lower_bound=False,
        status='qualified' if info['qualified'] else 'unqualified')


def fit_profile(candidate,payload,initial,*,seed=0,multistart=4,max_iter=500):
    started=perf_counter();data=prepare(candidate,payload);a=constraint(len(initial));attempts=[];records=[]
    rng=np.random.default_rng(seed)
    starts=[initial]+[project(initial+rng.normal(0,.08,len(initial))) for _ in range(multistart-1)]
    def fg(z):
        value=_profile_h(data,np.einsum('aje,e->aj',data['basis'],z))
        grad=np.einsum('aj,aje->e',value['gradient_H'],data['basis'])
        return value['loss']/data['count'],grad/data['count']
    for start in starts:
        result=minimize(fg,start,jac=True,method='SLSQP',bounds=Bounds(0,1),
            constraints=LinearConstraint(a,-np.inf,1.),options=dict(ftol=1e-12,maxiter=max_iter))
        point=project(result.x);value,gradient=fg(point);pg=float(np.max(abs(project(point-gradient*data['count'])-point)))
        attempts.append(dict(loss=value*data['count'],projected_gradient=pg,iterations=int(result.nit),
                             theta=point,success=bool(result.success),message=str(result.message)))
        records.append((value,point))
    _,point=min(records,key=lambda v:v[0]);polish_trace=[]
    for _ in range(24):
        profile=_profile_h(data,np.einsum('aje,e->aj',data['basis'],point))
        gradient=np.einsum('aj,aje->e',profile['gradient_H'],data['basis'])
        polish_trace.append(profile['loss'])
        if np.max(abs(project(point-gradient)-point))<=1e-5:break
        # Profiled Gauss--Newton: analytically eliminate latent P/Q in the
        # quadratic step too. This repairs numerical stops, not the model.
        design=np.einsum('aje,tj->tae',data['basis'],profile['latent_x'])
        weighted=np.linalg.solve(profile['covariance'],design)
        hess=2*np.einsum('tae,taf->ef',design,weighted)+1e-7*np.eye(len(point))
        target,_=solve_quadratic(hess,gradient-hess@point,0.,point)
        moved=False
        for power in range(16):
            trial=project(point+(target-point)*(.5**power))
            value=fg(trial)[0]*data['count']
            if value<profile['loss']:
                point=trial;moved=True;break
        if not moved:break
    profile=_profile_h(data,np.einsum('aje,e->aj',data['basis'],point))
    r,x=np.split(data['transform']@point,2);gradient=np.einsum('aj,aje->e',profile['gradient_H'],data['basis'])
    pg=float(np.max(abs(project(point-gradient)-point)))
    return dict(loss=profile['loss'],J=profile['J'],theta=point,r=r,x=x,p=profile['p'],q=profile['q'],
                projected_gradient=pg,qualified=pg<=1e-4,status='qualified' if pg<=1e-4 else 'unqualified',
                attempts=attempts,polish_trace=polish_trace,seconds=perf_counter()-started,ac_calls=0,truth_used=False)


def score_fixed(candidate,payload,r,x,*,covariance=None):
    data=prepare(candidate,payload)
    # Build H directly from R/X; no inverse or truth-specific parameter map.
    if payload['root_meter'] is None:direction=-contrast_matrix(candidate.n)
    else:direction=np.eye(candidate.n)[payload['voltage_positions']]
    h=2*np.c_[(direction@candidate.M*np.asarray(r))@candidate.M.T,
               (direction@candidate.M*np.asarray(x))@candidate.M.T]
    profile=_profile_h(data,h,derivatives=False)
    if covariance is not None:
        cov=np.broadcast_to(np.asarray(covariance)[0],(data['count'],*np.shape(covariance)[1:]))
        residual=profile['residual'];loss=float(np.sum(residual*np.linalg.solve(cov,residual[...,None])[...,0]))
        profile.update(loss=loss,J=loss/2)
    return dict(loss=profile['loss'],J=profile['J'],qualified=True,lines_updated=False,
                p=profile['p'],q=profile['q'],truth_used=False)
