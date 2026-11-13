"""Original QCQP with signed box capacity; no ADMM or device-state reconstruction."""
from time import perf_counter
import numpy as np
from ..linear_ac_population_20261113.solver import solve as unbounded,finite_attribute,along


def box_upper(model,alpha):
    spectral=np.array([float(q.max()) for q,V in model.spectral])
    first=2*alpha*np.sum(abs(model.a),axis=(1,2))+spectral*alpha**2*model.dimension
    radius=alpha*np.sum(abs(model.M),axis=2)
    second=np.sum(2*abs(model.b)*radius[:,None]+radius[:,None]**2,axis=(1,2))
    return np.minimum(first,second)


def solve(model,gamma,alpha,*,seconds=.4,env=None):
    if alpha is None:return unbounded(model,gamma,seconds=min(seconds,.15),env=env)
    began=perf_counter();initial=unbounded(model,gamma,seconds=0.,env=env)
    upper=box_upper(model,alpha)
    if initial.get('lower_bound') is not None and initial['lower_bound']>alpha**2+1e-10*max(1.,alpha**2):
        return dict(qualified=False,status='cost_lower_bound_proves_infeasible',source='unbounded_lower_bound_exceeds_box_max_cost',
            seconds=perf_counter()-began,cost=None,lower_bound=initial['lower_bound'],relative_gap=None,
            global_optimality_verified=False,infeasibility_proved=True,gain_upper=upper,alpha=alpha)
    if np.any(upper<gamma-1e-10*max(1.,gamma)):
        return dict(qualified=False,status='box_upper_proves_infeasible',source='valid_outer_box_bound',
            seconds=perf_counter()-began,cost=None,lower_bound=None,relative_gap=None,
            global_optimality_verified=False,infeasibility_proved=True,gain_upper=upper,alpha=alpha)
    if initial['qualified'] and np.max(abs(initial['x']))<=alpha+1e-10:
        initial.update(alpha=alpha,infeasibility_proved=False,seconds=perf_counter()-began)
        return initial
    lower=initial.get('lower_bound') or 0.;candidates=[]
    if initial['qualified']:
        d=initial['x'];clip=np.clip(d,-alpha,alpha)
        if min(model.gains(clip))>=gamma-1e-8:candidates.append(('clipped_unbounded',clip))
    else:d=np.sin(np.arange(model.dimension)).reshape(model.shape)
    directions=[d,-d,np.sum(model.a,axis=0)]
    directions.extend(model.a)
    directions.extend([np.sin(np.arange(model.dimension).reshape(model.shape)*np.sqrt(2)+.3),np.ones(model.shape)])
    for direction in directions:
        for sign in (1.,-1.):
            corner=alpha*np.where(sign*direction>=0,1.,-1.)
            point=along(model,corner,gamma)
            if point is not None and np.max(abs(point))<=alpha+1e-10 and min(model.gains(point))>=gamma-1e-8:
                candidates.append(('bounded_direction_seed',point))
    best=min(candidates,key=lambda p:model.norm_cost(p[1])) if candidates else None
    seed_cost=None if best is None else model.norm_cost(best[1])/model.dimension
    if best and (seed_cost-lower)/max(seed_cost,1e-30)<=1e-5:
        x=best[1];return dict(qualified=True,x=x,status='qualified',source=best[0],seconds=perf_counter()-began,
            cost=seed_cost,lower_bound=lower,relative_gap=max(0.,seed_cost-lower)/max(seed_cost,1e-30),
            global_optimality_verified=True,gains=model.direct_gains(x),alpha=alpha,infeasibility_proved=False,
            initialization_cost=seed_cost,native=None)
    import gurobipy as gp
    owns=env is None
    if owns:
        env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    native=None;source=None if best is None else best[0]
    try:
        with gp.Model('bounded_bidirectional_QCQP',env=env) as m:
            for k,v in dict(OutputFlag=0,Threads=1,NonConvex=2,MIPGap=.01,FeasibilityTol=1e-8,NumericFocus=1).items():m.setParam(k,v)
            v=m.addMVar(model.shape,lb=-alpha,ub=alpha)
            # First obtain a cheap convex sufficient-condition start at zero.
            linear=[]
            for j in range(len(model.M)):linear.append(m.addConstr(2*(model.a[j]*v).sum()>=gamma*(1+1e-8)))
            m.setObjective((v*v).sum());m.Params.TimeLimit=max(.001,min(.025,seconds-(perf_counter()-began)))
            m.optimize()
            qp_status=int(m.Status)
            if m.SolCount:
                x=np.asarray(v.X)
                if min(model.gains(x))>=gamma-1e-8 and (best is None or model.norm_cost(x)<model.norm_cost(best[1])):
                    best=('zero_supporting_plane_QP_seed',x);source=best[0]
                    seed_cost=model.norm_cost(x)/model.dimension
            m.remove(linear);m.update()
            for j in range(len(model.M)):
                quadratic=sum(v[h]@model.Q[j]@v[h] for h in range(24))
                m.addConstr((quadratic+2*(model.a[j]*v).sum())/gamma>=1.)
            if best:v.Start=best[1]
            m.Params.TimeLimit=max(.001,seconds-(perf_counter()-began));m.optimize()
            bound=finite_attribute(m,'ObjBound');gap=finite_attribute(m,'MIPGap') if m.SolCount else None
            native=dict(status=int(m.Status),solutions=int(m.SolCount),bound=bound,gap=gap,sufficient_QP_status=qp_status)
            if m.SolCount:
                x=np.asarray(v.X)
                if np.max(abs(x))<=alpha+1e-8 and min(model.gains(x))>=gamma-1e-8 and (best is None or model.norm_cost(x)<model.norm_cost(best[1])):
                    best=('native_bounded_QCQP_incumbent',x);source=best[0]
            if best is not None and bound is not None:
                cost=model.norm_cost(best[1])/model.dimension
                if bound/model.dimension<=cost+1e-7*max(cost,1e-20):lower=max(lower,bound/model.dimension)
                else:native['bound_withdrawn']=True
            if m.Status==gp.GRB.INFEASIBLE and best is None:
                return dict(qualified=False,status='native_proves_infeasible',source='native_global_box_QCQP',
                    seconds=perf_counter()-began,cost=None,lower_bound=None,relative_gap=None,
                    global_optimality_verified=False,infeasibility_proved=True,native=native,alpha=alpha,gain_upper=upper)
            if m.Status==gp.GRB.INFEASIBLE and best is not None:native['infeasibility_withdrawn_verified_seed_exists']=True
    finally:
        if owns:env.dispose()
    if best is None:
        return dict(qualified=False,status='time_limit_no_quote',seconds=perf_counter()-began,cost=None,lower_bound=lower,
            relative_gap=None,global_optimality_verified=False,infeasibility_proved=False,native=native,alpha=alpha,gain_upper=upper)
    x=best[1];gains=model.direct_gains(x);cost=model.norm_cost(x)/model.dimension
    if min(gains)<gamma-1e-8 or np.max(abs(x))>alpha+1e-8:raise ArithmeticError('Bounded quote violates original model')
    gap=max(0.,cost-lower)/max(cost,1e-30)
    return dict(qualified=True,x=x,status='qualified',source=source,cost=cost,lower_bound=lower,relative_gap=gap,
        global_optimality_verified=gap<=1e-5,seconds=perf_counter()-began,native=native,alpha=alpha,
        initialization_cost=seed_cost,gains=gains,infeasibility_proved=False)
