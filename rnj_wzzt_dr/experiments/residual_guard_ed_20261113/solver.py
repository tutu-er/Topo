"""Direct bounded QCQP, with optional two-sided historical-residual guards."""
from time import perf_counter
import numpy as np
from ..linear_ac_population_20261113.solver import solve as unbounded,finite_attribute
from ..bidirectional_price_dr_20261113.solver import box_upper
from .model import Evidence,along


def solve(base,gamma,alpha,*,means=None,seconds=.8,env=None,warm=None,previous_lower=0.):
    began=perf_counter();evidence=Evidence(base,means);required=evidence.thresholds(gamma)
    nominal=unbounded(base,gamma,seconds=0.,env=env);lower=max(previous_lower,nominal.get('lower_bound') or 0.)
    def failed(status,proof=False,**extra):
        return dict(qualified=False,status=status,infeasibility_proved=proof,cost=None,lower_bound=lower,
            relative_gap=None,global_optimality_verified=False,seconds=perf_counter()-began,
            guard_count=len(evidence.pairs),**extra)
    if any(np.all(M==0) for M in base.M):return failed('uncontrollable_competitor',True)
    upper=box_upper(base,alpha)
    if lower>alpha**2+1e-10*max(1.,alpha**2):return failed('cost_lower_bound_proves_infeasible',True)
    if min(upper)<gamma-1e-10*max(1.,gamma):return failed('box_upper_proves_infeasible',True)
    candidates=[]
    def add(name,x):
        if x is None:return
        x=np.asarray(x).reshape(base.shape)
        if np.max(abs(x))<=alpha+1e-9 and min(evidence.values(x)-required)>=-1e-8:
            candidates.append((name,x.copy()))
    add('reused_verified_previous_quote',warm)
    if nominal['qualified']:
        add(nominal['source'],nominal['x']);add('clipped_analytic_seed',np.clip(nominal['x'],-alpha,alpha))
    def cheapest():return min(candidates,key=lambda p:base.norm_cost(p[1])) if candidates else None
    best=cheapest()
    def result(best,native,initial_cost,initial_source,source):
        x=best[1];values=evidence.direct(x);cost=base.norm_cost(x)/base.dimension
        if min(values-required)<-1e-8 or np.max(abs(x))>alpha+1e-8:raise ArithmeticError('Original evidence/capacity violation')
        gap=max(0.,cost-lower)/max(cost,1e-30)
        return dict(qualified=True,x=x,status='qualified',source=source,cost=cost,lower_bound=lower,
            relative_gap=gap,global_optimality_verified=gap<=1e-5,infeasibility_proved=False,
            seconds=perf_counter()-began,gains=values[:len(base.M)],guard_gains=values[len(base.M):],
            guard_count=len(evidence.pairs),native=native,initialization_cost=initial_cost,
            initialization_source=initial_source,initialization_verified=initial_cost is not None)
    if best:
        cost=base.norm_cost(best[1])/base.dimension
        if (cost-lower)/max(cost,1e-30)<=1e-5:return result(best,None,None,None,best[0])
    directions=[np.sum(base.a,axis=0),np.sum(evidence.a,axis=0)]
    if nominal['qualified']:directions.insert(0,nominal['x'])
    directions.extend(evidence.a[:12]);directions.extend([np.ones(base.shape),np.sin(np.arange(base.dimension).reshape(base.shape)*np.sqrt(2)+.3)])
    for direction in directions:
        for sign in (1.,-1.):
            add('verified_direction_seed',along(evidence,sign*direction,required,alpha))
            corner=alpha*np.where(sign*direction>=0,1.,-1.)
            add('verified_box_direction_seed',along(evidence,corner,required,alpha))
    best=cheapest();source=None if best is None else best[0]
    initial_cost=None if best is None else base.norm_cost(best[1])/base.dimension
    initial_source=source
    if best and (initial_cost-lower)/max(initial_cost,1e-30)<=1e-5:
        return result(best,None,initial_cost,initial_source,source)
    import gurobipy as gp
    owns=env is None
    if owns:env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    native=None
    try:
        with gp.Model('residual_guard_QCQP',env=env) as m:
            for key,value in dict(OutputFlag=0,Threads=1,NonConvex=2,MIPGap=1e-5,FeasibilityTol=1e-9,NumericFocus=1).items():m.setParam(key,value)
            v=m.addMVar(base.shape,lb=-alpha,ub=alpha);m.setObjective((v*v).sum())
            for Q,a,rhs in zip(evidence.Q,evidence.a,required,strict=True):
                quadratic=sum(v[h]@Q@v[h] for h in range(24))
                m.addConstr((quadratic+2*(a*v).sum())/gamma>=rhs/gamma)
            if best:v.Start=best[1]
            m.Params.TimeLimit=max(.001,seconds-(perf_counter()-began));m.optimize()
            native=dict(status=int(m.Status),solutions=int(m.SolCount),bound=finite_attribute(m,'ObjBound'),
                gap=finite_attribute(m,'MIPGap') if m.SolCount else None)
            if m.SolCount:
                point=np.asarray(v.X)
                if np.max(abs(point))<=alpha+1e-8 and min(evidence.direct(point)-required)>=-1e-8:
                    if best is None or base.norm_cost(point)<base.norm_cost(best[1]):best=('native_QCQP_incumbent',point);source=best[0]
            if best is not None and native['bound'] is not None:
                cost=base.norm_cost(best[1])/base.dimension;bound=native['bound']/base.dimension
                if bound<=cost+1e-7*max(cost,1e-20):lower=max(lower,min(bound,cost))
                else:native['bound_withdrawn']=True
            if m.Status==gp.GRB.INFEASIBLE:
                if best is None:return failed('native_proves_infeasible',True,native=native)
                native['infeasibility_withdrawn_verified_seed_exists']=True
    finally:
        if owns:env.dispose()
    if best is None:return failed('time_limit_no_quote',native=native)
    return result(best,native,initial_cost,initial_source,source)
