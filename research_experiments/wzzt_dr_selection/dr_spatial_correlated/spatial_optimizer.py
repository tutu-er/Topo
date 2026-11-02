"""Capacity-fair spatial DR with the existing exact conditional L1 probability.

Sequential convex steps are only a numerical method for the original finite
probability problem. Every accepted point is checked with that probability.
No positive lower capacity bound, integer site count, or model refit is used.
"""
import sys,time
from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB

REPO=Path(r'C:/Users/23761/.codex/worktrees/wzzt-dr-selection/Topo')
LEGACY=REPO/'research_experiments/wzzt_dr_selection/dr_conditional_gaussian'
sys.path.insert(0,str(LEGACY))
from problem import ConditionalProblem
from response_law import draw,MAIN_LAW


def cost(u,a,dt=.25,beta_per_kw=.5):
    u=np.asarray(u,float);a=np.asarray(a,float)
    if u.ndim!=2 or a.shape!=(u.shape[0],) or np.any(a<0):raise ValueError('capacity shape/domain')
    if np.any(u[a==0]!=0):raise ValueError('zero capacity cannot deliver response')
    return float(dt/beta_per_kw*np.sum(np.divide(np.sum(u*u,axis=1),a,out=np.zeros_like(a),where=a>0)))


def feasible(u,a,caps,total=2.,tol=1e-7):
    u,a,caps=np.asarray(u),np.asarray(a),np.asarray(caps)
    return bool(np.isfinite(u).all() and np.isfinite(a).all() and np.all(a>=-tol) and
        np.all(a<=caps+tol) and a.sum()<=total+tol and np.all(np.abs(u)<=a[:,None]+tol))


class SpatialProbability:
    def __init__(self,packet,samples,base_seed,response_seed):
        self.core=ConditionalProblem(packet,samples,base_seed)
        all_multiplier=draw(response_seed,samples,packet['all_node_count'],packet['shape'][1],MAIN_LAW)
        self.core.multiplier=all_multiplier[:,packet['active_positions'],:]
        self.core.last_x=self.core.last_result=None
        self.shape=tuple(packet['shape'])
    def evaluate(self,u):return self.core.evaluate(np.asarray(u).ravel())


def cone_step(u0,a0,prob,gradient,caps,*,target=.95,total=2.,radius=.3,
              penalty=1000.,fixed_capacity=None,seconds=15.):
    """Exact convex perspective cost; only probability constraint is linearized."""
    m,T=u0.shape
    model=gp.Model('spatial_probability_step');model.Params.OutputFlag=0
    model.Params.Threads=1;model.Params.TimeLimit=float(seconds)
    model.Params.FeasibilityTol=1e-8;model.Params.OptimalityTol=1e-8
    a=model.addMVar(m,lb=np.zeros(m),ub=caps,name='capacity')
    if fixed_capacity is not None:
        a.lb=np.asarray(fixed_capacity);a.ub=np.asarray(fixed_capacity)
    uu=model.addMVar((m,T),lb=-caps[:,None]*np.ones((m,T)),ub=caps[:,None]*np.ones((m,T)),name='u')
    epi=model.addMVar(m,lb=0,name='energy_over_capacity')
    slack=model.addVar(lb=0,name='linearized_probability_shortfall')
    model.addConstr(a.sum()<=total)
    for i in range(m):
        model.addConstr(uu[i,:]<=a[i]);model.addConstr(uu[i,:]>=-a[i])
        model.addQConstr(gp.quicksum(uu[i,t].item()*uu[i,t].item() for t in range(T))<=a[i].item()*epi[i].item())
    model.addConstr(uu<=u0+radius);model.addConstr(uu>=u0-radius)
    linear=prob+(gradient*(uu-u0)).sum()
    model.addConstr(linear+slack>=target)
    model.setObjective(.5*epi.sum()+penalty*slack,GRB.MINIMIZE)
    model.optimize()
    info=dict(status=int(model.Status),runtime=float(model.Runtime),solutions=int(model.SolCount))
    if model.SolCount==0:
        model.dispose();return None,info
    raw_u,raw_a=np.array(uu.X),np.array(a.X)
    aa=np.clip(raw_a,0,caps)
    if fixed_capacity is not None:aa=np.asarray(fixed_capacity).copy()
    elif aa.sum()>total:aa*=total/aa.sum()
    u=np.clip(raw_u,-aa[:,None],aa[:,None]);u[aa==0]=0.
    info.update(objective=float(model.ObjVal),slack=float(slack.X),
        projection_max=float(max(np.max(abs(u-raw_u)),np.max(abs(aa-raw_a)))))
    model.dispose();return (u,aa),info


def solve_local(problem,caps,start_u,start_a,settings,*,fixed_capacity=None,callback=None):
    u=np.asarray(start_u,float).copy();a=np.asarray(start_a,float).copy()
    if not feasible(u,a,caps):raise ValueError('infeasible numerical start')
    started=time.perf_counter();radius=settings['initial_radius'];penalty=settings['penalty'];target=settings['target']
    trace=[];candidates=[];status='iteration_limit'
    def record(u,a,p):
        if not feasible(u,a,caps):raise AssertionError('accepted infeasible resources')
        candidates.append(dict(mean_u=u.tolist(),capacity_kw=a.tolist(),cost=cost(u,a),planning_probability=float(p)))
    initial=problem.evaluate(u)
    if initial['gradient'] is None:raise ArithmeticError('invalid initial probability gradient')
    record(u,a,initial['probability'])
    for it in range(settings['maxiter']):
        if time.perf_counter()-started>=settings['seconds']:status='time_limit';break
        r=problem.evaluate(u);p=r['probability'];g=r['gradient']
        if g is None:status='probability_gradient_unresolved';break
        remaining=settings['seconds']-(time.perf_counter()-started)
        if remaining<=0:status='time_limit';break
        step,info=cone_step(u,a,p,g.reshape(u.shape),caps,target=target,radius=radius,penalty=penalty,
            fixed_capacity=fixed_capacity,seconds=min(15.,remaining))
        row=dict(iteration=it+1,probability=p,cost=cost(u,a),radius=radius,conic=info)
        if step is None:
            trace.append(dict(row,accepted=False));status='conic_step_unresolved';break
        v,b=step;rv=problem.evaluate(v);pv=rv['probability'];cv=cost(v,b)
        old_merit=cost(u,a)+penalty*max(target-p,0.)
        new_merit=cv+penalty*max(target-pv,0.)
        predicted=cost(u,a)+penalty*max(target-p,0.)-(cv+penalty*max(target-(p+float(np.sum(g.reshape(u.shape)*(v-u)))),0.))
        actual=old_merit-new_merit;ratio=actual/max(predicted,1e-12)
        accepted=bool(rv['gradient'] is not None and actual>1e-8)
        # Feasible evaluated candidates are retained, even if a trust step is rejected.
        if rv['gradient'] is not None:record(v,b,pv)
        row.update(trial_probability=pv,trial_cost=cv,accepted=accepted,merit_reduction=actual,ratio=ratio,
                   elapsed=time.perf_counter()-started)
        trace.append(row)
        if accepted:
            move=float(np.max(abs(v-u)));u,a=v,b
            if ratio>.75:radius=min(settings['max_radius'],radius*1.5)
            elif ratio<.25:radius*=.5
            if move<1e-5 and abs(actual)<1e-5:status='small_step';break
        else:radius*=.5
        if callback is not None:callback(trace,candidates)
        if radius<settings['min_radius']:status='trust_region_stalled';break
    valid=[c for c in candidates if c['planning_probability']>=target-1e-7]
    best=min(valid,key=lambda c:c['cost']) if valid else max(candidates,key=lambda c:(c['planning_probability'],-c['cost']))
    return dict(status=status,elapsed=time.perf_counter()-started,trace=trace,best=best,
        planning_feasible=bool(valid),returned_candidates=len(candidates),global_optimum_proven=False,
        algorithm='sequential conic steps on original finite conditional probability; exact merit acceptance')
