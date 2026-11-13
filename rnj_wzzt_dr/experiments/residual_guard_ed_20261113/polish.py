"""Feasible tangent-QP improvements on the SAME old nominal QCQP problems.

Tangent constraints are sufficient because each original gain is convex.
Their QP bounds are never used as original-problem lower bounds.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from time import perf_counter
import argparse,traceback
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,utc
from ..linear_ac_population_20261113.solver import Model


def improve(model,x,gamma,alpha,seconds=1.,iterations=20):
    import gurobipy as gp
    point=np.array(x,copy=True);began=perf_counter();trace=[];initial=model.norm_cost(point)/model.dimension
    require(min(model.direct_gains(point))>=gamma-1e-8 and np.max(abs(point))<=alpha+1e-8,'Initial point is not feasible')
    env=gp.Env(empty=True);env.setParam('OutputFlag',0);env.start()
    try:
        with gp.Model('feasible_tangent_QP',env=env) as m:
            for k,v in dict(OutputFlag=0,Threads=1,NonConvex=0,FeasibilityTol=1e-9,OptimalityTol=1e-9,NumericFocus=1).items():m.setParam(k,v)
            y=m.addMVar(model.shape,lb=-alpha,ub=alpha);m.setObjective((y*y).sum());constraints=[]
            for iteration in range(iterations):
                if perf_counter()-began>=seconds:break
                m.remove(constraints);m.update();constraints=[]
                for Q,a in zip(model.Q,model.a,strict=True):
                    gradient=2*(point@Q+a);constant=float(np.einsum('hi,ij,hj',point,Q,point))
                    constraints.append(m.addConstr((gradient*y).sum()>=gamma+constant))
                y.Start=point;m.Params.TimeLimit=max(.001,seconds-(perf_counter()-began));m.optimize()
                if not m.SolCount:trace.append(dict(iteration=iteration,status=int(m.Status),accepted=False));break
                candidate=np.array(y.X);old=model.norm_cost(point);new=model.norm_cost(candidate)
                feasible=min(model.direct_gains(candidate))>=gamma-1e-8 and np.max(abs(candidate))<=alpha+1e-8
                accepted=feasible and new<=old+1e-12*max(old,1e-20)
                trace.append(dict(iteration=iteration,status=int(m.Status),accepted=accepted,cost=new/model.dimension))
                if not accepted:break
                point=candidate
                if old-new<=1e-9*max(old,1e-20):break
    finally:env.dispose()
    return point,dict(initial_cost=initial,cost=model.norm_cost(point)/model.dimension,seconds=perf_counter()-began,
        iterations=len(trace),trace=trace,lower_bound_from_QP_used=False,initialization_verified=True,
        guarantee='original nominal constraints feasible and fee nonincreasing; local tangent iteration, not a global certificate')


def worker(task):
    out,oldrun,item=task;e=item['entry'];s=io.read(e['scores']['path']);H=np.array([p['H'] for p in s['parameters']])
    with np.load(item['planner']['path'],allow_pickle=False) as p:F=p['forecast'].copy()
    with np.load(Path(oldrun)/'jobs'/item['key']/'controls.npz',allow_pickle=False) as p:
        keys=p.files;key='seconds30' if 'seconds30' in keys else 'seconds10';x=p[key].copy()
    old=io.read(Path(oldrun)/'jobs'/item['key']/'complete.json');base=Model(H,e['target'],e['competitors'],e['scale'],s['sigma2']).prepare(F)
    gamma=2. if item['label']=='g2' else 4.;point,result=improve(base,x,gamma,item['alpha'])
    # The carried bound is valid for the original QCQP; no tangent-QP dual bound
    # is promoted to a bound on the larger original feasible set.
    lower=old['new_lower'];gap=max(0.,result['cost']-lower)/max(result['cost'],1e-30)
    result.update(key=item['key'],id=item['id'],case=e['request']['case'],old_04_cost=item['old_plan']['cost'],
        long_native_cost=old['new_cost'],lower_bound=lower,relative_gap=gap,global_gap_closed=gap<=1e-5,
        original_to_polished_improvement=(item['old_plan']['cost']-result['cost'])/item['old_plan']['cost'])
    directory=Path(out)/'jobs'/item['key'];directory.mkdir(parents=True)
    np.savez_compressed(directory/'point.npz',x=point);io.write(directory/'result.json',result);return result


def run(old,out):
    require(not out.exists(),'Preserve previous improvements');out.mkdir(parents=True);cfg=io.read(old/'protocol.json')
    io.write(out/'protocol.json',dict(created_utc=utc(),source=io.ref(__file__),previous=io.ref(old/'protocol.json'),
        same_12_problems=True,seconds_per_problem=1.,maximum_iterations=20,local_only=True,new_independent_days=0))
    records=[];failures=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(worker,(str(out),str(old),item)):item for item in cfg['selection']}
        for f in as_completed(futures):
            try:records.append(f.result())
            except Exception as exc:failures.append(dict(key=futures[f]['key'],error=repr(exc),traceback=traceback.format_exc()))
    io.write(out/'summary.json',records);io.write(out/'complete.json',dict(status='passed' if not failures else 'complete_with_failures',
        completed=len(records),problems=12,failures=failures,new_independent_days=0,source=io.ref(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.previous.resolve(),a.output.resolve())
