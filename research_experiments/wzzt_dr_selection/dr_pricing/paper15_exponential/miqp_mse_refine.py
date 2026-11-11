"""One feasible MIQP refinement, with original nonconvex global bound retained.

For convex score q(u)=||b+Fu||², q(u)>=q(uk)+grad q(uk)'(u-uk).
Requiring this affine lower bound>=h guarantees the original probability
sufficient condition. The anchor is feasible, so the inner optimum cannot
cost more. Its bound applies to the INNER problem only. The earlier global
nonconvex bound is kept separately; this is not a global optimality proof.
"""
from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
import miqp_mse as mse
from vendor.solver import build_device_graph,_values,_attribute


def refine(case,parent,out):
    import gurobipy as gp
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'request.json').exists():raise ValueError('Existing refinement preserved')
    parent=Path(parent).resolve();previous=mse.base.read(parent/'summary.json')
    tau=previous['tau'];case,cfg,p,a,g=mse.build(case,tau,full=True)
    saved=mse.base.arrays(parent/'quotes.npz');anchor=saved['optimized'];warm=saved['optimized_hourly_requests']
    if not g.bound(anchor)['sufficient']:raise ValueError('A feasible anchor is required')
    gradient=2*g.factor.T@(g.offset+g.factor@anchor[g.resources].ravel())
    rhs=float(g.h-g.value(anchor)+gradient@anchor[g.resources].ravel())
    request=copy.deepcopy(mse.base.read(parent/'request.json'))
    request.update(parent_plan=mse.base.ref(parent/'policy_freeze.json'),
        refinement='one affine lower bound at the frozen feasible anchor',inner_seconds=60.,
        global_lower_bound=previous['solver']['lower_bound'])
    request['sources']['miqp_mse_refine.py']=mse.base.sha(__file__)
    mse.base.write(out/'request.json',request)
    # Exact separable zero-effect reduction, not a resource-capacity change.
    opt=copy.copy(p);opt.device=copy.copy(p.device)
    null=sorted(set(range(p.device.R))-set(g.resources.tolist()))
    for name in ('delta_lower_kw','delta_upper_kw','scale'):
        value=getattr(p.device,name).copy();value[null]=0.;setattr(opt.device,name,value)
    opt.lower=p.lower.copy();opt.upper=p.upper.copy()
    mask=np.isin(p.active//p.device.H,null);opt.lower[mask]=0.;opt.upper[mask]=0.
    began=time.perf_counter()
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        with gp.Model('physical_MSE_inner_MIQP',env=env) as model:
            for key,value in dict(OutputFlag=1,LogToConsole=0,LogFile=str(out/'gurobi.log'),
                TimeLimit=60.,MIPGap=.02,Threads=1,Seed=0,NonConvex=0,MIPFocus=1,
                FeasibilityTol=1e-8,IntFeasTol=1e-8,OptimalityTol=1e-8,DualReductions=0).items():model.setParam(key,value)
            x,u,cost=build_device_graph(model,opt,warm)
            linear=gp.LinExpr()
            for coeff,term in zip(gradient,u[g.resources].ravel()):
                if coeff!=0:linear+=float(coeff)*term.expr
            model.addConstr(linear>=rhs)
            model.setObjective(cost,gp.GRB.MINIMIZE);model.update();model.write(str(out/'model.mps.gz'))
            model.optimize()
            status=dict(status=int(model.Status),has_solution=bool(model.SolCount),
                        inner_lower_bound=_attribute(model,'ObjBound'),
                        inner_gap=_attribute(model,'MIPGap') if model.SolCount else None,
                        seconds=time.perf_counter()-began)
            best=p.nominal(p.requests_to_values(warm));origin='anchor'
            if model.SolCount:
                requests=_values(x)
                requests=p.expand(np.clip(requests.ravel()[p.active],p.lower,p.upper))
                point=p.nominal(p.requests_to_values(requests))
                if (p.audit_delivery(point['mean_u'])['passed'] and g.bound(point['mean_u'])['margin']>=-1e-7
                        and point['fee']<best['fee']):best=point;origin='inner_MIQP'
    u=best['mean_u'];fee=float(best['fee']);rotate,rot=mse.base.rotation(p,fee)
    np.savez_compressed(out/'quotes.npz',optimized=u,zero=np.zeros_like(u),rotation=rotate,
                        optimized_hourly_requests=best['x'])
    mse.base.write(out/'policy_freeze.json',dict(status='frozen',
        quote=dict(path='quotes.npz',sha256=mse.base.sha(out/'quotes.npz')),
        request=dict(path='request.json',sha256=mse.base.sha(out/'request.json')),
        no_actual_future_or_truth_used=True))
    global_lower=previous['solver']['lower_bound']
    summary=dict(status='complete',tau=tau,kappa=g.kappa,K=g.K,history_S=g.history_S,delta=g.delta,
        anchor_fee=previous['fees']['optimized'],optimized_bound=g.bound(u),zero_bound=g.bound(np.zeros_like(u)),
        fees=dict(optimized=fee,zero=0.,rotation=p.device.fees(rotate)),inner_solver=status,
        global_lower_bound=global_lower,global_gap=None if global_lower is None else max(0.,fee-global_lower)/max(fee,1e-12),
        retained_source=origin,rotation=rot,device_audit=p.audit_delivery(u),
        changed_hourly_requests=int(np.sum(np.abs(best['x']-warm)>1e-8)),
        active_resources=int(np.sum(np.any(np.abs(u)>1e-8,axis=1))),
        scope='feasible inner refinement; no global optimality claim')
    mse.base.write(out/'summary.json',summary);mse.base.write(out/'complete.json',dict(status='complete'))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(refine(args.case,args.parent,args.out),allow_nan=False))
