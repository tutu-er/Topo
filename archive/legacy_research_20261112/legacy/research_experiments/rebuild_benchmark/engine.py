"""Shared Gurobi backend, original forward MILP and reversible local MILP.

All native models use P/Q/Y multiplied by 1000. Public point objectives and
solver bounds use the original squared-voltage units. Model construction,
warm-start LPs and native solves count against the search wall budget.
"""
from __future__ import annotations

from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix

from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import build_model, canonical, compatible, frontiers
from research_experiments.wzzt_retune.engine import PoolLP, extension_start

SCALE=1000.
LP_SECONDS=5.


def loss(data, point):
    R,X=core.build_matrices_from_atoms(data.n,point['supports'],point['r'],point['x'])
    return sum(float(abs(y-p@R.T-q@X.T).sum()) for p,q,y in zip(data.p,data.q,data.target))/data.observation_count


def prepare(obs):
    p,q,v,root=obs
    return core._prepare_scenarios([dict(P_terminal=p,Q_terminal=q,
        drop_target=root.to_numpy()[:,None]**2-v.to_numpy()**2)])


class Bank:
    def __init__(self, data, pool):
        self.data=data
        self.scaled=core._PreparedScenarios(data.labels,tuple(p*SCALE for p in data.p),
            tuple(q*SCALE for q in data.q),tuple(y*SCALE for y in data.target))
        self.lp=PoolLP(None,pool,'raw',bound=2.,data=(self.scaled.p[0],self.scaled.q[0],self.scaled.target[0]))
        self.lp.model.Params.TimeLimit=LP_SECONDS
        self.failures=[]

    def extend(self,supports):
        missing=set(canonical(supports))-set(self.lp.pool)
        if not missing:return
        cache,solves=self.lp.cache,self.lp.solves
        other=PoolLP(None,[*self.lp.pool,*missing],'raw',bound=2.,data=(self.lp.p,self.lp.q,self.lp.y))
        self.lp.close();self.lp=other
        self.lp.model.Params.TimeLimit=LP_SECONDS
        self.lp.cache.update(cache);self.lp.solves=solves

    def fit(self,family,*,preserve=False):
        family=canonical(family);self.extend(family)
        try:q=self.lp.fit(family)
        except Exception as exc:
            self.failures.append(dict(status=int(self.lp.model.Status),family=family,error=str(exc)))
            raise
        if preserve:
            lookup={tuple(s):(r,x) for s,r,x in zip(q['supports'],q['r'],q['x'])}
            p=dict(supports=family,r=[lookup.get(s,(0.,0.))[0] for s in family],
                   x=[lookup.get(s,(0.,0.))[1] for s in family])
        else:p=dict(supports=canonical(q['supports']),r=q['r'],x=q['x'])
        p['objective']=loss(self.data,p)
        if abs(p['objective']-q['objective']/SCALE)>1e-8:raise RuntimeError('LP physical reproduction failed')
        return p

    def close(self):self.lp.close()


def audit_vector(model,values):
    if values is None:return dict(valid=False,has_incumbent=False)
    values=np.asarray(values)
    if values.shape!=(model.variables.size,) or not np.isfinite(values).all():
        return dict(valid=False,has_incumbent=False,error='malformed incumbent')
    rows=model.rows.build(model.variables.size);a=rows.A@values
    violation=float(max(np.max(rows.lb-a,initial=0.),np.max(a-rows.ub,initial=0.),
        np.max(np.asarray(model.variables.lower)-values,initial=0.),np.max(values-np.asarray(model.variables.upper),initial=0.)))
    integer=np.asarray(model.variables.integrality,dtype=bool)
    integral=float(np.max(abs(values[integer]-np.rint(values[integer])),initial=0.))
    return dict(valid=max(violation,integral)<=1e-7,has_incumbent=True,
                maximum_violation=violation,integrality_violation=integral,
                objective=float(np.asarray(model.variables.objective)@values)/SCALE)


def solve(bank,model,start,*,seconds):
    import gurobipy as gp
    then=perf_counter();raw=model.rows.build(model.variables.size);A=csr_matrix(raw.A)
    lower,upper=raw.lb,raw.ub;eq=np.isfinite(lower)&(lower==upper)
    types=np.where(np.asarray(model.variables.integrality,dtype=bool),gp.GRB.BINARY,gp.GRB.CONTINUOUS)
    start_audit=audit_vector(model,start)
    if not start_audit['valid']:raise RuntimeError(f'bad MIP start: {start_audit}')
    with gp.Model('paired_topology_MILP',env=bank.lp.environment) as m:
        for key,value in dict(Threads=1,OutputFlag=0,TimeLimit=max(.05,seconds),MIPGap=.001,
            MIPGapAbs=1e-7,FeasibilityTol=1e-9,IntFeasTol=1e-9,OptimalityTol=1e-9,
            DualReductions=0,Seed=20261030).items():m.setParam(key,value)
        v=m.addMVar(model.variables.size,lb=model.variables.lower,
            ub=np.where(np.isposinf(model.variables.upper),gp.GRB.INFINITY,model.variables.upper),
            obj=model.variables.objective,vtype=types)
        for mask,sense,rhs in ((eq,'=',lower),(~eq&np.isfinite(lower),'>',lower),(~eq&np.isfinite(upper),'<',upper)):
            if np.any(mask):m.addMConstr(A[mask],v,sense,rhs[mask])
        v.Start=start;m.optimize()
        values=np.asarray(v.X) if m.SolCount else None
        audit=audit_vector(model,values)
        obj=float(m.ObjVal)/SCALE if m.SolCount else None
        bound=float(m.ObjBound)/SCALE if abs(m.ObjBound)<gp.GRB.INFINITY else None
        consistent=bound is None or bound<=start_audit['objective']+1e-8
        record=dict(status=int(m.Status),objective=obj,dual_bound=bound,
            relative_gap=float(m.MIPGap) if m.SolCount else None,nodes=float(m.NodeCount),
            seconds=perf_counter()-then,variable_count=model.variables.size,
            binary_count=int(np.count_nonzero(model.variables.integrality)),constraint_count=model.rows.size,
            warm_start=start_audit,audit=audit,bound_consistent_with_known_upper=consistent,
            solver='gurobi',scale=SCALE,requested_relative_gap=.001)
        if not consistent:record['certificate_error']='bound exceeds verified feasible witness'
        return values if audit['valid'] and consistent else None,record


def hierarchy_children(family,n):
    nodes=canonical([*family,*((i,) for i in range(n)),tuple(range(n))])
    children={}
    for v in nodes:
        proper=[s for s in nodes if set(s)<set(v)]
        children[v]=canonical(s for s in proper if not any(set(s)<set(t) for t in proper))
    return children


def regions(family,n,b,kernel):
    candidates=[]
    for v,blocks in frontiers(family,n,max_blocks=b):candidates.append((v,blocks,False))
    # A large polytomy still offers legal unions of sibling subtrees. Choose
    # bounded neighborhoods by training-only RNJ distances, never by truth.
    D=np.diag(kernel)[:,None]+np.diag(kernel)[None,:]-2*kernel
    for v,children in hierarchy_children(family,n).items():
        if len(children)<=b:continue
        for child in children:
            other=sorted((s for s in children if s!=child),
                key=lambda s:(float(D[np.ix_(child,s)].mean()),s))
            blocks=canonical([child,*other[:b-1]])
            region=tuple(sorted(i for s in blocks for i in s))
            candidates.append((region,blocks,True))
    candidates=sorted(set(candidates),key=lambda q:(-len(q[0]),-len(q[1]),q[0],q[1]))
    # Round robin distinct regions before alternative frontiers of one region.
    first,rest,seen=[],[],set()
    for q in candidates:
        if q[0] in seen:rest.append(q)
        else:first.append(q);seen.add(q[0])
    return first+rest


def any_extension(family,n,pool):
    f=set(canonical(family))
    for c in canonical(pool):
        if c not in f and all(compatible(c,s) for s in f):return c
    for v,children in hierarchy_children(family,n).items():
        for left,right in combinations(children,2):
            c=tuple(sorted([*left,*right]))
            if c not in f and all(compatible(c,s) for s in f):return c
    return None


def reconstruction_start(bank,model,meta,point):
    values=np.zeros(model.variables.size)
    lookup={tuple(s):(r,x) for s,r,x in zip(point['supports'],point['r'],point['x'])}
    for k,s in enumerate(meta['all_supports']):values[model.r.start+k],values[model.x.start+k]=lookup.get(s,(0.,0.))
    for k,s in enumerate(meta['candidates']):values[meta['selection'].start+k]=float(s in lookup)
    R,X=core.build_matrices_from_atoms(bank.data.n,point['supports'],point['r'],point['x'])
    values[model.absolute]=abs(bank.lp.y-bank.lp.p@R.T-bank.lp.q@X.T).ravel()
    return values


def search(bank,initial,kernel,*,kind,b=5,relative=.001,seconds=20.,per_MIP=2.,warm_limit=4):
    then=perf_counter();preserve=kind=='forward'
    current=bank.fit(initial,preserve=preserve);path=[current];attempts=[];tried=set();reason='wall_limit'
    for iteration in range(3*bank.data.n):
        family=canonical(current['supports']);remaining=seconds-(perf_counter()-then)
        if remaining<=.05:break
        if kind=='forward':
            new=any_extension(family,bank.data.n,bank.lp.pool)
            if new is None:reason='structurally_complete';break
            model=core.build_extension_model(bank.scaled,family,r_upper_bound=2.,x_upper_bound=2.)
            warm=current
            # Fixed-family training LPs provide legal stronger starts. None
            # scans the whole pool, subject to the same search wall budget.
            # These starts never shrink the unrestricted support domain.
            possible=[c for c in bank.lp.pool if c not in family and all(compatible(c,s) for s in family)]
            for c in possible[:warm_limit]:
                if seconds-(perf_counter()-then)<per_MIP+.1:break
                q=bank.fit([*family,c],preserve=True)
                if q['objective']<warm['objective']-1e-12:warm,new=q,c
            warm_scaled={**warm,'objective':warm['objective']*SCALE}
            start=extension_start(model,family,new,warm_scaled,bank.lp.p,bank.lp.q,bank.lp.y)
            remaining=seconds-(perf_counter()-then)
            if remaining<=.05:break
            primal,event=solve(bank,model,start,seconds=min(per_MIP,remaining))
            point=warm if warm['objective']<current['objective'] else None
            if primal is not None:
                candidate=core._round_and_validate_support(primal[model.z],family,bank.data.n)
                point=dict(supports=[*family,candidate],r=np.r_[primal[model.r],primal[model.new_r]].clip(0,2).tolist(),
                    x=np.r_[primal[model.x],primal[model.new_x]].clip(0,2).tolist())
                point['objective']=loss(bank.data,point)
                if warm['objective']<point['objective']:point=warm
            event.update(kind='original_free_extension',iteration=iteration)
        elif kind=='rebuild':
            todo=[q for q in regions(family,bank.data.n,b,kernel) if (family,q) not in tried]
            if not todo:reason='neighborhood_scan_complete';break
            v,blocks,virtual=todo[0];tried.add((family,todo[0]))
            augmented=canonical([*family,v]) if virtual and v not in family else family
            model,meta=build_model(bank.scaled,augmented,v,blocks,bound=2.,cuts='subset')
            start=reconstruction_start(bank,model,meta,current)
            remaining=seconds-(perf_counter()-then)
            if remaining<=.05:break
            primal,event=solve(bank,model,start,seconds=min(per_MIP,remaining))
            point=None
            if primal is not None:
                chosen=primal[meta['selection']]>.5
                active=list(range(len(meta['keep'])))+[len(meta['keep'])+k for k,a in enumerate(chosen) if a]
                all_s=meta['all_supports'];rr=primal[model.r].clip(0,2);xx=primal[model.x].clip(0,2)
                active=[i for i in active if len(all_s[i])==1 or max(rr[i],xx[i])>1e-9]
                point=dict(supports=[all_s[i] for i in active],r=rr[active].tolist(),x=xx[active].tolist())
                if not core.is_laminar_family(point['supports']):raise RuntimeError('decoded rebuild is not laminar')
                point['objective']=loss(bank.data,point)
            event.update(kind='local_rebuild',iteration=iteration,region=v,blocks=blocks,virtual_region=virtual)
        else:raise ValueError(kind)
        if point is not None:
            # Close the continuous refit at every accepted integer family,
            # including when the MIP stops with a feasible heuristic incumbent.
            point=bank.fit(point['supports'],preserve=preserve)
        if primal is not None and point is not None and event['objective'] is not None and point['objective']>event['objective']+1e-8:
            # A different LP warm start may be better than the solver incumbent;
            # its objective can be lower but never exceeds its own epigraph UB.
            raise RuntimeError('decoded objective exceeds solver upper bound')
        gain=0. if point is None else current['objective']-point['objective']
        threshold=1e-10+relative*current['objective']
        accepted=point is not None and gain>threshold
        event.update(gain=gain,threshold=threshold,accepted=accepted,before=current['objective'],
                     decoded_objective=None if point is None else point['objective'])
        attempts.append(event)
        if accepted:current=point;path.append(current)
        elif kind=='forward':
            bound=event['dual_bound']
            reason='bound_excludes_gain' if event['bound_consistent_with_known_upper'] and bound is not None and current['objective']-bound<=threshold else 'no_gain_unresolved'
            break
    else:reason='step_limit'
    return dict(path=path,point=current,attempts=attempts,seconds=perf_counter()-then,stop_reason=reason,
                relative_gain=relative,max_blocks=b if kind=='rebuild' else None)


def pool_repair(bank,initial,*,seconds=20.,relative=.001):
    then=perf_counter();current=bank.fit(initial);path=[current];attempts=[];reason='step_limit'
    for iteration in range(3*bank.data.n):
        best=None;complete=True;checked=0
        for c in bank.lp.pool:
            if len(c)<2 or c in canonical(current['supports']):continue
            if perf_counter()-then>=seconds:complete=False;break
            retained=[s for s in current['supports'] if compatible(s,c)]
            p=bank.fit([*retained,c]);checked+=1
            if best is None or p['objective']<best['objective']-1e-12:best=p
        gain=0. if best is None else current['objective']-best['objective']
        accepted=gain>1e-10+relative*current['objective']
        attempts.append(dict(iteration=iteration,complete=complete,checked=checked,gain=gain,accepted=accepted))
        if accepted:current=best;path.append(current)
        else:reason='pool_stationary' if complete else 'wall_limit';break
        if not complete:reason='wall_limit';break
    return dict(path=path,point=current,attempts=attempts,seconds=perf_counter()-then,stop_reason=reason)
