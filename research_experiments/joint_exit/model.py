"""One global selection MILP over the old joint pool and revocable alternatives."""
from __future__ import annotations
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.sparse import csr_matrix
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical,compatible,unions
from research_experiments.joint_five.model import open_small,build_joint
from research_experiments.rebuild_benchmark.engine import SCALE,audit_vector,reconstruction_start,regions,loss
from research_experiments.revocable.engine import clean_zero
from research_experiments.rnj_count.engine import criterion


def candidate_pool(bank,initial,current,kernel,policy):
    keep,_=open_small(current['supports'],bank.data.n)
    _,old=build_joint(bank.scaled,keep)
    if not hasattr(bank,'joint_exit_seed_pool'):
        bank.joint_exit_seed_pool=canonical(bank.lp.pool)
    pool=set(canonical([*initial,*current['supports'],*bank.joint_exit_seed_pool,*old['all_supports']]))
    frontiers=[]
    if policy=='expanded':
        for family in (initial,current['supports']):
            for v,blocks,virtual in regions(family,bank.data.n,5,kernel):
                pool.update(s for _,s in unions(blocks));frontiers.append((v,blocks,virtual))
    elif policy!='compact':
        raise ValueError(policy)
    # The original multiregion MILP's entire domain remains representable.
    assert set(old['all_supports'])<=pool
    return canonical(pool),dict(old_pool=old['all_supports'],old_candidates=len(old['candidates']),
        policy=policy,frontier_count=len(frontiers),pool_count=len(pool))


def build(data,pool,*,penalty,hard=()):
    n=data.n;basis=canonical([*((i,) for i in range(n)),tuple(range(n))])
    candidates=canonical(s for s in pool if 1<len(s)<n)
    hard=set(canonical(hard))
    if not hard<=set(candidates)|set(basis):
        raise ValueError('hard priors missing from pool')
    model=core.build_fixed_model(data,[*basis,*candidates],r_upper_bound=2.,x_upper_bound=2.)
    a=model.variables.add('select',len(candidates),upper=1.,integral=True,objective=penalty*SCALE)
    for k,s in enumerate(candidates):
        col=a.start+k
        if s in hard:model.variables.lower[col]=1.
        for weights in (model.r,model.x):
            model.rows.add({weights.start+len(basis)+k:1.,col:-2.},upper=0.)
    conflicts=[]
    for i,j in combinations(range(len(candidates)),2):
        if not compatible(candidates[i],candidates[j]):
            model.rows.add({a.start+i:1.,a.start+j:1.},upper=1.);conflicts.append((i,j))
    cuts=0
    for W in [*candidates,tuple(range(n))]:
        inside={a.start+k:1. for k,s in enumerate(candidates) if set(s)<set(W)}
        if inside:
            model.rows.add(inside,upper=float(len(W)-2));cuts+=1
    # Crossing cliques are valid stronger cuts; no binary-resolution equality.
    adjacency={i:set() for i in range(len(candidates))}
    for i,j in conflicts:adjacency[i].add(j);adjacency[j].add(i)
    cliques=set()
    for i in range(len(candidates)):
        clique=[i]
        for j in sorted(adjacency[i],key=lambda j:(-len(adjacency[j]),j)):
            if all(j in adjacency[k] for k in clique):clique.append(j)
        if len(clique)>2:cliques.add(tuple(sorted(clique)))
    for clique in cliques:model.rows.add({a.start+i:1. for i in clique},upper=1.)
    return model,dict(keep=basis,candidates=candidates,all_supports=(*basis,*candidates),selection=a,
        conflicts=len(conflicts),cardinality_cuts=cuts,clique_cuts=len(cliques))


def solve_checkpoints(bank,model,meta,current,*,prefix_seconds,budgets=(20.,40.)):
    """Capture only incumbents/bounds available by each wall-clock checkpoint."""
    import gurobipy as gp
    then=perf_counter();start=reconstruction_start(bank,model,meta,current)
    warm=audit_vector(model,start)
    if not warm['valid']:raise RuntimeError('infeasible joint-exit warm start')
    snapshots={float(b):dict(values=start.copy(),bound=None,incumbent_at=prefix_seconds) for b in budgets}
    raw=model.rows.build(model.variables.size);A=csr_matrix(raw.A)
    lower,upper=raw.lb,raw.ub;eq=np.isfinite(lower)&(lower==upper)
    types=np.where(np.asarray(model.variables.integrality,dtype=bool),gp.GRB.BINARY,gp.GRB.CONTINUOUS)
    last=np.asarray(start);inc_at=prefix_seconds;events=[]
    def remember(now,bound=None,values=None):
        nonlocal last,inc_at
        if values is not None:last=np.asarray(values);inc_at=now
        for b,s in snapshots.items():
            if now<=b:
                if values is not None:s.update(values=last.copy(),incumbent_at=inc_at)
                if bound is not None:s['bound']=bound
    def callback(m,where):
        now=prefix_seconds+perf_counter()-then
        if where==gp.GRB.Callback.MIPSOL:
            values=m.cbGetSolution(v);remember(now,values=values)
            events.append(dict(at=now,objective=float(m.cbGet(gp.GRB.Callback.MIPSOL_OBJ))/SCALE))
        elif where==gp.GRB.Callback.MIP:
            bound=float(m.cbGet(gp.GRB.Callback.MIP_OBJBND))/SCALE
            if abs(bound)<gp.GRB.INFINITY:remember(now,bound=bound)
    remaining=max(budgets)-prefix_seconds-(perf_counter()-then)
    if remaining<=.05:
        return snapshots,dict(solved=False,seconds=perf_counter()-then,warm_start=warm,events=events)
    with gp.Model('multiregion_with_joint_exit',env=bank.lp.environment) as m:
        for key,value in dict(Threads=1,OutputFlag=0,TimeLimit=max(.05,remaining),MIPGap=.001,MIPGapAbs=1e-7,
                FeasibilityTol=1e-9,IntFeasTol=1e-9,OptimalityTol=1e-9,DualReductions=0,Seed=20261031).items():
            m.setParam(key,value)
        v=m.addMVar(model.variables.size,lb=model.variables.lower,
            ub=np.where(np.isposinf(model.variables.upper),gp.GRB.INFINITY,model.variables.upper),obj=model.variables.objective,vtype=types)
        for mask,sense,rhs in ((eq,'=',lower),(~eq&np.isfinite(lower),'>',lower),(~eq&np.isfinite(upper),'<',upper)):
            if np.any(mask):m.addMConstr(A[mask],v,sense,rhs[mask])
        # Native model construction counts against each checkpoint as well.
        m.Params.TimeLimit=max(.05,max(budgets)-prefix_seconds-(perf_counter()-then))
        v.Start=start;m.optimize(callback)
        now=prefix_seconds+perf_counter()-then
        if m.SolCount:
            bound=float(m.ObjBound)/SCALE
            remember(now,bound=bound if abs(bound)<gp.GRB.INFINITY else None,values=np.asarray(v.X))
        record=dict(solved=True,status=int(m.Status),seconds=perf_counter()-then,warm_start=warm,events=events,
            final_bound=float(m.ObjBound)/SCALE if abs(m.ObjBound)<gp.GRB.INFINITY else None,
            final_objective=float(m.ObjVal)/SCALE if m.SolCount else None,
            binary_count=len(meta['candidates']),variable_count=model.variables.size,constraint_count=model.rows.size)
    return snapshots,record


def joint_exit(bank,initial,current,kernel,*,policy,coefficient,reference,prefix_seconds,budgets=(20.,40.)):
    then=perf_counter();penalty=coefficient*reference
    pool,domain=candidate_pool(bank,initial,current,kernel,policy)
    model,meta=build(bank.scaled,pool,penalty=penalty)
    construction=perf_counter()-then
    snapshots,record=solve_checkpoints(bank,model,meta,current,prefix_seconds=prefix_seconds+construction,budgets=budgets)
    outputs={}
    for budget,snapshot in snapshots.items():
        values=snapshot.pop('values');audit=audit_vector(model,values)
        if not audit['valid']:raise RuntimeError('invalid checkpoint incumbent')
        indices=[*range(len(meta['keep'])),*[len(meta['keep'])+k for k,a in enumerate(values[meta['selection']]) if a>.5]]
        rr=values[model.r];xx=values[model.x]
        raw=dict(supports=[meta['all_supports'][i] for i in indices],r=rr[indices].tolist(),x=xx[indices].tolist())
        raw['objective']=loss(bank.data,raw)
        score=criterion(raw,bank.data.n,penalty);before=criterion(current,bank.data.n,penalty)
        if score>audit['objective']+1e-8:raise RuntimeError('checkpoint physical objective exceeds epigraph')
        if snapshot['bound'] is not None and snapshot['bound']>score+1e-8:raise RuntimeError('invalid checkpoint lower bound')
        eligible=prefix_seconds+construction<budget and before-score>1e-10+.001*current['objective']
        # Continuous closure is postprocessing, not an additional structural search.
        point=bank.fit(raw['supports'],preserve=True) if eligible else current
        if eligible:point,_=clean_zero(bank,point)
        outputs[f'{budget:g}']=dict(point=point,accepted=eligible,raw_point=raw,raw_score=score,before_score=before,
            score=criterion(point,bank.data.n,penalty),budget=budget,penalty=penalty,audit=audit,**snapshot,
            bound_scope='same full candidate-selection domain; basis only mandatory',
            postprocessing_excluded_from_checkpoint=True)
    return outputs,dict(**domain,**record,construction_seconds=construction,coefficient=coefficient,penalty=penalty,
        candidates=meta['candidates'],pool=pool,cardinality_cuts=meta['cardinality_cuts'],clique_cuts=meta['clique_cuts'])
