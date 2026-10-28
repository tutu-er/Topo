from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from research_experiments.wzzt_systematic.common import matrices,raw_model,write


def arrays(obs,preprocess):
    p,q,v,root=obs
    values=[p.to_numpy().copy(),q.to_numpy().copy(),root.to_numpy()[:,None]**2-v.to_numpy()**2]
    if preprocess=="demean":
        for rows in np.array_split(np.arange(len(p)),3):
            for value in values:value[rows]-=value[rows].mean(axis=0)
    elif preprocess!="raw":raise ValueError(preprocess)
    return values


def compatible(family,candidate):
    c=set(candidate)
    return all(not(c&set(s)) or c<=set(s) or set(s)<=c for s in family)


def canonical(family):return tuple(sorted(set(tuple(sorted(s)) for s in family),key=lambda s:(len(s),s)))


class PoolLP:
    """One reusable exact LP; inactive candidate coefficients have upper=0."""
    def __init__(self,obs,pool,preprocess="raw",bound=2.,data=None):
        import gurobipy as gp
        self.gp=gp
        self.p,self.q,self.y=arrays(obs,preprocess) if data is None else data
        self.n=self.p.shape[1]
        self.pool=canonical(pool)
        self.lookup={s:i for i,s in enumerate(self.pool)}
        self.bound=bound
        self.preprocess=preprocess
        self.cache={}
        z=np.zeros((len(self.pool),self.n))
        for row,s in zip(z,self.pool):row[list(s)]=1
        self.fr=np.column_stack([((self.p@row)[:,None]*row).ravel() for row in z])
        self.fx=np.column_stack([((self.q@row)[:,None]*row).ravel() for row in z])
        self.features=np.column_stack((self.fr,self.fx))
        self.environment=gp.Env(empty=True)
        self.environment.setParam("OutputFlag",0)
        self.environment.start()
        self.model=gp.Model("candidate_family_l1",env=self.environment)
        for key,value in dict(Threads=1,Method=1,FeasibilityTol=1e-9,OptimalityTol=1e-9,TimeLimit=15).items():
            self.model.setParam(key,value)
        self.weights=self.model.addMVar(2*len(self.pool),lb=0,ub=0)
        self.t=self.model.addMVar(self.y.size,lb=0,obj=1/self.y.size)
        f=csc_matrix(self.features)
        self.model.addConstr(f@self.weights-self.t<=self.y.ravel())
        self.model.addConstr(f@self.weights+self.t>=self.y.ravel())
        self.solves=[]

    def fit(self,family):
        family=canonical(family)
        if family in self.cache:return self.cache[family]
        if any(not compatible(family[:i],s) for i,s in enumerate(family)):
            raise ValueError("nonlaminar family passed to LP")
        then=perf_counter()
        k=len(self.pool)
        upper=np.zeros(2*k)
        active=[self.lookup[s] for s in family]
        upper[active]=upper[np.asarray(active)+k]=self.bound
        self.weights.UB=upper
        self.model.optimize()
        if self.model.Status!=self.gp.GRB.OPTIMAL:
            raise RuntimeError(f"fixed-family LP status {self.model.Status}")
        values=np.clip(self.weights.X,0.,self.bound)
        values[upper==0]=0.
        r=values[active];x=values[np.asarray(active)+k]
        prediction=self.features@values
        physical=float(np.mean(abs(prediction-self.y.ravel())))
        if not np.isclose(physical,self.model.ObjVal,atol=1e-8,rtol=1e-8):raise RuntimeError("LP physical objective mismatch")
        kept=[i for i,s in enumerate(family) if len(s)==1 or max(r[i],x[i])>1e-9]
        point=dict(supports=[list(family[i]) for i in kept],r=r[kept].tolist(),x=x[kept].tolist(),
                   objective=physical,preprocess=self.preprocess,solver_objective=float(self.model.ObjVal))
        audit=dict(family=family,objective=float(self.model.ObjVal),physical=physical,
                   max_constraint_violation=float(self.model.ConstrVio),max_bound_violation=float(self.model.BoundVio),
                   status=int(self.model.Status),seconds=perf_counter()-then)
        self.solves.append(audit)
        self.cache[family]=point
        return point

    def close(self):self.model.dispose();self.environment.dispose()

    def add_support(self,support):
        if tuple(support) in self.lookup:return
        cache,solves=self.cache,self.solves
        expanded=PoolLP(None,[*self.pool,support],self.preprocess,self.bound,data=(self.p,self.q,self.y))
        self.close()
        self.__dict__.update(expanded.__dict__)
        self.cache.update(cache);self.solves=solves


def build_bank(obs):
    from rnj_wzzt.estimation.multiscenario import fit_projected_sensitivity
    from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
    from rnj_wzzt.graph.rooted_neighbor_joining import rooted_neighbor_joining
    from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
    from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor
    labels=list(obs[0].columns)
    n=len(labels)
    pos={v:i for i,v in enumerate(labels)}
    bank=[];fronts={};calibrations={}
    pool={tuple([i]) for i in range(n)}|{tuple(range(n))}
    for preprocessing in ("raw","demean"):
        p,q,y=arrays(obs,preprocessing)
        scenes=[dict(P_terminal=pd.DataFrame(p,columns=labels),Q_terminal=pd.DataFrame(q,columns=labels),
                     drop_target=pd.DataFrame(y,columns=labels))]
        diag={};r,x,_,_=fit_projected_sensitivity(scenes,diagnostics=diag)
        fronts[preprocessing]=dict(R=r.tolist(),X=x.tolist(),diagnostics=diag)
        cal=calibrate_rx75_floor(r,x)
        calibrations[preprocessing]=cal
        for mode,alpha in (("R",1.),("RX75",.75),("X",0.)):
            kernel=alpha*r/cal["mu_r"]+(1-alpha)*x/cal["mu_x"]
            for tolerance in (0.,.04,.08,.16,.24):
                nj=infer_classical_nj(kernel,kernel,labels,1,collapse_factor=tolerance)
                rnj=rooted_neighbor_joining(kernel,np.diag(kernel),labels,1,
                    group_tolerance=tolerance*max(float(np.median(np.diag(kernel))),1e-12))
                for method,tree in (("NJ",nj),("RNJ",rnj)):
                    clades=rooted_clades(tree.edges,1,labels)
                    family=canonical([tuple([i]) for i in range(n)]+[tuple(pos[v] for v in c) for c in clades])
                    pool.update(family)
                    bank.append(dict(preprocess=preprocessing,mode=mode,tolerance=tolerance,method=method,
                        family=family,config=f"{preprocessing}_{mode}_{tolerance:g}"))
    return bank,canonical(pool),fronts,calibrations


def search(lp,initial,*,seconds=25.,repair=False,max_steps=None):
    started=perf_counter()
    current=lp.fit(initial)
    path=[current]
    decisions=[]
    reason="step_limit"
    for iteration in range(max_steps or 3*lp.n):
        family=canonical(current["supports"])
        best=None;candidate_records=[]
        full_sweep=True
        for candidate in lp.pool:
            if len(candidate)<2 or candidate in family:continue
            if perf_counter()-started>=seconds:
                full_sweep=False;break
            if not compatible(family,candidate):
                if not repair:continue
                retained=[s for s in family if compatible([s],candidate)]
            else:retained=list(family)
            proposed=canonical([*retained,candidate])
            fitted=lp.fit(proposed)
            candidate_records.append(dict(candidate=candidate,removed=[s for s in family if s not in retained],objective=fitted["objective"]))
            if best is None or fitted["objective"]<best["objective"]-1e-12:best=fitted
        gain=0 if best is None else current["objective"]-best["objective"]
        decisions.append(dict(iteration=iteration,current=current["objective"],gain=gain,full_sweep=full_sweep,candidates=candidate_records))
        if best is not None and gain>1e-10+.001*current["objective"]:
            current=best;path.append(current)
        else:
            reason="pool_local_optimum" if full_sweep else "pool_wall_limit"
            break
        if not full_sweep:reason="pool_wall_limit";break
    return dict(path=path,reason=reason,decisions=decisions,seconds=perf_counter()-started)


def extension_start(model,family,new,point,p,q,y):
    """Construct every lifted product from one legal fixed-family LP model."""
    values=np.zeros(model.variables.size)
    lookup={tuple(s):(r,x) for s,r,x in zip(point["supports"],point["r"],point["x"],strict=True)}
    for i,s in enumerate(family):
        values[model.r.start+i],values[model.x.start+i]=lookup.get(tuple(s),(0.,0.))
    nr,nx=lookup.get(tuple(new),(0.,0.))
    values[model.new_r]=nr;values[model.new_x]=nx
    z=np.zeros(p.shape[1]);z[list(new)]=1
    values[model.z]=z
    slices=model.variables.slices
    products=np.array([z[i]*z[j] for i in range(len(z)) for j in range(i,len(z))])
    values[slices["y"]]=products
    values[slices["u_r"]]=nr*products;values[slices["u_x"]]=nx*products
    for i,s in enumerate(family):
        which=0 if set(new)<set(s) else 1 if set(s)<set(new) else 2
        values[slices["relations"].start+3*i+which]=1
    r,x=matrices(len(z),point["supports"],point["r"],point["x"])
    values[slices["absolute_residual"]]=abs(p@r.T+q@x.T-y).ravel()
    return values


def free_search(lp,output,*,seconds=40.,per_solve=5.):
    from rnj_wzzt.estimation import laminar_l1_milp as core
    from research_experiments.wzzt_solver_tuning.policy import extension_decision
    from research_experiments.wzzt_solver_tuning.profile_gaps import primal_audit
    from .solver import solve
    output.mkdir(parents=True)
    labels=list(range(lp.n))
    scenes=[dict(P_terminal=pd.DataFrame(lp.p,columns=labels),Q_terminal=pd.DataFrame(lp.q,columns=labels),
                 drop_target=pd.DataFrame(lp.y,columns=labels))]
    prepared=core._prepare_scenarios(scenes)
    started=perf_counter()
    current=lp.fit([(i,) for i in range(lp.n)])
    path=[current];attempts=[];reason="wall_limit"
    for step in range(3*lp.n):
        family=canonical(current["supports"])
        if len(family)>=2*lp.n-1:reason="family_limit";break
        if seconds-(perf_counter()-started)<.5:break
        best=None;new=None
        for candidate in lp.pool:
            if len(candidate)<2 or candidate in family or not compatible(family,candidate):continue
            if seconds-(perf_counter()-started)<.5:break
            point=lp.fit([*family,candidate])
            if best is None or point["objective"]<best["objective"]:
                best,new=point,candidate
        model=core.build_extension_model(prepared,family,r_upper_bound=lp.bound,x_upper_bound=lp.bound)
        raw=raw_model(model)
        start=None if best is None else extension_start(model,family,new,best,lp.p,lp.q,lp.y)
        if start is not None and not primal_audit(raw,start)["feasible"]:raise RuntimeError("warm start violates original free MILP")
        remaining=seconds-(perf_counter()-started)
        if remaining<.25:
            if best is not None and current["objective"]-best["objective"]>1e-10+.001*current["objective"]:
                path.append(best)
            break
        target=output/f"step{step}"
        target.mkdir()
        np.savez_compressed(target/"model.npz",**raw)
        if start is not None:np.savez_compressed(target/"start.npz",primal=start)
        primal,info=solve(raw,start=start,current=current["objective"],seconds=min(per_solve,remaining))
        selected_new=new
        info.update(current_objective=current["objective"],family=family,slices={k:[v.start,v.stop] for k,v in model.variables.slices.items()})
        if primal is not None:
            np.savez_compressed(target/"primal.npz",primal=primal)
        if primal is not None and info["primal_audit"]["feasible"]:
            support=core._round_and_validate_support(model.expand_z_values(primal[model.z]),family,lp.n)
            lp.add_support(support)
            lp2=lp.fit([*family,support])
            if best is None or lp2["objective"]<best["objective"]:best=lp2;selected_new=support
        gain=0 if best is None else current["objective"]-best["objective"]
        info["accepted_physical_gain"]=gain
        info["accepted"]=bool(gain>1e-10+.001*current["objective"])
        info["bounded_decision"]=extension_decision(current["objective"],info["objective"],info["bound"],feasible=info["primal_audit"]["feasible"],uncertainty_fraction=.5)["decision"]
        if best is not None:
            selected_primal=extension_start(model,family,selected_new,best,lp.p,lp.q,lp.y)
            selected_audit=primal_audit(raw,selected_primal)
            if not selected_audit["feasible"]:raise RuntimeError("selected candidate violates original free MILP")
            np.savez_compressed(target/"selected_primal.npz",primal=selected_primal)
            info["selected_primal_audit"]=selected_audit
            info["selected_upper"]=best["objective"]
            info["selected_bounded_decision"]=extension_decision(current["objective"],best["objective"],info["bound"],
                feasible=True,uncertainty_fraction=.5)["decision"]
        write(target/"result.json",info);attempts.append(info)
        if info["accepted"]:
            current=best;path.append(current)
        else:
            reason="certified_no_gain" if info["bounded_decision"]=="stop_bound_proves_no_material_gain" else "no_gain_found_unresolved"
            break
    return dict(path=path,reason=reason,attempts=attempts,seconds=perf_counter()-started)
