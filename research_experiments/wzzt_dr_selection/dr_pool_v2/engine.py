"""Adapted finite-pool LP and crossing replacement; provenance.json records source.
Input day boundaries, root label, explicit QP budget and failure recording are adapted.
Full pre-pruning coefficients are retained as audit metadata; decisions are unchanged.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[_key]="1"

from time import perf_counter
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix

from .observations import arrays, matrices


def compatible(family,candidate):
    c=set(candidate)
    return all(not(c&set(s)) or c<=set(s) or set(s)<=c for s in family)


def canonical(family):return tuple(sorted(set(tuple(sorted(s)) for s in family),key=lambda s:(len(s),s)))


class PoolLP:
    """One reusable exact LP; inactive candidate coefficients have upper=0."""
    def __init__(self,obs,pool,preprocess="raw",bound=2.,data=None,lp_method=1):
        if isinstance(lp_method,(bool,np.bool_)) or not isinstance(lp_method,(int,np.integer)) or lp_method not in (0,1,2):
            raise ValueError("lp_method must be an integer in {0, 1, 2}")
        import gurobipy as gp
        self.gp=gp
        self.p,self.q,self.y=arrays(obs,preprocess) if data is None else data
        self.n=self.p.shape[1]
        self.pool=canonical(pool)
        self.lookup={s:i for i,s in enumerate(self.pool)}
        self.bound=bound
        self.lp_method=int(lp_method)
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
        for key,value in dict(Threads=1,Method=self.lp_method,FeasibilityTol=1e-9,OptimalityTol=1e-9,TimeLimit=15).items():
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
            self.solves.append(dict(family=family,status=int(self.model.Status),lp_method=int(self.model.Params.Method),
                seconds=perf_counter()-then,accepted=False,solution_count=int(self.model.SolCount)))
            raise RuntimeError(f"fixed-family LP status {self.model.Status}")
        values=np.clip(self.weights.X,0.,self.bound)
        values[upper==0]=0.
        r=values[active];x=values[np.asarray(active)+k]
        prediction=self.features@values
        physical=float(np.mean(abs(prediction-self.y.ravel())))
        if not np.isclose(physical,self.model.ObjVal,atol=1e-8,rtol=1e-8):raise RuntimeError("LP physical objective mismatch")
        kept=[i for i,s in enumerate(family) if len(s)==1 or max(r[i],x[i])>1e-9]
        point=dict(supports=[list(family[i]) for i in kept],r=r[kept].tolist(),x=x[kept].tolist(),
                   objective=physical,preprocess=self.preprocess,solver_objective=float(self.model.ObjVal),
                   full_fit=dict(supports=[list(s) for s in family],r=r.tolist(),x=x.tolist(),
                                 objective=physical,weight_upper_bound=float(self.bound),support_prune_tolerance=1e-9,
                                 semantics="clipped coefficients before numerical support pruning; not physical zero or an optimal-face certificate"))
        audit=dict(family=family,objective=float(self.model.ObjVal),physical=physical,
                   max_constraint_violation=float(self.model.ConstrVio),max_bound_violation=float(self.model.BoundVio),
                   status=int(self.model.Status),lp_method=int(self.model.Params.Method),seconds=perf_counter()-then)
        audit["accepted"] = True
        self.solves.append(audit)
        if audit["max_constraint_violation"] > 1e-7 or audit["max_bound_violation"] > 1e-7:
            raise RuntimeError("fixed-family LP numerical constraint audit failed")
        self.cache[family]=point
        return point

    def close(self):self.model.dispose();self.environment.dispose()

    def add_support(self,support):
        if tuple(support) in self.lookup:return
        cache,solves=self.cache,self.solves
        expanded=PoolLP(None,[*self.pool,support],self.preprocess,self.bound,data=(self.p,self.q,self.y),lp_method=self.lp_method)
        self.close()
        self.__dict__.update(expanded.__dict__)
        self.cache.update(cache);self.solves=solves


def build_bank(obs, *, day_ids, root_label=1):
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
        p,q,y=arrays(obs,preprocessing,day_ids=day_ids)
        scenes=[dict(P_terminal=pd.DataFrame(p,columns=labels),Q_terminal=pd.DataFrame(q,columns=labels),
                     drop_target=pd.DataFrame(y,columns=labels))]
        diag={};r,x,_,_=fit_projected_sensitivity(scenes,alpha=0.,constraint_mode="ordered",
            diagonal_margin_ratio=1e-6,constraint_refine_iterations=5000,diagnostics=diag)
        fronts[preprocessing]=dict(R=r.tolist(),X=x.tolist(),diagnostics=diag)
        cal=calibrate_rx75_floor(r,x)
        calibrations[preprocessing]=cal
        for mode,alpha in (("R",1.),("RX75",.75),("X",0.)):
            kernel=alpha*r/cal["mu_r"]+(1-alpha)*x/cal["mu_x"]
            for tolerance in (0.,.04,.08,.16,.24):
                nj=infer_classical_nj(kernel,kernel,labels,root_label,collapse_factor=tolerance)
                rnj=rooted_neighbor_joining(kernel,np.diag(kernel),labels,root_label,
                    group_tolerance=tolerance*max(float(np.median(np.diag(kernel))),1e-12))
                for method,tree in (("NJ",nj),("RNJ",rnj)):
                    clades=rooted_clades(tree.edges,root_label,labels)
                    family=canonical([tuple([i]) for i in range(n)]+[tuple(pos[v] for v in c) for c in clades])
                    pool.update(family)
                    bank.append(dict(preprocess=preprocessing,mode=mode,tolerance=tolerance,method=method,
                        family=family,config=f"{preprocessing}_{mode}_{tolerance:g}"))
    return bank,canonical(pool),fronts,calibrations


def search(lp,initial,*,seconds=25.,repair=False,max_steps=None,progress_callback=None):
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
        if progress_callback is not None:
            progress_callback(dict(path=path,reason="in_progress",decisions=decisions,seconds=perf_counter()-started))
        if not full_sweep:reason="pool_wall_limit";break
    return dict(path=path,reason=reason,decisions=decisions,seconds=perf_counter()-started)

