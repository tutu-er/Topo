"""Warm-started free-domain solve; valid incumbents survive a time limit."""
from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix
from research_experiments.wzzt_solver_tuning.policy import extension_decision
from research_experiments.wzzt_solver_tuning.profile_gaps import primal_audit


def solve(raw,*,start,current,seconds):
    import gurobipy as gp
    then=perf_counter()
    c=raw["objective_coefficients"]
    matrix=coo_matrix((raw["constraint_data"],(raw["constraint_row_index"],raw["constraint_col_index"])),
                     shape=(len(raw["constraint_lower"]),len(c))).tocsr()
    lo,hi=raw["constraint_lower"],raw["constraint_upper"]
    eq=np.isfinite(lo)&(lo==hi)
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag",0);env.start()
        with gp.Model("seeded_free_extension",env=env) as model:
            for key,value in dict(Threads=1,TimeLimit=max(.05,seconds),MIPGap=.01,MIPGapAbs=1e-10,
                FeasibilityTol=1e-9,IntFeasTol=1e-9,OptimalityTol=1e-9,DualReductions=0).items():model.setParam(key,value)
            types=np.where(raw["integrality"],gp.GRB.INTEGER,gp.GRB.CONTINUOUS)
            types[(raw["integrality"]==1)&(raw["upper_bounds"]==1)&(raw["lower_bounds"]==0)]=gp.GRB.BINARY
            v=model.addMVar(len(c),lb=raw["lower_bounds"],ub=raw["upper_bounds"],vtype=types,obj=c)
            for mask,sense,rhs in ((eq,"=",lo),(~eq&np.isfinite(lo),">",lo),(~eq&np.isfinite(hi),"<",hi)):
                if np.any(mask):model.addMConstr(matrix[mask],v,sense,rhs[mask])
            if start is not None:v.Start=start
            def callback(m,where):
                if where!=gp.GRB.Callback.MIP:return
                upper=float(m.cbGet(gp.GRB.Callback.MIP_OBJBST));lower=float(m.cbGet(gp.GRB.Callback.MIP_OBJBND))
                upper=upper if abs(upper)<gp.GRB.INFINITY else None
                lower=max(0.,lower) if abs(lower)<gp.GRB.INFINITY else None
                if extension_decision(current,upper,lower,feasible=upper is not None,uncertainty_fraction=.5)["decision"]!="refine_or_report_unresolved":m.terminate()
            model.optimize(callback)
            primal=np.asarray(v.X) if model.SolCount else None
            audit=primal_audit(raw,primal)
            ub=float(model.ObjVal) if model.SolCount else None
            lower=float(model.ObjBound)
            lower=max(0.,lower) if abs(lower)<gp.GRB.INFINITY else None
            return primal,dict(raw_status=int(model.Status),objective=ub,bound=lower,
                relative_gap=None if ub is None or lower is None or ub==0 else (ub-lower)/abs(ub),
                primal_audit=audit,seconds=perf_counter()-then,warm_start_valid=None if start is None else primal_audit(raw,start)["feasible"],
                certificate_scope="bounded free single extension; feasible-gain fallback is heuristic")
