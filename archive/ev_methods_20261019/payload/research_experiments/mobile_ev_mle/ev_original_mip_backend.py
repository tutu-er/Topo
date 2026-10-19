"""Silent linear MILP backend with the SciPy minimization result contract.

The backend only translates the supplied matrix problem. It does not change
objectives, constraints, integrality, bounds or the mathematical proxy. A
SciPy fallback is opt-in with backend='auto' (or 'gurobi_or_scipy').
"""
from __future__ import annotations
from time import perf_counter
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,OptimizeResult,milp
from scipy.sparse import csr_matrix


def _inputs(c,integrality,bounds,constraints,start,time_limit,mip_rel_gap):
    c=np.asarray(c,float)
    if c.ndim!=1 or not len(c) or not np.isfinite(c).all():raise ValueError('c must be a finite nonempty vector')
    n=len(c);itype=np.broadcast_to(np.asarray(0 if integrality is None else integrality,float),(n,)).copy()
    if not np.isfinite(itype).all() or not np.isin(itype,[0,1,2,3]).all():raise ValueError('integrality must contain 0, 1, 2 or 3')
    itype=itype.astype(int)
    if bounds is None:bounds=Bounds(np.zeros(n),np.full(n,np.inf))
    if not isinstance(bounds,Bounds):raise TypeError('bounds must be scipy.optimize.Bounds')
    lo=np.broadcast_to(np.asarray(bounds.lb,float),(n,)).copy();hi=np.broadcast_to(np.asarray(bounds.ub,float),(n,)).copy()
    if np.isnan(lo).any() or np.isnan(hi).any() or np.isposinf(lo).any() or np.isneginf(hi).any():raise ValueError('invalid variable bounds')
    # Native Gurobi S/N domains agree with the stated SciPy semi-continuous /
    # semi-integer domains for a positive finite lower and finite upper bound.
    semi=np.isin(itype,[2,3])
    if np.any(semi&((lo<=0)|~np.isfinite(lo)|~np.isfinite(hi))):
        raise ValueError('semi-continuous/integer variables require positive finite lower and finite upper bounds')
    if constraints is None:groups=[]
    elif isinstance(constraints,LinearConstraint):groups=[constraints]
    else:groups=list(constraints)
    clean=[]
    for con in groups:
        if not isinstance(con,LinearConstraint):raise TypeError('each constraint must be LinearConstraint')
        A=csr_matrix(con.A,dtype=float)
        if A.ndim!=2 or A.shape[1]!=n or not np.isfinite(A.data).all():raise ValueError('finite constraint matrix with matching columns required')
        m=A.shape[0];lower=np.broadcast_to(np.asarray(con.lb,float),(m,)).copy();upper=np.broadcast_to(np.asarray(con.ub,float),(m,)).copy()
        if np.isnan(lower).any() or np.isnan(upper).any() or np.isposinf(lower).any() or np.isneginf(upper).any():raise ValueError('invalid constraint bounds')
        clean.append(LinearConstraint(A,lower,upper))
    if not np.isfinite(time_limit) or time_limit<0 or not np.isfinite(mip_rel_gap) or mip_rel_gap<0:
        raise ValueError('time_limit and mip_rel_gap must be finite and nonnegative')
    if start is not None:
        start=np.asarray(start,float)
        if start.shape!=(n,) or np.isinf(start).any():raise ValueError('start must match c; NaN denotes an unspecified coordinate')
        start=start.copy()
    return c,itype,Bounds(lo,hi),clean,start


def _metadata(result,requested,used,start,*,fallback=False,reason=None,seconds=0.):
    result.backend_requested=requested;result.backend_used=used
    result.fallback_used=bool(fallback);result.fallback_reason=reason
    result.start_supplied=bool(start is not None)
    result.start_provided_count=0 if start is None else int(np.isfinite(start).sum())
    result.start_used=bool(used=='gurobi' and result.start_provided_count)
    result.start_used_meaning='submitted as a solver MIP start; acceptance or improvement is not inferred'
    result.start_policy=('partial/full MIP start; unspecified coordinates are GRB.UNDEFINED' if used=='gurobi'
        else 'SciPy milp does not accept a MIP start; supplied start is explicitly ignored')
    result.objective_sense='min';result.objective_transform='none';result.backend_seconds=float(seconds)
    return result


def _error_result(error,requested,start,seconds):
    r=OptimizeResult(x=None,fun=None,status=4,message='Gurobi backend unavailable or failed: '+type(error).__name__+': '+str(error),
        mip_dual_bound=None,mip_gap=None,mip_node_count=None,success=False,gurobi_status=None)
    r=_metadata(r,requested,'gurobi',start,seconds=seconds)
    r.start_used=False;r.start_policy='MIP-start submission is unconfirmed because the backend failed'
    return r


def _finite_attribute(model,name):
    try:value=float(getattr(model,name))
    except (AttributeError,RuntimeError):return None
    return value if np.isfinite(value) else None


def _gurobi(c,integrality,bounds,constraints,time_limit,mip_rel_gap,start):
    import gurobipy as gp
    GRB=gp.GRB
    # Configure OutputFlag before env.start(), suppressing license banners as
    # well as solve output. Each invocation has an isolated single-thread env.
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag',0);env.start()
        with gp.Model(env=env) as model:
            model.Params.OutputFlag=0;model.Params.Threads=1;model.Params.Seed=0
            model.Params.TimeLimit=float(time_limit);model.Params.MIPGap=float(mip_rel_gap)
            types=np.full(len(c),GRB.CONTINUOUS,dtype='<U1')
            types[integrality==1]=GRB.INTEGER;types[integrality==2]=GRB.SEMICONT;types[integrality==3]=GRB.SEMIINT
            binary=(integrality==1)&(bounds.lb>=0)&(bounds.ub<=1)
            types[binary]=GRB.BINARY
            lo=np.where(np.isneginf(bounds.lb),-GRB.INFINITY,bounds.lb)
            hi=np.where(np.isposinf(bounds.ub),GRB.INFINITY,bounds.ub)
            x=model.addMVar(len(c),lb=lo,ub=hi,obj=c,vtype=types)
            model.ModelSense=GRB.MINIMIZE
            for con in constraints:
                A=csr_matrix(con.A);lower=np.asarray(con.lb);upper=np.asarray(con.ub)
                equal=np.isfinite(lower)&np.isfinite(upper)&(lower==upper)
                above=np.isfinite(lower)&~equal;below=np.isfinite(upper)&~equal
                for mask,sense,rhs in ((equal,'=',lower),(above,'>',lower),(below,'<',upper)):
                    if np.any(mask):model.addMConstr(A[mask],x,sense,rhs[mask])
            if start is not None:x.Start=np.where(np.isnan(start),GRB.UNDEFINED,start)
            model.optimize();native=int(model.Status)
            limited={GRB.ITERATION_LIMIT,GRB.NODE_LIMIT,GRB.TIME_LIMIT,GRB.SOLUTION_LIMIT,GRB.INTERRUPTED,
                GRB.USER_OBJ_LIMIT,GRB.WORK_LIMIT,GRB.MEM_LIMIT}
            status=(0 if native==GRB.OPTIMAL else 1 if native in limited else 2 if native==GRB.INFEASIBLE
                else 3 if native==GRB.UNBOUNDED else 4)
            names={getattr(GRB,name):name for name in ('LOADED','OPTIMAL','INFEASIBLE','INF_OR_UNBD','UNBOUNDED','CUTOFF',
                'ITERATION_LIMIT','NODE_LIMIT','TIME_LIMIT','SOLUTION_LIMIT','INTERRUPTED','NUMERIC','SUBOPTIMAL',
                'INPROGRESS','USER_OBJ_LIMIT','WORK_LIMIT','MEM_LIMIT')}
            sol=np.asarray(x.X,float).copy() if model.SolCount else None
            value=None if sol is None else float(c@sol)
            is_mip=bool(model.IsMIP)
            if is_mip:
                dual=_finite_attribute(model,'ObjBound');gap=_finite_attribute(model,'MIPGap')
            else:dual=value if status==0 else None;gap=0. if status==0 else None
            node_count=_finite_attribute(model,'NodeCount')
            return OptimizeResult(x=sol,fun=value,status=status,message='Gurobi '+names.get(native,str(native)),
                mip_dual_bound=dual,mip_gap=gap,mip_node_count=node_count,success=status==0,
                gurobi_status=native,gurobi_runtime=_finite_attribute(model,'Runtime'),
                gurobi_version='.'.join(map(str,gp.gurobi.version())),threads=1,seed=0,
                solver_time_limit=float(time_limit),is_mip=is_mip)


def solve_linear_mip(c,integrality,bounds,constraints=None,time_limit=60.,mip_rel_gap=1e-4,
                     start=None,backend='gurobi'):
    """Solve exactly the supplied linear MIN problem, with explicit backend.

    Status follows SciPy: 0 optimal, 1 limit/interruption, 2 infeasible,
    3 unbounded, 4 other/unavailable/ambiguous. `fun` is an incumbent MIN
    objective; `mip_dual_bound` is a MIN lower bound when finite. An absent
    incumbent/bound is None. No sign reversal is applied for negative costs.

    `backend='gurobi'` never silently falls back. `backend='scipy'` explicitly
    ignores starts because scipy.optimize.milp has no MIP-start argument.
    `backend='auto'`/`'gurobi_or_scipy'` opt into a recorded SciPy fallback
    only if Gurobi import, license, construction or execution raises an error.
    Solver limit or infeasibility statuses do not trigger a fallback.
    """
    began=perf_counter();requested=backend
    if backend not in ('gurobi','scipy','auto','gurobi_or_scipy'):raise ValueError('unknown linear MIP backend')
    c,it,bounds,groups,start=_inputs(c,integrality,bounds,constraints,start,time_limit,mip_rel_gap)
    fallback=False;reason=None
    if backend!='scipy':
        try:answer=_gurobi(c,it,bounds,groups,time_limit,mip_rel_gap,start)
        except (ImportError,RuntimeError) as error:
            # GurobiError subclasses Exception, not RuntimeError on all versions.
            if backend=='gurobi':return _error_result(error,requested,start,perf_counter()-began)
            fallback=True;reason=type(error).__name__+': '+str(error)
        except Exception as error:
            try:
                import gurobipy as gp
                is_solver_error=isinstance(error,gp.GurobiError)
            except ImportError:is_solver_error=False
            if not is_solver_error:raise
            if backend=='gurobi':return _error_result(error,requested,start,perf_counter()-began)
            fallback=True;reason=type(error).__name__+': '+str(error)
        else:return _metadata(answer,requested,'gurobi',start,seconds=perf_counter()-began)
    answer=milp(c,integrality=it,bounds=bounds,constraints=groups,
        options=dict(time_limit=float(time_limit),mip_rel_gap=float(mip_rel_gap),presolve=True))
    # Ensure the advertised fields also exist on continuous or failed results.
    for key in ('mip_dual_bound','mip_gap','mip_node_count'):
        if not hasattr(answer,key):answer[key]=None
    if not np.any(it) and answer.status==0:
        answer.mip_dual_bound=answer.fun;answer.mip_gap=0.;answer.mip_node_count=0.
    return _metadata(answer,requested,'scipy',start,fallback=fallback,reason=reason,seconds=perf_counter()-began)


def gurobi_available():
    """Read-only silent license/runtime probe used to skip unavailable tests."""
    try:
        import gurobipy as gp
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            with gp.Model(env=env) as model:
                model.Params.OutputFlag=0;model.Params.Threads=1
                model.addVar(lb=0.,ub=1.,obj=1.);model.optimize()
                if model.Status!=gp.GRB.OPTIMAL:return dict(available=False,reason='probe status '+str(model.Status))
        return dict(available=True,version='.'.join(map(str,gp.gurobi.version())),reason=None)
    except (ImportError,RuntimeError) as error:return dict(available=False,reason=type(error).__name__+': '+str(error))
    except Exception as error:
        try:
            import gurobipy as gp
            if isinstance(error,gp.GurobiError):return dict(available=False,reason=type(error).__name__+': '+str(error))
        except ImportError:pass
        raise
