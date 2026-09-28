"""Solve a standard-form LP/MILP with gurobipy and SciPy-style inputs."""
from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_matrix

# Public diagnostics use SciPy's convention: 0 optimal, 1 limit/interruption,
# 2 infeasible, 3 unbounded, 4 other. Each entry stores (name, unified status).
_STATUSES = {
    1: ("LOADED", 4), 2: ("OPTIMAL", 0), 3: ("INFEASIBLE", 2),
    4: ("INF_OR_UNBD", 4), 5: ("UNBOUNDED", 3), 6: ("CUTOFF", 4),
    7: ("ITERATION_LIMIT", 1), 8: ("NODE_LIMIT", 1), 9: ("TIME_LIMIT", 1),
    10: ("SOLUTION_LIMIT", 1), 11: ("INTERRUPTED", 1), 12: ("NUMERIC", 4),
    13: ("SUBOPTIMAL", 4), 14: ("INPROGRESS", 4), 15: ("USER_OBJ_LIMIT", 1),
    16: ("WORK_LIMIT", 1), 17: ("MEM_LIMIT", 1),
}


def _normalized_status(raw_status: int) -> int:
    return _STATUSES.get(raw_status, ("UNKNOWN", 4))[1]


def solve_gurobi_milp(*, c, integrality, bounds, constraints, options):
    """Solve min c.T v with the same inputs as scipy.optimize.milp.

    Bounds and LinearConstraint encode lb <= v <= ub and lb <= A v <= ub.
    Only continuous/integer domains (integrality 0/1) are used by this model.
    """
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ImportError(
            "solver='gurobi' requires gurobipy and a valid Gurobi license; "
            "install it with `python -m pip install gurobipy`."
        ) from exc

    grb = gp.GRB
    if not np.isin(integrality, (0, 1)).all():
        raise ValueError("integrality must contain only 0 (continuous) or 1 (integer)")
    types = np.where(integrality, grb.INTEGER, grb.CONTINUOUS)
    types[(integrality == 1) & (bounds.lb == 0) & (bounds.ub == 1)] = grb.BINARY
    # Preserve free variables and unbounded nonnegative residual variables.
    lower = np.where(np.isneginf(bounds.lb), -grb.INFINITY, bounds.lb)
    upper = np.where(np.isposinf(bounds.ub), grb.INFINITY, bounds.ub)
    matrix = csr_matrix(constraints.A)
    row_lower, row_upper = constraints.lb, constraints.ub
    equal = np.isfinite(row_lower) & (row_lower == row_upper)

    def finite_attribute(model, name):
        value = float(getattr(model, name))
        return value if np.isfinite(value) and abs(value) < grb.INFINITY else None

    # Set logging before starting the environment, including license messages.
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", int(options["disp"]))
        env.start()
        with gp.Model("milp", env=env) as model:
            model.Params.Presolve = -1 if options["presolve"] else 0
            model.Params.MIPGap = options["mip_rel_gap"]
            model.Params.MIPGapAbs = options["mip_abs_gap"]
            model.Params.FeasibilityTol = 1e-9
            model.Params.IntFeasTol = 1e-9
            model.Params.OptimalityTol = 1e-9
            model.Params.Threads = 1
            # Distinguish infeasible from unbounded when presolve terminates.
            model.Params.DualReductions = 0
            if "time_limit" in options:
                model.Params.TimeLimit = options["time_limit"]
            v = model.addMVar(
                c.size, lb=lower, ub=upper, vtype=types, obj=c, name="v",
            )
            model.ModelSense = grb.MINIMIZE
            for mask, sense, rhs in (
                (equal, "=", row_lower),
                (~equal & np.isfinite(row_lower), ">", row_lower),
                (~equal & np.isfinite(row_upper), "<", row_upper),
            ):
                if np.any(mask):
                    model.addMConstr(matrix[mask], v, sense, rhs[mask])
            model.optimize()

            raw_status = int(model.Status)
            status_name = _STATUSES.get(raw_status, ("UNKNOWN", 4))[0]
            status = _normalized_status(raw_status)
            has_solution = model.SolCount > 0
            objective = float(model.ObjVal) if has_solution else None
            is_mip = bool(model.IsMIP)
            # A limit may still provide a valid lower bound without an incumbent.
            # LP bound attributes are not MIP certificates: use the optimum only.
            bound = objective if status == 0 else None
            if is_mip:
                bound = finite_attribute(model, "ObjBound")
            gap = finite_attribute(model, "MIPGap") if is_mip and has_solution else None
            return OptimizeResult(
                x=np.asarray(v.X, dtype=float) if has_solution else None,
                fun=objective,
                status=status,
                success=status == 0,
                message=f"Gurobi {status_name} (status {raw_status})",
                mip_dual_bound=bound,
                mip_gap=gap,
                mip_node_count=int(model.NodeCount) if is_mip else None,
                solver="gurobi",
                raw_status=raw_status,
            )
