"""Gurobi solve with an explicit improvement-bound callback and raw evidence."""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.sparse import coo_matrix

from .policy import extension_decision
from .profile_gaps import primal_audit


def solve(raw, *, current, seconds=60., relative_gain=.001, uncertainty_fraction=.5,
          requested_gap=.01):
    import gurobipy as gp
    grb = gp.GRB
    started = perf_counter()
    c = raw["objective_coefficients"]
    matrix = coo_matrix((raw["constraint_data"],
        (raw["constraint_row_index"], raw["constraint_col_index"])),
        shape=(len(raw["constraint_lower"]), len(c))).tocsr()
    lower, upper = raw["constraint_lower"], raw["constraint_upper"]
    equal = np.isfinite(lower) & (lower == upper)
    callback_decisions = []
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        with gp.Model("bounded_wzzt", env=env) as model:
            for key, value in dict(Threads=1, Presolve=-1, MIPGap=requested_gap, MIPGapAbs=1e-10,
                FeasibilityTol=1e-9, IntFeasTol=1e-9, OptimalityTol=1e-9,
                DualReductions=0, TimeLimit=seconds).items():
                model.setParam(key, value)
            types = np.where(raw["integrality"], grb.INTEGER, grb.CONTINUOUS)
            types[(raw["integrality"] == 1) & (raw["lower_bounds"] == 0) & (raw["upper_bounds"] == 1)] = grb.BINARY
            v = model.addMVar(len(c), lb=raw["lower_bounds"], ub=raw["upper_bounds"], vtype=types, obj=c)
            for mask, sense, rhs in ((equal, "=", lower),
                (~equal & np.isfinite(lower), ">", lower), (~equal & np.isfinite(upper), "<", upper)):
                if np.any(mask):
                    model.addMConstr(matrix[mask], v, sense, rhs[mask])

            def callback(m, where):
                if where != grb.Callback.MIP:
                    return
                ub = float(m.cbGet(grb.Callback.MIP_OBJBST))
                lb = float(m.cbGet(grb.Callback.MIP_OBJBND))
                ub = ub if abs(ub) < grb.INFINITY else None
                lb = max(0., lb) if abs(lb) < grb.INFINITY else None
                decision = extension_decision(current, ub, lb, feasible=ub is not None,
                    relative_gain=relative_gain, uncertainty_fraction=uncertainty_fraction)
                if decision["decision"] != "refine_or_report_unresolved":
                    callback_decisions.append({"seconds": perf_counter()-started, "upper": ub,
                        "lower": lb, **decision})
                    m.terminate()

            model.optimize(callback)
            primal = np.asarray(v.X) if model.SolCount else None
            audit = primal_audit(raw, primal)
            ub = float(model.ObjVal) if model.SolCount else None
            lb = float(model.ObjBound)
            lb = max(0., lb) if abs(lb) < grb.INFINITY else None
            decision = extension_decision(current, ub, lb, feasible=audit["feasible"],
                relative_gain=relative_gain, uncertainty_fraction=uncertainty_fraction)
            info = {"raw_status": int(model.Status), "objective": ub, "bound": lb,
                "relative_gap": None if ub is None or lb is None or ub == 0 else (ub-lb)/abs(ub),
                "runtime_seconds": perf_counter()-started, "node_count": float(model.NodeCount),
                "requested_gap": requested_gap, "primal_audit": audit,
                "callback_decisions": callback_decisions, **decision,
                "certificate_scope": "objective improvement interval for this bounded single extension; approximate support choice"}
    return primal, info
