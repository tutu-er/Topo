"""Re-estimated training-L1 families and operational 3% validation margins.

Uses the existing PoolLP, not a new topology identifier. Optimal-face ranges
are numerical bounds for a declared epsilon-optimal training set; they are
neither topology probabilities nor statistical coverage statements.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix

from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP, canonical
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import arrays, matrices


def features(p, q, pool):
    cols_r, cols_x = [], []
    for support in pool:
        z = np.zeros(p.shape[1])
        z[list(support)] = 1.
        cols_r.append(((p@z)[:, None]*z).ravel())
        cols_x.append(((q@z)[:, None]*z).ravel())
    return np.column_stack(cols_r+cols_x)


def _bound(model, name):
    try:
        value = float(model.getAttr(name))
        return value if np.isfinite(value) and abs(value) < 1e80 else None
    except Exception:
        return None


def validation(point, data):
    p, q, y = data
    r, x = matrices(p.shape[1], point["supports"], point["r"], point["x"])
    return float(np.mean(np.abs(p@r.T+q@x.T-y)))


def validation_range(lp, family, point, valid, *, face_epsilon=1e-8, seconds=8.):
    """Min/max validation L1 over ALL weights in the training optimal face.

    Lower endpoint is LP. Upper endpoint maximizes exact absolute residuals
    using bounded residual variables and Gurobi ABS mixed-integer constraints.
    A timed-out upper problem returns its upper bound, not its incumbent.
    """
    import gurobipy as gp
    if not np.isfinite((face_epsilon, seconds)).all() or face_epsilon < 0 or seconds <= 0:
        raise ValueError("nonnegative numerical face epsilon and positive time limit required")
    family = canonical(family)
    k = len(lp.pool)
    active = [lp.lookup[s] for s in family]
    upper = np.zeros(2*k)
    upper[active] = upper[np.asarray(active)+k] = lp.bound
    lp.weights.UB = upper
    lp.model.update()
    av = features(valid[0], valid[1], lp.pool)
    target = valid[2].ravel()
    full = point["full_fit"]
    theta = np.zeros(2*k)
    for support, r, x in zip(full["supports"], full["r"], full["x"], strict=True):
        j = lp.lookup[tuple(support)]
        theta[j], theta[j+k] = r, x
    train_value = float(np.mean(np.abs(lp.features@theta-lp.y.ravel())))
    if abs(train_value-full["objective"]) > 1e-9:
        raise AssertionError("training objective replay differs")
    # Numerical support pruning only affects non-singletons; bound its effect.
    nonleaf = [i for i,s in enumerate(lp.pool) if len(s) > 1]
    pruning = 1e-9*float(np.abs(av[:, nonleaf+list(np.asarray(nonleaf)+k)]).sum()/len(target))
    out = dict(face_epsilon=face_epsilon, training_optimum=train_value,
               parameter_domain=[0., float(lp.bound)], pruning_validation_bound=pruning,
               lower=None, upper=None, problems=[])
    residual_lower = np.minimum(av, 0.)@upper-target
    residual_upper = np.maximum(av, 0.)@upper-target
    for kind in ("minimum", "maximum"):
        model = lp.model.copy()
        try:
            model.Params.OutputFlag = 0
            model.Params.Threads = 1
            model.Params.TimeLimit = seconds
            model.Params.FeasibilityTol = 1e-9
            model.Params.OptimalityTol = 1e-9
            model.Params.NumericFocus = 2
            model.Params.DualReductions = 0
            model.Params.Method = 2
            all_vars = model.getVars()
            w = gp.MVar.fromlist(all_vars[:2*k])
            t = gp.MVar.fromlist(all_vars[2*k:])
            assert len(all_vars) == 2*k+lp.y.size
            model.addConstr(t.sum()/lp.y.size <= train_value+face_epsilon, name="training_epsilon_optimal")
            residual = model.addMVar(len(target), lb=residual_lower, ub=residual_upper, name="validation_residual")
            model.addConstr(csc_matrix(av)@w-residual == target)
            absolute = model.addMVar(len(target), lb=0.,
                        ub=np.maximum(np.abs(residual_lower), np.abs(residual_upper)), name="validation_abs")
            if kind == "minimum":
                model.addConstr(absolute >= residual)
                model.addConstr(absolute >= -residual)
                model.setObjective(absolute.sum()/len(target), gp.GRB.MINIMIZE)
            else:
                rv, tv = residual.tolist(), absolute.tolist()
                for i in range(len(target)):
                    if residual_lower[i] >= 0:
                        model.addConstr(tv[i] == rv[i])
                    elif residual_upper[i] <= 0:
                        model.addConstr(tv[i] == -rv[i])
                    else:
                        model.addGenConstrAbs(tv[i], rv[i], name=f"exact_abs_{i}")
                for v, value in zip(all_vars[:2*k], theta, strict=True):
                    v.Start = float(value)
                for v, value in zip(all_vars[2*k:], np.abs(lp.features@theta-lp.y.ravel()), strict=True):
                    v.Start = float(value)
                residual.Start = av@theta-target
                absolute.Start = np.abs(av@theta-target)
                model.Params.MIPGap = 1e-6
                model.setObjective(absolute.sum()/len(target), gp.GRB.MAXIMIZE)
            model.optimize()
            incumbent = _bound(model, "ObjVal") if model.SolCount else None
            bound = _bound(model, "ObjBound")
            if model.Status == gp.GRB.OPTIMAL and bound is None:
                bound = incumbent
            record = dict(kind=kind, status=int(model.Status), incumbent=incumbent,
                          best_bound=bound, runtime=float(model.Runtime),
                          exact_optimal=model.Status == gp.GRB.OPTIMAL,
                          variables=int(model.NumVars), binary_variables=int(model.NumBinVars))
            out["problems"].append(record)
            if kind == "minimum" and bound is not None:
                out["lower"] = max(0., bound-pruning)
            if kind == "maximum" and bound is not None:
                out["upper"] = bound+pruning
        finally:
            model.dispose()
    out["scope"] = "numerical profile interval over declared epsilon-optimal training weights; not statistical confidence"
    return out


def refit_pair(training, valid, coarse, fine, *, ranges=False, face_epsilon=1e-8, face_seconds=8.):
    """Refit every R/X coefficient on full training only; evaluate validation."""
    pval = arrays(valid)
    pool = canonical([*coarse, *fine])
    lp = PoolLP(training, pool, "raw", bound=2., lp_method=2)
    points, faces = {}, {}
    try:
        for key, family in (("coarse", coarse), ("fine", fine)):
            point = lp.fit(family)
            points[key] = dict(point, validation_mae=validation(point, pval))
            if ranges:
                faces[key] = validation_range(lp, family, point, pval,
                    face_epsilon=face_epsilon, seconds=face_seconds)
        j0, j1 = points["coarse"]["objective"], points["fine"]["objective"]
        v0, v1 = points["coarse"]["validation_mae"], points["fine"]["validation_mae"]
        result = dict(coarse_training_loss=j0, fine_training_loss=j1,
            coarse_validation_loss=v0, fine_validation_loss=v1,
            fitted_validation_exit_margin=v0-1.03*v1-1e-12,
            fitted_relative_validation_gap=v0/v1-1 if v1 else None,
            acceptance_margin=j0-j1-(1e-10+.001*j0),
            coarse_supports=points["coarse"]["supports"], fine_supports=points["fine"]["supports"],
            fine_weights=dict(r=points["fine"]["r"], x=points["fine"]["x"]),
            training_LP_solves=lp.solves, optimal_face_ranges=faces,
            safe_validation_exit_margin=None, optimistic_validation_exit_margin=None)
        if ranges:
            lo0, up0 = faces["coarse"]["lower"], faces["coarse"]["upper"]
            lo1, up1 = faces["fine"]["lower"], faces["fine"]["upper"]
            if lo0 is not None and up1 is not None:
                result["safe_validation_exit_margin"] = lo0-1.03*up1-1e-12
            if up0 is not None and lo1 is not None:
                result["optimistic_validation_exit_margin"] = up0-1.03*lo1-1e-12
        return result
    finally:
        lp.close()
