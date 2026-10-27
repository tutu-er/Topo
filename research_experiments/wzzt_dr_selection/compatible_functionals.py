"""Linear functionals over nonnegative, history-compatible WZZT coefficients.

The LP domain contains no artificial coefficient bounds. Every reported optimum
is checked against the original primal constraints and a maximization dual.
Unboundedness needs both a feasible point and a recession direction. These are
floating-point tolerance checks, not exact-arithmetic or statistical certificates.
"""
from __future__ import annotations

from collections.abc import Mapping
from operator import index

import numpy as np
from scipy.optimize import linprog

from .core import block_features, is_laminar


def _finite_vector(value, name, length=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (length is not None and result.size != length):
        raise ValueError(f"{name} must be a vector of length {length}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _constraint_pair(A, b, n, name):
    if A is None and b is None:
        return np.empty((0, n)), np.empty(0)
    if A is None or b is None:
        raise ValueError(f"{name} matrix and right hand side must be supplied together")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[1] != n or not np.isfinite(A).all():
        raise ValueError(f"{name} matrix must be finite with {n} columns")
    return A, _finite_vector(b, name + " right hand side", A.shape[0])


def _maximum(values):
    return float(np.max(values)) if np.size(values) else 0.0


def _primal_check(x, c, E, e, A, b):
    if x is None:
        return None
    try:
        x = np.asarray(x, dtype=float)
    except (ValueError, TypeError):
        return None
    if x.shape != c.shape or not np.isfinite(x).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        eq = _maximum(np.abs(E @ x - e))
        ub = max(0.0, _maximum(A @ x - b))
        nonnegative = max(0.0, _maximum(-x))
        value = float(c @ x)
    if not all(np.isfinite(v) for v in (eq, ub, nonnegative, value)):
        return None
    return {"x": x.tolist(), "value": value,
            "equality_violation": eq, "inequality_violation": ub,
            "nonnegative_violation": nonnegative,
            "violation": max(eq, ub, nonnegative)}


def _dual_check(result, c, E, e, A, b):
    """SciPy minimizes -c: negate its equality/inequality marginals."""
    try:
        u = -np.asarray(result.eqlin.marginals, dtype=float) if E.shape[0] else np.empty(0)
        w = -np.asarray(result.ineqlin.marginals, dtype=float) if A.shape[0] else np.empty(0)
    except (AttributeError, TypeError, ValueError):
        return None
    if u.shape != e.shape or w.shape != b.shape:
        return None
    if not np.isfinite(u).all() or not np.isfinite(w).all():
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        slack = E.T @ u + A.T @ w - c
        upper = float(e @ u + b @ w)
    if not np.isfinite(slack).all() or not np.isfinite(upper):
        return None
    violation = max(0.0, _maximum(-w), _maximum(-slack))
    return {"equality": u.tolist(), "inequality": w.tolist(),
            "constraint_slack": slack.tolist(), "value": upper,
            "violation": violation,
            "convention": "w>=0; A_ub.T*w+A_eq.T*u>=objective; upper=b_ub*w+b_eq*u"}


def maximize_nonnegative_lp(
    objective, A_eq=None, b_eq=None, A_ub=None, b_ub=None,
    feasible_x=None, tol=1e-7, options=None,
):
    """Maximize objective @ x with x>=0 and supplied linear constraints.

    value is a numerically feasible incumbent's objective, hence a numerical
    lower bound on the maximum; upper_bound uses a separately checked dual.
    An unbounded result retains a finite feasible value and uses
    supremum='positive_infinity', avoiding nonstandard JSON Infinity values.
    Infeasibility and failed ray searches remain unresolved unless a witness
    required for the positive result has actually been verified.
    """
    c = _finite_vector(objective, "objective")
    E, e = _constraint_pair(A_eq, b_eq, c.size, "equality")
    A, b = _constraint_pair(A_ub, b_ub, c.size, "inequality")
    tol = float(tol)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if options is not None and not isinstance(options, Mapping):
        raise ValueError("options must be a mapping or None")
    hint = None if feasible_x is None else _finite_vector(feasible_x, "feasible_x", c.size)
    hint_check = _primal_check(hint, c, E, e, A, b)
    hint_ok = hint_check is not None and hint_check["violation"] <= tol
    out = {
        "status": "unresolved", "success": False, "compatible": None,
        "bounded": None, "value": None, "upper_bound": None, "gap": None,
        "x": None, "dual": None, "direction": None, "ray_diagnostics": None,
        "primal_violation": None, "constraint_violation": None,
        "dual_violation": None, "primal_dual_violation": None,
        "solver_status": None, "message": "", "lp_calls": 0, "LPcalls": 0,
        "phase_results": [], "tolerance": tol,
        "verification_scope": "floating_point_tolerance",
        "strict_arithmetic_certificate": False,
        "feasible_hint_accepted": bool(hint_ok), "hint_diagnostics": hint_check,
        "supremum": None,
    }

    def retain_primal(check):
        if check is not None and check["violation"] <= tol:
            if out["value"] is None or check["value"] > out["value"]:
                out.update(x=check["x"], value=check["value"], compatible=True,
                           primal_violation=check["violation"],
                           constraint_violation=check["violation"])
                out["primal_diagnostics"] = check

    retain_primal(hint_check)

    def finish():
        out["LPcalls"] = out["lp_calls"]
        violations = [v for v in (out["primal_violation"], out["dual_violation"])
                      if v is not None]
        out["primal_dual_violation"] = max(violations) if violations else None
        return out

    def solve(cost, Eq, eq, Ub, ub, phase):
        out["lp_calls"] += 1
        try:
            result = linprog(
                cost, A_ub=Ub if Ub.shape[0] else None,
                b_ub=ub if Ub.shape[0] else None,
                A_eq=Eq if Eq.shape[0] else None,
                b_eq=eq if Eq.shape[0] else None,
                bounds=(0, None), method="highs",
                options=None if options is None else dict(options),
            )
            out["phase_results"].append({
                "phase": phase, "solver_status": int(result.status),
                "solver_success": bool(result.success), "message": str(result.message),
            })
            return result
        except Exception as exc:
            out["phase_results"].append({
                "phase": phase, "solver_status": None,
                "solver_success": False, "message": f"{type(exc).__name__}: {exc}",
            })
            return None

    if c.size == 0:
        empty_check = _primal_check(np.empty(0), c, E, e, A, b)
        retain_primal(empty_check)
        if out["x"] is not None:
            out.update(status="optimal", success=True, bounded=True,
                       upper_bound=0.0, gap=0.0, dual_violation=0.0,
                       message="Zero-variable feasible LP checked directly.",
                       dual={"equality": np.zeros(e.size).tolist(),
                             "inequality": np.zeros(b.size).tolist(),
                             "constraint_slack": [], "value": 0.0, "violation": 0.0})
        else:
            out.update(status="infeasible_verified", compatible=False,
                       message="Zero-variable constraints are inconsistent.")
        return finish()

    primary = solve(-c, E, e, A, b, "primary")
    if primary is None:
        out.update(status="solver_exception", message=out["phase_results"][-1]["message"])
        return finish()
    out.update(solver_status=int(primary.status), message=str(primary.message))
    check = _primal_check(getattr(primary, "x", None), c, E, e, A, b)
    if check is not None:
        out["solver_primal_diagnostics"] = check
    retain_primal(check)
    dual = _dual_check(primary, c, E, e, A, b)
    if dual is not None:
        out.update(dual=dual, dual_violation=dual["violation"])
        if dual["violation"] <= tol:
            out.update(upper_bound=dual["value"], bounded=True)
    if out["value"] is not None and out["upper_bound"] is not None:
        out["gap"] = out["upper_bound"] - out["value"]

    if primary.status == 0:
        gap_tol = tol * (1.0 + max(abs(out["value"] or 0.0),
                                 abs(out["upper_bound"] or 0.0)))
        out["gap_tolerance"] = gap_tol
        if (out["x"] is not None and out["upper_bound"] is not None
                and abs(out["gap"]) <= gap_tol):
            out.update(status="optimal", success=True)
        else:
            out["status"] = "optimal_unverified"
        return finish()
    if primary.status != 3:
        out["status"] = {1: "iteration_limit", 2: "infeasible_unverified",
                         4: "solver_error"}.get(int(primary.status), "solver_failure")
        return finish()

    out["status"] = "unbounded_unverified"
    if out["x"] is None:
        feasible = solve(np.zeros(c.size), E, e, A, b, "feasible_point")
        if feasible is not None:
            check = _primal_check(getattr(feasible, "x", None), c, E, e, A, b)
            out["feasibility_diagnostics"] = check
            retain_primal(check)
    if out["x"] is None:
        out["message"] += " A feasible point could not be verified."
        return finish()

    ray_A = np.vstack((A, -c[None, :]))
    ray_b = np.concatenate((np.zeros(b.size), [-1.0]))
    ray = solve(np.ones(c.size), E, np.zeros(e.size), ray_A, ray_b, "recession_direction")
    if ray is None:
        out["message"] += " The recession-direction solve failed."
        return finish()
    d_check = _primal_check(getattr(ray, "x", None), c, E, np.zeros(e.size),
                            ray_A, ray_b)
    out["ray_diagnostics"] = d_check
    if d_check is not None and d_check["violation"] <= tol:
        if out["upper_bound"] is not None:
            out["status"] = "inconsistent_solver_witnesses"
            out["message"] += " Both a numerical upper bound and an improving ray were returned."
        else:
            out.update(status="unbounded_verified", success=True, bounded=False,
                       direction=d_check["x"], supremum="positive_infinity",
                       gap=None)
            out["message"] += " Feasible point and improving recession direction verified numerically."
    else:
        out["message"] += " An improving recession direction could not be verified."
    return finish()


def maximize_history_functional(
    P, Q, Y, blocks, objective, epsilon=0, feasible_beta=None,
    tol=1e-7, options=None,
):
    """Maximize a linear coefficient functional over the full compatible set.

    P,Q,Y are n by T; beta is (r0,x0,r1,x1,...), in the supplied block order.
    The history constraint uses a sum of absolute residuals, not their mean.
    epsilon=0 uses equality constraints; positive epsilon uses an L1 epigraph.
    blocks must be laminar, but zero line weights remain admissible.
    """
    P, Q, Y = (np.asarray(v, dtype=float) for v in (P, Q, Y))
    if (P.ndim != 2 or min(P.shape) < 1 or Q.shape != P.shape
            or Y.shape != P.shape or not all(np.isfinite(v).all() for v in (P, Q, Y))):
        raise ValueError("P, Q, Y must have the same finite nonempty (n,T) shape")
    family = []
    for block in blocks:
        values = tuple(index(v) for v in block)
        if (not values or len(set(values)) != len(values)
                or min(values) < 0 or max(values) >= P.shape[0]):
            raise ValueError("Blocks must contain distinct valid node indices")
        family.append(tuple(sorted(values)))
    family = tuple(family)
    if len(set(family)) != len(family) or not is_laminar(family):
        raise ValueError("blocks must be a duplicate-free laminar family")
    F = block_features(P, Q, family)
    y = Y.ravel(order="C")
    k, m = F.shape[1], F.shape[0]
    c = _finite_vector(objective, "objective", k)
    epsilon = float(epsilon)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and nonnegative")
    hint = None if feasible_beta is None else _finite_vector(feasible_beta, "feasible_beta", k)
    if epsilon == 0:
        out = maximize_nonnegative_lp(c, A_eq=F, b_eq=y, feasible_x=hint,
                                      tol=tol, options=options)
    else:
        A = np.vstack((np.hstack((F, -np.eye(m))), np.hstack((-F, -np.eye(m))),
                       np.hstack((np.zeros((1, k)), np.ones((1, m))))))
        b = np.concatenate((y, -y, [epsilon]))
        full_hint = None if hint is None else np.concatenate((hint, np.abs(y - F @ hint)))
        out = maximize_nonnegative_lp(np.concatenate((c, np.zeros(m))),
                                      A_ub=A, b_ub=b, feasible_x=full_hint,
                                      tol=tol, options=options)
    out.update(blocks=[list(block) for block in family], epsilon=epsilon,
               history_shape=list(P.shape), beta=None, history_loss=None,
               history_budget_violation=None, direction_beta=None)
    if out["x"] is not None:
        beta = np.asarray(out["x"][:k])
        loss = float(np.abs(y - F @ beta).sum())
        budget_violation = max(0.0, loss - epsilon)
        out.update(beta=beta.tolist(), history_loss=loss,
                   history_budget_violation=budget_violation)
        if budget_violation > tol:
            out.update(status="history_budget_unverified", success=False, compatible=None,
                       candidate_beta=out["beta"], beta=None, value=None, gap=None)
    if out["direction"] is not None:
        db = np.asarray(out["direction"][:k])
        residual = F @ db
        out["direction_beta"] = db.tolist()
        out["history_ray_diagnostics"] = {
            "feature_residual_max": _maximum(np.abs(residual)),
            "feature_residual_l1": float(np.abs(residual).sum()),
            "objective_increase": float(c @ db),
            "nonnegative_violation": max(0.0, _maximum(-db)),
        }
        if (float(np.abs(residual).sum()) > tol
                or float(c @ db) < 1.0 - tol
                or max(0.0, _maximum(-db)) > tol):
            out.update(status="unbounded_unverified", success=False, bounded=None,
                       direction_beta=None, supremum=None)
    return out
