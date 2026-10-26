"""Numerical matrix-relaxation bounds for fixed-block omission.

R and X are symmetric, entrywise nonnegative, with each diagonal no smaller
than any off-diagonal in its row. These are relaxed matrix constraints, NOT a
legal-tree model. Orders constrain a*R[i,j]+b*X[i,j] <= a*R[k,j]+b*X[k,j];
k=-1 denotes the observed root and its known zero similarity.

For an omitted block C, the caller's hierarchy argument supplies, for every
j in C, a union of such order branches over i in C and k outside C. The bound
uses max_j min_branch(dual lower bound). It never substitutes a branch's
primal optimum for its dual lower bound. A j with any unverified branch is
excluded; all failures remain in the returned diagnostics.

Confirmed blocks add only block-external equalities. With these equalities,
the bound is conditional on those equalities; it must be compared with the
same confirmed-constrained completion class, not unconditional omission.
Zero matrix entries are allowed, and no coefficient upper bound is added.
All bounds are floating-point diagnostics, not strict arithmetic certificates.
"""

from __future__ import annotations

from itertools import combinations
from operator import index
from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linprog


def _data(P, Q, Y):
    P, Q, Y = (np.asarray(a, dtype=float) for a in (P, Q, Y))
    if P.ndim != 2 or Q.shape != P.shape or Y.shape != P.shape or min(P.shape) < 1:
        raise ValueError("P, Q, Y must have the same nonempty (n, T) shape")
    if not all(np.isfinite(a).all() for a in (P, Q, Y)):
        raise ValueError("P, Q, Y must be finite")
    return P, Q, Y


def _block(block, n):
    try:
        values = tuple(index(i) for i in block)
    except TypeError as exc:
        raise ValueError("Block indices must be integers") from exc
    if not values or len(set(values)) != len(values):
        raise ValueError("Blocks must be nonempty and contain distinct indices")
    if min(values) < 0 or max(values) >= n:
        raise ValueError("Block index out of range")
    return tuple(sorted(values))


def _confirmed(blocks, n):
    blocks = tuple(_block(block, n) for block in blocks)
    if len(set(blocks)) != len(blocks):
        raise ValueError("Duplicate confirmed blocks are not allowed")
    sets = tuple(set(block) for block in blocks)
    if any(a & b and not (a <= b or b <= a) for a, b in combinations(sets, 2)):
        raise ValueError("Confirmed blocks must form a laminar family")
    return blocks


def _orders(order, n):
    if order is None:
        return ()
    entries = tuple(order)
    if len(entries) == 3:
        try:
            single = tuple(index(i) for i in entries)
        except TypeError:
            single = None
        if single is not None:
            entries = (single,)
    result = []
    for triple in entries:
        try:
            triple = tuple(index(i) for i in triple)
        except TypeError as exc:
            raise ValueError("Order must be one triple or an iterable of triples") from exc
        if len(triple) != 3:
            raise ValueError("Each order must contain (i, j, k)")
        i, j, k = triple
        if not (0 <= i < n and 0 <= j < n and -1 <= k < n):
            raise ValueError("Only k may equal -1, denoting the observed root")
        result.append(triple)
    return tuple(result)


def _coefficient(value):
    coefficient = np.asarray(value, dtype=float)
    if coefficient.shape != (2,) or not np.isfinite(coefficient).all() or np.min(coefficient) < 0:
        raise ValueError("coefficient must be two finite nonnegative numbers")
    # (0, 0) is permitted and deliberately yields an uninformative order.
    return tuple(float(v) for v in coefficient)


def _number(value):
    return float(value) if np.isfinite(value) else None


def _basis(n):
    entries = tuple((i, j) for i in range(n) for j in range(i, n))
    matrices = []
    for i, j in entries:
        B = np.zeros((n, n))
        B[i, j] = B[j, i] = 1.
        matrices.append(B)
    return entries, matrices


def matrix_relaxation_fit(
    P, Q, Y, order=None, confirmed=(), coefficient=(1., 0.), *,
    certificate_tol: float = 1e-7, options: Mapping | None = None,
) -> dict:
    """Fit the relaxed matrices by sum L1, including all requested orders.

    The LP is min c.x, A.x<=b, E.x=0, x>=0. Its reported dual uses
    inequality multipliers l<=0, free equality multipliers nu, and
    A.T l + E.T nu <= c. Primal and dual feasibility and their gap are
    independently recomputed. No statistical or legal-tree claim is made.
    """
    P, Q, Y = _data(P, Q, Y)
    n, times = P.shape
    confirmed = _confirmed(confirmed, n)
    orders = _orders(order, n)
    coefficient = _coefficient(coefficient)
    if not np.isfinite(certificate_tol) or certificate_tol <= 0:
        raise ValueError("certificate_tol must be finite and positive")
    entries, bases = _basis(n)
    count, obs = len(entries), Y.size
    locations = {entry: k for k, entry in enumerate(entries)}
    def pos(i, j):
        return locations[min(i, j), max(i, j)]
    F = np.column_stack([(B @ P).ravel(order="C") for B in bases]
                        + [(B @ Q).ravel(order="C") for B in bases])
    variables = 2*count + obs
    y = Y.ravel(order="C")
    A = np.block([[F, -np.eye(obs)], [-F, -np.eye(obs)]])
    b = np.concatenate([y, -y])
    rows = []
    # Each diagonal >= each off-diagonal; no row-sum dominance is imposed.
    for offset in (0, count):
        for i in range(n):
            for j in range(n):
                if i != j:
                    row = np.zeros(variables)
                    row[offset + pos(i, j)] = 1.
                    row[offset + pos(i, i)] -= 1.
                    rows.append(row)
    for i, j, k in orders:
        row = np.zeros(variables)
        for offset, weight in zip((0, count), coefficient):
            row[offset + pos(i, j)] += weight
            if k != -1:
                row[offset + pos(k, j)] -= weight
        rows.append(row)
    if rows:
        A = np.vstack([A, np.asarray(rows)])
        b = np.concatenate([b, np.zeros(len(rows))])
    equalities = []
    for block in confirmed:
        outside = sorted(set(range(n))-set(block))
        anchor = block[0]
        for i in block[1:]:
            for ell in outside:
                for offset in (0, count):
                    row = np.zeros(variables)
                    row[offset + pos(i, ell)] += 1.
                    row[offset + pos(anchor, ell)] -= 1.
                    equalities.append(row)
    E = np.asarray(equalities).reshape((-1, variables))
    objective = np.concatenate([np.zeros(2*count), np.ones(obs)])
    result = linprog(objective, A_ub=A, b_ub=b,
                     A_eq=E if len(E) else None,
                     b_eq=np.zeros(len(E)) if len(E) else None,
                     bounds=(0, None), method="highs",
                     options=None if options is None else dict(options))
    status_names = {1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "solver_error"}
    report = {
        "status": status_names.get(result.status, "numerical_warning"),
        "success": False,
        "scope": "symmetric_nonnegative_diagonal_above_each_offdiagonal_matrix_relaxation",
        "legal_tree_model": False,
        "strict_arithmetic_certificate": False,
        "confirmed_conditional": bool(confirmed),
        "comparison_scope": ("same_confirmed_equalities_completion_class" if confirmed
                             else "unconditioned_completion_class"),
        "confirmed": [list(c) for c in confirmed],
        "orders": [list(t) for t in orders],
        "coefficient": list(coefficient),
        "coefficient_bounds": "entrywise_nonnegative_only",
        "certificate_tol": float(certificate_tol),
        "loss": None, "lp_objective": None, "dual_bound": None, "gap": None,
        "constraint_violation": None, "dual_violation": None,
        "numerically_primal_feasible": False, "numerically_dual_feasible": False,
        "R": None, "X": None, "dual": None,
        "solver_status": int(result.status), "solver_success": bool(result.success),
        "solver_message": str(result.message), "solver_nit": int(result.nit or 0),
        "solver_method": "highs",
        "dimensions": {"nodes": n, "times": times, "variables": variables,
                       "inequalities": len(b), "equalities": len(E)},
    }
    loss = lp_objective = gap = float("nan")
    primal_violation = dual_violation = float("inf")
    if result.x is not None and np.isfinite(result.x).all():
        x = np.asarray(result.x)
        R, X = np.zeros((n, n)), np.zeros((n, n))
        for k, (i, j) in enumerate(entries):
            R[i, j] = R[j, i] = x[k]
            X[i, j] = X[j, i] = x[count+k]
        loss = float(np.abs(Y - R@P - X@Q).sum())
        lp_objective = float(objective @ x)
        primal_violation = float(max(0., np.max(-x, initial=0.),
                                     np.max(A@x-b, initial=0.),
                                     np.max(np.abs(E@x), initial=0.)))
        report.update({"R": R.tolist(), "X": X.tolist(), "loss": loss,
                       "lp_objective": lp_objective,
                       "constraint_violation": primal_violation,
                       "numerically_primal_feasible": primal_violation <= certificate_tol})
    inequality = getattr(getattr(result, "ineqlin", None), "marginals", None)
    equality = getattr(getattr(result, "eqlin", None), "marginals", None)
    if not len(E):
        equality = np.empty(0)
    if (inequality is not None and equality is not None
            and np.shape(inequality) == (len(b),) and np.shape(equality) == (len(E),)
            and np.isfinite(inequality).all() and np.isfinite(equality).all()):
        inequality, equality = np.asarray(inequality), np.asarray(equality)
        reduced = objective - A.T@inequality - E.T@equality
        dual_violation = float(max(0., np.max(inequality, initial=0.),
                                   np.max(-reduced, initial=0.)))
        dual_objective = float(b@inequality)  # Equality right-hand sides are zero.
        report["dual"] = {
            "inequality_multipliers": inequality.tolist(),
            "equality_multipliers": equality.tolist(),
            "observation_vector": (inequality[:obs]-inequality[obs:2*obs]).tolist(),
            "reduced_cost": reduced.tolist(),
            "objective": dual_objective,
        }
        report["dual_violation"] = dual_violation
        report["numerically_dual_feasible"] = dual_violation <= certificate_tol
        if dual_violation <= certificate_tol:
            report["dual_bound"] = dual_objective
            gap = loss - dual_objective
            report["gap"] = _number(gap)
    scale = 1. + abs(loss) if np.isfinite(loss) else 1.
    verified = bool(result.success and primal_violation <= certificate_tol
                    and dual_violation <= certificate_tol and np.isfinite(gap)
                    and abs(gap) <= certificate_tol*scale
                    and abs(loss-lp_objective) <= certificate_tol*scale)
    if verified:
        report["status"] = "optimal"
        report["success"] = True
    return report


def block_omission_lower_bound(
    P, Q, Y, C, confirmed=(), coefficient=(1., 0.), *,
    certificate_tol: float = 1e-7, options: Mapping | None = None,
) -> dict:
    """Return max_j min_(i,k) numerical dual bounds for omission-order branches.

    i includes j. k ranges over observed terminals outside C; for C equal to
    all terminals, the only k is -1 (observed root, zero similarity).
    Singletons are accepted, although the principal use is internal blocks.

    A branch is usable only after an optimal, numerically verified solve.
    Any failed branch invalidates its whole j. Remaining complete j values
    may still yield a partial bound, with success=False and status='partial'.
    If all j values are invalid, lower_bound=None (trivial_bound remains 0).
    """
    P, Q, Y = _data(P, Q, Y)
    n = P.shape[0]
    C = _block(C, n)
    confirmed = _confirmed(confirmed, n)
    coefficient = _coefficient(coefficient)
    if C in confirmed:
        raise ValueError("Cannot test omission of a block supplied as confirmed")
    if not np.isfinite(certificate_tol) or certificate_tol <= 0:
        raise ValueError("certificate_tol must be finite and positive")
    outside = tuple(i for i in range(n) if i not in C) or (-1,)
    per_j, eligible, failed = [], [], []
    for j in C:
        branches = []
        for i in C:
            for k in outside:
                fit = matrix_relaxation_fit(
                    P, Q, Y, order=(i, j, k), confirmed=confirmed,
                    coefficient=coefficient, certificate_tol=certificate_tol,
                    options=options)
                usable = bool(fit["success"] and fit["status"] == "optimal"
                              and fit["dual_bound"] is not None
                              and np.isfinite(fit["dual_bound"]))
                branches.append({"i": i, "j": j, "k": k,
                                 "usable": usable, "fit": fit})
        complete = all(b["usable"] for b in branches)
        item = {"j": j, "all_branches_optimal": complete, "eligible": complete,
                "branches": branches, "branch_count": len(branches),
                "unclamped_min_dual_bound": None, "lower_bound": None,
                "min_primal_loss_diagnostic": None}
        primal_values = [b["fit"]["loss"] for b in branches
                         if b["fit"].get("loss") is not None and np.isfinite(b["fit"]["loss"])]
        if primal_values:
            item["min_primal_loss_diagnostic"] = float(min(primal_values))
        if complete:
            raw = float(min(b["fit"]["dual_bound"] for b in branches))
            item["unclamped_min_dual_bound"] = raw
            item["lower_bound"] = max(0., raw)  # Zero is a known feasible dual bound.
            eligible.append(item)
        else:
            failed.append(j)
        per_j.append(item)
    all_complete = not failed
    lower = max(item["lower_bound"] for item in eligible) if eligible else None
    return {
        "status": "optimal" if all_complete else ("partial" if eligible else "failed"),
        "success": all_complete,
        "scope": "max_min_of_numerical_matrix_relaxation_dual_bounds",
        "legal_tree_model": False,
        "strict_arithmetic_certificate": False,
        "C": list(C), "confirmed": [list(c) for c in confirmed],
        "confirmed_conditional": bool(confirmed),
        "comparison_scope": ("same_confirmed_equalities_completion_class" if confirmed
                             else "unconditioned_completion_class"),
        "coefficient": list(coefficient), "certificate_tol": float(certificate_tol),
        "lower_bound": lower, "trivial_bound": 0.,
        "bound_from_eligible_j_only": True,
        "eligible_j": [item["j"] for item in eligible],
        "incomplete_j": failed,
        "maximizing_j": [item["j"] for item in eligible
                         if item["lower_bound"] == lower],
        "all_branches_optimal": all_complete,
        "branch_count": sum(item["branch_count"] for item in per_j),
        "per_j": per_j,
    }
