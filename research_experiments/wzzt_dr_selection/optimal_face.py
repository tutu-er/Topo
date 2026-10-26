"""Historical-optimal-face diagnostics for a fixed-support nonnegative L1 model.

For finite, fixed matrices Fh, Ff and vectors yh, yf, set
    Lh(b) = ||yh - Fh b||_1, Lf(b) = ||yf - Ff b||_1, b >= 0,
    hstar = min Lh, d = min{Lf(b): b >= 0, Lh(b) = hstar}.
Then v(lam) = min_b Lh(b) + lam Lf(b) has v'(0+) = d. Compactness
of the coefficient set is unnecessary here: introducing nonnegative epigraph
variables gives a nonempty pointed polyhedron with finitely many vertices.
All objectives for lam>=0 are finite and bounded below by zero, attain their
minimum, and have a vertex optimum. Thus v is the minimum of finitely many
affine functions. Among the vertices optimal at zero, the smallest slope is d;
vertices with a positive intercept gap cannot win sufficiently close to zero.
The secondary LP is itself feasible, bounded below, and attains its minimum.
This argument is specific to these finite polyhedral problems, not a general
compactness-free envelope theorem.

For lam>=0, 0 <= v(lam)-hstar <= lam*d. Equality holds on some initial
interval but need not hold at lam=1. With a relaxed historical cap hstar+tau,
the returned secondary objective is a tolerance-dependent lower approximation
to d; the returned coefficient gives the feasible upper bound
    v(lam) <= Lh(b_face) + lam*Lf(b_face).
The floating-point hstar itself has a reported LP gap. No coefficient upper
bounds, strictly positive lower bounds, or regularizers are introduced.

This module considers one fixed support and one supplied predicted future Yf.
It does not establish completeness of a model pool, correctness of a selected
line, exchange of expectation and minimization for random future batches, or
exact finite-batch candidate rankings. All public outputs are JSON-friendly.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from scipy.optimize import linprog

from .core import block_features, fit_l1


def _data(P, Q, Y):
    P, Q, Y = (np.asarray(a, dtype=float) for a in (P, Q, Y))
    if P.ndim != 2 or Q.shape != P.shape or Y.shape != P.shape or min(P.shape) < 1:
        raise ValueError("P, Q, Y must have the same nonempty (n, T) shape")
    if not all(np.isfinite(a).all() for a in (P, Q, Y)):
        raise ValueError("All data must be finite")
    return P, Q, Y


def _number(value):
    return float(value) if np.isfinite(value) else None


def _history_summary(fit):
    return {
        "status": fit.status,
        "loss": _number(fit.loss),
        "dual_bound": _number(fit.dual_bound),
        "gap": _number(fit.gap),
        "constraint_violation": _number(fit.constraint_violation),
        "dual_violation": _number(fit.dual_violation),
        "solver_status": fit.solver_status,
        "message": fit.message,
    }


def fit_future_on_history_optimal_face(
    Ph, Qh, Yh, Pf, Qf, Yf, blocks: Iterable[Iterable[int]], *,
    history_atol: float = 0.0,
    history_rtol: float = 0.0,
    certificate_tol: float = 1e-7,
    frozen_beta=None,
    options: Mapping | None = None,
) -> dict:
    """Minimize future L1 loss subject to historical L1 loss <= hstar+tolerance.

    hstar is obtained from a verified numerical historical fit. Zero tolerance
    means no *additional* historical slack, not an exact-arithmetic certificate.
    frozen_beta defaults to the particular historical solution returned by the
    LP solver. Its historical compatibility is checked before comparing it.
    """
    Ph, Qh, Yh = _data(Ph, Qh, Yh)
    Pf, Qf, Yf = _data(Pf, Qf, Yf)
    if Ph.shape[0] != Pf.shape[0]:
        raise ValueError("History and future must have the same number of nodes")
    tolerances = np.asarray([history_atol, history_rtol, certificate_tol], dtype=float)
    if not np.isfinite(tolerances).all() or min(history_atol, history_rtol) < 0 or certificate_tol <= 0:
        raise ValueError("Face tolerances must be nonnegative; certificate_tol must be positive")
    history = fit_l1(Ph, Qh, Yh, tuple(blocks), certificate_tol=certificate_tol, options=options)
    report = {
        "scope": "fixed_support_fixed_predicted_future",
        "status": "history_fit_failed",
        "success": False,
        "blocks": [list(b) for b in history.blocks],
        "history_fit": _history_summary(history),
        "history_atol": float(history_atol),
        "history_rtol": float(history_rtol),
        "certificate_tol": float(certificate_tol),
        "exact_arithmetic_derivative_certificate": False,
        "coefficient_bounds": "nonnegative_only",
    }
    if not history.success:
        return report
    Fh = block_features(Ph, Qh, history.blocks)
    Ff = block_features(Pf, Qf, history.blocks)
    yh, yf = Yh.ravel(order="C"), Yf.ravel(order="C")
    mh, k = Fh.shape
    mf = Ff.shape[0]
    slack = float(history_atol + history_rtol * abs(history.loss))
    cap = float(history.loss + slack)
    # x = (beta, historical epigraph, future epigraph); every entry >= 0.
    A = np.block([
        [Fh, -np.eye(mh), np.zeros((mh, mf))],
        [-Fh, -np.eye(mh), np.zeros((mh, mf))],
        [Ff, np.zeros((mf, mh)), -np.eye(mf)],
        [-Ff, np.zeros((mf, mh)), -np.eye(mf)],
        [np.zeros((1, k)), np.ones((1, mh)), np.zeros((1, mf))],
    ])
    rhs = np.concatenate([yh, -yh, yf, -yf, [cap]])
    objective = np.concatenate([np.zeros(k + mh), np.ones(mf)])
    result = linprog(objective, A_ub=A, b_ub=rhs, bounds=(0, None),
                     method="highs", options=None if options is None else dict(options))
    report.update({
        "status": "secondary_fit_failed",
        "history_cap": cap,
        "added_history_slack": slack,
        "history_cap_minus_numerical_dual_bound": _number(cap - history.dual_bound),
        "solver_status": int(result.status),
        "message": str(result.message),
        "beta": None,
        "future_loss": None,
        "history_loss": None,
        "dual_bound": None,
        "gap": None,
        "constraint_violation": None,
        "dual_violation": None,
    })
    frozen = history.beta.copy() if frozen_beta is None else np.asarray(frozen_beta, dtype=float)
    if frozen.shape != (k,) or not np.isfinite(frozen).all():
        raise ValueError("frozen_beta must be a finite vector with two entries per block")
    frozen_h = float(np.abs(yh - Fh @ frozen).sum())
    frozen_f = float(np.abs(yf - Ff @ frozen).sum())
    frozen_ok = bool(np.min(frozen, initial=0.) >= -certificate_tol
                     and frozen_h <= cap + certificate_tol)
    report.update({
        "frozen_beta": frozen.tolist(),
        "frozen_history_loss": frozen_h,
        "frozen_future_loss": frozen_f,
        "frozen_history_compatible": frozen_ok,
        "frozen_minus_face_future_loss": None,
    })
    if result.x is None or not np.isfinite(result.x).all():
        return report
    beta = np.asarray(result.x[:k])
    h_loss = float(np.abs(yh - Fh @ beta).sum())
    f_loss = float(np.abs(yf - Ff @ beta).sum())
    lp_objective = float(objective @ result.x)
    violation = float(max(0., np.max(-result.x, initial=0.),
                          np.max(A @ result.x - rhs, initial=0.), h_loss - cap))
    report.update({
        "beta": beta.tolist(),
        "history_loss": h_loss,
        "history_excess_over_reference": h_loss - history.loss,
        "future_loss": f_loss,
        "lp_objective": lp_objective,
        "constraint_violation": violation,
    })
    marginal = getattr(getattr(result, "ineqlin", None), "marginals", None)
    dual_violation, gap = float("inf"), float("nan")
    if marginal is not None and np.isfinite(marginal).all():
        # Dual: max yh.qh+yf.qf-alpha*cap; |qh|<=alpha, |qf|<=1,
        # Fh.T qh+Ff.T qf<=0, alpha>=0.
        qh = marginal[:mh] - marginal[mh:2*mh]
        qf = marginal[2*mh:2*mh+mf] - marginal[2*mh+mf:2*mh+2*mf]
        alpha = float(-marginal[-1])
        dual_violation = float(max(0., -alpha,
                                   np.max(np.abs(qh) - alpha, initial=0.),
                                   np.max(np.abs(qf) - 1., initial=0.),
                                   np.max(Fh.T @ qh + Ff.T @ qf, initial=0.)))
        report.update({"dual_history": qh.tolist(), "dual_future": qf.tolist(),
                       "dual_history_cap_multiplier": alpha,
                       "dual_violation": dual_violation})
        if dual_violation <= certificate_tol:
            bound = float(yh @ qh + yf @ qf - alpha * cap)
            gap = f_loss - bound
            report.update({"dual_bound": bound, "gap": gap})
    scale = 1. + abs(f_loss)
    verified = bool(result.success and violation <= certificate_tol
                    and dual_violation <= certificate_tol and np.isfinite(gap)
                    and abs(gap) <= certificate_tol * scale
                    and abs(f_loss - lp_objective) <= certificate_tol * scale)
    report["success"] = verified
    if result.success:
        report["status"] = "optimal" if verified else "numerical_warning"
    if verified and frozen_ok:
        report["frozen_minus_face_future_loss"] = frozen_f - f_loss
    return report


def fixed_support_profile_diagnostic(
    Ph, Qh, Yh, Pf, Qf, Yf, blocks: Iterable[Iterable[int]], *,
    lambdas=(1e-6, 0.1, 1.0), frozen_beta=None,
    history_atol: float = 0.0, history_rtol: float = 0.0,
    certificate_tol: float = 1e-7, options: Mapping | None = None,
) -> dict:
    """Compare the optimal-face slope with finite-weight joint refits.

    All finite weights use the same historical data and predicted future Yf.
    The face solution's affine objective is a feasible upper bound, not an
    exact finite-weight loss or a proof of candidate ordering.
    """
    Ph, Qh, Yh = _data(Ph, Qh, Yh)
    Pf, Qf, Yf = _data(Pf, Qf, Yf)
    weights = tuple(float(value) for value in lambdas)
    if any(not np.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("Diagnostic lambdas must be finite and strictly positive")
    face = fit_future_on_history_optimal_face(
        Ph, Qh, Yh, Pf, Qf, Yf, blocks, frozen_beta=frozen_beta,
        history_atol=history_atol, history_rtol=history_rtol,
        certificate_tol=certificate_tol, options=options)
    report = {"optimal_face": face, "finite_weights": [],
              "claim": "fixed_support_local_slope_and_feasible_upper_bounds_only"}
    if not face["success"]:
        return report
    blocks = tuple(tuple(b) for b in face["blocks"])
    hstar = face["history_fit"]["loss"]
    for weight in weights:
        fitted = fit_l1(np.concatenate([Ph, weight*Pf], axis=1),
                        np.concatenate([Qh, weight*Qf], axis=1),
                        np.concatenate([Yh, weight*Yf], axis=1), blocks,
                        certificate_tol=certificate_tol, options=options)
        entry = {"lambda": weight, "status": fitted.status,
                 "value": _number(fitted.loss), "dual_bound": _number(fitted.dual_bound),
                 "gap": _number(fitted.gap), "secant_slope": None,
                 "face_solution_upper_bound": face["history_loss"] + weight*face["future_loss"],
                 "zero_added_slack_tangent_expression": hstar + weight*face["future_loss"]}
        if fitted.success:
            entry["secant_slope"] = float((fitted.loss - hstar) / weight)
            entry["upper_bound_minus_value"] = float(entry["face_solution_upper_bound"] - fitted.loss)
            entry["beta"] = fitted.beta.tolist()
        report["finite_weights"].append(entry)
    return report


def common_mode_counterexample() -> dict:
    """Two historically identical models: 2I and the all-ones shared block."""
    blocks = ((0,), (1,), (0, 1))
    Ph, Qh = np.ones((2, 1)), np.zeros((2, 1))
    Yh = np.full((2, 1), 2.)
    Pf, Qf = np.array([[1.], [0.]]), np.zeros((2, 1))
    Yf = np.ones((2, 1))
    diagonal = [2., 0., 2., 0., 0., 0.]
    shared = [0., 0., 0., 0., 1., 0.]
    report = fixed_support_profile_diagnostic(
        Ph, Qh, Yh, Pf, Qf, Yf, blocks, frozen_beta=diagonal)
    report["example"] = {
        "name": "common_history_diagonal_vs_shared",
        "Ph": Ph.tolist(), "Qh": Qh.tolist(), "Yh": Yh.tolist(),
        "Pf": Pf.tolist(), "Qf": Qf.tolist(), "Yf": Yf.tolist(),
        "diagonal_beta": diagonal, "shared_beta": shared,
        "hand_history_loss_both": 0., "hand_frozen_diagonal_future_loss": 2.,
        "hand_shared_future_loss": 0.,
        "noncompactness": "Qh=Qf=0 leaves all nonnegative x coefficients unbounded",
    }
    return report
