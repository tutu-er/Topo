"""Local joint-profile-margin prices on the existing EV operations MILP.

The only added rows are capped, scenario-wise affine margin models and a
power trust region.  The route, connection, SOC, loss, throughput, worker,
and terminal-service rows are copied unchanged from ``MobileServiceDomain``.
Solver bounds concern this *fixed affine surrogate*, not the AC objective.
An outer caller must refit every candidate and accept using the original
joint-profile score.  In particular, a zero gradient needs an inherited
nonzero-excitation plan; this module does not invent a quadratic signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..ev_mobile_service_domain import MobileServiceDomain
from ..ev_operations_milp import LinearMILP


@dataclass(frozen=True)
class MarginPlane:
    """One scenario's margin and derivative for an ordered candidate pair.

    ``gradient`` has shape [time, site], in margin units per kW.  ``value``
    is the joint-profile margin at the common parent ``u0``.  Pair labels
    may be hashable tuples, strings, or integers.  Scenario labels only
    annotate a plane; all planes with the same pair constrain that pair.
    """

    pair: Any
    value: float
    gradient: Any
    scenario: Any = None


def _finite_array(value, shape, name, *, nonnegative=False):
    try:
        answer = np.broadcast_to(np.asarray(value, float), shape).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must broadcast to {shape}") from exc
    if (not np.isfinite(answer).all()
            or (nonnegative and np.any(answer < 0))):
        raise ValueError(f"{name} must be finite"
                         + (" and nonnegative" if nonnegative else ""))
    return answer


def _plane_records(planes, margins, gradients, pair_ids, shape):
    if planes is not None:
        if margins is not None or gradients is not None or pair_ids is not None:
            raise ValueError("Use planes or dense margins/gradients, not both")
        records = []
        for plane in planes:
            if isinstance(plane, MarginPlane):
                records.append(plane)
            elif isinstance(plane, Mapping):
                records.append(MarginPlane(
                    plane["pair"], plane["value"], plane["gradient"],
                    plane.get("scenario")))
            else:
                if len(plane) not in (3, 4):
                    raise ValueError("A plane tuple is (pair, value, gradient[, scenario])")
                records.append(MarginPlane(*plane))
    else:
        if margins is None or gradients is None:
            raise ValueError("Supply both margins and gradients, or planes")
        values = np.asarray(margins, float)
        prices = np.asarray(gradients, float)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("margins must have nonempty shape [scenario,pair]")
        if prices.shape != values.shape + shape:
            raise ValueError("gradients must have shape [scenario,pair,time,site]")
        labels = list(range(values.shape[1])) if pair_ids is None else list(pair_ids)
        if len(labels) != values.shape[1] or len(set(labels)) != len(labels):
            raise ValueError("pair_ids must be distinct and match the pair axis")
        records = [MarginPlane(labels[p], values[s, p], prices[s, p], s)
                   for s in range(values.shape[0])
                   for p in range(values.shape[1])]
    if not records:
        raise ValueError("At least one margin plane is required")
    checked, labels = [], []
    for plane in records:
        try:
            hash(plane.pair)
        except TypeError as exc:
            raise ValueError("Pair labels must be hashable") from exc
        value = float(plane.value)
        gradient = np.asarray(plane.gradient, float)
        if (not np.isfinite(value) or gradient.shape != shape
                or not np.isfinite(gradient).all()):
            raise ValueError("Each plane needs a finite value and [time,site] gradient")
        if plane.pair not in labels:
            labels.append(plane.pair)
        checked.append(MarginPlane(plane.pair, value, gradient.copy(), plane.scenario))
    return checked, labels


def _pair_array(value, labels, name, *, nonnegative=False):
    if isinstance(value, Mapping):
        if set(value) != set(labels):
            raise ValueError(f"{name} mapping must contain exactly the plane pairs")
        value = [value[pair] for pair in labels]
    return _finite_array(value, (len(labels),), name, nonnegative=nonnegative)


def _linear_feasibility(model: LinearMILP, x, tolerance):
    if x is None:
        return dict(feasible=False, issues=["no_incumbent"], max_violation=np.inf)
    x = np.asarray(x, float)
    if x.shape != (model.n_variables,) or not np.isfinite(x).all():
        return dict(feasible=False, issues=["shape_or_nonfinite"], max_violation=np.inf)
    lower, upper = np.asarray(model.lower), np.asarray(model.upper)
    integer = np.asarray(model.integer, bool)
    rows = model.arrays()["constraints"]
    ax = rows.A @ x
    violations = dict(
        variable_bounds=float(max(np.max(lower - x, initial=0),
                                  np.max(x - upper, initial=0))),
        integrality=float(np.max(abs(x[integer] - np.round(x[integer])), initial=0)),
        linear_constraints=float(max(np.max(rows.lb - ax, initial=0),
                                     np.max(ax - rows.ub, initial=0))))
    return dict(feasible=max(violations.values()) <= tolerance,
                issues=[key for key, val in violations.items() if val > tolerance],
                max_violation=max(violations.values()), **violations)


def solve_local_margin_milp(
        domain: MobileServiceDomain, u0, *, margins=None, gradients=None,
        planes: Sequence[MarginPlane | Mapping | tuple] | None = None,
        pair_ids=None, tau=4., radius=1., pair_weights=None,
        time_limit=5., mip_rel_gap=1e-6, feasibility_tolerance=1e-6,
        **solver_options):
    """Maximize the capped worst-scenario affine margin of each pair.

    Dense input has margins [S,P] and gradients [S,P,T,J].  The sparse
    ``planes`` form permits different scenarios for different references.
    Pair weights default to 1/P, and are design weights, not probabilities.
    ``x`` is the original operation vector suitable for domain.check/decode;
    ``x_milp`` also includes pair scores.  The caller's domain is not mutated.

    The incumbent and best bound only bracket this frozen affine MILP.
    They do not bound any nonlinear profile likelihood or error probability.
    """
    if not isinstance(domain, MobileServiceDomain):
        raise TypeError("domain must be a MobileServiceDomain")
    shape = domain.indices.u.shape
    parent = np.asarray(u0, float)
    if parent.shape != shape or not np.isfinite(parent).all():
        raise ValueError("u0 must be finite [time,site]")
    if not np.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
        raise ValueError("feasibility_tolerance must be positive and finite")
    steps = _finite_array(radius, shape, "radius", nonnegative=True)
    records, labels = _plane_records(planes, margins, gradients, pair_ids, shape)
    caps = _pair_array(tau, labels, "tau")
    if np.any(caps <= 0):
        raise ValueError("tau must be positive")
    weights = (np.ones(len(labels)) if pair_weights is None else
               _pair_array(pair_weights, labels, "pair_weights", nonnegative=True))
    if weights.sum() <= 0:
        raise ValueError("At least one pair weight must be positive")
    weights = weights / weights.sum()
    model = domain.model.clone()
    original_size = domain.model.n_variables
    controls = domain.indices.u
    for i, center, step in zip(controls.ravel(), parent.ravel(), steps.ravel()):
        # Keep the physical power limits and every existing operation row.
        low = max(model.lower[int(i)], center - step)
        high = min(model.upper[int(i)], center + step)
        if low > high:
            raise ValueError("Power trust region misses the operation bounds")
        model.lower[int(i)], model.upper[int(i)] = float(low), float(high)
    scores = model.add_variables("joint_margin_score", len(labels),
                                 lower=-np.inf, upper=caps)
    pair_index = {pair: p for p, pair in enumerate(labels)}
    for plane in records:
        row = {int(scores[pair_index[plane.pair]]): 1.}
        for index, price in zip(controls.ravel(), plane.gradient.ravel()):
            if price:
                row[int(index)] = -float(price)
        rhs = plane.value - float(np.sum(plane.gradient * parent))
        model.add_row(row, upper=rhs)
    objective = {int(index): float(weight) for index, weight in zip(scores, weights)}
    result = model.solve(objective, cmax=True, time_limit=time_limit,
                         mip_rel_gap=mip_rel_gap, **solver_options)
    full_x = None if result.x is None else np.asarray(result.x, float).copy()
    operation_x = None if full_x is None else full_x[:original_size].copy()
    proxy_check = _linear_feasibility(model, full_x, feasibility_tolerance)
    physical_check = domain.check(operation_x, tolerance=feasibility_tolerance)
    u = None if operation_x is None else operation_x[controls].copy()
    values = None
    plane_values = None
    if u is not None:
        values = caps.copy()
        plane_values = np.array([plane.value + float(np.sum(plane.gradient * (u - parent)))
                                 for plane in records])
        for plane, value in zip(records, plane_values):
            p = pair_index[plane.pair]
            values[p] = min(values[p], value)
    parent_values = caps.copy()
    for plane in records:
        p = pair_index[plane.pair]
        parent_values[p] = min(parent_values[p], plane.value)
    incumbent = getattr(result, "objective_value", None)
    bound = getattr(result, "objective_bound", None)
    gap = (None if incumbent is None or bound is None
           else float(bound - incumbent))
    return dict(
        success=bool(proxy_check["feasible"] and physical_check["feasible"]),
        optimal=bool(result.success), solver_status=int(result.status),
        solver_message=str(result.message),
        bound_scope="fixed_affine_margin_MILP_only_not_AC_or_probability",
        surrogate_incumbent=None if incumbent is None else float(incumbent),
        surrogate_best_bound=None if bound is None else float(bound),
        surrogate_absolute_gap=gap,
        solver_mip_gap=(None if getattr(result, "mip_gap", None) is None
                        else float(result.mip_gap)),
        objective_recomputed=None if values is None else float(weights @ values),
        parent_affine_objective=float(weights @ parent_values),
        x=operation_x, operation_x=operation_x, x_milp=full_x, u=u, u_kw=u,
        pair_ids=labels, pair_weights=weights, pair_values=values,
        plane_values=plane_values, feasibility=physical_check, check=physical_check,
        surrogate_feasibility=proxy_check,
        all_gradient_zero=all(not np.any(p.gradient) for p in records),
        all_pairs_capped_at_parent=bool(np.all(parent_values >= caps - 1e-12)),
        n_operation_variables=original_size, n_milp_variables=model.n_variables,
        n_planes=len(records), radius=steps)


# A concise alias for experiment callers; both names have the same scope.
solve_margin_price_milp = solve_local_margin_milp
