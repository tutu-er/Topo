"""History-only priorities for local uncertain-support design tasks.

The weights solve an entropy-regularized planning problem, followed by a
strictly positive uniform floor. They are not topology posterior probabilities.
All three necessary target fits remain required, including low-weight ones.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np


def history_target_weights(candidates, historical_fits, *, epsilon=.15, tau=1.):
    """Return a complete target weight record; any required failure keeps all.

    Costs are attained historical half-SSE values. The task mixture is
    epsilon/3+(1-epsilon)*softmax(-cost/tau), not a predictive density.
    Invalid user configuration raises. Missing/failed fits return a failed
    record with every necessary support and weights=None, never a reduced set.
    """
    if (not np.isfinite(epsilon) or not 0 < epsilon <= 1
            or not np.isfinite(tau) or tau <= 0):
        raise ValueError("positive floor fraction at most one and positive finite half-SSE temperature required")
    expected = tuple(combinations(range(3), 2))
    rows, failures, by_support = [], [], {}
    seen_names = set()
    for candidate in candidates:
        name = str(candidate.name)
        if name in seen_names:
            failures.append(dict(candidate=name, kind="duplicate_candidate_name"))
        seen_names.add(name)
        if candidate.n != 3:
            failures.append(dict(candidate=name, kind="candidate_not_three_terminal"))
            continue
        proper = [tuple(sorted(s)) for s in candidate.supports if 1 < len(s) < candidate.n]
        if not proper:  # star remains a final likelihood candidate; no internal task.
            continue
        if len(proper) != 1 or proper[0] not in expected:
            failures.append(dict(candidate=name, kind="invalid_internal_target_family"))
            continue
        support = proper[0]
        if support in by_support:
            failures.append(dict(candidate=name, target_support=list(support), kind="duplicate_target_support"))
        by_support.setdefault(support, []).append(candidate)
    for support in expected:
        entries = by_support.get(support, [])
        row = dict(target_support=list(support), candidate=None,
                   success=False, loss_full_SSE=None, cost_half_SSE=None, weight=None)
        rows.append(row)
        if len(entries) != 1:
            row["failure"] = "missing_target_candidate" if not entries else "ambiguous_target_candidate"
            failures.append(dict(target_support=list(support), kind=row["failure"]))
            continue
        candidate = entries[0]
        row["candidate"] = candidate.name
        fit = historical_fits.get(candidate.name)
        if fit is None:
            row["failure"] = "missing_historical_fit"
        elif not bool(getattr(fit, "success", False)):
            row["failure"] = "historical_fit_unsuccessful"
        else:
            try:
                loss = float(fit.loss)
            except (ValueError, TypeError, AttributeError):
                loss = np.nan
            if not np.isfinite(loss) or loss < 0:
                row["failure"] = "invalid_historical_loss"
            else:
                row.update(success=True, loss_full_SSE=loss, cost_half_SSE=.5*loss)
        if not row["success"]:
            failures.append(dict(candidate=candidate.name, target_support=list(support), kind=row["failure"]))
    success = not failures and all(row["success"] for row in rows)
    weights = None
    if success:
        costs = np.array([row["cost_half_SSE"] for row in rows])
        shifted = -(costs-costs.min())/float(tau)
        raw = np.exp(shifted)
        soft = raw/raw.sum()
        values = float(epsilon)/3+(1-float(epsilon))*soft
        weights = {support: float(value) for support,value in zip(expected,values)}
        for row,value,base in zip(rows,values,soft):
            row.update(weight=float(value), entropy_regularized_component=float(base))
    return dict(success=bool(success), weights=weights, target_records=rows,
        failures=failures, epsilon=float(epsilon), tau=float(tau), tau_half_SSE=float(tau),
        formula="epsilon/3+(1-epsilon)*softmax(-(F_s(H)-min_s F_s(H))/tau)",
        entropy_objective="sum_s w_s F_s(H)+tau*sum_s w_s log(3*w_s)",
        cost_scope="attained_H_only_AC_half_SSE", minimum_weight_floor=float(epsilon)/3,
        necessary_targets_retained=[list(s) for s in expected],
        weights_are_topology_probabilities=False, uses_truth=False, uses_future_noise=False,
        policy_parameters_frozen_before_independent_calibration_required=True)