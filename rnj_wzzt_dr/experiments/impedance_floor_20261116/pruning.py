"""Training-only, one-support-at-a-time contraction followed by ordinary QP.

This is a deterministic deletion heuristic, not a globally optimal constrained
topology search. It never adds a singleton/common basis or deletes observations.
The positive threshold is checked again after every fixed-support refit.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np

from ...observations import fingerprint
from ...identification import trees
from ...identification.optimization import errors


NUMERICAL_RELATIVE_TOLERANCE = 1e-10


class PruningError(RuntimeError):
    """Preserve the completed deletion trace and last valid point on failure."""

    def __init__(self, message, *, receipt, last_point):
        super().__init__(message)
        self.receipt = receipt
        self.last_point = last_point


def calibration_from_matrices(R, X, fit_diagnostics=None):
    """Freeze the old RX75 normalization, with no truth or final-tree input.

    The caller must supply the *training pre-tree ordered sensitivity* fit.
    Positive pair-distance means have the legacy 1e-12 numerical floor and
    fallback 1 when no positive distances exist. A nonpositive median depth
    is rejected: silently dividing by a new depth floor would change the rule.
    """
    r, x = np.asarray(R, float), np.asarray(X, float)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or not r.size or x.shape != r.shape:
        raise ValueError("Calibration R/X must be equal, nonempty square matrices")
    if not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError("Calibration R/X must be finite")
    means, counts = [], []
    for matrix in (r, x):
        distance = np.diag(matrix)[:, None] + np.diag(matrix)[None, :] - 2*matrix
        if not np.isfinite(distance).all():
            raise ValueError("Calibration pair distances overflowed")
        positive = distance[distance > 0.]
        counts.append(int(positive.size))
        means.append(max(float(positive.mean()) if positive.size else 1., 1e-12))
    a, b = .75/means[0], .25/means[1]
    median = float(np.median(np.diag(a*r+b*x)))
    if not np.isfinite([a, b, median]).all() or median <= 0.:
        raise ValueError("RX75 median root depth must be finite and positive")
    return dict(a=a, b=b, median_root_depth=median, m=median,
                mu_r=means[0], mu_x=means[1], terminal_count=r.shape[0],
                positive_distance_counts=counts, pair_distance_numerical_floor=1e-12,
                units="dimensionless normalized RX75 response / median root depth",
                formula="(a*r+b*x)/median_root_depth",
                fit_diagnostics=deepcopy(fit_diagnostics),
                calibration_contract="once from training-only trees.fit_sensitivity; frozen across points, thresholds and deletions",
                validation_used=False, truth_used=False)


def calibrate(train):
    """Fit the current ordered sensitivity model exactly once for this input."""
    started = perf_counter()
    R, X, diagnostic = trees.fit_sensitivity(train)
    result = calibration_from_matrices(R, X, diagnostic)
    result.update(training_sha256=fingerprint(train), labels=list(train.labels),
                  training_rows=int(len(train.y)), seconds=perf_counter()-started)
    return result


def _validate_point(point, n):
    supports = point["supports"]
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    if r.shape != (len(supports),) or x.shape != r.shape:
        raise ValueError("Point supports and coefficient vectors disagree")
    if not np.isfinite(r).all() or not np.isfinite(x).all() or np.any(r < 0.) or np.any(x < 0.):
        raise ValueError("Response coefficients must be finite and nonnegative")
    family = []
    for support in supports:
        if not support or any(isinstance(i, (bool, np.bool_)) or int(i) != i for i in support):
            raise ValueError("Each support must contain distinct integer column positions")
        C = tuple(sorted(map(int, support)))
        if len(C) != len(set(C)) or min(C) < 0 or max(C) >= n:
            raise ValueError("Support positions must be distinct and within the observed labels")
        family.append(C)
    if len(set(family)) != len(family) or not trees.is_laminar(family):
        raise ValueError("Point supports must form a duplicate-free laminar family")
    return family, r, x


def strengths(point, *, metric="rx75", calibration=None, zbase_ohm=None):
    """Return one strength per stored support, including zero coefficients.

    Response weights in Y=Vroot^2-V^2 are twice per-unit line impedances.
    Thus physical |Z|=(Zbase/2)*sqrt(r^2+x^2), in ohms.
    """
    r, x = np.asarray(point["r"], float), np.asarray(point["x"], float)
    if metric == "rx75":
        if calibration is None:
            raise ValueError("A frozen training-only RX75 calibration is required")
        a, b = float(calibration["a"]), float(calibration["b"])
        m = float(calibration["median_root_depth"])
        if not np.isfinite([a, b, m]).all() or min(a, b, m) <= 0.:
            raise ValueError("Calibration a, b and median depth must be finite and positive")
        values = (a*r+b*x)/m
    elif metric == "ohm":
        if zbase_ohm is None or not np.isfinite(zbase_ohm) or zbase_ohm <= 0.:
            raise ValueError("Physical strength requires positive Zbase in ohms")
        values = .5*float(zbase_ohm)*np.hypot(r, x)
    else:
        raise ValueError("metric must be 'rx75' or 'ohm'")
    if not np.isfinite(values).all():
        raise ValueError("Support strengths must be finite")
    return values


def _fit_remaining(fitter, family):
    key = trees.canonical(family)
    hit = key in fitter.cache
    before = len(fitter.records)
    if not key and not hit:
        # BVLS does not accept a zero-column design. Its unique prediction is
        # exactly zero; objective and global lower bound coincide analytically.
        mse = float(np.mean(np.asarray(fitter.data.y, float)**2))
        objective = mse/float(fitter.reference_mse)
        point = dict(supports=[], r=[], x=[], objective=objective, mse=mse,
                     reference_mse=float(fitter.reference_mse), fit_loss="normalized_MSE")
        record = dict(family=(), solver="closed_form_empty_support", status="complete",
                      objective=objective, lower_bound=objective, certificate_gap=0.,
                      seconds=0., iterations=0, rank=0, coefficient_bound=float(fitter.bound),
                      analytic_prediction="zero H with all observed columns retained")
        fitter.cache[key] = point
        fitter.records.append(record)
    point = fitter.cache[key] if hit or not key else fitter.fit(key)
    if len(fitter.records) > before:
        record = fitter.records[-1]
    else:
        record = next((r for r in reversed(fitter.records)
                       if trees.canonical(r.get("family", ())) == key), None)
    return point, dict(cache_hit=hit, new_fit_records=len(fitter.records)-before,
                       record=deepcopy(record),
                       certificate_gap=None if record is None else record.get("certificate_gap"))


def prune(point, fitter, threshold, *, metric="rx75", scope="all", calibration=None, zbase_ohm=None):
    """Return (point, receipt); positive thresholds restart from this point.

    A shared existing QPFitter provides train-only fits and cache reuse. The
    threshold zero baseline returns the very same object, without cleanup or
    refitting. Each positive-threshold iteration removes just the weakest
    violating support; exact-strength ties use (support length, positions).
    """
    started = perf_counter()
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0.:
        raise ValueError("threshold must be finite and nonnegative")
    if scope not in ("all", "non_singleton") or metric not in ("rx75", "ohm"):
        raise ValueError("Unknown threshold metric or support scope")
    n = int(fitter.data.n)
    family, _, _ = _validate_point(point, n)
    labels = list(map(int, fitter.data.labels))
    if len(labels) != n:
        raise ValueError("Observed labels must be retained in their original column order")
    if calibration is not None:
        if int(calibration.get("terminal_count", n)) != n:
            raise ValueError("Calibration terminal count differs from this training input")
        if "training_sha256" in calibration and calibration["training_sha256"] != fingerprint(fitter.data):
            raise ValueError("Calibration was fitted from a different training input")
        if "labels" in calibration and list(calibration["labels"]) != labels:
            raise ValueError("Calibration observation-label order changed")
    tolerance = NUMERICAL_RELATIVE_TOLERANCE*threshold
    receipt = dict(status="running", metric=metric, threshold=threshold, scope=scope,
                   units="ohm" if metric == "ohm" else "dimensionless",
                   zbase_ohm=None if zbase_ohm is None else float(zbase_ohm),
                   calibration=deepcopy(calibration), numerical_absolute_tolerance=tolerance,
                   numerical_tolerance_rule="1e-10 * threshold; no absolute floor",
                   original_supports=[list(C) for C in family], observed_labels=labels,
                   observed_terminal_count=n, removed_singletons=[], removed_common=[],
                   steps=[], maximum_checks=len(family)+1, fixed_supports_added=False,
                   observations_removed=False, validation_used=False, truth_used=False,
                   guarantee="final retained eligible coefficients meet the threshold within numerical tolerance; greedy deletion is not globally optimal")
    original_losses = errors(fitter.data, point)
    receipt["initial_training_loss"] = original_losses
    current = point
    if threshold == 0.:
        available = calibration is not None if metric == "rx75" else zbase_ohm is not None
        values = strengths(point, metric=metric, calibration=calibration, zbase_ohm=zbase_ohm) if available else None
        eligible = [i for i, C in enumerate(family) if scope == "all" or len(C) > 1]
        receipt.update(status="complete", stop_reason="zero_threshold_exact_original",
                       baseline_identity_preserved=True, refits=0, deletions=0,
                       floor_satisfied=True, minimum_final_strength=None if values is None else
                       min((float(values[i]) for i in eligible), default=None),
                       final_training_loss=original_losses, final_supports=[list(C) for C in family],
                       final_strengths=None if values is None else
                       [dict(support=list(C), strength=float(v), eligible=scope == "all" or len(C) > 1)
                        for C, v in zip(family, values, strict=True)],
                       seconds=perf_counter()-started)
        return point, receipt
    try:
        for check in range(len(family)+1):
            current_family, _, _ = _validate_point(current, n)
            values = strengths(current, metric=metric, calibration=calibration, zbase_ohm=zbase_ohm)
            eligible = [i for i, C in enumerate(current_family) if scope == "all" or len(C) > 1]
            violating = [i for i in eligible if values[i]+tolerance < threshold]
            if not violating:
                receipt.update(status="complete", stop_reason="retained_support_threshold_stable",
                               baseline_identity_preserved=False, floor_satisfied=True,
                               minimum_final_strength=min((float(values[i]) for i in eligible), default=None),
                               final_strengths=[dict(support=list(C), strength=float(v),
                                                    eligible=scope == "all" or len(C) > 1)
                                                for C, v in zip(current_family, values, strict=True)],
                               final_supports=[list(C) for C in current_family],
                               final_training_loss=errors(fitter.data, current),
                               refits=sum(not s["fit"]["cache_hit"] for s in receipt["steps"]),
                               deletions=len(receipt["steps"]), checks=check+1,
                               seconds=perf_counter()-started)
                return current, receipt
            index = min(violating, key=lambda i: (float(values[i]), len(current_family[i]), current_family[i]))
            removed = current_family[index]
            remaining = trees.canonical(C for i, C in enumerate(current_family) if i != index)
            event = dict(iteration=check+1, removed_support=list(removed),
                         removed_node_labels=[labels[i] for i in removed],
                         removed_strength=float(values[index]), threshold=threshold,
                         was_singleton=len(removed) == 1, was_common=len(removed) == n,
                         violating_count=len(violating), support_count_before=len(current_family),
                         support_count_after=len(remaining), training_loss_before=errors(fitter.data, current))
            receipt["steps"].append(event)
            previous_record_count = len(fitter.records)
            try:
                new_point, fit = _fit_remaining(fitter, remaining)
            except Exception:
                # QPFitter appends its objective/bound before reporting a soft
                # time-limit breach. Keep that evidence without accepting it.
                appended = fitter.records[previous_record_count:]
                event["fit"] = dict(status="failed", new_fit_records=len(appended),
                                    records=deepcopy(appended))
                raise
            new_family, _, _ = _validate_point(new_point, n)
            if trees.canonical(new_family) != remaining:
                raise RuntimeError("Fixed-support refit changed the requested family")
            event.update(fit=fit, training_loss_after=errors(fitter.data, new_point))
            event["training_mse_change"] = event["training_loss_after"]["mse"]-event["training_loss_before"]["mse"]
            if len(removed) == 1:
                receipt["removed_singletons"].append(dict(support=list(removed), label=labels[removed[0]],
                                                        evaluation="count as a removed structural singleton, never remove its observation"))
            if len(removed) == n:
                receipt["removed_common"].append(list(removed))
            current = new_point
        raise RuntimeError("Deletion exceeded the initial-support-count termination bound")
    except Exception as exc:
        receipt.update(status="failed", stop_reason="refit_or_threshold_audit_failed",
                       error_type=type(exc).__name__, error=str(exc),
                       floor_satisfied=False, seconds=perf_counter()-started,
                       last_valid_supports=deepcopy(current["supports"]))
        raise PruningError(str(exc), receipt=receipt, last_point=current) from exc
