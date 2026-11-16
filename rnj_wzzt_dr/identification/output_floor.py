"""Training-only output contraction, before final validation model selection.

Only non-singleton supports are eligible, including an all-terminal common
support. The search path and its acceptance rules are not changed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from time import perf_counter

import numpy as np

if __package__:
    from ..observations import fingerprint
    from . import trees
    from .optimization import errors
else:
    from observations import fingerprint
    import trees
    from optimization import errors


NUMERICAL_RELATIVE_TOLERANCE = 1e-10
METRIC = 'rx75'
SCOPE = 'non_singleton'


class OutputFloorError(RuntimeError):
    """A failed output bank, with partial evidence but no accepted partial bank."""
    def __init__(self, message, *, receipt, qp_records=(), last_point=None):
        super().__init__(message)
        self.receipt = deepcopy(receipt)
        self.output_contraction = self.receipt
        self.qp_records = deepcopy(list(qp_records))
        self.last_point = deepcopy(last_point)


def _threshold(value):
    value = float(value)
    if not np.isfinite(value) or value < 0.:
        raise ValueError('internal_floor must be finite and nonnegative')
    return value


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        raise TimeoutError('Output contraction exhausted the original method deadline')


def calibrate(guidance, train=None):
    """Reuse existing training R/X arrays; this function performs no fitting."""
    R, X = np.asarray(guidance['R'], float), np.asarray(guidance['X'], float)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or not R.size or X.shape != R.shape:
        raise ValueError('Training calibration requires equally sized square R/X')
    if not np.isfinite(R).all() or not np.isfinite(X).all():
        raise ValueError('Training calibration R/X must be finite')
    means, counts = [], []
    for matrix in (R, X):
        distance = np.diag(matrix)[:, None] + np.diag(matrix)[None, :] - 2*matrix
        positive = distance[distance > 0.]
        counts.append(int(positive.size))
        means.append(max(float(positive.mean()) if positive.size else 1., 1e-12))
    a, b = .75 / means[0], .25 / means[1]
    median = float(np.median(np.diag(a*R + b*X)))
    if not np.isfinite([a, b, median]).all() or median <= 0.:
        raise ValueError('Training RX75 median root depth must be finite and positive')
    result = dict(a=a, b=b, m=median, median_root_depth=median, mu_r=means[0], mu_x=means[1],
        terminal_count=R.shape[0], positive_distance_counts=counts,
        formula='(.75*r/mu_r+.25*x/mu_x)/median_root_depth',
        source='Existing prepare_common training sensitivity arrays; no repeated ordered QP',
        pair_distance_numerical_floor=1e-12, validation_used=False, truth_used=False)
    if train is not None:
        if train.n != R.shape[0]:
            raise ValueError('Calibration dimension differs from training data')
        result.update(training_sha256=fingerprint(train), labels=list(train.labels), training_rows=train.points)
    return result


def _point(point, n):
    supports = point['supports']
    for C in supports:
        if not C or any(isinstance(i, (bool, np.bool_)) or int(i) != i for i in C):
            raise ValueError('Support coordinates must be integer-valued observed-column indices')
    family = [tuple(map(int, C)) for C in supports]
    if any(not C or tuple(sorted(set(C))) != C or min(C) < 0 or max(C) >= n for C in family):
        raise ValueError('Point supports must be ordered, unique, nonempty observed-column sets')
    if len(family) != len(set(family)) or not trees.is_laminar(family):
        raise ValueError('Point supports must be a unique laminar family')
    r, x = np.asarray(point['r'], float), np.asarray(point['x'], float)
    if r.shape != (len(family),) or x.shape != r.shape:
        raise ValueError('Point coefficient dimensions differ from supports')
    if not np.isfinite(r).all() or not np.isfinite(x).all() or np.any(r < 0.) or np.any(x < 0.):
        raise ValueError('Point coefficients must be finite and nonnegative')
    return family, r, x


def parameter_sha256(point):
    payload = dict(supports=point['supports'], r=point['r'], x=point['x'])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _strengths(point, calibration, n):
    family, r, x = _point(point, n)
    if calibration is None or int(calibration['terminal_count']) != n:
        raise ValueError('A matching frozen training calibration is required')
    a, b, m = (float(calibration[k]) for k in ('a', 'b', 'median_root_depth'))
    if not np.isfinite([a, b, m]).all() or min(a, b, m) <= 0.:
        raise ValueError('Invalid frozen training calibration')
    strength = (a*r + b*x) / m
    if not np.isfinite(strength).all():
        raise ValueError('Nonfinite normalized support strength')
    return family, strength


def _fit(fitter, family, deadline):
    """Retain the old convex fit; its noninterruptible soft budget is explicit."""
    _check_deadline(deadline)
    key = trees.canonical(family)
    before = len(fitter.records)
    cached = key in fitter.cache
    if not key and not cached:
        mse = float(np.mean(fitter.data.y**2))
        objective = mse / fitter.reference_mse
        fitter.cache[key] = dict(supports=[], r=[], x=[], objective=objective, mse=mse,
                                reference_mse=fitter.reference_mse, fit_loss='normalized_MSE')
        fitter.records.append(dict(family=(), solver='closed_form_empty_support', status='complete',
            objective=objective, lower_bound=objective, certificate_gap=0., seconds=0.,
            coefficient_bound=fitter.bound, rank=0, iterations=0))
    if key in fitter.cache:
        point = fitter.cache[key]
    else:
        previous_soft_limit = fitter.qp_seconds
        if deadline is not None:
            fitter.qp_seconds = min(previous_soft_limit, max(1e-12, deadline-perf_counter()))
        try:
            point = fitter.fit(key)
        finally:
            fitter.qp_seconds = previous_soft_limit
    _check_deadline(deadline)
    record = next((r for r in reversed(fitter.records)
                   if trees.canonical(r.get('family', ())) == key), None)
    return point, dict(cache_hit=cached, new_fit_records=len(fitter.records)-before,
                       record=deepcopy(record))


def contract_point(point, fitter, threshold, calibration, deadline=None):
    threshold = _threshold(threshold)
    if threshold == 0.:
        return point, None
    started = perf_counter()
    n = fitter.data.n
    current = point
    receipt = dict(status='running', threshold=threshold, metric=METRIC, scope=SCOPE,
        numerical_absolute_tolerance=NUMERICAL_RELATIVE_TOLERANCE*threshold,
        original_parameter_sha256=parameter_sha256(point), original_supports=deepcopy(point['supports']),
        steps=[], observed_labels=list(fitter.data.labels), observations_removed=False,
        basis_added=False, validation_used=False, truth_used=False,
        guarantee='Deterministic output contraction, not global constrained topology optimization')
    try:
        _check_deadline(deadline)
        family, _ = _strengths(point, calibration, n)
        if 'training_sha256' in calibration and calibration['training_sha256'] != fingerprint(fitter.data):
            raise ValueError('Calibration belongs to different training observations')
        if 'labels' in calibration and calibration['labels'] != list(fitter.data.labels):
            raise ValueError('Calibration observation labels changed')
        receipt['initial_training_loss'] = errors(fitter.data, point)
        for iteration in range(len(family)+1):
            _check_deadline(deadline)
            current_family, values = _strengths(current, calibration, n)
            eligible = [i for i, C in enumerate(current_family) if len(C) > 1]
            bad = [i for i in eligible if values[i]+receipt['numerical_absolute_tolerance'] < threshold]
            if not bad:
                receipt.update(status='complete', floor_satisfied=True,
                    final_parameter_sha256=parameter_sha256(current), final_supports=deepcopy(current['supports']),
                    final_strengths=[dict(support=list(C), strength=float(v), eligible=len(C)>1)
                                     for C, v in zip(current_family, values, strict=True)],
                    minimum_final_strength=min((float(values[i]) for i in eligible), default=None),
                    deletions=len(receipt['steps']), seconds=perf_counter()-started,
                    final_training_loss=errors(fitter.data, current),
                    stop_reason='eligible_supports_above_frozen_floor')
                _check_deadline(deadline)
                return current, receipt
            i = min(bad, key=lambda j: (float(values[j]), len(current_family[j]), current_family[j]))
            removed = current_family[i]
            remaining = trees.canonical(C for j, C in enumerate(current_family) if j != i)
            event = dict(iteration=iteration+1, removed_support=list(removed),
                removed_strength=float(values[i]), was_common=len(removed)==n,
                was_singleton=False, point_before=deepcopy(current),
                training_loss_before=errors(fitter.data, current))
            receipt['steps'].append(event)
            record_count = len(fitter.records)
            try:
                fitted, fit_receipt = _fit(fitter, remaining, deadline)
            except Exception:
                event['fit'] = dict(status='failed', new_records=deepcopy(fitter.records[record_count:]))
                raise
            if trees.canonical(fitted['supports']) != remaining:
                raise ValueError('Fixed-family QP changed the requested support family')
            event.update(fit=fit_receipt, point_after=deepcopy(fitted),
                         training_loss_after=errors(fitter.data, fitted))
            current = fitted
        raise RuntimeError('Output contraction exceeded the finite support-deletion bound')
    except Exception as exc:
        receipt.update(status='failed', floor_satisfied=False, seconds=perf_counter()-started,
                       error=type(exc).__name__ + ': ' + str(exc),
                       deadline_exceeded=deadline is not None and perf_counter() >= deadline)
        raise OutputFloorError(str(exc), receipt=receipt, qp_records=fitter.records, last_point=current) from exc


def contract_bank(points, fitter, threshold, calibration, deadline=None):
    threshold = _threshold(threshold)
    if threshold == 0.:
        return points, None
    started = perf_counter()
    receipt = dict(status='running', threshold=threshold, metric=METRIC, scope=SCOPE,
        calibration=deepcopy(calibration), candidate_count=len(points), candidates=[],
        qp_record_start=len(fitter.records), original_deadline=deadline,
        observations_removed=False, partial_bank_accepted=False,
        validation_used=False, truth_used=False)
    contracted = []
    try:
        if not points:
            raise ValueError('Output candidate bank must be nonempty')
        for index, point in enumerate(points):
            try:
                q, detail = contract_point(point, fitter, threshold, calibration, deadline)
            except OutputFloorError as exc:
                receipt['candidates'].append(dict(index=index, **exc.receipt))
                raise
            receipt['candidates'].append(dict(index=index, **detail))
            contracted.append(q)
        _check_deadline(deadline)
        receipt.update(status='complete', seconds=perf_counter()-started,
                       qp_record_stop=len(fitter.records), completed_candidates=len(contracted))
        validate_bank(contracted, receipt, fitter.data.n)
        return contracted, receipt
    except Exception as exc:
        receipt.update(status='failed', seconds=perf_counter()-started,
            qp_record_stop=len(fitter.records), completed_candidates=len(contracted),
            error=type(exc).__name__ + ': ' + str(exc),
            deadline_exceeded=deadline is not None and perf_counter() >= deadline)
        raise OutputFloorError(str(exc), receipt=receipt, qp_records=fitter.records,
                               last_point=getattr(exc, 'last_point', contracted[-1] if contracted else None)) from exc


def validate_bank(points, receipt, n):
    """Check declared post-contraction parameters; never repair or refit them."""
    if not isinstance(receipt, dict) or receipt.get('status') != 'complete':
        raise ValueError('Missing or incomplete output-contraction receipt')
    threshold = _threshold(receipt.get('threshold', 0.))
    if threshold <= 0. or receipt.get('metric') != METRIC or receipt.get('scope') != SCOPE:
        raise ValueError('Unexpected output-contraction rule')
    rows = receipt.get('candidates', [])
    if receipt.get('candidate_count') != len(points) or len(rows) != len(points):
        raise ValueError('Output-contraction candidate count is stale')
    for index, (point, row) in enumerate(zip(points, rows, strict=True)):
        if row.get('index') != index or row.get('status') != 'complete':
            raise ValueError('Output-contraction candidate mapping is stale')
        if (row.get('threshold') != threshold or row.get('metric') != METRIC or row.get('scope') != SCOPE):
            raise ValueError('Candidate output-contraction rule differs from the bank')
        family, values = _strengths(point, receipt['calibration'], n)
        if parameter_sha256(point) != row.get('final_parameter_sha256') or point['supports'] != row.get('final_supports'):
            raise ValueError('Candidate parameters changed after output contraction')
        final = row.get('final_strengths', [])
        if len(final) != len(family):
            raise ValueError('Missing final strength evidence')
        for C, value, recorded in zip(family, values, final, strict=True):
            if (list(C) != recorded.get('support') or recorded.get('eligible') != (len(C)>1)
                    or not np.isclose(value, recorded.get('strength', np.nan), rtol=1e-13, atol=1e-15)):
                raise ValueError('Final support strength evidence changed')
            if len(C)>1 and value+NUMERICAL_RELATIVE_TOLERANCE*threshold < threshold:
                raise ValueError('Candidate does not satisfy the declared output floor')
    return dict(passed=True, candidate_count=len(points), threshold=threshold, metric=METRIC, scope=SCOPE)
