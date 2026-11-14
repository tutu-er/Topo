"""Optional line-segment initial points; never imported by the running solver.

For a convex physical resource domain, convex combinations of two resource-
feasible endpoints preserve that domain.  Probability/guard constraints along
the segment are polynomials of degree at most four.  This module enumerates
their numerical real roots and the piecewise-linear fee breakpoints, then
checks each returned control in the original formulas.  It does not solve or
certify the global QCQP, and numerical root enumeration is not an infeasibility
certificate.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from .solver import BUFFER, direct_audit


def line_coefficients(q, a, c, anchor, direction):
    """Ascending coefficients of a quadratic restricted to a straight line."""
    q, a = np.asarray(q, float), np.asarray(a, float)
    if q.ndim == 2:
        q = np.broadcast_to(q, (len(anchor),)+q.shape)
    q = (q+q.swapaxes(-1, -2))/2
    return np.array([
        np.einsum('hi,hij,hj->', anchor, q, anchor)+2*np.sum(a*anchor)+float(c),
        2*np.einsum('hi,hij,hj->', anchor, q, direction)+2*np.sum(a*direction),
        np.einsum('hi,hij,hj->', direction, q, direction)], float)


def constraints_on_line(evidence, anchor, direction, buffer=BUFFER):
    rows = []
    if evidence.mode == 'normal_fixed':
        for k, row in enumerate(evidence.rows):
            mu = line_coefficients(evidence.mu_q[k], evidence.mu_a[k],
                evidence.mu_c[k]-buffer, anchor, direction)
            variance = line_coefficients(evidence.variance_q[k], evidence.variance_a[k],
                evidence.variance_c[k], anchor, direction)
            z = float(evidence.z[k])
            squared = np.pad(np.polynomial.polynomial.polymul(mu, mu), (0, 4))[:5]
            squared[:3] -= z*z*variance
            rows.extend([dict(name=f'normal_nonnegative_mean_{k}', coefficients=mu),
                         dict(name=f'normal_squared_{k}', coefficients=squared)])
    else:
        for k in range(len(evidence.rows)):
            rows.append(dict(name=f'mgf_{k}', coefficients=line_coefficients(
                evidence.gate_q[k], evidence.gate_a[k], evidence.gate_c[k]-buffer, anchor, direction)))
    for k, (_, _, _, q, a) in enumerate(evidence.guards):
        rows.append(dict(name=f'guard_{k}', coefficients=line_coefficients(q, a, 0., anchor, direction)))
    return rows


def _scaled_value(coefficients, t):
    c = np.asarray(coefficients, np.longdouble)
    scale = max(np.max(np.abs(c), initial=np.longdouble(0)), np.longdouble(1e-300))
    return float(np.polynomial.polynomial.polyval(np.longdouble(t), c/scale))


def _root_partition(constraints, tolerance=1e-9):
    points, records = [0., 1.], []
    for row in constraints:
        c = np.trim_zeros(np.asarray(row['coefficients'], float), trim='b')
        record = dict(name=row['name'], degree=max(0, len(c)-1), accepted_roots=[], rejected_roots=[])
        if len(c) > 1:
            try:
                roots = np.polynomial.polynomial.polyroots(c/np.max(np.abs(c)))
                for root in roots:
                    real, imag = float(root.real), float(root.imag)
                    close_real = abs(imag) <= tolerance*(1+abs(real))
                    if close_real and -tolerance <= real <= 1+tolerance:
                        t = min(1., max(0., real))
                        residual = abs(_scaled_value(c, t))
                        if residual <= 1e-7:
                            points.append(t)
                            record['accepted_roots'].append(dict(t=t, imaginary_part=imag, scaled_residual=residual))
                        else:
                            record['rejected_roots'].append(dict(real=real, imaginary_part=imag,
                                reason='large_scaled_polynomial_residual', scaled_residual=residual))
                    elif close_real:
                        # Real roots outside the segment are irrelevant, not failures.
                        continue
                    elif abs(imag) <= 100*tolerance*(1+abs(real)) and -tolerance <= real <= 1+tolerance:
                        record['rejected_roots'].append(dict(real=real, imaginary_part=imag,
                            reason='near_real_but_outside_root_tolerance'))
            except Exception as exc:
                record['root_error'] = repr(exc)
        records.append(record)
    # Keep very narrow intervals; only exactly repeated floats are removed.
    return sorted(set(points)), records


def _polynomials_feasible(constraints, t, tolerance=1e-10):
    return all(_scaled_value(row['coefficients'], t) >= -tolerance for row in constraints)


def _candidate_points(constraints, anchor, direction):
    roots, records = _root_partition(constraints)
    zeros = []
    nonzero = direction != 0
    for t in (-anchor[nonzero]/direction[nonzero]).ravel():
        if np.isfinite(t) and 0 <= t <= 1:
            zeros.append(float(t))
    candidates, intervals = [], []
    for lo, hi in zip(roots[:-1], roots[1:]):
        if hi <= lo:
            continue
        midpoint = lo+(hi-lo)/2
        if not _polynomials_feasible(constraints, midpoint):
            continue
        intervals.append([lo, hi])
        # Constraints have no enumerated root within this interval.  A linear
        # absolute-action fee can change slope only at an action zero.
        for t in zeros:
            if lo < t < hi:
                candidates.append((t, 'fee_breakpoint'))
        candidates.append((midpoint, 'interval_midpoint'))
        for fraction in (1e-10, 1e-8, 1e-6, 1e-4, .01):
            candidates.append((lo+(hi-lo)*fraction, 'inward_from_left_root'))
            candidates.append((hi-(hi-lo)*fraction, 'inward_from_right_root'))
    for t in roots:
        if _polynomials_feasible(constraints, t):
            if t not in (0., 1.) and any(t == lo or t == hi for lo, hi in intervals):
                continue
            kind = 'original_endpoint' if t in (0., 1.) else 'isolated_or_boundary_root'
            candidates.append((t, kind))
    # Numerical boundary candidates are attempted only after inward points of
    # essentially equal fee.  Isolated feasible roots have no inward option.
    unique = {}
    for t, kind in candidates:
        if 0 <= t <= 1:
            unique.setdefault(float(t), kind)
    return [(t, kind) for t, kind in unique.items()], intervals, records, len(set(zeros))


def search_line(case, evidence, anchor, endpoint, *, name='line', buffer=BUFFER):
    began = perf_counter()
    anchor, endpoint = case.array(anchor).copy(), case.array(endpoint).copy()
    receipt = dict(name=name, feasible=False, source='polynomial_line_enumeration',
        global_qcqp_optimality_claimed=False, infeasibility_proved=False,
        endpoint_resource_audits=[case.audit(anchor), case.audit(endpoint)], buffer=float(buffer))
    if not all(a['passed'] for a in receipt['endpoint_resource_audits']):
        receipt.update(status='invalid_resource_endpoint', seconds=perf_counter()-began)
        return receipt, None
    direction = endpoint-anchor
    constraints = constraints_on_line(evidence, anchor, direction, buffer)
    candidates, intervals, root_records, zero_count = _candidate_points(constraints, anchor, direction)
    receipt.update(root_records=root_records, feasible_open_intervals=intervals,
        narrow_interval_count=sum(hi-lo < 1e-7 for lo, hi in intervals),
        fee_breakpoint_count=zero_count, candidate_count=len(candidates), direct_checks=[],
        coefficients=[dict(name=r['name'], coefficients=r['coefficients']) for r in constraints])
    # Prefer an inward point to its boundary when fees differ by <1e-9 yuan.
    ordered = sorted(candidates, key=lambda item: (round(float(case.fee(anchor+item[0]*direction)), 9),
        item[1] == 'isolated_or_boundary_root', float(case.fee(anchor+item[0]*direction)), item[0]))
    selected, best_fee = None, np.inf
    for t, kind in ordered:
        u = anchor+t*direction
        fee = float(case.fee(u))
        if selected is not None and fee > best_fee+1e-9:
            break
        try:
            audit = direct_audit(case, evidence, u)
            info = audit['evidence']
            gate = np.asarray(info['margins'], float)-buffer
            # A root may be rounded to either side.  Direct acceptance is
            # necessary; this additional small gate tolerance preserves the
            # intended numerical buffer without inventing a new risk model.
            passed = bool(audit['passed_acceptance'] and np.all(gate >= -1e-10))
            record = dict(t=t, kind=kind, activation_yuan=fee, passed=passed,
                minimum_encoded_gate=float(np.min(gate)) if len(gate) else None,
                minimum_guard=info['minimum_guard_gain'])
            receipt['direct_checks'].append(record)
            if passed and (selected is None or fee < best_fee-1e-9):
                selected, best_fee = u.copy(), fee
                receipt.update(feasible=True, selected_t=t, selected_kind=kind,
                    cost_yuan=fee, final_audit=audit)
        except Exception as exc:
            receipt['direct_checks'].append(dict(t=t, kind=kind, activation_yuan=fee,
                passed=False, error=repr(exc)))
    receipt.update(status='verified_line_point' if selected is not None else 'no_verified_point_on_enumerated_line',
        rejected_direct_candidates=sum(not r['passed'] for r in receipt['direct_checks']),
        numerical_root_issues=sum(len(r['rejected_roots'])+int('root_error' in r) for r in root_records),
        seconds=perf_counter()-began,
        scope='Only numerical line-segment candidates; near-multiple roots can create very narrow numerical intervals; no global-QCQP or numerical-root completeness certificate',
        resource_scope='Convex linear physical resource domain; each returned point also independently audited')
    return receipt, selected


def search_lines(case, evidence, anchor, seed_points, *, buffer=BUFFER):
    """Return the least-fee verified candidate among old-anchor/seed lines."""
    began = perf_counter()
    if not isinstance(seed_points, dict):
        seed_points = {f'seed_{i}': u for i, u in enumerate(seed_points)}
    records, best, best_name, best_fee = [], None, None, np.inf
    for name, endpoint in seed_points.items():
        try:
            receipt, point = search_line(case, evidence, anchor, endpoint, name=str(name), buffer=buffer)
        except Exception as exc:
            receipt, point = dict(name=str(name), feasible=False, status='exception', error=repr(exc)), None
        records.append(receipt)
        if point is not None and case.fee(point) < best_fee:
            best, best_name, best_fee = point.copy(), str(name), float(case.fee(point))
    return dict(feasible=best is not None, selected_line=best_name,
        cost_yuan=None if best is None else best_fee, lines=records, line_count=len(records),
        feasible_lines=sum(r['feasible'] for r in records), exceptions=sum(r['status']=='exception' for r in records),
        seconds=perf_counter()-began, optimization_calls=0,
        scope='New optional initialization only; existing formal solver and plans unchanged',
        global_qcqp_optimality_claimed=False, infeasibility_proved=False), best
