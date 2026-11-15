"""Read-only outer-ellipsoid upper bounds; no QCQP, fitting or AC execution.

If u'Wu<=b and ||u||²<=R², then u'(W+eps I)u<=b+eps R².
Whiten this positive-definite ellipsoid. For lambda>max(0,lmax(A)),
 z'Az+2a'z <= lambda ||z||²+a'(lambda I-A)^-1 a.
This is a valid upper bound over the original resource/budget domain. Dropping
other constraints only enlarges that domain. Scalar minimization tightens the
certificate; it does not generate or change a procurement action.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from ... import io


def ellipsoid_upper(A, a, c, W, budget, radius2, epsilon):
    """A[hour,r,r], a[hour,r], W[r,r]; return a floating-point upper bound."""
    A, a, W = np.asarray(A, float), np.asarray(a, float), np.asarray(W, float)
    if (A.ndim != 3 or a.shape != A.shape[:2] or W.shape != A.shape[1:]
            or not all(np.isfinite(v).all() for v in (A, a, W))
            or not np.isfinite([c, budget, radius2, epsilon]).all()
            or min(budget, radius2) < 0 or epsilon <= 0):
        raise ValueError('Finite aligned block quadratics and positive epsilon required')
    Q = (W+W.T)/2+epsilon*np.eye(len(W))
    L = np.linalg.cholesky(Q)
    inverse = np.linalg.solve(L, np.eye(len(W)))
    whitened = np.einsum('ij,hjk,lk->hil', inverse, A, inverse)
    whitened = (whitened+whitened.swapaxes(1, 2))/2
    values, vectors = np.linalg.eigh(whitened)
    linear = a@inverse.T
    projections = np.einsum('hij,hi->hj', vectors, linear)
    values, squares = values.ravel(), np.square(projections).ravel()
    B = float(budget+epsilon*radius2)
    largest = float(values.max(initial=0.))
    delta = 1e-10*max(1., abs(largest))
    lower = largest+delta
    def derivative(lam):
        return B-float(np.sum(squares/(lam-values)**2))
    if derivative(lower) >= 0:
        lam = lower
    else:
        upper = max(2*lower, lower+1.)
        for _ in range(200):
            if derivative(upper) >= 0:
                break
            upper = 2*upper+1
        else:
            raise ArithmeticError('Could not bracket the scalar dual bound')
        lam = float(brentq(derivative, lower, upper, xtol=1e-12, rtol=1e-12))
    separation = lam-values
    if not np.all(separation > 0) or lam <= 0:
        raise ArithmeticError('Invalid ellipsoid dual multiplier')
    quadratic_allowance = float(lam*B)
    linear_allowance = float(np.sum(squares/separation))
    raw = float(c+quadratic_allowance+linear_allowance)
    # Visible conservative numerical allowance, not interval-arithmetic proof.
    numerical_allowance = 1e-8*(1+abs(c)+abs(quadratic_allowance)+abs(linear_allowance))
    return dict(upper_bound=raw+numerical_allowance, raw_upper_bound=raw,
        numerical_allowance=numerical_allowance, epsilon=float(epsilon),
        outer_ellipsoid_budget=B, lambda_dual=lam,
        maximum_whitened_eigenvalue=largest, minimum_dual_spectral_gap=float(separation.min()),
        regularized_budget_condition=float(np.linalg.cond(Q)),
        floating_point_scope='Numerically checked sufficient bound; not an interval-arithmetic certificate')


def checked(ref):
    path = Path(ref['path'])
    if io.ref(path) != ref:
        raise ValueError('Source hash mismatch: '+str(path))
    return path


def run(source, out):
    from ...observations import Observation
    from ..reactive_flex_20261114.domain import PQIncrementCase
    from ..distribution_gate_20261114.evidence import DistributionEvidence
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    protocol = io.read(source/'protocol.json')
    entries = io.read(checked(protocol['inputs']))
    records = []
    for entry in entries:
        if entry['case'] != 'soumalas11':
            continue
        directory = source/'cases'/entry['id']
        metadata_ref = io.ref(directory/'guard_geometry.json')
        metadata = io.read(metadata_ref['path'])
        with np.load(checked(metadata['arrays']), allow_pickle=False) as z:
            matrices = {k:z[k].copy() for k in z.files}
        seal = io.read(checked(entry['prior_seal']))
        with np.load(checked(seal['planner']), allow_pickle=False) as z:
            forecast = z['forecast'].copy()
        score = io.read(checked(entry['score']))
        with np.load(checked(entry['files']['validation.npz']), allow_pickle=False) as z:
            val = Observation(z['P'], z['Q'], z['Y'], tuple(z['labels']), z['V'], z['root'])
        context = io.read(checked(entry['resource_context']))
        case = PQIncrementCase(context, val.labels, forecast, entry['base_kw'],
            q_resources=entry['q_resources'], fraction=.1)
        radius2 = float(np.maximum(abs(case.lower), abs(case.upper)).ravel()@np.maximum(abs(case.lower), abs(case.upper)).ravel())
        ws = []
        for guard in metadata['guards']:
            if guard['kind'] == 'response':
                Q = matrices[guard['matrix_key']]
                if not np.allclose(Q, np.kron(np.eye(case.H), Q[:case.R, :case.R]), rtol=0, atol=1e-15):
                    raise ValueError('Budget is not identical hour blocks')
                ws.append((guard['id'], Q[:case.R, :case.R]))
        if len(ws) > 1:
            ws.append(('mean_of_target_budgets', np.mean([w for _, w in ws], axis=0)))
        for mode in ('mgf_fixed', 'mgf_forecast'):
            evidence = DistributionEvidence(case, score, val, forecast, entry['base_kw'], mode=mode,
                meter_relative_sd=entry['meter_V_relative_sd'], forecast_relative_sd=entry['forecast_relative_sd'])
            for fraction in (.25, .35):
                gate_bounds = []
                for k, row in enumerate(evidence.rows):
                    candidates = []
                    for label, W in ws:
                        scale = max(float(np.linalg.eigvalsh(W).max()), np.finfo(float).tiny)
                        for relative_epsilon in (1e-6, 1e-4, 1e-2, 1.):
                            bound = ellipsoid_upper(evidence.gate_q[k], evidence.gate_a[k],
                                evidence.gate_c[k], W, fraction**2, radius2, scale*relative_epsilon)
                            candidates.append(dict(budget=label, relative_epsilon=relative_epsilon, **bound))
                    best = min(candidates, key=lambda r:r['upper_bound'])
                    gate_bounds.append(dict(target_native=row['target_native'], competitor_native=row['competitor_native'],
                        gate_buffer=1e-8, lambda_mgf=row['lambda_selected'], best=best,
                        sufficient_conflict=best['upper_bound'] < 1e-8, candidates=candidates))
                minimum = min(gate_bounds, key=lambda r:r['best']['upper_bound'])
                record = dict(id=entry['id'], mode=mode, response_rms_fraction=fraction,
                    resource_box_radius_squared=radius2, score=entry['score'], validation=entry['files']['validation.npz'],
                    forecast=seal['planner'], geometry=metadata_ref, resource_context=entry['resource_context'],
                    worst_gate=minimum, conflict_count=sum(r['sufficient_conflict'] for r in gate_bounds), gates=gate_bounds)
                records.append(record)
                print(entry['id'], mode, fraction, 'upper', minimum['best']['upper_bound'],
                    'pair', minimum['target_native'], minimum['competitor_native'], flush=True)
    result = dict(source_protocol=io.ref(source/'protocol.json'), source=io.ref(__file__), records=records,
        scope='Posthoc planning-only upper-bound diagnostic; no actions or original solver results changed; no truth, future observations, fitting, AC or QCQP calls')
    io.write(out/'report.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.source, args.out)
