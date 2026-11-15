"""Historical-only quadratic restrictions on frozen-model extrapolation.

The response statistic is the *incremental prediction variance* of the
unconstrained fixed-support linear design under a declared voltage-meter law.
It retains all R/X columns, including coefficients at a BVLS bound. It is not a
bound on systematic response error, nor a confidence set for the selected,
bounded fit. No fitting, optimizer, physical network or future data is read.
"""
from __future__ import annotations

import numpy as np

from ... import io
from ..absolute_identification_20261114.solver import target_set


def design_blocks(p, q, supports):
    """Return time x output x parameter; order is all R then all X."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim != 2 or p.shape != q.shape or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Aligned finite P/Q matrices required')
    n = p.shape[1]
    sets = [tuple(map(int, s)) for s in supports]
    if (not sets or len(set(sets)) != len(sets) or any(not s or tuple(sorted(set(s))) != s
            or min(s) < 0 or max(s) >= n for s in sets)):
        raise ValueError('Unique nonempty sorted supports in terminal index order required')
    z = np.array([[k in s for k in range(n)] for s in sets], float)
    return np.concatenate([(values @ z.T)[:, None, :] * z.T[None]
                           for values in (p, q)], axis=2)


def parameter_covariance(blocks, root_variance, individual_variance):
    """Full-rank OLS sandwich; column scaling changes no mathematical estimator.

    A rank-deficient design is rejected. In particular, no pseudoinverse turns
    an unobserved parameter direction into zero uncertainty.
    """
    blocks = np.asarray(blocks, float)
    if blocks.ndim != 3 or not np.isfinite(blocks).all():
        raise ValueError('Finite time/output/parameter design required')
    if (not np.isfinite([root_variance, individual_variance]).all()
            or min(root_variance, individual_variance) < 0):
        raise ValueError('Finite nonnegative voltage-noise variances required')
    t, n, m = blocks.shape
    F = blocks.reshape(t*n, m)
    norms = np.linalg.norm(F, axis=0)
    if np.any(norms == 0):
        raise ValueError('Unobserved fixed-support parameter column; no finite design covariance')
    U, singular, Vh = np.linalg.svd(F / norms, full_matrices=False)
    threshold = np.finfo(float).eps * max(F.shape) * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    if rank != m:
        raise ValueError(f'Rank-deficient fixed-support design: rank {rank} of {m}')
    # L=(F'F)^-1 F', obtained by full-rank SVD without squaring the condition.
    L = ((Vh.T / singular) @ U.T) / norms[:, None]
    common = L.reshape(m, t, n).sum(axis=2)
    C = individual_variance*(L @ L.T) + root_variance*(common @ common.T)
    C = (C + C.T)/2
    return C, F.T @ F, dict(rank=rank, parameter_count=m,
        singular_threshold=float(threshold), column_scaled_condition=float(singular[0]/singular[-1]),
        singular_values=singular.tolist(), column_norms=norms.tolist(),
        inverse_design_identity_error=float(np.max(np.abs(L @ F - np.eye(m)))))


def _resource_map(case, base_kw):
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError('Positive finite common P/Q power base required')
    # The public response map supports P, Q and coupled columns alike.
    if hasattr(case, 'response_effect'):
        B = np.asarray(case.response_effect(np.eye(2*case.N), base_kw), float)
    elif hasattr(case, 'B'):
        B = np.asarray(case.B, float)/base_kw
    else:
        raise ValueError('Explicit linear P/Q resource response map required')
    if B.shape != (2*case.N, case.R) or not np.isfinite(B).all():
        raise ValueError('Resource response map dimension or finite-value mismatch')
    return B


def _psd_receipt(Q):
    Q = np.asarray(Q, float)
    symmetry = float(np.max(abs(Q-Q.T)))
    minimum = float(np.linalg.eigvalsh((Q+Q.T)/2).min(initial=0.))
    tolerance = 1e-11*max(1., float(np.linalg.norm(Q, 2)))
    if symmetry > tolerance or minimum < -tolerance:
        raise ArithmeticError('Constructed geometry is not numerically symmetric PSD')
    return dict(symmetry_error=symmetry, minimum_eigenvalue=minimum, psd_tolerance=tolerance)


def build(case, score, train, validation, base_kw, meter_relative_sd, target_mass=.99, *,
          pq_rho=4., response_rms_fraction=.25, target_native_indices=None):
    """Return arrays for an NPZ and JSON-safe metadata for two guard variants.

    ``pq`` is the mean node-standardized squared departure, limit rho^2.
    ``response_native{k}`` is incremental response-variance RMS squared divided
    by score sigma^2, limit response_rms_fraction^2. Both full matrices use
    hourly row-major resource controls. No forecast is needed for construction.
    """
    H, R, n = int(case.H), int(case.R), int(case.N)
    if H <= 0 or R <= 0 or case.repeat <= 0:
        raise ValueError('Positive decision and repeat dimensions required')
    labels = tuple(case.labels)
    if tuple(train.labels) != labels or tuple(validation.labels) != labels or tuple(score['labels']) != labels:
        raise ValueError('History, score and resource terminal order differ')
    sigma2 = float(score['sigma2'])
    if (not np.isfinite([sigma2, meter_relative_sd, pq_rho, response_rms_fraction]).all()
            or sigma2 <= 0 or meter_relative_sd < 0 or pq_rho < 0 or response_rms_fraction < 0):
        raise ValueError('Invalid score/noise scale or guard budget')
    p, q = np.asarray(train.p, float), np.asarray(train.q, float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != n or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Aligned finite training P/Q required')
    B = _resource_map(case, base_kw)
    centered_p, centered_q = p-p.mean(axis=0), q-q.mean(axis=0)
    p_energy = np.sum(centered_p**2, axis=0)
    if np.any(p_energy <= np.finfo(float).tiny):
        raise ValueError('Constant training P prevents an identifiable affine P/Q slope')
    slope = np.sum(centered_p*centered_q, axis=0)/p_energy
    intercept = q.mean(axis=0)-slope*p.mean(axis=0)
    residual = q-p*slope[None]-intercept[None]
    residual_variance = np.mean(residual**2, axis=0)
    departure_scale = float(residual_variance.sum())
    residual_floor = np.finfo(float).eps*np.maximum(np.mean(q*q, axis=0), np.finfo(float).tiny)
    if np.any(residual_variance <= residual_floor):
        raise ValueError('Zero or numerically negligible node affine residual scale; no implicit floor is applied')
    departure = B[n:] - slope[:, None]*B[:n]
    standardized_departure = departure/np.sqrt(residual_variance)[:, None]
    pq_hour = standardized_departure.T @ standardized_departure / (H*n)
    arrays = dict(pq=np.kron(np.eye(H), pq_hour), resource_map_pu=B,
        pq_slope=slope, pq_intercept_pu=intercept, pq_residual_rms_pu=np.sqrt(residual_variance))
    guard_rows = [dict(id='pq', matrix_key='pq', kind='pq', limit=float(pq_rho**2),
        units='dimensionless: mean over hours and nodes of departure squared / node training affine residual variance',
        source='training P/Q; centered affine slopes; no intercept in action departures',
        **_psd_receipt(pq_hour))]
    voltage, root = validation.voltage, validation.root
    if voltage is None or root is None:
        raise ValueError('Historical validation V/root required; no unit-voltage fallback')
    voltage, root = np.asarray(voltage, float), np.asarray(root, float)
    if (voltage.shape != validation.y.shape or root.shape != (len(voltage),)
            or not np.isfinite(voltage).all() or not np.isfinite(root).all()
            or np.any(voltage <= 0) or np.any(root <= 0)):
        raise ValueError('Aligned positive finite historical V/root required')
    root_median, voltage_median = float(np.median(root)), float(np.median(voltage))
    root_variance = (2*meter_relative_sd*root_median**2)**2
    individual_variance = (2*meter_relative_sd*voltage_median**2)**2
    parameters = score['parameters']
    selected, weights = target_set([r['history_sse'] for r in parameters], sigma2, target_mass)
    native = [int(parameters[i]['index']) for i in selected]
    if len(set(native)) != len(native):
        raise ValueError('Duplicate target native index')
    if target_native_indices is not None and list(map(int, target_native_indices)) != native:
        raise ValueError('Provided target native indices differ from the frozen historical target set')
    details = []
    for rep in selected:
        parameter = parameters[rep]
        index, point = int(parameter['index']), parameter['point']
        supports = point['supports']
        blocks = design_blocks(p, q, supports)
        theta = np.r_[point['r'], point['x']]
        frozen_H = np.asarray(parameter['H'], float)
        if (frozen_H.shape != (n, 2*n) or theta.shape != (blocks.shape[-1],)
                or not np.isfinite(theta).all()):
            raise ValueError('Frozen support/parameter/response dimensions differ')
        identity_blocks = design_blocks(np.eye(2*n)[:, :n], np.eye(2*n)[:, n:], supports)
        reconstructed_H = np.einsum('tnk,k->nt', identity_blocks, theta)
        if not np.allclose(frozen_H, reconstructed_H, rtol=1e-11, atol=1e-13):
            raise ValueError('Frozen supports and coefficients do not reconstruct the scoring matrix')
        if 'native_bank' in score and not np.array_equal(frozen_H, np.asarray(score['native_bank'][index]['H'])):
            raise ValueError('Representative scoring matrix differs from its native point')
        C, G, info = parameter_covariance(blocks, root_variance, individual_variance)
        action_blocks = design_blocks(B[:n].T, B[n:].T, supports)  # resource x output x parameter
        Qhour = np.einsum('rnk,kl,snl->rs', action_blocks, C, action_blocks, optimize=True)
        # repeat * sum(hour variances)/(H*repeat*N*sigma^2); repeat cancels.
        Qhour = (Qhour+Qhour.T)/(2*H*n*sigma2)
        key = f'response_native{index}'
        arrays[key] = np.kron(np.eye(H), Qhour)
        arrays[f'covariance_native{index}'] = C
        arrays[f'gram_native{index}'] = G
        guard_rows.append(dict(id=key, matrix_key=key, kind='response', target_rep=int(rep),
            target_native=index, limit=float(response_rms_fraction**2),
            units='dimensionless: mean incremental predicted Y variance / frozen score variance',
            source='original training fixed-support design; declared historical voltage-meter sandwich',
            **_psd_receipt(Qhour)))
        details.append(dict(target_rep=int(rep), target_native=index, supports=supports,
            parameter_order='all R supports in listed order, then all X supports in same order',
            all_bound_parameters_retained=True, coefficient_count=len(theta),
            at_zero_count=int(np.count_nonzero(theta <= 1e-12)),
            covariance_key=f'covariance_native{index}', gram_key=f'gram_native{index}', **info))
    metadata = dict(schema='frozen_rx_guard_geometry_v1', source=io.ref(__file__),
        labels=list(labels), hours=H, repeats=int(case.repeat), resources=R,
        decision_shape=[H, R], decision_order='hour-major, then unchanged resource order',
        power_base_kva=float(base_kw), score_sigma2=sigma2, training_rows=len(p), validation_rows=len(voltage),
        target_mass=float(target_mass), target_representatives=list(map(int, selected)),
        target_native_indices=native, representative_weights=weights.tolist(),
        pq_rho=float(pq_rho), response_rms_fraction=float(response_rms_fraction),
        pq_summed_training_residual_variance_pu=departure_scale,
        pq_scaling='Per-node affine residual RMS; average over all decision hours and measured nodes',
        pq_residual_variance_rejection_thresholds=residual_floor.tolist(),
        meter_relative_sd=float(meter_relative_sd), root_median=root_median, voltage_median=voltage_median,
        root_variance=root_variance, individual_variance=individual_variance,
        noise='Independent time blocks; within-time root_variance*11T + individual_variance*I',
        training_p_sha256=io.contract_hash(p), training_q_sha256=io.contract_hash(q),
        validation_voltage_sha256=io.contract_hash(voltage), validation_root_sha256=io.contract_hash(root),
        guards=guard_rows, targets=details,
        covariance_scope='Frozen scoring H, not row-scaled K: unconstrained fixed-support linear-design approximation; all bounded-fit directions retained',
        excluded_uncertainty=['selected support', 'systematic R/X or AC-model bias', 'P/Q input measurement error',
                              'temporal correlation', 'accuracy of the historical row-scaled planning mean'],
        interpretation='Incremental variance design restriction only; no total-response-error or 90% coverage guarantee',
        matrices_sha256={key:io.contract_hash(value) for key, value in arrays.items()})
    return arrays, metadata


def build_budgets(case, score, train, validation, base_kw, *, targets=None,
                  meter_relative_sd=.0005, target_mass=.99, pq_rho=4.,
                  response_rms_fraction=.25):
    """Runner adapter: targets are representative positions, not native IDs."""
    matrices, metadata = build(case, score, train, validation, base_kw, meter_relative_sd,
        target_mass, pq_rho=pq_rho, response_rms_fraction=response_rms_fraction)
    if targets is not None and list(map(int, targets)) != metadata['target_representatives']:
        raise ValueError('Runner target representatives differ from historical selection')
    return matrices, metadata


def guards(matrices, metadata, kind):
    if kind not in ('pq', 'response'):
        raise ValueError('Guard kind must be pq or response')
    result = []
    dimension = int(np.prod(metadata['decision_shape']))
    for row in metadata['guards']:
        if row['kind'] != kind:
            continue
        Q = np.asarray(matrices[row['matrix_key']], float)
        if (Q.shape != (dimension, dimension) or not np.isfinite(Q).all()
                or io.contract_hash(Q) != metadata['matrices_sha256'][row['matrix_key']]):
            raise ValueError('Sealed guard matrix shape/content mismatch')
        result.append(dict(row, Q=Q))
    return result


def direct_audit(u, matrices, metadata, kind=None, tolerance=1e-9):
    """Recompute through node injections/design covariance, not the Q matrices."""
    u = np.asarray(u, float)
    if u.shape != tuple(metadata['decision_shape']) or not np.isfinite(u).all():
        raise ValueError('Finite controls in the sealed hourly resource shape required')
    if kind not in (None, 'pq', 'response'):
        raise ValueError('Unknown guard kind')
    H, R = u.shape
    n, sigma2 = len(metadata['labels']), metadata['score_sigma2']
    B = np.asarray(matrices['resource_map_pu'], float)
    pq = u @ B.T
    rows = []
    for row in metadata['guards']:
        if kind is not None and row['kind'] != kind:
            continue
        if row['kind'] == 'pq':
            departure = pq[:, n:] - pq[:, :n]*matrices['pq_slope'][None]
            value = float(np.mean((departure/matrices['pq_residual_rms_pu'][None])**2))
        else:
            details = next(t for t in metadata['targets'] if t['target_native'] == row['target_native'])
            F = design_blocks(pq[:, :n], pq[:, n:], details['supports']).reshape(H*n, -1)
            C = matrices[details['covariance_key']]
            value = float(np.einsum('tk,kl,tl->', F, C, F, optimize=True)/(H*n*sigma2))
        quadratic = float(u.ravel() @ matrices[row['matrix_key']] @ u.ravel())
        difference = abs(value-quadratic)
        if difference > 1e-9*(1+abs(value)):
            raise ArithmeticError('Direct incremental geometry and sealed quadratic differ')
        rows.append(dict(id=row['id'], kind=row['kind'], value=value, limit=row['limit'],
            margin=row['limit']-value, rms_ratio=float(np.sqrt(max(0., value))),
            expansion_error=difference, passed=bool(value <= row['limit']+tolerance)))
    return dict(passed=all(row['passed'] for row in rows), rows=rows,
                scope='Additional planning restrictions; physical resource audit is separate')
