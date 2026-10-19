"""Conservative local-response QP envelopes, separate from empirical scores.

The score weights dominate the Gaussian *linear* part of squared-voltage and
P/Q noise over the declared physical domain.  AC loss remainders have to be
supplied as valid contrast-space boxes.  Squared-voltage noise is handled by
a separate finite-sample union bound.  No observed ranking or true topology
enters the rule.  ``gate_passed`` defaults to False: computing an envelope
does not activate a rejection claim.

All optimization bounds use floating-point arithmetic with an explicit guard,
not outward-rounded interval arithmetic.  A failed optimization never rejects.
"""
from __future__ import annotations

from itertools import product
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.stats import chi2, norm

from . import scoring


def alpha_split(alpha, *, source='meter'):
    """The two predeclared levels; no tuning based on the observed losses.

    A meter-derived P/Q rectangle consumes 20 percent of alpha, the quadratic
    voltage-noise event 20 percent, and the Gaussian event 60 percent.  Public
    physical P/Q envelopes need no P/Q event and allocate 80 percent to the
    Gaussian event.  The union bound does not require event independence.
    """
    if not any(np.isclose(float(alpha), value, rtol=0., atol=1e-14)
               for value in (.05, .01)):
        raise ValueError('only predeclared alpha=.05 and alpha=.01 are supported')
    if source not in ('meter', 'public'):
        raise ValueError("source must be 'meter' or 'public'")
    alpha = float(alpha)
    return dict(alpha=alpha, source=source,
                alpha_pq=.2*alpha if source == 'meter' else 0.,
                alpha_square=.2*alpha,
                alpha_gaussian=(.6 if source == 'meter' else .8)*alpha)


def project_box_metric(residual, covariance, lower, upper):
    """Exact 2-D covariance-metric projection onto independent time boxes.

    For each time minimize ``(residual-delta)' covariance^-1 (residual-delta)``
    over its two-dimensional box.  An optimum is either the residual itself
    or the constrained minimum on one of four edges.  Corners are included by
    clipping each edge minimum.  This is a constant-size QP, not an action
    library or a search over whole trajectories.
    """
    residual = np.asarray(residual, float)
    lower = np.broadcast_to(np.asarray(lower, float), residual.shape)
    upper = np.broadcast_to(np.asarray(upper, float), residual.shape)
    if residual.ndim != 2 or residual.shape[1] != 2:
        raise ValueError('two independent contrast coordinates are required')
    covariance = np.broadcast_to(np.asarray(covariance, float),
                                 (len(residual), 2, 2))
    if not all(np.isfinite(a).all() for a in (residual, lower, upper, covariance)):
        raise ValueError('finite residual, covariance and box values required')
    if np.any(lower > upper):
        raise ValueError('delta lower bound exceeds upper bound')
    precision = np.linalg.inv(covariance)
    if np.min(np.linalg.eigvalsh(covariance)) <= 0:
        raise ValueError('positive-definite covariance required')
    candidates = [np.clip(residual, lower, upper)]
    for fixed in (lower[:, 0], upper[:, 0]):
        free = residual[:, 1]-precision[:, 1, 0]/precision[:, 1, 1]*(fixed-residual[:, 0])
        candidates.append(np.c_[fixed, np.clip(free, lower[:, 1], upper[:, 1])])
    for fixed in (lower[:, 1], upper[:, 1]):
        free = residual[:, 0]-precision[:, 0, 1]/precision[:, 0, 0]*(fixed-residual[:, 1])
        candidates.append(np.c_[np.clip(free, lower[:, 0], upper[:, 0]), fixed])
    candidates = np.asarray(candidates)
    errors = residual[None, :, :]-candidates
    losses = np.einsum('cti,tij,ctj->ct', errors, precision, errors)
    best = np.argmin(losses, axis=0)
    delta = candidates[best, np.arange(len(residual))]
    error = residual-delta
    weight = np.einsum('tij,tj->ti', precision, error)
    return dict(delta=delta, residual=error, weight=weight,
                loss=float(np.sum(error*weight)), precision=precision)


def _qp_value_gradient(point, y, design, covariance, delta_lower, delta_upper):
    value = project_box_metric(y-np.einsum('tae,e->ta', design, point),
                               covariance, delta_lower, delta_upper)
    # Envelope theorem.  The unique projection makes this convex reduced
    # objective differentiable even when its box active set changes.
    gradient = -2*np.einsum('tae,ta->e', design, value['weight'])
    return value, gradient


def _reduced_hessian(value, design, covariance, delta_lower, delta_upper):
    """Exact quadratic Hessian on the current projection active region."""
    delta = value['delta']
    free = ((delta > np.asarray(delta_lower)+1e-12) &
            (delta < np.asarray(delta_upper)-1e-12))
    metric = value['precision'].copy()
    both = np.all(free, axis=1)
    metric[both] = 0.
    # A free delta coordinate is analytically eliminated by a Schur
    # complement.  In two dimensions the remaining metric is 1/variance.
    first_only = free[:, 0] & ~free[:, 1]
    second_only = free[:, 1] & ~free[:, 0]
    metric[first_only] = 0.
    metric[first_only, 1, 1] = 1./covariance[first_only, 1, 1]
    metric[second_only] = 0.
    metric[second_only, 0, 0] = 1./covariance[second_only, 0, 0]
    return 2*np.einsum('tae,tab,tbf->ef', design, metric, design)


def fit_delta_qp(candidate, obs, sigma, domain, delta_lower, delta_upper, *,
                 v_ref=1., max_iter=1000, tol=1e-11):
    """Minimize the fixed-weight QP over the convex outer line domain and box.

    The independent two-dimensional delta blocks are eliminated exactly;
    only a few shared line coefficients remain in the convex minimization.
    A feasible point supplies the primal upper value.  A convex tangent plus
    an explicitly evaluated nonnegative-multiplier box dual supplies the
    lower value.  Thus solver status alone never acts as an optimality bound.
    """
    started = perf_counter()
    try:
        data = scoring.prepare(obs, sigma, v_ref=v_ref)
        if data['n'] != 3:
            raise ValueError('this first implementation requires three terminals')
        covariance = scoring.fixed_covariance(candidate, data, domain, weight='upper')
        b, active, design, lower, upper, a = scoring._qp_geometry(candidate, data, domain)
        rhs = np.asarray(b.get('linear_rhs', np.zeros(a.shape[0])), float)
        dlo = np.broadcast_to(np.asarray(delta_lower, float), data['y'].shape).copy()
        dhi = np.broadcast_to(np.asarray(delta_upper, float), data['y'].shape).copy()
        if not np.isfinite(dlo).all() or not np.isfinite(dhi).all() or np.any(dlo > dhi):
            raise ValueError('finite ordered AC remainder bounds required')
        scale, count = b['zhi'], data['count']
        radius, angle = np.sqrt(b['zlo']*b['zhi']), np.mean(b['angles'])
        initial = np.r_[np.full(len(active), radius*np.cos(angle)),
                        np.full(len(active), radius*np.sin(angle))]

        def function(u):
            value, gradient = _qp_value_gradient(scale*u, data['y'], design,
                                                 covariance, dlo, dhi)
            return value['loss']/count, gradient*scale/count

        result = minimize(function, initial/scale, jac=True, method='SLSQP',
                          bounds=Bounds(lower/scale, upper/scale),
                          constraints=LinearConstraint(a*scale, -np.inf, rhs),
                          options=dict(maxiter=int(max_iter), ftol=float(tol), disp=False))
        point = np.clip(np.asarray(result.x)*scale, lower, upper)
        value, gradient = _qp_value_gradient(point, data['y'], design, covariance, dlo, dhi)
        polish_count = 0
        for _ in range(4):
            hessian = _reduced_hessian(value, design, covariance, dlo, dhi)
            linear = gradient-hessian @ point
            polished, legal = scoring._polish_qp(point, hessian, linear, a, rhs,
                                                 lower, upper, scale)
            if not legal:
                break
            new_value, new_gradient = _qp_value_gradient(polished, data['y'], design,
                                                        covariance, dlo, dhi)
            if new_value['loss'] > value['loss']+1e-10:
                break
            difference = np.max(abs(polished-point))
            point, value, gradient = polished, new_value, new_gradient
            polish_count += 1
            if difference <= 1e-13:
                break
        feasibility = max(0., float(np.max(a @ point-rhs)), float(np.max(lower-point)),
                          float(np.max(point-upper)))
        certificate = scoring.convex_first_order_bound(point, value['loss'], gradient,
                                                       lower, upper, a, rhs)
        gap = max(0., value['loss']-certificate['lower_bound'])
        finite = all(np.isfinite(x) for x in (value['loss'], certificate['lower_bound'], gap))
        qualified = bool(finite and feasibility <= 1e-9 and
                         gap <= max(1e-5, 1e-7*max(1., value['loss'])))
        return dict(status='qualified' if qualified else 'unqualified', qualified=qualified,
                    loss=value['loss'], J=.5*value['loss'],
                    lower_bound=certificate['lower_bound'], absolute_gap=gap,
                    relative_gap=gap/max(1., abs(value['loss'])),
                    point=point.tolist(), delta=value['delta'].tolist(),
                    delta_lower=dlo.tolist(), delta_upper=dhi.tolist(),
                    covariance=covariance.tolist(), gradient=gradient.tolist(),
                    active_edges=active.tolist(), certificate=certificate,
                    constraint_violation=feasibility, solver_success=bool(result.success),
                    solver_message=str(result.message), iterations=int(result.nit),
                    active_face_polish_count=polish_count,
                    seconds=perf_counter()-started, truth_used=False,
                    model='linear_response_convex_outer_domain_AC_remainder_box',
                    covariance_rule='public_domain_upper',
                    strict_rejection_activated=False,
                    arithmetic='floating_point_with_guard_not_interval')
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return dict(status='failed', qualified=False, error=repr(error),
                    seconds=perf_counter()-started, truth_used=False,
                    strict_rejection_activated=False)


def square_noise_bound(data, covariance, *, alpha_square):
    """Finite-sample bound for D*(voltage_error**2-sigma_v**2).

    A simultaneous normal interval covers all raw voltage errors.  The
    transformed centered-square errors lie in a box; a norm's maximum over
    this box is attained at a vertex.  With three terminals there are eight
    vertices per time.  Correlation with the linear Gaussian part is allowed.
    """
    sv = np.asarray(data['sigma_v'], float)
    d = np.asarray(data['D'], float)
    if not 0 < alpha_square < 1:
        raise ValueError('alpha_square must lie in (0,1)')
    covariance = np.broadcast_to(np.asarray(covariance, float),
                                 (sv.shape[0], d.shape[0], d.shape[0]))
    if sv.ndim != 2 or sv.shape[1] != d.shape[1] or np.any(sv <= 0):
        raise ValueError('voltage scales must have (time,terminal) shape')
    if d.shape[1] > 12:
        raise ValueError('vertex bound is limited to twelve terminals')
    cutoff = float(norm.ppf(1.-alpha_square/(2*sv.size)))
    raw_lower, raw_upper = -sv**2, (cutoff**2-1.)*sv**2
    choices = np.asarray(tuple(product((False, True), repeat=d.shape[1])), bool)
    vertices = np.where(choices[:, None, :], raw_upper[None, :, :], raw_lower[None, :, :])
    contrast = np.einsum('ai,cti->cta', d, vertices)
    precision = np.linalg.inv(covariance)
    norm2 = np.einsum('cta,tab,ctb->ct', contrast, precision, contrast)
    time_max = np.max(norm2, axis=0)
    return dict(bound=float(np.sqrt(np.sum(time_max))),
                alpha_square=float(alpha_square), raw_standard_normal_cutoff=cutoff,
                raw_voltage_count=int(sv.size), vertex_count=len(choices),
                per_time_squared_bound=time_max.tolist(),
                construction='simultaneous_normal_interval_then_exact_box_vertex_norm')


def rejection_envelope(qp_result, data, alpha, *, source='meter', gate_passed=False):
    """Audit the conditional rejection envelope; a failed/gated QP stays in.

    Required assumptions: the candidate family covers the physical truth;
    the public line/voltage domain covers it; the supplied delta boxes cover
    all of its AC remainders on the stated P/Q event; and the original meter
    errors are independent Gaussian with the declared known scales.  The
    Gaussian dimension is ``time*(terminal-1)`` without subtracting fitted
    coefficients.  These assumptions are not inferred from the observed fit.
    """
    split = alpha_split(alpha, source=source)
    df = int(data['count']*(data['n']-1))
    gaussian = float(np.sqrt(chi2.ppf(1.-split['alpha_gaussian'], df)))
    if not qp_result.get('qualified', False) or 'covariance' not in qp_result:
        return dict(alpha=split, df=df, gate_passed=bool(gate_passed), reject=False,
                    status='optimization_failure_kept',
                    gaussian_radius=gaussian, truth_used=False)
    square = square_noise_bound(data, qp_result['covariance'],
                                alpha_square=split['alpha_square'])
    threshold = gaussian+square['bound']
    lower = max(0., float(qp_result['lower_bound']))
    exceeds = bool(np.sqrt(lower) > threshold)
    return dict(alpha=split, df=df, gate_passed=bool(gate_passed),
                lower_bound=lower, sqrt_lower_bound=float(np.sqrt(lower)),
                gaussian_radius=gaussian, square_noise=square,
                radius_threshold=threshold, SSE_threshold=threshold**2,
                J_threshold=.5*threshold**2, envelope_exceeded=exceeds,
                reject=bool(gate_passed and exceeds), truth_used=False,
                status='activated' if gate_passed else 'audit_only_gate_not_passed',
                interpretation='conditional_false_exclusion_bound_not_topology_probability',
                arithmetic=qp_result.get('arithmetic', 'unspecified'))


__all__ = ['alpha_split', 'project_box_metric', 'fit_delta_qp',
           'square_noise_bound', 'rejection_envelope']
