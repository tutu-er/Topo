"""Topology-constrained local voltage-response scores from noisy P/Q/V.

Only measured data, declared meter scales and public line domains enter these
functions.  A squared-voltage contrast removes the unknown common level in
the *linear* response model.  It does not remove nonlinear AC loss terms.

``profile`` analytically minimizes over unrestricted latent terminal P/Q.
``upper`` uses a fixed covariance upper bound and an outer convex line domain;
its convex QP lower bound concerns that declared linear-response model.  No
AC rejection or Gaussian confidence statement is returned by this module.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def contrast_matrix(n):
    """Helmert rows: ``D @ ones = 0`` and ``D @ D.T = I``."""
    if int(n) != n or n < 2:
        raise ValueError('at least two terminals are required')
    d = np.zeros((int(n)-1, int(n)))
    for j in range(1, int(n)):
        d[j-1, :j] = 1./np.sqrt(j*(j+1))
        d[j-1, j] = -j/np.sqrt(j*(j+1))
    return d


def _broadcast(value, shape, name):
    result = np.broadcast_to(np.asarray(value, float), shape).copy()
    if not np.isfinite(result).all():
        raise ValueError(name+' must be finite')
    return result


def prepare(obs, sigma, *, v_ref=1.):
    """Debias squared voltage and freeze its variance at a declared reference.

    ``sigma`` accepts both ``p/q/v`` and ``sigma_p/sigma_q/sigma_v`` names.
    ``v_ref`` is declared independently of the measured voltage; the default
    is 1 pu.  The first two moments of squared Gaussian voltage are used, but
    squared-voltage noise itself is not Gaussian.
    """
    p = np.asarray(obs['p'], float)
    if p.ndim != 2 or min(p.shape) < 1 or p.shape[1] < 2:
        raise ValueError('nonempty (time, terminal) P/Q/V arrays are required')
    q, v = (_broadcast(obs[key], p.shape, key) for key in ('q', 'v'))
    if not np.isfinite(p).all():
        raise ValueError('p must be finite')
    sig = {key: _broadcast(sigma.get(key, sigma.get('sigma_'+key)), p.shape, key)
           for key in ('p', 'q', 'v')}
    if any(np.any(value <= 0) for value in sig.values()):
        raise ValueError('all meter scales must be strictly positive')
    reference = _broadcast(v_ref, p.shape, 'v_ref')
    if np.any(reference <= 0):
        raise ValueError('v_ref must be positive')
    d = contrast_matrix(p.shape[1])
    variance = 4*sig['v']**2*reference**2+2*sig['v']**4
    covariance = np.einsum('ai,ti,bi->tab', d, variance, d)
    return dict(D=d, x=np.c_[p, q], y=(v**2-sig['v']**2) @ d.T,
                sigma_x=np.c_[sig['p'], sig['q']], sigma_v=sig['v'],
                var_y=variance, cov_y=covariance, v_ref=reference,
                n=p.shape[1], count=p.shape[0])


def _response_basis(candidate, d):
    # H = sum_e r_e Lr_e + x_e Lx_e.  The same scalar tree
    # coefficients act at every time, instead of arbitrary response entries.
    g = -2*np.einsum('ae,je->aje', d @ candidate.M, candidate.M)
    zero = np.zeros_like(g)
    return np.concatenate((np.concatenate((g, zero), axis=1),
                           np.concatenate((zero, g), axis=1)), axis=2)


def response_matrix(candidate, r, x, D=None):
    """Map concatenated terminal P/Q to squared-voltage contrasts."""
    d = contrast_matrix(candidate.n) if D is None else np.asarray(D, float)
    r, x = np.asarray(r, float), np.asarray(x, float)
    if r.shape != (candidate.m,) or x.shape != r.shape:
        raise ValueError('r/x length must equal candidate edge count')
    return np.einsum('aje,e->aj', _response_basis(candidate, d), np.r_[r, x])


def _profile_h(data, h, *, derivatives=True):
    observed_x, observed_y = data['x'], data['y']
    sx2 = data['sigma_x']**2
    residual = observed_y-observed_x @ h.T
    omega = data['cov_y']+np.einsum('ai,ti,bi->tab', h, sx2, h)
    w = np.linalg.solve(omega, residual[..., None])[..., 0]
    latent = observed_x+sx2*(w @ h)
    result = dict(loss=float(np.sum(residual*w)), residual=residual,
                  covariance=omega, H=h, latent_x=latent,
                  p=latent[:, :data['n']], q=latent[:, data['n']:])
    result['J'] = .5*result['loss']
    if derivatives:
        # Envelope theorem for min_x ||x_obs-x||_Sx^2+||y_obs-Hx||_Sy^2.
        # This includes the derivative of H Sx H', not just d residual/dH.
        result['gradient_H'] = -2*np.einsum('ta,ti->ai', w, latent)
    return result


def profile_at(candidate, data, r, x, *, derivatives=True):
    """Exact latent-P/Q profile loss for the declared linear response model.

    There is no log determinant: latent P/Q are optimized fixed quantities,
    rather than integrated random variables.  Latent P/Q here have no hard
    cap; consequently this is an outer relaxation of a capped EIV model.
    """
    h = response_matrix(candidate, r, x, data['D'])
    result = _profile_h(data, h, derivatives=derivatives)
    if derivatives:
        result['gradient_rx'] = np.einsum('aj,aje->e', result['gradient_H'],
                                         _response_basis(candidate, data['D']))
    return result


def _domain(candidate, domain, d):
    zlo, zhi = float(domain['z_min']), float(domain['z_max'])
    ratio_lo = float(domain.get('ratio_min', domain.get('ratio_bounds', (.25, 1.5))[0]))
    ratio_hi = float(domain.get('ratio_max', domain.get('ratio_bounds', (.25, 1.5))[1]))
    if not 0 < zlo <= zhi or not 0 <= ratio_lo < ratio_hi:
        raise ValueError('positive ordered magnitude and X/R bounds required')
    angles = np.arctan([ratio_lo, ratio_hi])
    # Component boxes are consequences of the original polar domain; they
    # intentionally do not enforce its circular lower/upper magnitudes.
    rl, ru = zlo*np.cos(angles[1]), zhi*np.cos(angles[0])
    xl, xu = zlo*np.sin(angles[0]), zhi*np.sin(angles[1])
    active = np.flatnonzero(np.linalg.norm(d @ candidate.M, axis=0) > 1e-12)
    if not len(active):
        raise ValueError('candidate has no observable contrast response')
    return dict(zlo=zlo, zhi=zhi, angles=angles, ratio_lo=ratio_lo,
                ratio_hi=ratio_hi, active=active, rlo=rl, rhi=ru,
                xlo=xl, xhi=xu)


def _full_rx(candidate, active, r, x, bounds):
    # The annihilated common trunk is not estimated.  A public-domain legal
    # representative is supplied only so full candidate arrays retain shape.
    radius = np.sqrt(bounds['zlo']*bounds['zhi'])
    angle = np.mean(bounds['angles'])
    rr = np.full(candidate.m, radius*np.cos(angle))
    xx = np.full(candidate.m, radius*np.sin(angle))
    rr[active], xx[active] = r, x
    return rr, xx


def fixed_covariance(candidate, data, domain, *, weight='upper'):
    """Fixed QP weights, without reading a fitted or true voltage/impedance.

    ``upper`` dominates covariance for every line in the original physical
    domain and terminal voltage <= v_max.  A Frobenius-norm bound on H Sx H'
    is used.  ``nominal`` freezes covariance at the geometric-midpoint line
    model and 1-pu reference; it has no lower-bound interpretation.
    """
    d = data['D']
    bounds = _domain(candidate, domain, d)
    if weight == 'nominal':
        radius = np.sqrt(bounds['zlo']*bounds['zhi'])
        angle = np.mean(bounds['angles'])
        r = np.full(candidate.m, radius*np.cos(angle))
        x = np.full(candidate.m, radius*np.sin(angle))
        h = response_matrix(candidate, r, x, d)
        return data['cov_y']+np.einsum('ai,ti,bi->tab', h, data['sigma_x']**2, h)
    if weight != 'upper':
        raise ValueError('fixed weight must be upper or nominal')
    basis = _response_basis(candidate, d)
    absolute_h = np.einsum('aje,e->aj', abs(basis),
                           np.r_[np.full(candidate.m, bounds['rhi']),
                                 np.full(candidate.m, bounds['xhi'])])
    # H Sx H' <= trace(H Sx H') I <= sum_ij |H_ij|max^2 sx_j^2 I.
    pq_upper = np.einsum('aj,tj->t', absolute_h**2, data['sigma_x']**2)
    vmax = float(domain.get('v_max', domain.get('v_bounds', (.9, 1.1))[1]))
    if vmax < np.max(data['v_ref']):
        raise ValueError('v_max must cover declared variance reference')
    sv = data['sigma_v']
    covariance_y = np.einsum('ai,ti,bi->tab', d, 4*sv**2*vmax**2+2*sv**4, d)
    return covariance_y+pq_upper[:, None, None]*np.eye(candidate.n-1)


def score_fixed(candidate, obs, sigma, r, x, *, v_ref=1., weight='profile', domain=None):
    """Score new observations with training coefficients held completely fixed."""
    data = prepare(obs, sigma, v_ref=v_ref)
    result = profile_at(candidate, data, r, x, derivatives=False)
    if weight != 'profile':
        if domain is None:
            raise ValueError('fixed covariance scoring requires the declared domain')
        covariance = fixed_covariance(candidate, data, domain, weight=weight)
        residual = result['residual']
        w = np.linalg.solve(covariance, residual[..., None])[..., 0]
        result.update(loss=float(np.sum(residual*w)), covariance=covariance,
                      score_kind='fixed_covariance_'+weight)
        result['J'] = .5*result['loss']
    else:
        result['score_kind'] = 'analytic_PQ_profile'
    result['fixed_training_lines'] = True
    result['truth_used'] = False
    return result


def _qp_geometry(candidate, data, domain):
    b = _domain(candidate, domain, data['D'])
    active = b['active']; width = len(active)
    index = np.r_[active, candidate.m+active]
    basis = _response_basis(candidate, data['D'])[:, :, index]
    design = np.einsum('aje,tj->tae', basis, data['x'])
    lower = np.r_[np.full(width, b['rlo']), np.full(width, b['xlo'])]
    upper = np.r_[np.full(width, b['rhi']), np.full(width, b['xhi'])]
    # Linear outer hull of the allowed annular sector: ratio sides, a lower
    # supporting chord, and eight upper tangent halfspaces.  Every original
    # polar-domain point is included; this is not a fitted support region.
    upper_angles = np.linspace(b['angles'][0], b['angles'][1], 8)
    rows_per_edge = 3+len(upper_angles)
    a = np.zeros((rows_per_edge*width, 2*width))
    rhs = np.zeros(rows_per_edge*width)
    mid = np.mean(b['angles']); half = .5*np.diff(b['angles'])[0]
    for j in range(width):
        row = rows_per_edge*j
        a[row, j] = b['ratio_lo']; a[row, width+j] = -1.
        a[row+1, j] = -b['ratio_hi']; a[row+1, width+j] = 1.
        a[row+2, j] = -np.cos(mid); a[row+2, width+j] = -np.sin(mid)
        rhs[row+2] = -b['zlo']*np.cos(half)
        for k, angle in enumerate(upper_angles):
            a[row+3+k, j] = np.cos(angle)
            a[row+3+k, width+j] = np.sin(angle)
            rhs[row+3+k] = b['zhi']
    b['linear_rhs'] = rhs
    b['geometry_version'] = 'sector_chord_eight_upper_tangents_v1'
    return b, active, design, lower, upper, a


def convex_first_order_bound(point, value, gradient, lower, upper, a, b=None,
                             *, guard=1e-9):
    """Checkable lower bound for a nonnegative convex differentiable loss.

    A linear program selects nonnegative multipliers for ``A x <= b``.  Even
    an inaccurate LP solution is safe: for any nonnegative multiplier lambda,
    ``min_box (g+A'lambda)'x-lambda'b`` is an LP dual lower bound.  The
    floating-point guard is recorded; this is not interval arithmetic.
    """
    point, gradient = np.asarray(point, float), np.asarray(gradient, float)
    lower, upper, a = np.asarray(lower, float), np.asarray(upper, float), np.asarray(a, float)
    b = np.zeros(a.shape[0]) if b is None else np.asarray(b, float)
    solution = linprog(gradient, A_ub=a, b_ub=b,
                       bounds=list(zip(lower, upper)), method='highs')
    multipliers = np.zeros(len(b))
    if solution.success and solution.ineqlin is not None:
        multipliers = np.maximum(-np.asarray(solution.ineqlin.marginals), 0.)
    adjusted = gradient+a.T @ multipliers
    minimizing_corner = np.where(adjusted >= 0, lower, upper)
    dual_linear = float(adjusted @ minimizing_corner-multipliers @ b)
    raw = float(value-gradient @ point+dual_linear)
    scale = max(1., abs(value), abs(gradient @ point), abs(dual_linear),
                float(np.sum(abs(adjusted)*np.maximum(abs(lower), abs(upper)))))
    margin = float(guard*scale)
    return dict(lower_bound=max(0., raw-margin), raw_lower_bound=raw,
                numerical_guard=margin, multiplier=multipliers.tolist(),
                adjusted_gradient=adjusted.tolist(), minimizing_corner=minimizing_corner.tolist(),
                dual_linear=dual_linear, lp_success=bool(solution.success),
                lp_status=int(solution.status), lp_primal=float(solution.fun) if solution.success else None,
                construction='convex_tangent_plus_box_inequality_lagrange_dual',
                arithmetic='floating_point_with_guard_not_interval')


def _polish_qp(point, hessian, linear, a, rhs, lower, upper, scale):
    """Solve the small QP on its current active face and accept legal descent.

    SLSQP's objective-change test can stop before its gradient gives a small
    certified gap.  The quadratic active-face equations are linear, so a
    direct KKT solve supplies a cheap precision correction.  Qualification
    still uses the independently reconstructed convex dual bound.
    """
    u = np.asarray(point)/scale
    au = a*scale
    lo, hi = lower/scale, upper/scale
    active_matrix = np.r_[au, -np.eye(len(u)), np.eye(len(u))]
    active_rhs = np.r_[rhs, -lo, hi]
    slack = active_rhs-active_matrix @ u
    near = slack <= 1e-7
    rows, targets = active_matrix[near], active_rhs[near]
    h = hessian*scale**2
    g = linear*scale
    objective_scale = max(1., float(np.linalg.norm(h, ord=2)))
    h, g = h/objective_scale, g/objective_scale
    kkt = np.block([[h, rows.T], [rows, np.zeros((len(rows), len(rows)))]])
    rhs_kkt = np.r_[-g, targets]
    candidate = np.linalg.lstsq(kkt, rhs_kkt, rcond=1e-12)[0][:len(u)]
    if not np.isfinite(candidate).all():
        return point, False
    if np.max(active_matrix @ candidate-active_rhs) > 1e-9:
        return point, False
    return candidate*scale, True


def fit_qp(candidate, obs, sigma, domain, *, weight='upper', v_ref=1., max_iter=1000,
           tol=1e-11):
    """Globally convex fixed-weight response QP on a public outer line domain."""
    started = perf_counter()
    data = prepare(obs, sigma, v_ref=v_ref)
    covariance = fixed_covariance(candidate, data, domain, weight=weight)
    b, active, design, lower, upper, a = _qp_geometry(candidate, data, domain)
    inv_design = np.linalg.solve(covariance, design)
    inv_y = np.linalg.solve(covariance, data['y'][..., None])[..., 0]
    hessian = 2*np.einsum('tae,taf->ef', design, inv_design)
    linear = -2*np.einsum('tae,ta->e', design, inv_y)
    constant = float(np.sum(data['y']*inv_y))
    scale = b['zhi']; count = data['count']
    def loss(theta):
        residual = data['y']-np.einsum('tae,e->ta', design, theta)
        w = np.linalg.solve(covariance, residual[..., None])[..., 0]
        return float(np.sum(residual*w)), -2*np.einsum('tae,ta->e', design, w)
    def function(u):
        value, gradient = loss(scale*u)
        return value/count, gradient*scale/count
    initial = .5*(lower+upper)
    # The midpoint obeys ratio inequalities because the midpoint of two legal
    # points is legal; construct it from the polar midpoint directly instead
    # to avoid relying on separate component-box midpoints.
    radius = np.sqrt(b['zlo']*b['zhi']); angle = np.mean(b['angles'])
    initial[:] = np.r_[np.full(len(active), radius*np.cos(angle)),
                       np.full(len(active), radius*np.sin(angle))]
    result = minimize(function, initial/scale, jac=True, method='SLSQP',
                      bounds=Bounds(lower/scale, upper/scale),
                      constraints=LinearConstraint(a*scale, -np.inf, b['linear_rhs']),
                      options=dict(maxiter=int(max_iter), ftol=float(tol), disp=False))
    point = np.clip(np.asarray(result.x)*scale, lower, upper)
    primal, gradient = loss(point)
    polished, polish_feasible = _polish_qp(point, hessian, linear, a, b['linear_rhs'], lower, upper, scale)
    polish_accepted = False
    if polish_feasible:
        polished_loss, polished_gradient = loss(polished)
        if polished_loss <= primal+1e-10:
            point, primal, gradient = polished, polished_loss, polished_gradient
            polish_accepted = True
    feasibility = max(0., float(np.max(a @ point-b['linear_rhs'])), float(np.max(lower-point)),
                      float(np.max(point-upper)))
    certificate = convex_first_order_bound(point, primal, gradient, lower, upper, a, b['linear_rhs'])
    gap = max(0., primal-certificate['lower_bound'])
    r, x = _full_rx(candidate, active, point[:len(active)], point[len(active):], b)
    h = response_matrix(candidate, r, x, data['D'])
    qualified = bool(feasibility <= 1e-9 and gap <= max(1e-5, 1e-7*max(1., primal)))
    return dict(status='qualified' if qualified else 'unqualified', qualified=qualified,
                loss=primal, J=.5*primal, lower_bound=certificate['lower_bound'],
                absolute_gap=gap, relative_gap=gap/max(1., abs(primal)),
                r=r.tolist(), x=x.tolist(), H=h.tolist(), active_edges=active.tolist(),
                unidentified_edges=np.setdiff1d(np.arange(candidate.m), active).tolist(),
                covariance=covariance.tolist(), gradient_rx_active=gradient.tolist(),
                constraint_violation=feasibility, solver_success=bool(result.success),
                active_face_polish_accepted=polish_accepted,
                solver_message=str(result.message), iterations=int(result.nit),
                certificate=certificate, seconds=perf_counter()-started, truth_used=False,
                weight=weight, line_domain='convex_component_box_ratio_sector_chord_outer_domain',
                lower_bound_for_profile_model=weight == 'upper',
                AC_lower_bound=False, exact_Gaussian_squared_voltage=False,
                quadratic_hessian=hessian.tolist(), quadratic_linear=linear.tolist(),
                quadratic_constant=constant)


def fit_profile(candidate, obs, sigma, domain, *, v_ref=1., multistart=4,
                max_iter=500, gtol=1e-7, seed=0):
    """Local multistart response fit in the original magnitude/X:R domain."""
    if multistart < 1 or max_iter < 1 or gtol <= 0:
        raise ValueError('positive starts, iterations and tolerance required')
    started = perf_counter()
    data = prepare(obs, sigma, v_ref=v_ref)
    b = _domain(candidate, domain, data['D'])
    active = b['active']; m = len(active)
    lower = np.r_[np.full(m, np.log(b['zlo'])), np.full(m, b['angles'][0])]
    upper = np.r_[np.full(m, np.log(b['zhi'])), np.full(m, b['angles'][1])]
    initializer = fit_qp(candidate, obs, sigma, domain, weight='nominal', v_ref=v_ref,
                         max_iter=min(500, max_iter))
    rr, xx = np.asarray(initializer['r'])[active], np.asarray(initializer['x'])[active]
    first = np.clip(np.r_[np.log(np.hypot(rr, xx)), np.arctan2(xx, rr)], lower, upper)
    rng = np.random.default_rng(seed)
    starts = [first]+[rng.uniform(lower, upper) for _ in range(multistart-1)]
    calls, records, attempts = 0, [], []
    def function(theta):
        nonlocal calls
        radius, angle = np.exp(theta[:m]), theta[m:]
        r, x = _full_rx(candidate, active, radius*np.cos(angle), radius*np.sin(angle), b)
        value = profile_at(candidate, data, r, x)
        calls += 1
        gr, gx = value['gradient_rx'][active], value['gradient_rx'][candidate.m+active]
        gradient = np.r_[gr*r[active]+gx*x[active], -gr*x[active]+gx*r[active]]
        return value['loss']/data['count'], gradient/data['count']
    for number, initial in enumerate(starts):
        try:
            result = minimize(function, initial, jac=True, method='L-BFGS-B',
                              bounds=Bounds(lower, upper),
                              options=dict(maxiter=int(max_iter), gtol=gtol/data['count'],
                                           ftol=1e-15, maxls=50, maxcor=20))
            value, gradient = function(result.x)
            # Loss-scaled polar projected gradient has same convention as AC.
            full_gradient = gradient*data['count']
            pg = float(np.max(abs(np.clip(-full_gradient, lower-result.x, upper-result.x))))
            qualified = bool(np.isfinite(value) and pg <= max(1e-4, 10*gtol))
            radius, angle = np.exp(result.x[:m]), result.x[m:]
            r, x = _full_rx(candidate, active, radius*np.cos(angle), radius*np.sin(angle), b)
            record = dict(start=number, loss=float(value*data['count']),
                          projected_gradient=pg, qualified=qualified,
                          solver_success=bool(result.success), message=str(result.message),
                          iterations=int(result.nit), r=r.tolist(), x=x.tolist())
            attempts.append(record); records.append((record, r, x))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempts.append(dict(start=number, qualified=False, error=repr(error)))
    if not records:
        return dict(status='failed', qualified=False, attempts=attempts,
                    seconds=perf_counter()-started, truth_used=False, response_calls=calls,
                    initializer=initializer)
    record, r, x = min(records, key=lambda item: item[0]['loss'])
    value = profile_at(candidate, data, r, x, derivatives=False)
    theta = np.r_[np.log(np.hypot(r, x)), np.arctan2(x, r)]
    return dict(status='qualified' if record['qualified'] else 'unqualified',
                qualified=record['qualified'], loss=record['loss'], J=.5*record['loss'],
                r=r.tolist(), x=x.tolist(), theta=theta.tolist(), H=value['H'].tolist(),
                p=value['p'].tolist(), q=value['q'].tolist(), residual=value['residual'].tolist(),
                covariance=value['covariance'].tolist(), active_edges=active.tolist(),
                unidentified_edges=np.setdiff1d(np.arange(candidate.m), active).tolist(),
                attempts=attempts, projected_gradient=record['projected_gradient'],
                seconds=perf_counter()-started, response_calls=calls, truth_used=False,
                initializer_seconds=initializer['seconds'],
                line_domain='original_polar_magnitude_and_ratio',
                PQ_domain='unrestricted_profile_latent_inputs',
                global_optimality_certified=False, AC_lower_bound=False)


__all__ = ['contrast_matrix', 'prepare', 'response_matrix', 'profile_at',
           'fixed_covariance', 'score_fixed', 'convex_first_order_bound',
           'fit_qp', 'fit_profile']
