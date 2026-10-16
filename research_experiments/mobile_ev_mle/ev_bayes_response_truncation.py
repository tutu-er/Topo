"""Moments of a Gaussian restricted to the original line-impedance domain.

Coordinates are interleaved ``eta=(log R_0, log(X_0/R_0), ...)``.  The sampler
is the sequential conditional truncated-Normal importance construction (GHK):
the proposal normalizes each one-dimensional conditional interval, and the
importance weight is the product of those interval masses.  Independent
scrambled Sobol replicates diagnose integration error; their reported SEs are
numerical diagnostics, not statistical coverage guarantees.

Only the low-dimensional line marginal is integrated here.  A caller can
recover the other jointly Gaussian states and their cross-covariance by
Gaussian conditioning, without drawing all historical states.
"""
from __future__ import annotations

from math import log, pi

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, logsumexp, ndtr, ndtri, ndtri_exp
from scipy.stats import qmc


_LOG2PI = log(2.0 * pi)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


def _edge_intervals(value, m, name):
    value = np.asarray(value, dtype=float)
    if value.shape == (2,):
        value = np.tile(value, (m, 1))
    if value.shape != (m, 2) or not np.isfinite(value).all():
        raise ValueError(name + ' must be a finite pair or [edges,2] array')
    if np.any(value <= 0.0) or np.any(value[:, 0] >= value[:, 1]):
        raise ValueError(name + ' needs strictly positive, nonempty intervals')
    return value


def _bounds(bounds, dimension=None):
    if not isinstance(bounds, dict):
        raise ValueError('bounds must be the line-domain dictionary')
    try:
        m = int(bounds['edges'])
        lr = np.asarray(bounds['log_R'], float)
        lx = np.asarray(bounds['log_X'], float)
        lb = np.asarray(bounds['log_ratio'], float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('bounds require edges/log_R/log_X/log_ratio') from exc
    if m < 1 or any(a.shape != (m, 2) for a in (lr, lx, lb)):
        raise ValueError('matching [edges,2] log bounds required')
    if dimension is not None and dimension != 2*m:
        raise ValueError('mean/covariance do not match the interleaved line domain')
    if any(not np.isfinite(a).all() or np.any(a[:, 0] >= a[:, 1]) for a in (lr, lx, lb)):
        raise ValueError('finite, strictly nonempty log intervals required')
    outer = np.column_stack((np.maximum(lr[:, 0], lx[:, 0]-lb[:, 1]),
                             np.minimum(lr[:, 1], lx[:, 1]-lb[:, 0])))
    if np.any(outer[:, 0] >= outer[:, 1]):
        raise ValueError('original box/cone has zero-volume or empty line domain')
    return dict(edges=m, log_R=lr, log_X=lx, log_ratio=lb, a_outer=outer)


def eta_to_rx(eta):
    """Return R and X from an array whose last axis is interleaved eta."""
    eta = np.asarray(eta, float)
    if eta.ndim == 0 or eta.shape[-1] % 2 or not np.isfinite(eta).all():
        raise ValueError('finite interleaved eta with an even last dimension required')
    with np.errstate(over='raise', invalid='raise'):
        try:
            r = np.exp(eta[..., 0::2])
            x = np.exp(eta[..., 0::2] + eta[..., 1::2])
        except FloatingPointError as exc:
            raise ValueError('eta produces nonfinite physical impedance') from exc
    if np.any(r <= 0.) or np.any(x <= 0.):
        raise ValueError('eta underflows positive physical impedance')
    return r, x


def eta_jacobian(r, x, ordering='interleaved'):
    """Jacobian of physical R/X with respect to interleaved eta.

    ``ordering='stacked'`` gives rows ``(R_all,X_all)`` as used by the existing
    full AC line Jacobian.  The default rows are ``(R_0,X_0,R_1,X_1,...)``.
    """
    r, x = np.asarray(r, float), np.asarray(x, float)
    if r.ndim != 1 or x.shape != r.shape or not r.size or not all(np.isfinite(a).all() and np.all(a > 0.) for a in (r, x)):
        raise ValueError('matching positive finite one-dimensional R/X required')
    m = r.size
    result = np.zeros((2*m, 2*m))
    if ordering == 'interleaved':
        ri, xi = np.arange(0, 2*m, 2), np.arange(1, 2*m, 2)
    elif ordering == 'stacked':
        ri, xi = np.arange(m), np.arange(m, 2*m)
    else:
        raise ValueError("ordering must be 'interleaved' or 'stacked'")
    a, b = np.arange(0, 2*m, 2), np.arange(1, 2*m, 2)
    result[ri, a] = r
    result[xi, a] = x
    result[xi, b] = x
    return result


def contains_eta(eta, bounds, tolerance=0.):
    """Original R/X-box and ratio-cone membership, expressed in log coordinates."""
    eta = np.asarray(eta, float)
    if eta.ndim == 0:
        raise ValueError('interleaved eta array required')
    spec = _bounds(bounds, eta.shape[-1])
    a, b = eta[..., 0::2], eta[..., 1::2]
    valid = np.isfinite(eta).all(axis=-1)
    for value, interval in ((a, spec['log_R']), (b, spec['log_ratio']), (a+b, spec['log_X'])):
        valid &= np.all((value >= interval[:, 0]-tolerance) & (value <= interval[:, 1]+tolerance), axis=-1)
    return valid


def _logdiff(high, low):
    """log(exp(high)-exp(low)), including exact zero and infinite tails."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return high + np.log(-np.expm1(np.minimum(0., low-high)))


def _log_interval(a, b):
    """Stable standard-Normal interval log mass, vectorized."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    ans = np.full(a.shape, -np.inf)
    valid = a < b
    pos = valid & (a >= 0.)
    neg = valid & (b <= 0.)
    mid = valid & ~pos & ~neg
    ans[pos] = _logdiff(log_ndtr(-a[pos]), log_ndtr(-b[pos]))
    ans[neg] = _logdiff(log_ndtr(b[neg]), log_ndtr(a[neg]))
    with np.errstate(divide='ignore', invalid='ignore'):
        ans[mid] = np.log(ndtr(b[mid])-ndtr(a[mid]))
    # CDF subtraction can lose all bits for a very narrow interval.  Integrate
    # the smooth density locally; eight Gauss-Legendre points are far beyond
    # floating-point accuracy when width * max(1,|midpoint|) <= 1e-4.
    finite = valid & np.isfinite(a) & np.isfinite(b)
    width = np.where(finite, b-a, np.inf)
    center = np.where(finite, a+width*.5, 0.)
    narrow = finite & ((width*np.maximum(1., np.abs(center)) <= 1e-4) | ~np.isfinite(ans))
    if np.any(narrow):
        points = center[narrow, None] + width[narrow, None]*.5*_GL_X
        ans[narrow] = (np.log(width[narrow]) - log(2.) - .5*_LOG2PI
                       + logsumexp(-.5*points**2 + np.log(_GL_W), axis=1))
    return ans


def _truncated_quantile(u, a, b):
    """Inverse truncated standard-Normal CDF without subtracting tiny tails."""
    u, a, b = np.broadcast_arrays(np.asarray(u, float), np.asarray(a, float), np.asarray(b, float))
    u = np.clip(u, np.nextafter(0., 1.), np.nextafter(1., 0.))
    z = np.empty(u.shape)
    pos = a >= 0.
    neg = b <= 0.
    mid = ~pos & ~neg
    lu, l1u = np.log(u), np.log1p(-u)
    z[pos] = -ndtri_exp(np.logaddexp(l1u[pos]+log_ndtr(-a[pos]), lu[pos]+log_ndtr(-b[pos])))
    z[neg] = ndtri_exp(np.logaddexp(l1u[neg]+log_ndtr(a[neg]), lu[neg]+log_ndtr(b[neg])))
    z[mid] = ndtri(ndtr(a[mid])+u[mid]*(ndtr(b[mid])-ndtr(a[mid])))
    finite = np.isfinite(a) & np.isfinite(b)
    width = np.where(finite, b-a, np.inf)
    center = np.where(finite, a+width*.5, 0.)
    narrow = finite & (width*np.maximum(1., np.abs(center)) <= 1e-4)
    if np.any(narrow):
        aa, bb, uu = a[narrow], b[narrow], u[narrow]
        zz = aa+(bb-aa)*uu
        whole = _log_interval(aa, bb)
        # Starting from the uniform quantile, two Newton corrections invert
        # the same locally integrated Gaussian CDF to floating-point precision.
        for _ in range(2):
            part = _log_interval(aa, zz)
            ratio = np.exp(part-whole)
            step = (ratio-uu)*np.exp(whole+.5*zz**2+.5*_LOG2PI)
            zz = np.clip(zz-step, aa, bb)
        z[narrow] = zz
    return np.clip(z, a, b)


def line_prior_mass(mean, cov, bounds, *, epsabs=1e-12, epsrel=1e-10):
    """Product of independent-edge, diagonal-Normal line-domain integrals.

    Each edge is a one-dimensional quadrature over log R, with the conditional
    log-ratio interval integrated analytically.  Correlated posteriors must use
    ``bounded_line_moments`` instead.
    """
    mean, cov = np.asarray(mean, float), np.asarray(cov, float)
    spec = _bounds(bounds, mean.size)
    if mean.shape != (2*spec['edges'],) or cov.shape != (mean.size, mean.size) or not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError('matching finite mean/covariance required')
    if np.any(np.diag(cov) <= 0.) or np.any(cov != np.diag(np.diag(cov))):
        raise ValueError('prior quadrature requires a strictly positive diagonal covariance')
    masses, errors = [], []
    for edge in range(spec['edges']):
        ma, mb = mean[2*edge:2*edge+2]
        sa, sb = np.sqrt(np.diag(cov)[2*edge:2*edge+2])
        alo, ahi = spec['a_outer'][edge]
        blo, bhi = spec['log_ratio'][edge]
        xlo, xhi = spec['log_X'][edge]
        za, zb = (alo-ma)/sa, (ahi-ma)/sa
        outer_mass = float(_log_interval(za, zb))
        # Integrating over the truncated a quantile avoids losing a narrow or
        # far-tail a interval.  The remaining integrand lies in [0,1].
        def integrand(t):
            a = ma+sa*float(_truncated_quantile(t, za, zb))
            low, high = max(blo, xlo-a), min(bhi, xhi-a)
            return float(np.exp(_log_interval((low-mb)/sb, (high-mb)/sb)))
        # Breakpoints at the original cone/box intersections improve quad's
        # reliability at the two changes of active b bound.
        points = []
        for a in (xlo-blo, xlo-bhi, xhi-blo, xhi-bhi):
            if alo < a < ahi:
                logpart = float(_log_interval(za, (a-ma)/sa))
                points.append(float(np.exp(logpart-outer_mass)))
        val, err = quad(integrand, 0., 1., points=sorted(set(points)), epsabs=epsabs, epsrel=epsrel, limit=160)
        if val <= 0. or not np.isfinite(val):
            raise ArithmeticError('diagonal prior normalization failed')
        masses.append(outer_mass+log(val))
        errors.append(float(err/val))
    return dict(log_mass=float(sum(masses)), mass=float(np.exp(sum(masses))),
                edge_log_mass=np.asarray(masses), edge_relative_quad_error=np.asarray(errors),
                method='independent_edge_one_dimensional_quadrature')


def line_prior(m, r_bounds, x_bounds, ratio_bounds):
    """Public proper Gaussian prior, then restricted by the original domain.

    The untruncated a/b means are the log-R/log-ratio interval midpoints; their
    SDs are the corresponding widths/sqrt(12).  X bounds only condition this
    prior and do not change its untruncated mean or covariance.
    """
    if isinstance(m, (bool, np.bool_)) or int(m) != m or m < 1:
        raise ValueError('m must be a positive integer number of edges')
    m = int(m)
    r, x, ratio = (_edge_intervals(v, m, n) for v, n in
                   ((r_bounds, 'r_bounds'), (x_bounds, 'x_bounds'), (ratio_bounds, 'ratio_bounds')))
    bounds = dict(edges=m, r_bounds=r, x_bounds=x, ratio_bounds=ratio,
                  log_R=np.log(r), log_X=np.log(x), log_ratio=np.log(ratio),
                  coordinate_order='interleaved_logR_logXoverR')
    bounds['a_outer'] = _bounds(bounds)['a_outer']
    mean = np.empty(2*m)
    sd = np.empty(2*m)
    mean[0::2], mean[1::2] = bounds['log_R'].mean(axis=1), bounds['log_ratio'].mean(axis=1)
    sd[0::2], sd[1::2] = np.diff(bounds['log_R'], axis=1).ravel()/np.sqrt(12.), np.diff(bounds['log_ratio'], axis=1).ravel()/np.sqrt(12.)
    cov = np.diag(sd**2)
    normalizer = line_prior_mass(mean, cov, bounds)
    return dict(mean=mean, cov=cov, bounds=bounds, log_mass=normalizer['log_mass'],
                normalization=normalizer, prior='proper_diagonal_normal_conditioned_on_original_line_domain')


def _psd_cholesky(cov):
    """Unpivoted conditional factor, allowing exact deterministic coordinates."""
    if not np.allclose(cov, cov.T, atol=1e-13, rtol=1e-11):
        raise ValueError('covariance must be symmetric')
    cov = (cov+cov.T)*.5
    scale = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    tol = 256*np.finfo(float).eps*cov.shape[0]*scale
    work, L = cov.copy(), np.zeros_like(cov)
    corrections = []
    for j in range(cov.shape[0]):
        v = float(work[j, j])
        if v < -tol:
            raise ValueError('covariance is not positive semidefinite')
        if v > 0.:
            L[j, j] = np.sqrt(v)
            L[j+1:, j] = work[j+1:, j]/L[j, j]
            work[j+1:, j+1:] -= np.outer(L[j+1:, j], L[j+1:, j])
            work[j+1:, j+1:] = (work[j+1:, j+1:]+work[j+1:, j+1:].T)*.5
        else:
            if np.any(np.abs(work[j+1:, j]) > tol):
                raise ValueError('zero conditional variance has nonzero residual covariance')
            if v < 0.:
                corrections.append(dict(coordinate=j, negative_pivot=v))
    error = float(np.max(np.abs(L@L.T-cov)))
    if error > 8*tol:
        raise ValueError('conditional covariance factor is numerically unstable')
    return L, dict(rank=int(np.count_nonzero(np.diag(L))),
                   reconstruction_max_error=error, negative_roundoff_pivots=corrections,
                   correction_tolerance=tol)


def _draw_replicate(mean, L, spec, seed, power):
    n, d = 2**power, mean.size
    uniform = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(power)
    z, x = np.zeros((n, d)), np.empty((n, d))
    logw, alive = np.zeros(n), np.ones(n, bool)
    clipping = 0
    for j in range(d):
        edge = j//2
        conditional = mean[j]+z[:, :j]@L[j, :j]
        if j % 2 == 0:
            low = np.full(n, spec['a_outer'][edge, 0])
            high = np.full(n, spec['a_outer'][edge, 1])
        else:
            a = x[:, j-1]
            low = np.maximum(spec['log_ratio'][edge, 0], spec['log_X'][edge, 0]-a)
            high = np.minimum(spec['log_ratio'][edge, 1], spec['log_X'][edge, 1]-a)
        if L[j, j] == 0.:
            x[:, j] = conditional
            alive &= (conditional >= low) & (conditional <= high)
            logw[~alive] = -np.inf
            continue
        aa, bb = (low-conditional)/L[j, j], (high-conditional)/L[j, j]
        masses = _log_interval(aa, bb)
        active = alive & np.isfinite(masses)
        alive &= active
        logw[~alive] = -np.inf
        # Dead particles carry a finite value solely to keep later conditional
        # algebra finite; they retain exactly zero importance weight.
        if np.any(active):
            zz = _truncated_quantile(uniform[active, j], aa[active], bb[active])
            xx = conditional[active]+L[j, j]*zz
            clipped = np.clip(xx, low[active], high[active])
            clipping += int(np.count_nonzero(xx != clipped))
            x[active, j] = clipped
            z[active, j] = (clipped-conditional[active])/L[j, j]
            logw[active] += masses[active]
        x[~active, j] = np.clip(conditional[~active], low[~active], high[~active])
    good = np.isfinite(logw)
    if not np.any(good):
        return dict(log_mass=-np.inf, mean=None, cov=None, ess=0.,
                    samples=x, log_weights=logw, clipping=clipping)
    shift = float(np.max(logw[good]))
    w = np.zeros(n)
    w[good] = np.exp(logw[good]-shift)
    total = float(w.sum())
    w /= total
    mu = w@x
    centered = x-mu
    covariance = (centered*w[:, None]).T@centered
    return dict(log_mass=shift+log(total)-log(n), mean=mu,
                cov=(covariance+covariance.T)*.5, ess=1./float(w@w),
                samples=x, log_weights=logw, clipping=clipping)


def _combine(records, dimension):
    logs = np.asarray([r['log_mass'] for r in records])
    finite = np.isfinite(logs)
    if not np.any(finite):
        return None
    shift = float(np.max(logs[finite]))
    a = np.zeros(len(records))
    a[finite] = np.exp(logs[finite]-shift)
    normalized = a/a.sum()
    means = np.asarray([r['mean'] if r['mean'] is not None else np.zeros(dimension) for r in records])
    covs = np.asarray([r['cov'] if r['cov'] is not None else np.zeros((dimension, dimension)) for r in records])
    mu = normalized@means
    offsets = means-mu
    centered_second = covs+offsets[:, :, None]*offsets[:, None, :]
    cov = np.einsum('r,rij->ij', normalized, centered_second)
    reps = len(records)
    relative_se = float(np.std(a, ddof=1)/np.sqrt(reps)/np.mean(a))
    mean_influence = (a/np.mean(a))[:, None]*(means-mu)
    covariance_influence = (a/np.mean(a))[:, None, None]*(centered_second-cov)
    mean_se = np.std(mean_influence, axis=0, ddof=1)/np.sqrt(reps)
    covariance_se = np.std(covariance_influence, axis=0, ddof=1)/np.sqrt(reps)
    sd = np.sqrt(np.maximum(0., np.diag(cov)))
    # Deterministic coordinates have zero SE; a spurious nonzero SE at zero
    # variance is diagnosed as infinite rather than silently divided away.
    ratio = np.divide(mean_se, sd, out=np.where(mean_se == 0., 0., np.inf), where=sd > 0.)
    norm = np.linalg.norm(cov, ord='fro')
    cov_ratio = float(np.linalg.norm(covariance_se, ord='fro')/norm) if norm > 0. else (0. if not np.any(covariance_se) else np.inf)
    all_logw = np.concatenate([r['log_weights'] for r in records])
    logsum = float(logsumexp(all_logw))
    weights = np.exp(all_logw-logsum)
    ess = 1./float(weights@weights)
    return dict(mean=mu, cov=(cov+cov.T)*.5, log_mass=shift+log(float(np.mean(a))),
                relative_se=relative_se, mean_se=mean_se, covariance_se=covariance_se,
                mean_standardized_se_max=float(np.max(ratio)), covariance_relative_se_fro=cov_ratio,
                weights=weights, ess=ess, replicate_log_mass=logs,
                replicate_mean=means, replicate_cov=covs)


def bounded_line_moments(mean, cov, bounds, seed=0, power=10, max_power=16, replicates=8, *,
                         relative_se_tolerance=.01, min_ess=512., min_ess_fraction=.02,
                         mean_se_tolerance=.01, covariance_se_tolerance=.02,
                         return_samples=False):
    """Approximate the exact bounded-Gaussian target, with explicit diagnostics.

    Powers are increased using the same nested Sobol replicates until mass SE,
    weight concentration, and moment SE targets pass or ``max_power`` is hit.
    A finite estimate that fails these checks has ``status='precision_not_met'``;
    a caller must preserve that failure rather than drop the candidate.  ESS is
    an importance-weight concentration diagnostic, not an iid sample count.
    ``samples``/normalized ``weights`` are returned only if requested.
    """
    mean, cov = np.asarray(mean, float), np.asarray(cov, float)
    if mean.ndim != 1 or not mean.size or cov.shape != (mean.size, mean.size) or not np.isfinite(mean).all() or not np.isfinite(cov).all():
        raise ValueError('matching finite line mean and covariance required')
    spec = _bounds(bounds, mean.size)
    for value, name in ((power, 'power'), (max_power, 'max_power'), (replicates, 'replicates'), (seed, 'seed')):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
            raise ValueError(name+' must be a nonnegative integer')
    if power > max_power or max_power > 24 or replicates < 2:
        raise ValueError('require power<=max_power<=24 and at least two independent replicates')
    checks = (relative_se_tolerance, min_ess, min_ess_fraction, mean_se_tolerance, covariance_se_tolerance)
    if not all(np.isfinite(t) and t >= 0. for t in checks) or min_ess_fraction > 1.:
        raise ValueError('finite nonnegative precision thresholds and ESS fraction <=1 required')
    L, factor = _psd_cholesky(cov)
    seeds = [int(np.random.SeedSequence([int(seed), rep]).generate_state(1)[0]) for rep in range(int(replicates))]
    history, combined, records = [], None, None
    for current in range(int(power), int(max_power)+1):
        records = [_draw_replicate(mean, L, spec, s, current) for s in seeds]
        combined = _combine(records, mean.size)
        count = int(replicates)*2**current
        if combined is None:
            history.append(dict(power=current, points=count, log_mass=-np.inf, passed=False,
                                reason='no_positive_importance_weight'))
            if factor['rank'] == 0:
                break
            continue
        c = combined
        required_ess = max(float(min_ess), float(min_ess_fraction)*count)
        passed = (c['relative_se'] <= relative_se_tolerance and c['ess'] >= required_ess
                  and c['mean_standardized_se_max'] <= mean_se_tolerance
                  and c['covariance_relative_se_fro'] <= covariance_se_tolerance)
        history.append(dict(power=current, points=count, log_mass=c['log_mass'],
                            mass_relative_se=c['relative_se'], ess=c['ess'], required_ess=required_ess,
                            mean_standardized_se_max=c['mean_standardized_se_max'],
                            covariance_relative_se_fro=c['covariance_relative_se_fro'], passed=bool(passed)))
        if passed:
            break
    status = 'ok' if combined is not None and history[-1]['passed'] else 'precision_not_met'
    if combined is None:
        status = 'zero_mass' if factor['rank'] == 0 else 'integration_failed'
        result = dict(mean=np.full(mean.shape, np.nan), cov=np.full(cov.shape, np.nan), log_mass=-np.inf)
    else:
        result = {key: combined[key] for key in ('mean', 'cov', 'log_mass')}
    diag = dict(method='sequential_conditional_truncated_normal_scrambled_sobol',
                coordinate_order='interleaved_logR_logXoverR', status=status,
                precision_met=status == 'ok', initial_power=int(power), final_power=history[-1]['power'],
                replicates=int(replicates), points=history[-1]['points'], replicate_seeds=seeds,
                factor=factor, history=history,
                mass_se_is_delta_approximation=True, ess_is_weight_concentration_only=True,
                conditional_roundoff_clips=sum(r['clipping'] for r in records))
    if combined is not None:
        diag.update({key: combined[key] for key in ('ess', 'relative_se', 'mean_se', 'covariance_se',
                     'mean_standardized_se_max', 'covariance_relative_se_fro',
                     'replicate_log_mass', 'replicate_mean', 'replicate_cov')})
        diag['ess_fraction'] = combined['ess']/history[-1]['points']
        diag['log_mass_se'] = result['log_mass']+log(combined['relative_se']) if combined['relative_se'] > 0. else -np.inf
        diag['min_covariance_eigenvalue'] = float(np.linalg.eigvalsh(result['cov'])[0])
    result.update(status=status, mass=float(np.exp(result['log_mass'])), diagnostics=diag,
                  samples=np.concatenate([r['samples'] for r in records]) if return_samples else None,
                  weights=combined['weights'] if return_samples and combined is not None else None)
    return result
