"""Source-profile values and envelope gradients for fixed-line response audits.

This diagnostic does not select an EV plan or alter a frozen policy. Input P/Q
are calibrated direct inputs, including EV exactly once. Every response object
owns one fixed R/X vector. Retained fits are local source explanations.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import least_squares


def profile_pair(reference, competitor, p, q, source_reference, source_initial,
                 root_v, power_kw, sigma, *, base_kw=20., source_bounds=(-.5, .5),
                 multistart=2, max_nfev=180, tol=1e-11, seed=0):
    """Fit the rival source and differentiate the attained half-SSE in kW.

    Values are time-major [terminal V, root P, root Q]. Source columns are
    independent across time; their exact block Jacobian is assembled here.
    Bounds are independent of the EV input. The voltage range is an audit,
    not a constrained optimizer: use derivatives strictly inside that range.
    """
    began = perf_counter()
    p, q, u = (np.asarray(v, float) for v in (p, q, power_kw))
    if p.ndim != 2 or p.shape != q.shape or p.shape != u.shape:
        raise ValueError('P/Q/power must have matching (time,terminal) shapes')
    T, n = p.shape
    if not T or not n or not np.isfinite([p, q, u]).all():
        raise ValueError('nonempty finite input arrays required')
    if not np.isfinite(base_kw) or base_kw <= 0:
        raise ValueError('positive base_kw required')
    if int(multistart) != multistart or multistart < 1:
        raise ValueError('positive integer multistart required')
    root = np.broadcast_to(np.asarray(root_v, float), (T,)).copy()
    sr = np.asarray(source_reference, float)
    si = np.asarray(source_initial, float)
    if sr.shape != (T, 2) or si.shape != sr.shape or not np.isfinite([sr, si]).all():
        raise ValueError('source arrays must be finite (time,2)')
    sig = np.broadcast_to(np.asarray(sigma, float), (T, n+2)).copy()
    if not np.isfinite(sig).all() or (sig <= 0).any():
        raise ValueError('positive finite output sigma required')
    raw_bounds = np.asarray(source_bounds, float)
    if raw_bounds.shape == (2,):
        lower, upper = np.full((T, 2), raw_bounds[0]), np.full((T, 2), raw_bounds[1])
    elif raw_bounds.shape == (2, T, 2):
        lower, upper = raw_bounds.copy()
    else:
        raise ValueError('source_bounds must be (lower,upper) or (2,time,2)')
    if not np.isfinite(raw_bounds).all() or np.any(lower >= upper):
        raise ValueError('finite ordered source bounds required')
    if np.any(sr < lower) or np.any(sr > upper):
        raise ValueError('reference source lies outside the physical box')
    total = p + u/base_kw
    try:
        ref = reference.evaluate(total, q, sr[:, 0], sr[:, 1], root)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return dict(success=False, failure='reference_response: '+type(exc).__name__+':'+str(exc),
            attempts=[], seconds=perf_counter()-began, line_parameters_fixed=True,
            source_interface=1, global_optimality_certified=False)
    target = ref['values']
    cache = {}

    def evaluate(vector, derivatives=False):
        if 'source' not in cache or not np.array_equal(vector, cache['source']):
            s = np.asarray(vector).reshape(T, 2)
            out = competitor.evaluate(total, q, s[:, 0], s[:, 1], root,
                                      derivatives=derivatives)
            residual = (out['values']-target)/sig
            cache.clear()
            cache.update(source=vector.copy(), output=out, residual=residual)
        elif derivatives and cache['output']['jac'] is None:
            s = np.asarray(vector).reshape(T, 2)
            cache['output'] = competitor.evaluate(total, q, s[:, 0], s[:, 1], root,
                                                  derivatives=True)
        return cache['residual'].ravel()

    def jacobian(vector):
        evaluate(vector, derivatives=True)
        jac = cache['output']['jac'][:, :, -2:]/sig[:, :, None]
        full = np.zeros((T*(n+2), 2*T))
        for t in range(T):
            full[t*(n+2):(t+1)*(n+2), 2*t:2*t+2] = jac[t]
        return full

    def kkt_violation(source_gradient, active):
        return np.where(active < 0, np.minimum(source_gradient, 0.),
            np.where(active > 0, np.maximum(source_gradient, 0.), source_gradient))

    margin = np.minimum(1e-10*np.maximum(upper-lower, 1.), .01*(upper-lower))
    start = np.clip(si, lower+margin, upper-margin).ravel()
    rng = np.random.default_rng(seed)
    starts = [start]
    for _ in range(1, int(multistart)):
        starts.append(np.clip(start+rng.normal(0., .01, 2*T),
                              (lower+margin).ravel(), (upper-margin).ravel()))
    attempts, retained = [], []
    for index, x0 in enumerate(starts):
        row = dict(index=index, success=False, source_initial=x0.reshape(T, 2).tolist())
        try:
            fit = least_squares(evaluate, x0, jac=jacobian,
                bounds=(lower.ravel(), upper.ravel()), x_scale=.01,
                ftol=tol, xtol=tol, gtol=tol, max_nfev=max_nfev)
            residual = evaluate(fit.x).copy()
            value = float(residual@residual)/2
            voltage = cache['output']['values'][:, :n]
            margin_v = float(min(voltage.min()-.85, 1.15-voltage.max()))
            valid = bool(fit.success and np.isfinite(value) and margin_v >= 0.)
            source_gradient = jacobian(fit.x).T@residual
            kkt = float(np.max(abs(kkt_violation(source_gradient, fit.active_mask)), initial=0.))
            row.update(success=valid, solver_success=bool(fit.success),
                half_SSE=value, nfev=int(fit.nfev), optimality=float(fit.optimality),
                source_kkt_violation=kkt,
                active_mask=fit.active_mask.reshape(T, 2).tolist(),
                terminal_voltage_margin=margin_v, message=str(fit.message))
            retained.append((not valid, value, fit, row))
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            row.update(failure=type(exc).__name__+':'+str(exc))
        attempts.append(row)
    result = dict(success=False, attempts=attempts, seconds=perf_counter()-began,
        line_parameters_fixed=True, source_interface=1,
        voltage_check_scope='terminal-only audit; not an active optimization constraint',
        global_optimality_certified=False)
    if not retained:
        result['failure'] = 'all source restarts failed'
        return result
    # Numerical loss ties should not prefer a visibly less stationary restart
    # because of a few last floating-point digits. Keep the attained score.
    usable = [item for item in retained if not item[0]] or retained
    best_value = min(item[1] for item in usable)
    tie_tolerance = 1e-11*(1.+abs(best_value))
    tied = [item for item in usable if item[1] <= best_value+tie_tolerance]
    _, value, fit, row = min(tied, key=lambda item:(item[3]['source_kkt_violation'], item[1]))
    residual = evaluate(fit.x, derivatives=True).reshape(T, n+2).copy()
    out = cache['output']
    B = (out['jac'][:, :, :n]-ref['jac'][:, :, :n])/(sig[:, :, None]*base_kw)
    gradient = np.einsum('toj,to->tj', B, residual)
    source_gradient = jacobian(fit.x).T@residual.ravel()
    active = fit.active_mask
    # KKT: at a lower bound a positive derivative is allowed; at an upper
    # bound a negative derivative is allowed. Interior derivatives must vanish.
    violation = kkt_violation(source_gradient, active)
    result.update(success=row['success'], half_SSE=value, source=fit.x.reshape(T, 2).copy(),
        residual=residual, gradient_per_kw=gradient, active_mask=active.reshape(T, 2),
        source_kkt_violation=float(np.max(abs(violation), initial=0.)),
        scaled_source_kkt_violation=.01*float(np.max(abs(violation), initial=0.)),
        terminal_voltage_margin=row['terminal_voltage_margin'],
        reference_values=target.copy(), competitor_values=out['values'].copy(),
        total_nfev=sum(a.get('nfev', 0) for a in attempts), seconds=perf_counter()-began,
        restart_tie_policy='minimum projected source gradient within 1e-11*(1+best_half_SSE)',
        restart_loss_tie_tolerance=tie_tolerance, restart_numerical_ties=len(tied),
        derivative_scope='envelope on a stable attained local source branch; lines fixed')
    return result


def nearest_summary(pairs):
    """Equal-reference nearest-rival score; retain every signed pair value."""
    groups = {}
    for pair in pairs:
        if not pair.get('success'):
            return dict(success=False, failure='a required pair failed')
        groups.setdefault(pair['a'], []).append(pair)
    selected = []
    for a, rows in sorted(groups.items()):
        best = min(r['half_SSE'] for r in rows)
        active = [r for r in rows if np.isclose(r['half_SSE'], best, rtol=1e-10, atol=1e-10)]
        selected.append(dict(a=a, half_SSE=best, active=[r['b'] for r in active]))
    return dict(success=True, mean_nearest_half_SSE=float(np.mean([r['half_SSE'] for r in selected])),
                nearest=selected)
