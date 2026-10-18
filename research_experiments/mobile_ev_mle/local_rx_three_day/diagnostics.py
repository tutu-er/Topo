"""Observation-only local AC diagnostics; derived quantities add no likelihood.

The public entry point accepts a candidate, P/Q/V readings, declared meter and
physical domains, and an attained fit. It never generates a case or reads truth.
R supplies a genuinely additional cut-voltage meter; K fixes the cut voltage.
"""
from __future__ import annotations

import numpy as np

from .. import likelihood as lk


def _arrays(observed, sigma):
    obs = {k: np.asarray(observed[k], float) for k in ('p', 'q', 'v')}
    if obs['p'].ndim != 2 or any(a.shape != obs['p'].shape for a in obs.values()):
        raise ValueError('P/Q/V must have matching time-by-terminal shapes')
    sig = {k: np.broadcast_to(np.asarray(sigma[k], float), obs['p'].shape)
           for k in ('p', 'q', 'v')}
    if any(not np.isfinite(a).all() for a in (*obs.values(), *sig.values())):
        raise ValueError('finite measurements and standard deviations required')
    if any(np.any(a <= 0) for a in sig.values()):
        raise ValueError('meter standard deviations must be positive')
    return obs, sig


def _polar_map(r, x):
    return np.block([[np.diag(r), -np.diag(x)],
                     [np.diag(x), np.diag(r)]])


def _physical_and_residual(candidate, observed, sigma, theta, local,
                           root_known, root_meter, root_scale=.01):
    n, m = candidate.n, candidate.m
    z = np.exp(theta[:m] + 1j*theta[m:])
    p = observed['p'] + sigma['p']*local[:, :n]
    q = observed['q'] + sigma['q']*local[:, n:2*n]
    b = (np.broadcast_to(np.asarray(root_known, float), (len(p),))
         if root_known is not None else 1. + root_scale*local[:, 2*n])
    physical = lk.ac_predict(candidate, z.real, z.imag, p, q, root_v=b)
    residual = np.concatenate(((p-observed['p'])/sigma['p'],
                               (q-observed['q'])/sigma['q'],
                               (physical['v']-observed['v'])/sigma['v']), axis=1)
    if root_meter is not None:
        measured, std = (np.broadcast_to(np.asarray(a, float), (len(p),)) for a in root_meter)
        if np.any(std <= 0) or not np.isfinite(measured).all() or not np.isfinite(std).all():
            raise ValueError('finite root measurements and positive sigma required')
        residual = np.c_[residual, (b-measured)/std]
    return physical, residual, p, q, b


def derivative_blocks(candidate, observed, sigma, fitted, *, root_known=None,
                      root_meter=None):
    """Rebuild analytic white-residual blocks in scaled fitting coordinates."""
    if root_known is not None and root_meter is not None:
        raise ValueError('known and metered boundary modes are distinct')
    obs, sig = _arrays(observed, sigma)
    n, m, count = candidate.n, candidate.m, len(obs['p'])
    r, x = (np.asarray(fitted[k], float) for k in ('r', 'x'))
    if r.shape != (m,) or x.shape != (m,) or np.any(r <= 0) or np.any(x <= 0):
        raise ValueError('one positive R/X pair required per candidate edge')
    theta = np.r_[np.log(np.hypot(r, x)), np.arctan2(x, r)]
    p, q = (np.asarray(fitted[k], float) for k in ('p', 'q'))
    b = np.broadcast_to(np.asarray(fitted['root_v'], float), (count,))
    local = np.c_[(p-obs['p'])/sig['p'], (q-obs['q'])/sig['q']]
    if root_known is None:
        local = np.c_[local, (b-1.)/.01]
    physical, residual, p, q, b = _physical_and_residual(
        candidate, obs, sig, theta, local, root_known, root_meter)
    dv = lk._response_derivatives(candidate, physical, False)['v']
    shared = np.zeros((count, residual.shape[1], 2*m))
    shared[:, 2*n:3*n] = (dv[:, :, :2*m] @ _polar_map(r, x))/sig['v'][:, :, None]
    state = np.zeros((count, residual.shape[1], local.shape[1]))
    for j in range(2*n):
        state[:, j, j] = 1.
    state[:, 2*n:3*n, :n] = dv[:, :, 2*m:2*m+n]*sig['p'][:, None, :]/sig['v'][:, :, None]
    state[:, 2*n:3*n, n:2*n] = dv[:, :, 2*m+n:]*sig['q'][:, None, :]/sig['v'][:, :, None]
    if root_known is None:
        db = (physical['v']-2*np.einsum('tnj,tj->tn', dv[:, :, 2*m:], np.c_[p, q]))/b[:, None]
        state[:, 2*n:3*n, 2*n] = db*.01/sig['v']
        if root_meter is not None:
            std = np.broadcast_to(np.asarray(root_meter[1], float), (count,))
            state[:, 3*n, 2*n] = .01/std
    return dict(observed=obs, sigma=sig, theta=theta, local=local,
                shared=shared, state=state, residual=residual,
                physical=physical, p=p, q=q, root_v=b)


def _projected(shared, state):
    # State columns have independent P/Q residual rows and (when present) a
    # nonzero root response, hence full column rank in this model.
    q, _ = np.linalg.qr(state, mode='reduced')
    return (shared-q @ (q.transpose(0, 2, 1) @ shared)).reshape(-1, shared.shape[2])


def _spectrum(matrix):
    singular = np.linalg.svd(matrix, compute_uv=False)
    tol = max(float(singular[0])*1e-7, 1e-9) if len(singular) else 1e-9
    return dict(rank=int(np.count_nonzero(singular > tol)),
                width=int(matrix.shape[1]), singular_values=singular.tolist(),
                singular_value_tolerance=tol)


def _excitation(p, q):
    rows = []
    for j in range(p.shape[1]):
        pp, qq = p[:, j], q[:, j]
        kappa = float(pp @ qq/max(float(pp @ pp), 1e-30))
        pc, qc = pp-pp.mean(), qq-qq.mean()
        denominator = float(np.linalg.norm(pc)*np.linalg.norm(qc))
        corr = float(pc @ qc/denominator) if denominator > 1e-30 else None
        rows.append(dict(kappa=kappa, centered_correlation=corr,
                         q_orthogonal_fraction=float(np.linalg.norm(qq-kappa*pp)/max(np.linalg.norm(qq), 1e-30)),
                         p_coefficient_of_variation=float(np.std(pp)/max(abs(pp.mean()), 1e-30)),
                         q_coefficient_of_variation=float(np.std(qq)/max(abs(qq.mean()), 1e-30))))
    return rows


def star_local_identity(candidate, physical, r, x, p, q):
    """Exact receiving-end DistFlow check, not an extra measured residual."""
    if candidate.m != candidate.n+1 or any(candidate.parents[e] != 0 for e in candidate.terminal_edges):
        return dict(supported=False, reason='formula applies to direct leaves of the common junction')
    ids = np.asarray(candidate.terminal_edges)
    rr, xx = np.asarray(r)[ids], np.asarray(x)[ids]
    w = np.asarray(physical['v'])**2
    s = (p*p+q*q)/w
    loss = 2*(rr*p+xx*q)+(rr*rr+xx*xx)*s
    wc = abs(np.asarray(physical['voltage'])[:, 0])**2
    # Orthonormal contrasts: D D^T=I and D 1=0. All three pair differences
    # are redundant; these two retain their complete contrast space.
    if candidate.n == 3:
        contrast = np.array([[1., -1., 0.], [1., 1., -2.]])/np.sqrt([[2.], [6.]])
    else:
        contrast = np.linalg.qr(np.c_[np.ones(candidate.n), np.eye(candidate.n)[:, :-1]])[0][:, 1:].T
    local_error = w+loss-wc[:, None]
    difference_error = (w+loss) @ contrast.T
    return dict(supported=True, branch_max_abs=float(np.max(abs(local_error))),
                contrast_max_abs=float(np.max(abs(difference_error))),
                dL_dr=(2*p+2*rr*s).tolist(), dL_dx=(2*q+2*xx*s).tolist(),
                contrast=contrast.tolist(),
                interpretation='exact derived physical identity; not additional likelihood')


def canonical_trunk_check(candidate, physical, domain, r, x, p, q, anchor):
    """Try a supplied common-trunk anchor without changing the fit or domain."""
    z = complex(anchor)
    legal_z = (z.real > 0 and z.imag > 0 and domain['z_min'] <= abs(z) <= domain['z_max']
               and domain['ratio_min'] <= z.imag/z.real <= domain['ratio_max'])
    if not legal_z:
        return dict(legal=False, reason='anchor_outside_original_line_domain')
    b = abs(physical['voltage'][:, 0]+z*physical['branch_current'][:, 0])
    if np.any(b < domain['root_min']) or np.any(b > domain['root_max']):
        return dict(legal=False, reason='mapped_root_outside_original_domain',
                    mapped_root_min=float(b.min()), mapped_root_max=float(b.max()))
    rr, xx = np.array(r, copy=True), np.array(x, copy=True)
    rr[0], xx[0] = z.real, z.imag
    mapped = lk.ac_predict(candidate, rr, xx, p, q, root_v=b)
    all_v = abs(mapped['voltage'])
    legal = bool(np.min(all_v) >= domain['v_min'] and np.max(all_v) <= domain['v_max']
                 and np.all(np.hypot(p, q) <= np.asarray(domain['power_caps'])+1e-10))
    return dict(legal=legal, reason='legal_equivalent_response' if legal else 'mapped_state_outside_original_domain',
                mapped_root_min=float(b.min()), mapped_root_max=float(b.max()),
                terminal_voltage_max_abs_change=float(np.max(abs(mapped['v']-physical['v']))),
                fitted_parameters_changed=False, original_domain_changed=False)


def diagnose(candidate, observed, sigma, domain, fitted, *, root_known=None,
             root_meter=None, fd_step=1e-5, canonical_anchor=None):
    """Return JSON-safe, observation-only fit/local-information diagnostics.

    ``root_meter=(readings, sigma)`` represents the additional R-mode sensor.
    It is never synthesized here. ``root_known`` is the separate ideal K mode.
    """
    value = derivative_blocks(candidate, observed, sigma, fitted,
                              root_known=root_known, root_meter=root_meter)
    a, c, residual = (value[k] for k in ('shared', 'state', 'residual'))
    m, n = candidate.m, candidate.n
    loss = float(np.sum(residual**2))
    projected = _projected(a, c)
    info = _spectrum(projected)
    trunk = [0, m]
    leaf = [e for e in range(2*m) if e not in trunk]
    # Marginalize the common trunk before evaluating downstream parameters.
    u, s, _ = np.linalg.svd(projected[:, trunk], full_matrices=False)
    active = s > max(float(s[0])*1e-7, 1e-9) if len(s) else np.zeros(0, bool)
    leaf_matrix = projected[:, leaf]
    if np.any(active):
        leaf_matrix = leaf_matrix-u[:, active] @ (u[:, active].T @ leaf_matrix)
    info['downstream_after_trunk_profile'] = _spectrum(leaf_matrix)
    if root_meter is not None:
        info['without_added_root_meter_at_same_point'] = _spectrum(_projected(a[:, :3*n], c[:, :3*n]))
    checks = []
    directions = [np.eye(2*m)[0], np.eye(2*m)[m]]
    directions += [np.cos(np.arange(2*m)+.3), np.sin(np.arange(2*m)+.7)]
    for number, direction in enumerate(directions):
        direction = direction/np.linalg.norm(direction)
        args = (candidate, value['observed'], value['sigma'])
        plus = _physical_and_residual(*args, value['theta']+fd_step*direction,
            value['local'], root_known, root_meter)[1]
        minus = _physical_and_residual(*args, value['theta']-fd_step*direction,
            value['local'], root_known, root_meter)[1]
        finite = (plus-minus)/(2*fd_step)
        analytic = np.einsum('tri,i->tr', a, direction)
        checks.append(dict(direction=number,
            residual_relative_error=float(np.linalg.norm(finite-analytic)/max(np.linalg.norm(analytic), 1e-12)),
            residual_max_abs_error=float(np.max(abs(finite-analytic))),
            J_gradient_analytic=float(np.sum(analytic*residual)),
            J_gradient_finite=float((np.sum(plus**2)-np.sum(minus**2))/(4*fd_step))))
    # One mixed local direction also audits latent P/Q and root derivatives.
    direction = np.cos(np.arange(c.shape[2])+.4)
    direction /= np.linalg.norm(direction)
    delta = np.broadcast_to(fd_step*direction, value['local'].shape)
    plus = _physical_and_residual(candidate, value['observed'], value['sigma'],
        value['theta'], value['local']+delta, root_known, root_meter)[1]
    minus = _physical_and_residual(candidate, value['observed'], value['sigma'],
        value['theta'], value['local']-delta, root_known, root_meter)[1]
    finite = (plus-minus)/(2*fd_step)
    analytic = np.einsum('tri,i->tr', c, direction)
    checks.append(dict(direction='mixed_time_local',
        residual_relative_error=float(np.linalg.norm(finite-analytic)/max(np.linalg.norm(analytic), 1e-12)),
        residual_max_abs_error=float(np.max(abs(finite-analytic)))))
    r, x = (np.asarray(fitted[k], float) for k in ('r', 'x'))
    components = {key: float(np.sum(residual[:, j*n:(j+1)*n]**2))
                  for j, key in enumerate(('p', 'q', 'v'))}
    if root_meter is not None:
        components['root_meter'] = float(np.sum(residual[:, 3*n]**2))
    result = dict(schema='local_three_day_observation_diagnostics_v1', truth_used=False,
        extra_likelihood_terms=0, loss_recomputed=loss,
        loss_reported=float(fitted['loss']), loss_absolute_difference=abs(loss-float(fitted['loss'])),
        loss_components=components, profile_information=info,
        finite_direction_checks=checks, observed_pq_excitation=_excitation(value['observed']['p'], value['observed']['q']),
        observed_common_pq_excitation=_excitation(value['observed']['p'].sum(axis=1, keepdims=True),
                                                value['observed']['q'].sum(axis=1, keepdims=True))[0],
        local_star_identity=star_local_identity(candidate, value['physical'], r, x, value['p'], value['q']),
        interpretation='derived diagnostics; local information is not a confidence set')
    if canonical_anchor is not None:
        if root_known is not None or root_meter is not None:
            result['canonical_trunk'] = dict(legal=False, reason='only_an_equivalence_for_unmetered_boundary')
        else:
            result['canonical_trunk'] = canonical_trunk_check(candidate, value['physical'], domain,
                r, x, value['p'], value['q'], canonical_anchor)
    return result
