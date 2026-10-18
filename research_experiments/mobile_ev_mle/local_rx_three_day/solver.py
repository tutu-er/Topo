"""Local attained AC EIV fits, with an optional noisy boundary-voltage meter.

The nonlinear objective is always the complete raw Gaussian P/Q/V loss.
The optional boundary meter adds one residual per time, never a fixed input.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import lsq_linear

from ..local_rx_identifiability.fitting import (
    evaluate as _evaluate, eliminated_step, polar_lines, profile_information, _scales,
)


def _root_meter(root_meter, count):
    if root_meter is None:
        return None
    measured, sigma = root_meter
    measured = np.broadcast_to(np.asarray(measured, float), (count,)).copy()
    sigma = np.broadcast_to(np.asarray(sigma, float), (count,)).copy()
    if not np.isfinite(measured).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('boundary observations must be finite and their sigma positive')
    return measured, sigma


def evaluate(candidate, obs, sigma, theta, local, *, root_known=None,
             root_meter=None, derivatives=True):
    """Residual blocks; the boundary meter is the final channel when present."""
    if root_known is not None and root_meter is not None:
        raise ValueError('choose a fixed boundary or a noisy boundary meter')
    value = _evaluate(candidate, obs, sigma, theta, local,
                      root_known=root_known, derivatives=derivatives)
    meter = _root_meter(root_meter, len(obs['p']))
    if meter is not None:
        measured, scale = meter
        value['residual'] = np.c_[value['residual'], (value['root_v']-measured)/scale]
        if derivatives:
            a, c = value['shared'], value['state']
            value['shared'] = np.concatenate((a, np.zeros((len(a), 1, a.shape[2]))), axis=1)
            extra = np.zeros((len(c), 1, c.shape[2]))
            extra[:, 0, -1] = .01/scale
            value['state'] = np.concatenate((c, extra), axis=1)
    return value


def _physical(value, domain, cap):
    voltage = abs(value['physical']['voltage'])
    root = value['root_v']
    return bool(np.min(voltage) >= domain['v_min'] and np.max(voltage) <= domain['v_max']
                and np.all(np.hypot(value['p'], value['q']) <= cap+1e-10)
                and np.min(root) >= domain['root_min']-1e-12
                and np.max(root) <= domain['root_max']+1e-12)


def _gradient(a, c, residual, theta, local, lower, upper, ll, lu):
    ga = np.einsum('tri,tr->i', a, residual)
    gc = np.einsum('tri,tr->ti', c, residual)
    pg = max(float(np.max(abs(np.clip(-ga, lower-theta, upper-theta)))),
             float(np.max(abs(np.clip(-gc, ll-local, lu-local)))) if gc.size else 0.)
    return ga, gc, pg


def _step(a, c, residual, damping, precondition):
    # Coordinate scaling affects only the local GN solve, not the loss or data.
    if not precondition:
        return eliminated_step(a, c, residual, damping)
    sa = 1./np.maximum(np.linalg.norm(a, axis=(0, 1)), 1e-8)
    sc = 1./np.maximum(np.linalg.norm(c, axis=1), 1e-8)
    da, dc = eliminated_step(a*sa, c*sc[:, None, :], residual, damping)
    return sa*da, sc*dc


def fit(candidate, obs, sigma, domain, *, root_known=None, root_meter=None,
        initial=None, multistart=2, max_iter=180, gtol=1e-5, seed=0,
        precondition=False):
    """Bounded damped GN, with time-local block elimination and exact AC checks.

    Every accepted step decreases the original SSE.  Box-active coordinates
    stay fixed in the GN subproblem; all other physical constraints are checked
    by AC back-substitution.  Results are attained local fits, not global MLEs.
    """
    if root_known is not None and root_meter is not None:
        raise ValueError('choose a fixed boundary or a noisy boundary meter')
    if multistart < 1 or max_iter < 1 or gtol <= 0:
        raise ValueError('positive iterations, starts and gradient tolerance required')
    started = perf_counter()
    n, m, count = candidate.n, candidate.m, len(obs['p'])
    meter = _root_meter(root_meter, count)
    sig = _scales(obs, sigma)
    if any(not np.isfinite(s).all() or (s <= 0).any() for s in sig.values()):
        raise ValueError('measurement scales must be finite and positive')
    lo_z, hi_z = float(domain['z_min']), float(domain['z_max'])
    lower = np.r_[np.full(m, np.log(lo_z)), np.full(m, np.arctan(domain['ratio_min']))]
    upper = np.r_[np.full(m, np.log(hi_z)), np.full(m, np.arctan(domain['ratio_max']))]
    cap = np.asarray(domain['power_caps'], float)
    ll = np.c_[(-cap-obs['p'])/sig['p'], (-cap-obs['q'])/sig['q']]
    lu = np.c_[(cap-obs['p'])/sig['p'], (cap-obs['q'])/sig['q']]
    root_init = np.broadcast_to(np.asarray(root_known if root_known is not None
                               else meter[0] if meter is not None else 1., float), (count,))
    if root_known is None:
        ll = np.c_[ll, np.full(count, (domain['root_min']-1.)/.01)]
        lu = np.c_[lu, np.full(count, (domain['root_max']-1.)/.01)]
    local0 = np.zeros_like(ll)
    if root_known is None:
        local0[:, -1] = (np.clip(root_init, domain['root_min'], domain['root_max'])-1.)/.01
    branch_p, branch_q = np.asarray(obs['p']) @ candidate.M, np.asarray(obs['q']) @ candidate.M
    design = np.concatenate((2*candidate.M[None, :, :]*branch_p[:, None, :],
                             2*candidate.M[None, :, :]*branch_q[:, None, :]), axis=2)
    target = root_init[:, None]**2-np.asarray(obs['v'])**2
    if root_known is None and meter is None:
        projection = np.eye(n)-np.ones((n, n))/n
        design = np.einsum('ij,tjk->tik', projection, design)
        target = target @ projection
    initial_linear = lsq_linear(design.reshape(-1, 2*m), target.ravel(),
                               bounds=(np.zeros(2*m), np.full(2*m, hi_z)), tol=1e-9)
    z = initial_linear.x[:m]+1j*initial_linear.x[m:]
    theta0 = np.clip(np.r_[np.log(np.maximum(abs(z), lo_z)), np.angle(z)], lower, upper)
    if initial is not None:
        theta0 = np.asarray(initial['theta'], float).copy()
        if np.shape(initial.get('local')) == local0.shape:
            local0 = np.asarray(initial['local'], float).copy()
    rng = np.random.default_rng(seed)
    starts = [theta0]+[np.clip(theta0+np.r_[rng.normal(0, .45, m), rng.normal(0, .15, m)],
                              lower, upper) for _ in range(multistart-1)]
    attempts, records, calls = [], [], 0
    for number, start in enumerate(starts):
        theta, local = np.clip(start, lower, upper), np.clip(local0, ll, lu)
        damping, reason, trace = 1e-3, 'iteration_limit', []
        try:
            for iteration in range(max_iter):
                value = evaluate(candidate, obs, sig, theta, local,
                                 root_known=root_known, root_meter=meter)
                calls += 1
                residual, a, c = value['residual'], value['shared'], value['state']
                loss = float(np.sum(residual**2))
                if not trace:
                    trace.append(loss)
                ga, gc, pg = _gradient(a, c, residual, theta, local, lower, upper, ll, lu)
                if pg <= gtol:
                    reason = 'projected_gradient'; break
                active_a = ((theta <= lower+1e-10) & (ga > 0)) | ((theta >= upper-1e-10) & (ga < 0))
                active_c = ((local <= ll+1e-10) & (gc > 0)) | ((local >= lu-1e-10) & (gc < 0))
                free = ~active_a
                moved = False
                for _ in range(16):
                    da, dc = _step(a[:, :, free], c*(~active_c)[:, None, :], residual,
                                   damping, precondition)
                    full_da = np.zeros_like(theta); full_da[free] = da
                    trial_theta, trial_local = np.clip(theta+full_da, lower, upper), np.clip(local+dc, ll, lu)
                    try:
                        trial = evaluate(candidate, obs, sig, trial_theta, trial_local,
                                         root_known=root_known, root_meter=meter, derivatives=False)
                        calls += 1
                        legal = _physical(trial, domain, cap)
                        trial_loss = float(np.sum(trial['residual']**2))
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        legal, trial_loss = False, np.inf
                    if legal and trial_loss < loss:
                        theta, local = trial_theta, trial_local
                        trace.append(trial_loss); damping = max(damping/3., 1e-10)
                        moved = True; break
                    damping *= 10.
                if not moved:
                    reason = 'no_accepted_step'; break
            value = evaluate(candidate, obs, sig, theta, local, root_known=root_known, root_meter=meter)
            calls += 1
            _, _, pg = _gradient(value['shared'], value['state'], value['residual'],
                                 theta, local, lower, upper, ll, lu)
            physical = _physical(value, domain, cap)
            record = dict(start=number, loss=float(np.sum(value['residual']**2)),
                          projected_gradient=pg, iterations=iteration+1, reason=reason,
                          accepted_steps=len(trace)-1, trace=trace, physical=physical,
                          qualified=bool(physical and pg <= max(10*gtol, 1e-4)))
            attempts.append(record); records.append((record, theta.copy(), local.copy(), value))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            attempts.append(dict(start=number, qualified=False, trace=trace, error=repr(error)))
    if not records:
        return dict(status='failed', qualified=False, attempts=attempts,
                    seconds=perf_counter()-started, ac_calls=calls, truth_used=False)
    record, theta, local, value = min(records, key=lambda item:(not item[0]['physical'], item[0]['loss']))
    r, x = polar_lines(theta)
    information = profile_information(value['shared'], value['state'], theta)
    active = (theta <= lower+1e-7) | (theta >= upper-1e-7)
    information.update(active_line_coordinate_count=int(active.sum()),
                       active_line_coordinates=np.flatnonzero(active).tolist(),
                       interior_CRLB_interpretation_applicable=bool(information['rank'] == len(theta)
                                                                  and not active.any()))
    return dict(status='qualified' if record['qualified'] else 'unqualified',
                qualified=record['qualified'], loss=record['loss'], J=.5*record['loss'],
                theta=theta.tolist(), local=local.tolist(), r=r.tolist(), x=x.tolist(),
                p=value['p'].tolist(), q=value['q'].tolist(), root_v=value['root_v'].tolist(),
                v=value['physical']['v'].tolist(), residual=value['residual'].tolist(),
                information=information, projected_gradient=record['projected_gradient'],
                trace=record['trace'], attempts=attempts, seconds=perf_counter()-started,
                ac_calls=calls, root_model='known' if root_known is not None else
                'noisy_meter' if meter is not None else 'unmeasured',
                preconditioned=bool(precondition),
                ac_equation_residual=float(value['physical']['ac_equation_residual']),
                power_balance_residual=float(value['physical']['power_balance_residual']),
                global_optimality_certified=False, truth_used=False)
