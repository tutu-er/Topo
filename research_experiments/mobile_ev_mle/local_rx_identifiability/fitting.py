"""Raw P/Q/V error-in-variables AC fitting with time-local state elimination.

Only observed inputs are accepted. All retained solutions are local attained
fits. The high-voltage AC branch and analytic physical derivatives are reused
from likelihood.py; no estimated input is silently treated as noise-free.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.optimize import lsq_linear

from .. import likelihood as lk


def polar_lines(theta):
    m = len(theta)//2
    z = np.exp(theta[:m] + 1j*theta[m:])
    return z.real, z.imag


def _line_map(theta):
    r, x = polar_lines(theta)
    return np.block([[np.diag(r), -np.diag(x)],
                     [np.diag(x), np.diag(r)]])


def _scales(obs, sigma):
    return {k: np.broadcast_to(np.asarray(sigma[k], float), np.shape(obs[k])).copy()
            for k in ('p', 'q', 'v')}


def evaluate(candidate, obs, sigma, theta, local, *, root_known=None,
             root_scale=.01, derivatives=True, fixed_pq=None):
    """Return residual and derivative blocks in fixed dimensionless coordinates."""
    n, t = candidate.n, len(obs['p'])
    sig = _scales(obs, sigma)
    if fixed_pq is None:
        p = np.asarray(obs['p']) + sig['p']*local[:, :n]
        q = np.asarray(obs['q']) + sig['q']*local[:, n:2*n]
        bcol = 2*n
    else:
        p, q = (np.asarray(fixed_pq[k], float) for k in ('p', 'q'))
        bcol = 0
    b = (np.asarray(root_known, float) if root_known is not None
         else 1. + root_scale*local[:, bcol])
    r, x = polar_lines(theta)
    physical = lk.ac_predict(candidate, r, x, p, q, 0., 0., b)
    residual = np.concatenate(((p-obs['p'])/sig['p'],
                               (q-obs['q'])/sig['q'],
                               (physical['v']-obs['v'])/sig['v']), axis=1)
    result = dict(residual=residual, physical=physical, p=p, q=q, root_v=b)
    if not derivatives:
        return result
    dv = lk._response_derivatives(candidate, physical, False)['v']
    m = candidate.m
    shared = np.zeros((t, 3*n, 2*m))
    shared[:, 2*n:] = (dv[:, :, :2*m] @ _line_map(theta))/sig['v'][:, :, None]
    state = np.zeros((t, 3*n, local.shape[1]))
    if fixed_pq is None:
        for j in range(2*n):
            state[:, j, j] = 1.
        state[:, 2*n:, :n] = dv[:, :, 2*m:2*m+n]*sig['p'][:, None, :]/sig['v'][:, :, None]
        state[:, 2*n:, n:2*n] = dv[:, :, 2*m+n:]*sig['q'][:, None, :]/sig['v'][:, :, None]
    if root_known is None:
        # V(b,S)=b*V(1,S/b^2), including magnitude derivatives.
        root_derivative = (physical['v'] - 2*np.einsum('tnj,tj->tn',
            dv[:, :, 2*m:], np.c_[p, q]))/b[:, None]
        state[:, 2*n:, bcol] = root_derivative*root_scale/sig['v']
    result.update(shared=shared, state=state)
    return result


def eliminated_step(shared, state, residual, damping):
    """Exact block elimination of the unconstrained damped GN subproblem."""
    width, loc = shared.shape[2], state.shape[2]
    aa = np.einsum('tri,trj->ij', shared, shared) + damping*np.eye(width)
    ga = np.einsum('tri,tr->i', shared, residual)
    if not loc:
        return np.linalg.solve(aa, -ga), np.empty((len(shared), 0))
    cc = np.einsum('tri,trj->tij', state, state) + damping*np.eye(loc)
    ca = np.einsum('tri,trj->tij', state, shared)
    gc = np.einsum('tri,tr->ti', state, residual)
    rhs = np.concatenate((ca, gc[:, :, None]), axis=2)
    solved = np.linalg.solve(cc, rhs)
    za, zr = solved[:, :, :width], solved[:, :, width]
    reduced = aa - np.einsum('tli,tlj->ij', ca, za)
    gradient = ga - np.einsum('tli,tl->i', ca, zr)
    ds = np.linalg.solve(reduced, -gradient)
    dl = -zr - np.einsum('tli,i->tl', za, ds)
    return ds, dl


def profile_information(shared, state, theta):
    """Local information; null physical directions have unbounded uncertainty."""
    if state.shape[2]:
        # QR by time avoids a dense (T*channels) by (T*states) matrix.
        q, _ = np.linalg.qr(state, mode='reduced')
        perpendicular = shared - q @ (q.transpose(0, 2, 1) @ shared)
    else:
        perpendicular = shared
    _, singular, vt = np.linalg.svd(perpendicular.reshape(-1, shared.shape[2]), full_matrices=False)
    tolerance = max(singular[0]*1e-7, 1e-9) if len(singular) else 1e-9
    identified = singular > tolerance
    cov = (vt[identified].T/singular[identified]**2) @ vt[identified]
    transform = _line_map(theta)
    variance = np.maximum(np.diag(transform @ cov @ transform.T), 0)
    null_loading = np.linalg.norm(transform @ vt[~identified].T, axis=1)
    variance[null_loading > 1e-7*np.maximum(np.linalg.norm(transform, axis=1), 1e-15)] = np.inf
    return dict(rank=int(identified.sum()), width=shared.shape[2],
                singular_values=singular.tolist(),
                physical_standard_errors=np.sqrt(variance).tolist(),
                weak_polar_direction=vt[-1].tolist(),
                physical_null_loading=null_loading.tolist(),
                interpretation='local linear precision reference, not a confidence set')


def fit(candidate, obs, sigma, domain, *, root_known=None, initial=None,
        multistart=2, max_iter=120, gtol=1e-5, seed=0, fixed_pq=None):
    started = perf_counter()
    n, m, t = candidate.n, candidate.m, len(obs['p'])
    sig = _scales(obs, sigma)
    lo_z, hi_z = float(domain['z_min']), float(domain['z_max'])
    phi_lo, phi_hi = np.arctan(domain['ratio_min']), np.arctan(domain['ratio_max'])
    lower = np.r_[np.full(m, np.log(lo_z)), np.full(m, phi_lo)]
    upper = np.r_[np.full(m, np.log(hi_z)), np.full(m, phi_hi)]
    cap = np.asarray(domain['power_caps'], float)
    root_scale = .01
    if fixed_pq is None:
        ll = np.c_[(-cap-obs['p'])/sig['p'], (-cap-obs['q'])/sig['q']]
        lu = np.c_[(cap-obs['p'])/sig['p'], (cap-obs['q'])/sig['q']]
    else:
        ll, lu = np.empty((t, 0)), np.empty((t, 0))
    if root_known is None:
        ll = np.c_[ll, np.full(t, (domain['root_min']-1.)/root_scale)]
        lu = np.c_[lu, np.full(t, (domain['root_max']-1.)/root_scale)]
    local0 = np.zeros_like(ll)
    root_init = np.asarray(root_known if root_known is not None else np.ones(t))
    branch_p, branch_q = np.asarray(obs['p']) @ candidate.M, np.asarray(obs['q']) @ candidate.M
    design = np.concatenate((2*candidate.M[None, :, :]*branch_p[:, None, :],
                             2*candidate.M[None, :, :]*branch_q[:, None, :]), axis=2)
    target = root_init[:, None]**2 - np.asarray(obs['v'])**2
    if root_known is None:
        projection = np.eye(n)-np.ones((n, n))/n
        design = np.einsum('ij,tjk->tik', projection, design)
        target = target @ projection
    init_linear = lsq_linear(design.reshape(-1, 2*m), target.ravel(),
                            bounds=(np.zeros(2*m), np.full(2*m, hi_z)), tol=1e-9)
    zz = init_linear.x[:m]+1j*init_linear.x[m:]
    theta0 = np.r_[np.log(np.maximum(abs(zz), lo_z)), np.angle(zz)]
    theta0 = np.clip(theta0, lower, upper)
    if initial is not None:
        theta0 = np.asarray(initial['theta'], float).copy()
        if np.shape(initial.get('local')) == local0.shape:
            local0 = np.asarray(initial['local'], float).copy()
    starts = [theta0]
    rng = np.random.default_rng(seed)
    for _ in range(1, multistart):
        starts.append(np.clip(theta0+np.r_[rng.normal(0, .45, m), rng.normal(0, .15, m)], lower, upper))
    records, attempts, calls = [], [], 0
    for number, initial_theta in enumerate(starts):
        theta = np.clip(initial_theta, lower, upper)
        local = np.clip(local0, ll, lu)
        damping, accepted, reason = 1e-3, 0, 'iteration_limit'
        cache = None
        try:
            for iteration in range(max_iter):
                value = evaluate(candidate, obs, sig, theta, local, root_known=root_known,
                                 fixed_pq=fixed_pq)
                calls += 1
                residual, a, c = value['residual'], value['shared'], value['state']
                loss = float(np.sum(residual**2))
                ga = np.einsum('tri,tr->i', a, residual)
                gc = np.einsum('tri,tr->ti', c, residual)
                projected = max(float(np.max(abs(np.clip(-ga, lower-theta, upper-theta)))),
                                float(np.max(abs(np.clip(-gc, ll-local, lu-local)))) if gc.size else 0.)
                if projected <= gtol:
                    reason = 'projected_gradient'; cache = value; break
                moved = False
                active_shared = ((theta <= lower+1e-10) & (ga > 0)) | ((theta >= upper-1e-10) & (ga < 0))
                active_local = ((local <= ll+1e-10) & (gc > 0)) | ((local >= lu-1e-10) & (gc < 0))
                free_shared = ~active_shared
                # Solve on the current active face, rather than repeatedly
                # clipping a coupled unconstrained direction into a boundary.
                c_free = c * (~active_local)[:, None, :]
                for _ in range(16):
                    ds_free, dl = eliminated_step(a[:, :, free_shared], c_free, residual, damping)
                    ds = np.zeros_like(theta); ds[free_shared] = ds_free
                    trial_theta = np.clip(theta+ds, lower, upper)
                    trial_local = np.clip(local+dl, ll, lu)
                    try:
                        trial = evaluate(candidate, obs, sig, trial_theta, trial_local,
                                         root_known=root_known, derivatives=False, fixed_pq=fixed_pq)
                        calls += 1
                        voltage = abs(trial['physical']['voltage'])
                        legal = (np.min(voltage) >= domain['v_min'] and np.max(voltage) <= domain['v_max']
                                 and np.all(np.hypot(trial['p'], trial['q']) <= cap+1e-10))
                        trial_loss = float(np.sum(trial['residual']**2))
                    except (ValueError, RuntimeError, np.linalg.LinAlgError):
                        legal, trial_loss = False, np.inf
                    if legal and trial_loss < loss:
                        theta, local = trial_theta, trial_local
                        damping = max(damping/3, 1e-10)
                        accepted += 1; moved = True; break
                    damping *= 10
                if not moved:
                    reason = 'no_accepted_step'; cache = value; break
            cache = evaluate(candidate, obs, sig, theta, local, root_known=root_known,
                             fixed_pq=fixed_pq)
            calls += 1
            residual, a, c = cache['residual'], cache['shared'], cache['state']
            ga = np.einsum('tri,tr->i', a, residual)
            gc = np.einsum('tri,tr->ti', c, residual)
            projected = max(float(np.max(abs(np.clip(-ga, lower-theta, upper-theta)))),
                            float(np.max(abs(np.clip(-gc, ll-local, lu-local)))) if gc.size else 0.)
            v = abs(cache['physical']['voltage'])
            physical = bool(np.min(v) >= domain['v_min'] and np.max(v) <= domain['v_max']
                            and np.all(np.hypot(cache['p'], cache['q']) <= cap+1e-10))
            record = dict(start=number, loss=float(np.sum(residual**2)),
                          projected_gradient=projected, iterations=iteration+1,
                          accepted_steps=accepted, reason=reason, physical=physical,
                          qualified=bool(physical and projected <= max(10*gtol, 1e-4)))
            attempts.append(record)
            records.append((record, theta.copy(), local.copy(), cache))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            attempts.append(dict(start=number, qualified=False, error=repr(error)))
    if not records:
        return dict(status='failed', qualified=False, attempts=attempts,
                    seconds=perf_counter()-started, ac_calls=calls)
    # A lower legal objective is always retained; stationarity is reported independently.
    record, theta, local, value = min(records, key=lambda r:(not r[0]['physical'], r[0]['loss']))
    r, x = polar_lines(theta)
    info = profile_information(value['shared'], value['state'], theta)
    line_active = (theta <= lower+1e-7) | (theta >= upper-1e-7)
    info['active_line_coordinate_count'] = int(line_active.sum())
    info['active_line_coordinates'] = np.flatnonzero(line_active).tolist()
    info['interior_CRLB_interpretation_applicable'] = bool(info['rank'] == len(theta) and not line_active.any())
    return dict(status='qualified' if record['qualified'] else 'unqualified',
                qualified=record['qualified'], loss=record['loss'],
                theta=theta.tolist(), local=local.tolist(), r=r.tolist(), x=x.tolist(),
                p=value['p'].tolist(), q=value['q'].tolist(), root_v=value['root_v'].tolist(),
                v=value['physical']['v'].tolist(), residual=value['residual'].tolist(),
                information=info, projected_gradient=record['projected_gradient'],
                attempts=attempts, seconds=perf_counter()-started, ac_calls=calls,
                ac_equation_residual=float(value['physical']['ac_equation_residual']),
                power_balance_residual=float(value['physical']['power_balance_residual']),
                global_optimality_certified=False, truth_used=False)
