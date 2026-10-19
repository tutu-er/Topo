"""Fixed-training-line AC EIV scoring on independent noisy terminal meters.

Only time-local P/Q and an unmetered boundary magnitude are fitted.  No line
parameter is updated.  Batched GN steps preserve independent time blocks.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np

from ..local_rx_identifiability.fitting import evaluate


def fit_fixed(candidate, obs, sigma, domain, r, x, *, max_iter=100, gtol=1e-4):
    started = perf_counter()
    obs = {k: np.asarray(obs[k], float) for k in ('p', 'q', 'v')}
    count, n = obs['p'].shape
    sig = {k: np.broadcast_to(np.asarray(sigma[k], float), obs[k].shape) for k in obs}
    theta = np.r_[np.log(np.hypot(r, x)), np.arctan2(x, r)]
    cap = np.asarray(domain['power_caps'])
    lower = np.c_[(-cap-obs['p'])/sig['p'], (-cap-obs['q'])/sig['q'],
                  np.full(count, (domain['root_min']-1.)/.01)]
    upper = np.c_[(cap-obs['p'])/sig['p'], (cap-obs['q'])/sig['q'],
                  np.full(count, (domain['root_max']-1.)/.01)]
    calls = 0

    def ev(local, derivative=True):
        nonlocal calls
        calls += 1
        return evaluate(candidate, obs, sig, theta, local, derivatives=derivative)

    def legal(value):
        volts = abs(value['physical']['voltage'])
        return ((volts.min(axis=1) >= domain['v_min']) &
                (volts.max(axis=1) <= domain['v_max']) &
                np.all(np.hypot(value['p'], value['q']) <= cap+1e-10, axis=1))

    local = np.zeros_like(lower)
    # A LinDistFlow boundary initial value uses the current noisy measurements,
    # not the true boundary, and is only an initial value for full AC fitting.
    resistance = (candidate.M*np.asarray(r))@candidate.M.T
    reactance = (candidate.M*np.asarray(x))@candidate.M.T
    b2 = np.mean(obs['v']**2+2*(obs['p']@resistance.T+obs['q']@reactance.T), axis=1)
    local[:, -1] = (np.sqrt(np.maximum(b2, .1))-1.)/.01
    local = np.clip(local, lower, upper)
    damping = np.full(count, 1e-4)
    trace = []
    reason = 'iteration_limit'
    for iteration in range(max_iter):
        value = ev(local)
        residual, c = value['residual'], value['state']
        loss_t = np.sum(residual**2, axis=1)
        gradient = np.einsum('tri,tr->ti', c, residual)
        pg_t = np.max(abs(np.clip(-gradient, lower-local, upper-local)), axis=1)
        trace.append(dict(iteration=iteration, loss=float(loss_t.sum()),
                          projected_gradient=float(pg_t.max())))
        if pg_t.max() <= gtol:
            reason = 'projected_gradient'
            break
        active = ((local <= lower+1e-10)&(gradient>0)) | ((local >= upper-1e-10)&(gradient<0))
        cf = c*(~active)[:, None, :]
        hessian = np.einsum('tri,trj->tij', cf, cf)
        grad = np.einsum('tri,tr->ti', cf, residual)
        pending = pg_t > gtol
        moved = np.zeros(count, bool)
        for _ in range(18):
            step = np.linalg.solve(hessian+damping[:, None, None]*np.eye(local.shape[1]),
                                   -grad[:, :, None])[:, :, 0]
            trial_local = np.clip(local+step, lower, upper)
            trial_local[~pending] = local[~pending]
            trial = ev(trial_local, False)
            trial_loss = np.sum(trial['residual']**2, axis=1)
            accept = pending & legal(trial) & (trial_loss < loss_t)
            local[accept] = trial_local[accept]
            damping[accept] = np.maximum(damping[accept]/3., 1e-10)
            moved |= accept
            pending &= ~accept
            if not pending.any():
                break
            damping[pending] *= 10.
        if not moved.any():
            reason = 'no_decreasing_step'
            break
    value = ev(local)
    gradient = np.einsum('tri,tr->ti', value['state'], value['residual'])
    pg = float(np.max(abs(np.clip(-gradient, lower-local, upper-local))))
    physical = bool(legal(value).all())
    loss = float(np.sum(value['residual']**2))
    return dict(loss=loss, J=loss/2., r=np.asarray(r), x=np.asarray(x),
                theta=theta, local=local, p=value['p'], q=value['q'],
                v=value['physical']['v'], root_v=value['root_v'],
                residual=value['residual'], projected_gradient=pg,
                qualified=physical and pg <= gtol, physical=physical,
                status=reason, seconds=perf_counter()-started, ac_calls=calls,
                trace=trace, lines_updated=False, truth_used=False,
                global_optimality_certified=False,
                ac_equation_residual=value['physical']['ac_equation_residual'],
                power_balance_residual=value['physical']['power_balance_residual'])
