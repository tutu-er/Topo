"""Exact whitened AC Jacobian blocks, without the global source zero blocks."""
import numpy as np
from ..reference_benchmark.derivatives import fit_response_derivatives


def problem_blocks(problem, vector):
    """Return G[T,n+2,d] and S[T,n+2,2] in the optimizer coordinates.

    The reference implementation builds an entire (T*(n+2),d+2*T)
    array. Here only the shared columns and each time's two source columns
    are stored. The physical AC derivatives and the R/alpha map are unchanged.
    """
    problem.evaluate(vector)
    base = problem.base
    derivatives = fit_response_derivatives(base.candidate, base.cache['physical'],
                                           include_lines=not base.line_fixed)
    rows = []
    for name in ('v', 'master_p', 'master_q'):
        raw = derivatives[name]
        if name != 'v':
            raw = raw[:, None, :]
        sigma = np.asarray(base.sigmas[name])
        if name != 'v':
            sigma = sigma[:, None]
        rows.append(raw / sigma[:, :, None])
    physical = np.concatenate(rows, axis=1)
    source = physical[:, :, -2:]
    if base.line_fixed:
        return np.zeros((*source.shape[:2], 0)), source
    full = problem.expand(vector)
    _, _, dx_dr, dx_da = problem.mapping.forward(
        full[:problem.m], full[problem.m:problem.shared])
    jr, jx = physical[:, :, :problem.m], physical[:, :, problem.m:problem.shared]
    shared = np.concatenate((jr+jx*dx_dr[None, None, :],
                             jx*dx_da[None, None, :]), axis=2)
    return shared[:, :, problem.free[:problem.shared]], source


def matvec(line, source, vector):
    d = line.shape[-1]
    local = np.asarray(vector[d:]).reshape(source.shape[0], 2)
    return np.einsum('tkd,d->tk', line, vector[:d]) + np.einsum('tks,ts->tk', source, local)


def gradient(line, source, residual):
    residual = np.asarray(residual).reshape(source.shape[:2])
    return np.r_[np.einsum('tkd,tk->d', line, residual),
                 np.einsum('tks,tk->ts', source, residual).ravel()]


def column_norms(line, source):
    return np.r_[np.sqrt(np.einsum('tkd,tkd->d', line, line)),
                 np.sqrt(np.einsum('tks,tks->ts', source, source)).ravel()]


def projected_gradient(vector, grad, lower, upper, scale):
    """Infinity norm of the scaled gradient after projecting one scaled step."""
    scaled = np.asarray(grad)*scale
    projected = np.clip(-scaled, (lower-vector)/scale, (upper-vector)/scale)
    return float(np.max(abs(projected), initial=0.))


def dense_jacobian(line, source):
    """Small-fixture audit helper; the production solver never calls it."""
    t, k, d = line.shape
    answer = np.zeros((t*k, d+2*t))
    answer[:, :d] = line.reshape(t*k, d)
    for time in range(t):
        answer[time*k:(time+1)*k, d+2*time:d+2*time+2] = source[time]
    return answer
