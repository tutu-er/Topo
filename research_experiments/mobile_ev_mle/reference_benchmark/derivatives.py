"""Exact AC fit derivatives, omitting only unused terminal P/Q right sides.

The state equation and linear system are those of likelihood._response_derivatives.
Terminal P/Q are fixed during conditional fitting. Thus the fit needs shared
R/X and the two source derivatives, or just the two source derivatives when
lines are fixed. This changes no residual, physical equation, or approximation.
The original full derivative remains available for independent tests and EV
planning, which does need derivatives with respect to terminal P/Q.
"""
import numpy as np


def fit_response_derivatives(candidate, physical, *, include_lines):
    """Physical columns R, X, source P/Q, with R/X omitted for fixed lines."""
    voltage, loads, zbus = (physical[k] for k in ('voltage', 'loads', 'zbus'))
    t, m = voltage.shape
    width = 2*m+2 if include_lines else 2
    coefficient = np.conj(loads)/np.conj(voltage)**2
    k = zbus[None, :, :]*coefficient[:, None, :]
    eye = np.eye(m)[None, :, :]
    matrix = np.concatenate((np.concatenate((eye-k.real, -k.imag), axis=2),
                             np.concatenate((-k.imag, eye+k.real), axis=2)), axis=1)
    rhs = np.zeros((t, m, width), complex)
    if include_lines:
        edge = -candidate.B[None, :, :]*physical['branch_current'][:, None, :]
        rhs[:, :, :m], rhs[:, :, m:2*m] = edge, 1j*edge
    # The same fixed physical source interface as likelihood.ac_predict.
    ds = np.zeros((m, 2), complex)
    ds[0, 0], ds[0, 1] = 1., 1j
    direct = np.conj(ds)[None, :, :]/np.conj(voltage)[:, :, None]
    rhs[:, :, -2:] = -np.einsum('ij,tjk->tik', zbus, direct)
    solution = np.linalg.solve(matrix, np.concatenate((rhs.real, rhs.imag), axis=1))
    dv = solution[:, :m]+1j*solution[:, m:]
    di = -coefficient[:, :, None]*np.conj(dv)
    di[:, :, -2:] += direct
    master = physical['root_v'][:, None]*np.conj(di.sum(axis=1))
    ids = list(candidate.terminal_edges)
    magnitude = (np.conj(voltage[:, ids, None])*dv[:, ids]).real/abs(voltage[:, ids, None])
    return dict(v=magnitude, master_p=master.real, master_q=master.imag)


def physical_problem_jacobian(base, physical_vector):
    """Whiten/time-assemble the unchanged conditional-output physical Jacobian."""
    base.evaluate(physical_vector)
    derivatives = fit_response_derivatives(base.candidate, base.cache['physical'],
                                           include_lines=not base.line_fixed)
    answer = np.zeros((base.t*(base.n+2), base.width))
    width = base.shared+2
    for time in range(base.t):
        for name, offset, count in (('v', 0, base.n), ('master_p', base.n, 1), ('master_q', base.n+1, 1)):
            block = derivatives[name][time].reshape(count, width)
            block = block/np.asarray(base.sigmas[name][time]).reshape(-1, 1)
            rows = slice(time*(base.n+2)+offset, time*(base.n+2)+offset+count)
            if not base.line_fixed:
                answer[rows, :base.shared] = block[:, :base.shared]
            answer[rows, base.shared+2*time:base.shared+2*time+2] = block[:, -2:]
    return answer


def optimizer_problem_jacobian(problem, optimizer_vector):
    """Apply the original physical R/X → R/alpha chain rule and free mask."""
    if problem.base.line_fixed:
        return physical_problem_jacobian(problem.base, optimizer_vector)
    full = problem.expand(optimizer_vector)
    physical = physical_problem_jacobian(problem.base, problem.physical_vector(optimizer_vector))
    _, _, derivative_r, derivative_alpha = problem.mapping.forward(
        full[:problem.m], full[problem.m:problem.shared])
    jr, jx = physical[:, :problem.m], physical[:, problem.m:problem.shared]
    mapped = np.c_[jr+jx*derivative_r[None, :], jx*derivative_alpha[None, :],
                   physical[:, problem.shared:]]
    return mapped[:, problem.free]


__all__ = ['fit_response_derivatives', 'physical_problem_jacobian', 'optimizer_problem_jacobian']
