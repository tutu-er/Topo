"""Exact local AC loss terms and observation-only physical envelopes.

These are squared-voltage contrasts.  The common full-support edge cancels
exactly, including its loss interaction with every downstream edge.  A bound
computed from the public line domain covers all permitted line parameters;
``fixed_r/fixed_x`` instead gives a state envelope conditional on those lines.
Observed-meter rectangles spend the returned ``alpha_meter`` once over all
P/Q/V channels and times in this function call.  Numerical arithmetic uses
floating-point guards and is not an interval-arithmetic certificate.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .. import likelihood as lk


def contrast_matrix(n=3):
    """Orthonormal Helmert rows: D @ ones = 0 and D @ D.T = identity."""
    n = int(n)
    if n < 2:
        raise ValueError('at least two terminals required')
    result = np.zeros((n-1, n))
    for j in range(n-1):
        result[j, :j+1] = 1./np.sqrt((j+1)*(j+2))
        result[j, j+1] = -(j+1)/np.sqrt((j+1)*(j+2))
    return result


def _line_arrays(candidate, r, x):
    r, x = np.asarray(r, float), np.asarray(x, float)
    if r.shape != (candidate.m,) or x.shape != r.shape:
        raise ValueError('one R/X pair required per candidate edge')
    if not np.isfinite(r).all() or not np.isfinite(x).all() or np.any(r < 0) or np.any(x < 0):
        raise ValueError('finite nonnegative line parameters required')
    return r, x


def exact_remainder(candidate, r, x, p, q, *, physical=None, root_v=1.):
    """Compute the exact loss remainder at one supplied legal AC explanation.

    Write R=M diag(r) M.T and likewise X.  If w is terminal squared voltage,
    ``D w = -2 D(R p + X q) + contrast_remainder``.  For receiving-end
    constant-power terminal loads, branch current is the sum of downstream
    terminal injection currents.  Therefore, with ell_e=|I_e|^2,

      C_e=|z_e|^2 ell_e + 2 sum_{f strict descendant of e} Re(z_e conj(z_f)) ell_f,
      contrast_remainder = -(D M) C.

    This evaluated remainder is a feasible-point diagnostic, not a bound over
    other line parameters or latent states.  No case truth is accepted/read.
    """
    r, x = _line_arrays(candidate, r, x)
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.ndim == 1:
        p, q = p[None, :], q[None, :]
    if p.ndim != 2 or q.shape != p.shape or p.shape[1] != candidate.n or not len(p):
        raise ValueError('matching nonempty time-by-terminal P/Q required')
    if physical is None:
        physical = lk.ac_predict(candidate, r, x, p, q, root_v=root_v)
    d = contrast_matrix(candidate.n)
    dm = d @ np.asarray(candidate.M, float)
    # Exact algebraic cancellation prevents tiny roundoff in the common mode
    # from being incorrectly treated as a dependence on the common trunk.
    dm[:, np.all(candidate.M == 1, axis=0)] = 0.
    ell = np.abs(np.asarray(physical['branch_current']))**2
    if ell.shape != (len(p), candidate.m):
        raise ValueError('physical branch current shape does not match P/Q')
    correction = ell*(r*r+x*x)[None, :]
    for e, descendants in enumerate(candidate.descendants):
        if descendants:
            ids = np.asarray(descendants, int)
            correction[:, e] += 2.*(ell[:, ids] @ (r[e]*r[ids]+x[e]*x[ids]))
    remainder = -correction @ dm.T
    rr = (candidate.M*r) @ candidate.M.T
    xx = (candidate.M*x) @ candidate.M.T
    linear = -2.*(p @ rr.T+q @ xx.T) @ d.T
    exact = np.asarray(physical['v'])**2 @ d.T
    error = float(np.max(np.abs(exact-linear-remainder)))
    if error > 1e-9:
        raise ArithmeticError('supplied physical explanation violates local DistFlow identity')
    return dict(contrast=d, edge_loss_correction=correction,
                squared_branch_current=ell, linear_contrast=linear,
                contrast_remainder=remainder, exact_contrast=exact,
                identity_max_abs=error,
                interpretation='evaluated_feasible_point_not_uniform_envelope')


def _meter_array(meter, channel, shape):
    key = channel if channel in meter else 'sigma_'+channel
    values = np.broadcast_to(np.asarray(meter[key], float), shape)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('finite positive meter standard deviations required')
    return values


def _terminal_current_envelope(candidate, domain, observed, meter, alpha_meter):
    n = candidate.n
    caps = np.broadcast_to(np.asarray(domain['power_caps'], float), (n,))
    vmin, vmax = float(domain['v_min']), float(domain['v_max'])
    if not np.isfinite(caps).all() or np.any(caps < 0) or not 0 < vmin <= vmax:
        raise ValueError('nonnegative apparent-power caps and positive voltage range required')
    if observed is None:
        return dict(current_lower=np.zeros((1, n)),
                    current_upper=(caps/vmin)[None, :],
                    power_lower=np.zeros((1, n)), power_upper=caps[None, :],
                    voltage_lower=np.full((1, n), vmin),
                    voltage_upper=np.full((1, n), vmax),
                    alpha_meter=0., simultaneous_z=None,
                    envelope='public_physical_domain')
    if meter is None or not 0 < alpha_meter < 1:
        raise ValueError('observation tightening needs meter and alpha_meter in (0,1)')
    obs = {key: np.asarray(observed[key], float) for key in ('p', 'q', 'v')}
    shape = obs['p'].shape
    if (len(shape) != 2 or shape[0] < 1 or shape[1] != n
            or any(value.shape != shape or not np.isfinite(value).all() for value in obs.values())):
        raise ValueError('matching finite time-by-terminal P/Q/V required')
    sig = {key: _meter_array(meter, key, shape) for key in obs}
    # P(|epsilon_i| > z sigma_i) <= alpha/(3*T*n); independence is not
    # necessary for this union bound, but the marginal Gaussian law is.
    z = float(norm.isf(float(alpha_meter)/(2.*3.*np.prod(shape))))
    maxp = np.abs(obs['p'])+z*sig['p']
    maxq = np.abs(obs['q'])+z*sig['q']
    minp = np.maximum(np.abs(obs['p'])-z*sig['p'], 0.)
    minq = np.maximum(np.abs(obs['q'])-z*sig['q'], 0.)
    slo = np.hypot(minp, minq)
    shi = np.minimum(np.hypot(maxp, maxq), caps[None, :])
    vlo = np.maximum(obs['v']-z*sig['v'], vmin)
    vhi = np.minimum(obs['v']+z*sig['v'], vmax)
    if np.any(slo > shi) or np.any(vlo > vhi) or np.any(vlo <= 0):
        raise ValueError('simultaneous meter rectangle and physical domain do not intersect')
    return dict(current_lower=slo/vhi, current_upper=shi/vlo,
                power_lower=slo, power_upper=shi,
                voltage_lower=vlo, voltage_upper=vhi,
                alpha_meter=float(alpha_meter), simultaneous_z=z,
                envelope='public_domain_intersect_simultaneous_PQV_rectangle')


def remainder_bound(candidate, domain, observed=None, meter=None, *,
                    alpha_meter=.01, fixed_r=None, fixed_x=None,
                    numerical_guard=1e-12):
    """Bound D eta for every legal state in the supplied envelope.

    The default is uniform over the public impedance magnitude/angle domain.
    A supplied fixed R/X pair changes the scope to those lines only.  The
    returned observation-conditioned envelope has failure probability at most
    ``alpha_meter`` for all P/Q/V readings jointly, under the declared marginal
    Gaussian meter model.  It must be included in any subsequent alpha split.

    All quantities returned are squared-voltage units.  ``contrast_center``
    plus the component halfwidths can be used as an outer remainder set;
    ``radius_about_center`` is a Euclidean outer radius.  A large evaluated
    remainder at one fit is not substituted for this uniform envelope.
    """
    if numerical_guard < 0:
        raise ValueError('nonnegative numerical guard required')
    if (fixed_r is None) != (fixed_x is None):
        raise ValueError('fixed_r and fixed_x must be supplied together')
    terminal = _terminal_current_envelope(candidate, domain, observed, meter, alpha_meter)
    lo, hi = terminal['current_lower'], terminal['current_upper']
    # I_e=sum_{j in support(e)} I_j.  Triangle and reverse triangle
    # inequalities need no unobserved voltage phase assumptions.
    current_hi = hi @ candidate.M
    current_lo = np.zeros_like(current_hi)
    for e, support in enumerate(candidate.supports):
        ids = list(support)
        total_hi = hi[:, ids].sum(axis=1)
        current_lo[:, e] = np.maximum(0., np.max(lo[:, ids]+hi[:, ids]-total_hi[:, None], axis=1))
    ell_lo, ell_hi = current_lo**2, current_hi**2
    if fixed_r is None:
        zmin, zmax = float(domain['z_min']), float(domain['z_max'])
        philo, phihi = np.arctan(float(domain['ratio_min'])), np.arctan(float(domain['ratio_max']))
        if not 0 <= zmin <= zmax or not 0 <= philo <= phihi <= np.pi/2:
            raise ValueError('ordered nonnegative public magnitude and X/R domain required')
        diagonal_lo = np.full(candidate.m, zmin*zmin)
        diagonal_hi = np.full(candidate.m, zmax*zmax)
        cross_lo = np.full((candidate.m, candidate.m), zmin*zmin*np.cos(phihi-philo))
        cross_hi = np.full((candidate.m, candidate.m), zmax*zmax)
        scope = 'all_lines_in_public_domain'
    else:
        r, x = _line_arrays(candidate, fixed_r, fixed_x)
        diagonal_lo = diagonal_hi = r*r+x*x
        cross_lo = cross_hi = np.outer(r, r)+np.outer(x, x)
        scope = 'fixed_lines_only_not_public_parameter_domain'
    correction_lo = ell_lo*diagonal_lo
    correction_hi = ell_hi*diagonal_hi
    for e, descendants in enumerate(candidate.descendants):
        if descendants:
            ids = np.asarray(descendants, int)
            correction_lo[:, e] += 2.*(ell_lo[:, ids] @ cross_lo[e, ids])
            correction_hi[:, e] += 2.*(ell_hi[:, ids] @ cross_hi[e, ids])
    d = contrast_matrix(candidate.n)
    dm = d @ candidate.M
    common = np.all(candidate.M == 1, axis=0)
    dm[:, common] = 0.
    positive, negative = np.maximum(dm, 0.), np.minimum(dm, 0.)
    lower = -correction_hi @ positive.T-correction_lo @ negative.T
    upper = -correction_lo @ positive.T-correction_hi @ negative.T
    guard = float(numerical_guard)*(1.+np.maximum(np.abs(lower), np.abs(upper)))
    lower, upper = lower-guard, upper+guard
    center = (lower+upper)/2.
    halfwidth = (upper-lower)/2.
    return dict(contrast=d, contrast_lower=lower, contrast_upper=upper,
                contrast_center=center, contrast_halfwidth=halfwidth,
                radius_about_center=np.linalg.norm(halfwidth, axis=1),
                absolute_radius=np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper)), axis=1),
                edge_correction_lower=correction_lo, edge_correction_upper=correction_hi,
                squared_current_lower=ell_lo, squared_current_upper=ell_hi,
                terminal_envelope=terminal, scope=scope,
                alpha_meter=terminal['alpha_meter'],
                common_support_edges_cancelled=np.flatnonzero(common).tolist(),
                numerical_guard=float(numerical_guard),
                validation='floating_point_guard_not_interval_arithmetic',
                source='observations_meter_and_declared_domain_only')


__all__ = ['contrast_matrix', 'exact_remainder', 'remainder_bound']
