"""Broadcastable high-voltage AC coordinates for the declared 3-leaf prior.

Derivation and uniform nominal-domain certificate are in
continuous_ac_inverse_math.md, saved before this implementation. This module
does not access truth, observations, project fitting, future data, or actions.
All leading axes use ordinary NumPy broadcasting; callers insert singleton
particle/time/noise axes explicitly. Final axes are terminal (3) or edge (m).
"""
from __future__ import annotations

import numpy as np


def _layout(candidate):
    if int(candidate.n) != 3:
        raise ValueError("only the declared three-terminal star/cherries are supported")
    supports = tuple(tuple(map(int, s)) for s in candidate.supports)
    roots = [i for i, s in enumerate(supports) if tuple(sorted(s)) == (0, 1, 2)]
    shared = [i for i, s in enumerate(supports) if len(s) == 2]
    leaves = tuple(next(i for i, s in enumerate(supports) if s == (j,)) for j in range(3))
    if len(roots) != 1 or len(shared) > 1 or len(supports) != 4 + len(shared):
        raise ValueError("candidate must be exactly a reduced root/star/cherry")
    if tuple(candidate.terminal_edges) != leaves:
        raise ValueError("terminal_edges disagree with canonical singleton supports")
    pair = tuple(sorted(supports[shared[0]])) if shared else ()
    return roots[0], leaves, shared[0] if shared else None, pair


def _inputs(candidate, r, x, p, q, a, b, root_v):
    root_edge, leaf_edges, shared_edge, pair = _layout(candidate)
    arrays = [np.asarray(value, float) for value in (r, x, p, q, a, b, root_v)]
    m = len(candidate.supports)
    for value, width, name in zip(arrays[:4], (m, m, 3, 3), ("r", "x", "p", "q")):
        if value.ndim < 1 or value.shape[-1] != width:
            raise ValueError(f"{name} must have final axis {width}")
    shape = np.broadcast_shapes(*(v.shape[:-1] for v in arrays[:4]),
                                *(v.shape for v in arrays[4:]))
    r, x = (np.broadcast_to(v, shape + (m,)) for v in arrays[:2])
    p, q = (np.broadcast_to(v, shape + (3,)) for v in arrays[2:4])
    a, b, root = (np.broadcast_to(v, shape) for v in arrays[4:])
    return shape, r, x, p, q, a, b, root, root_edge, leaf_edges, shared_edge, pair


def _leaf(parent_tau, s, z):
    k = np.abs(s)**2
    h = (z * np.conj(s)).real
    a = np.abs(z)**2
    A = parent_tau - 2*h
    D = A*A - 4*a*k
    disc = np.sqrt(np.maximum(D, 0.0))
    denom = A + disc
    numerical = np.isfinite(D) & (D > 0) & (A > 0) & (denom > 0)
    safe = np.where(numerical, denom, 1.0)
    current2 = 2*k/safe
    receiving_tau = .5*denom
    derivative = -current2/np.where(disc > 0, disc, 1.0)
    flow = s + z*current2
    numerical &= np.isfinite(flow) & np.isfinite(derivative) & (receiving_tau > 0)
    return flow, z*derivative, receiving_tau, current2, numerical


def _cherry_equation(u, target, s_pair, z_pair, shared_z):
    flow, dflow, tau_leaf, current2, leaf_valid = _leaf(u[..., None], s_pair, z_pair)
    L, dL = flow.sum(axis=-1), dflow.sum(axis=-1)
    safe_u = np.where(u > 0, u, 1.0)
    kL = np.abs(L)**2
    dk_du = 2*(np.conj(L)*dL).real
    l_shared = kL/safe_u
    dl_du = dk_du/safe_u-kL/safe_u**2
    G = u+2*(shared_z*np.conj(L)).real+np.abs(shared_z)**2*l_shared
    Gprime = 1+2*(shared_z*np.conj(dL)).real+np.abs(shared_z)**2*dl_du
    valid = leaf_valid.all(axis=-1) & (u > 0) & np.isfinite(G) & np.isfinite(Gprime)
    return G-target, Gprime, L, dL, tau_leaf, l_shared, dl_du, valid


def inverse_ac(candidate, r, x, p, q, master_p, master_q, *, root_v=1.0,
               tol=2e-13, max_iter=40):
    """Map terminal totals and actual master power to source and voltages.

    source_jacobian[..., i, j] differentiates source coordinate i with respect
    to master coordinate j (P before Q). jacobian_abs is the source-to-master
    *density* coordinate factor, i.e. abs(det dsource/dmaster). valid records
    numerical equations; high_branch additionally checks all branch choices.
    certified_domain is a conservative subset of the proven nominal envelope.
    The mathematical certificate applies only at root_v=1 and to declared
    prior/action inputs. Other inputs may be solved but are not certified.
    """
    if tol <= 0 or max_iter < 1:
        raise ValueError("positive tol and max_iter required")
    (shape, r, x, p, q, P0, Q0, root, root_edge, leaf_edges,
     shared_edge, pair) = _inputs(candidate, r, x, p, q, master_p, master_q, root_v)
    z, s = r+1j*x, p+1j*q
    S0 = P0+1j*Q0
    z0 = z[..., root_edge]
    root2 = root*root
    safe_root = np.where(root > 0, root, 1.0)
    root_current2 = np.abs(S0)**2/np.where(root2 > 0, root2, 1.0)
    V1 = root-z0*np.conj(S0)/safe_root
    root_tau = np.abs(V1)**2
    leaf_z = z[..., list(leaf_edges)]
    terminal_tau = np.empty(shape+(3,), float)
    junction_tau = np.full(shape, np.nan)
    junction_derivative = np.ones(shape)
    shared_residual = np.zeros(shape)
    shared_high = np.ones(shape, bool)
    iterations = 0
    if shared_edge is None:
        flow, dflow, terminal_tau, _, leaf_valid = _leaf(root_tau[..., None], s, leaf_z)
        downstream, downstream_dtau = flow.sum(axis=-1), dflow.sum(axis=-1)
        numerical = leaf_valid.all(axis=-1)
    else:
        shared_z = z[..., shared_edge]
        s_pair, z_pair = s[..., list(pair)], leaf_z[..., list(pair)]
        lo = .94*root2
        hi = root_tau.copy()
        f_lo, _, _, _, _, _, _, valid_lo = _cherry_equation(lo, root_tau, s_pair, z_pair, shared_z)
        f_hi, _, _, _, _, _, _, valid_hi = _cherry_equation(hi, root_tau, s_pair, z_pair, shared_z)
        bracket_valid = valid_lo & valid_hi & (hi >= lo) & (f_lo <= tol) & (f_hi >= -tol)
        initial = root_tau-2*(shared_z*np.conj(s_pair.sum(axis=-1))).real
        u = np.minimum(np.maximum(initial, lo), hi)
        for iterations in range(1, max_iter+1):
            f, gp, *_rest, local_valid = _cherry_equation(u, root_tau, s_pair, z_pair, shared_z)
            active = bracket_valid & local_valid & (gp > 0)
            converged = active & (np.abs(f) <= tol)
            if np.all(~active | converged):
                break
            lo = np.where(active & (f < 0), u, lo)
            hi = np.where(active & (f >= 0), u, hi)
            next_u = u-f/np.where(gp > 0, gp, 1.0)
            interior = np.isfinite(next_u) & (next_u > lo) & (next_u < hi)
            next_u = np.where(interior, next_u, .5*(lo+hi))
            u = np.where(active & ~converged, next_u, u)
        (shared_residual, gp, L, dL, pair_tau, l_shared,
         dl_du, local_valid) = _cherry_equation(u, root_tau, s_pair, z_pair, shared_z)
        junction_tau = u
        junction_derivative = 1/np.where(gp > 0, gp, 1.0)
        B = L+shared_z*l_shared
        dB = (dL+shared_z*dl_du)*junction_derivative
        third = next(j for j in range(3) if j not in pair)
        flow, dflow, third_tau, _, third_valid = _leaf(root_tau, s[..., third], leaf_z[..., third])
        terminal_tau[..., list(pair)] = pair_tau
        terminal_tau[..., third] = third_tau
        downstream, downstream_dtau = B+flow, dB+dflow
        numerical = bracket_valid & local_valid & third_valid & (gp > 0) & (np.abs(shared_residual) <= tol)
        shared_high = u*u > np.abs(shared_z)**2*np.abs(L)**2

    source = S0-z0*root_current2-downstream
    a0 = np.abs(z0)**2
    grad_tau = np.stack((-2*r[..., root_edge]+2*a0*P0/np.where(root2 > 0, root2, 1.0),
                         -2*x[..., root_edge]+2*a0*Q0/np.where(root2 > 0, root2, 1.0)), axis=-1)
    z0_real = np.stack((z0.real, z0.imag), axis=-1)
    S0_real = np.stack((P0, Q0), axis=-1)
    down_real = np.stack((downstream_dtau.real, downstream_dtau.imag), axis=-1)
    jacobian = np.broadcast_to(np.eye(2), shape+(2, 2)).copy()
    jacobian -= 2*z0_real[..., :, None]*S0_real[..., None, :]/np.where(root2 > 0, root2, 1.0)[..., None, None]
    jacobian -= down_real[..., :, None]*grad_tau[..., None, :]
    determinant = jacobian[..., 0, 0]*jacobian[..., 1, 1]-jacobian[..., 0, 1]*jacobian[..., 1, 0]
    finite_inputs = (np.isfinite(r).all(axis=-1) & np.isfinite(x).all(axis=-1)
                     & np.isfinite(p).all(axis=-1) & np.isfinite(q).all(axis=-1)
                     & np.isfinite(P0) & np.isfinite(Q0) & np.isfinite(root))
    numerical &= finite_inputs & (root > 0) & (r >= 0).all(axis=-1) & (x >= 0).all(axis=-1)
    numerical &= np.isfinite(source) & np.isfinite(jacobian).all(axis=(-1, -2)) & (terminal_tau > 0).all(axis=-1)
    root_high = root_tau > a0*root_current2
    high_branch = numerical & root_high & shared_high

    certified = ((root == 1.0) & (P0 >= 0) & (P0 <= 1) & (Q0 >= 0) & (Q0 <= .5)
                 & (p >= .104-1e-14).all(axis=-1) & (p <= .436+1e-14).all(axis=-1)
                 & (q >= .033-1e-14).all(axis=-1) & (q <= .047+1e-14).all(axis=-1)
                 & (p.sum(axis=-1) <= .708+1e-14) & ((p > .136+1e-14).sum(axis=-1) <= 1)
                 & (r[..., root_edge] <= .006) & (x[..., root_edge] <= .0042)
                 & (r[..., list(leaf_edges)] <= .0408).all(axis=-1)
                 & (x[..., list(leaf_edges)] <= .018).all(axis=-1)
                 & (r >= 0).all(axis=-1) & (x >= 0).all(axis=-1))
    if shared_edge is not None:
        certified &= (r[..., shared_edge] <= .0216) & (x[..., shared_edge] <= .0108)
    certified &= finite_inputs
    safe_tau = np.maximum(terminal_tau, 0.0)
    return dict(v=np.where(numerical[..., None], np.sqrt(safe_tau), np.nan),
                source_p=np.where(numerical, source.real, np.nan),
                source_q=np.where(numerical, source.imag, np.nan),
                source_jacobian=np.where(numerical[..., None, None], jacobian, np.nan),
                determinant=np.where(numerical, determinant, np.nan),
                jacobian_abs=np.where(numerical, np.abs(determinant), np.nan),
                valid=numerical, high_branch=high_branch, certified_domain=certified,
                root_tau=root_tau, junction_tau=junction_tau,
                junction_derivative=junction_derivative,
                downstream=downstream, downstream_dtau=downstream_dtau,
                shared_residual=shared_residual, iterations=int(iterations))


def forward_ac(candidate, r, x, p, q, source_p=0.0, source_q=0.0, root_v=1.0,
               *, tol=2e-13, max_iter=100):
    """Vectorized project-equation AC generation for hypothetical prior draws.

    This duplicates the independently validated bus-impedance equation, not
    project evaluation state. valid requires convergence and full AC/balance
    residuals; high_branch flags voltage magnitudes above .9. See the math
    document for the uniform nominal-domain guarantee.
    """
    if tol <= 0 or max_iter < 1:
        raise ValueError("positive tol and max_iter required")
    (shape, r, x, p, q, sp, sq, root, root_edge, leaf_edges,
     shared_edge, pair) = _inputs(candidate, r, x, p, q, source_p, source_q, root_v)
    m = len(candidate.supports)
    z = r+1j*x
    B = np.asarray(candidate.B, float)
    if B.shape != (m, m):
        raise ValueError("candidate B must be square node-path/edge incidence")
    zbus = np.einsum("ie,...e,je->...ij", B, z, B)
    loads = np.zeros(shape+(m,), complex)
    loads[..., root_edge] = sp+1j*sq
    loads[..., list(leaf_edges)] += p+1j*q
    voltage = np.broadcast_to(root[..., None], shape+(m,)).astype(complex).copy()
    converged = np.zeros(shape, bool)
    for iteration in range(1, max_iter+1):
        safe_v = np.where(np.abs(voltage) > .1, voltage, 1+0j)
        new = root[..., None]-np.einsum("...ij,...j->...i", zbus, np.conj(loads/safe_v))
        error = np.max(np.abs(new-voltage), axis=-1)
        converged = error <= tol
        voltage = new
        if np.all(converged):
            break
    safe_v = np.where(np.abs(voltage) > .1, voltage, 1+0j)
    currents = np.conj(loads/safe_v)
    branch_current = np.einsum("...i,ie->...e", currents, B)
    master = root*np.conj(currents.sum(axis=-1))
    losses = (np.abs(branch_current)**2*z).sum(axis=-1)
    balance = master-loads.sum(axis=-1)-losses
    equation = voltage-root[..., None]+np.einsum("...ij,...j->...i", zbus, currents)
    equation_residual = np.max(np.abs(equation), axis=-1)
    balance_residual = np.abs(balance)
    valid = (converged & np.isfinite(voltage).all(axis=-1) & (root > 0)
             & (r >= 0).all(axis=-1) & (x >= 0).all(axis=-1)
             & (equation_residual <= 2e-10) & (balance_residual <= 2e-10))
    return dict(v=np.where(valid[..., None], np.abs(voltage[..., list(leaf_edges)]), np.nan),
                master_p=np.where(valid, master.real, np.nan),
                master_q=np.where(valid, master.imag, np.nan),
                source_p=sp, source_q=sq, voltage=voltage, branch_current=branch_current,
                valid=valid, high_branch=valid & (np.abs(voltage) >= .9).all(axis=-1),
                ac_equation_residual=equation_residual, power_balance_residual=balance_residual,
                iterations=int(iteration))


def background_logpdf(p_base, q_base):
    """Exact marginal prior density after integrating shared c over intervals.

    Final axis must have three terminals. p_base excludes the known EV P;
    q_base excludes known EV Q (zero in the certified nominal protocol).
    """
    p, q = np.broadcast_arrays(np.asarray(p_base, float), np.asarray(q_base, float))
    if p.ndim < 1 or p.shape[-1] != 3:
        raise ValueError("background P/Q must have final axis three terminals")
    lower = np.maximum(-.012, np.maximum((p-.12-.004).max(axis=-1),
                                         (4*(q-.04-.004)).max(axis=-1)))
    upper = np.minimum(.012, np.minimum((p-.12+.004).min(axis=-1),
                                        (4*(q-.04+.004)).min(axis=-1)))
    length = np.maximum(upper-lower, 0.0)
    finite = np.isfinite(p).all(axis=-1) & np.isfinite(q).all(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(length)-np.log(.024)-6*np.log(.008)
    return np.where(finite & (length > 0), result, -np.inf)


def source_logpdf(source_p, source_q):
    """Independent normalized nominal uniform source density, any leading axes."""
    p, q = np.broadcast_arrays(np.asarray(source_p, float), np.asarray(source_q, float))
    valid = np.isfinite(p) & np.isfinite(q) & (p >= .065) & (p <= .135) & (q >= .02) & (q <= .05)
    return np.where(valid, -np.log(.07*.03), -np.inf)
