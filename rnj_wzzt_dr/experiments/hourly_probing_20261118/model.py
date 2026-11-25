"""Candidate response model with the branch-flow loss term (manuscript (1)) and
the bounded refit with a resistance floor (manuscript (16)-(18)).

Loss term used here (equivalent to manuscript (1)): for support C with
downstream metered flow a_C = z_C'p, b_C = z_C'q,

    y_i = sum_{C ni i} r_C (a_C + L^P_C) + x_C (b_C + L^Q_C) + (r_C^2 + x_C^2) l_C / 4,

with L^P_C = sum_{D strictly inside C} (r_D/2) l_D (likewise Q), and
l_C = ((a_C+L^P_C)^2 + (b_C+L^Q_C)^2) / v_C at the receiving end of C, where v_C
is taken from the candidate's linear prediction (third-order error). This is
the DistFlow recursion of a balanced radial feeder with hidden zero-injection
nodes, so it is exact up to that third-order term for the true candidate.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import lsq_linear


def canonical_family(family, n):
    supports = {tuple(sorted(int(i) for i in C)) for C in family if len(C)}
    supports |= {(i,) for i in range(n)}
    return tuple(sorted(supports, key=lambda s: (len(s), s)))


def internal(family, n):
    """Non-singleton supports below the full set (the identifier's 'internal')."""
    return frozenset(C for C in family if 1 < len(C) < n)


def nonsingleton(family):
    return frozenset(C for C in family if len(C) > 1)


def demean_rows(A, n):
    """Per-terminal demeaning of rows ordered (t, i).

    Fitting on demeaned rows is the same as fitting a free offset per meter
    (Frisch-Waugh-Lovell), also when the other coefficients are bounded: for any
    fixed theta the best offset is the terminal's mean residual.
    """
    A = np.asarray(A, float)
    T = A.shape[0]//n
    B = A.reshape(T, n, -1)
    return (B-B.mean(axis=0, keepdims=True)).reshape(A.shape)


def root_gamma(V0sq, Y):
    """Whitening strength for the shared root-meter error.

    Every terminal's y = V0^2 - V_i^2 subtracts the same root reading, so the
    noise of one time slot has covariance c (I + rho 11'), rho = V0^4 / V^4.
    Its inverse square root is I - gamma 11'/n with gamma = 1 - 1/sqrt(1 + n rho).
    """
    Y = np.asarray(Y, float)
    n = Y.shape[1]
    v0sq = np.asarray(V0sq, float).reshape(-1)
    rho = float(np.mean(v0sq**2)/np.mean((v0sq[:, None]-Y)**2))
    return 1.-1./np.sqrt(1.+n*rho)


def whiten_slots(A, n, gamma):
    """Apply I - gamma 11'/n to every time slot of rows ordered (t, i) (GLS whitening)."""
    A = np.asarray(A, float)
    if not gamma:
        return A
    T = A.shape[0]//n
    B = A.reshape(T, n, -1)
    return (B-gamma*B.mean(axis=1, keepdims=True)).reshape(A.shape)


def identifiable_p(cand, P, Q, pq_sigma, demean=True, k=2.):
    """Number of identifiable parameters of a candidate on metered history P, Q (REAL_NETWORK_CASE.md).

    Each support C has r_C (always identifiable) and x_C, which is identifiable only if the reactive flow b_C
    varies independently of the active flow a_C. The share of b_C not explained by a_C (after removing the
    meter offsets' means), relative to the level of b_C, is about sqrt(2) pq_sigma or less when Q is a fixed
    fraction of P and only the P/Q meter errors differ; x_C counts only above k times that. On the real
    feeders, whose data set fixes the power factor, every support counts 1 (ratios 0.003-0.009). On the 541
    synthetic S1 inputs of DEVELOPMENT_RUNS sections 16-18 the count would differ from 2 x supports for 115 of
    75,573 candidates (0.15%); it is switched on only for the real feeders (settings.identifiable_params).
    """
    a, b = cand.flows(P, Q)
    ad, bd = (a-a.mean(0), b-b.mean(0)) if demean else (a, b)
    count = 0
    for s in range(cand.S):
        level = float(np.linalg.norm(b[:, s]))
        coef = float(ad[:, s]@bd[:, s])/max(float(ad[:, s]@ad[:, s]), 1e-300)
        resid = float(np.linalg.norm(bd[:, s]-coef*ad[:, s]))
        count += 1+(level > 0 and resid > k*math.sqrt(2)*pq_sigma*level)
    return count


class Candidate:
    def __init__(self, family, n, name=None):
        self.n = n
        self.family = canonical_family(family, n)
        self.name = name
        S = len(self.family)
        self.S = S
        self.Z = np.array([[i in C for i in range(n)] for C in self.family], float)
        self.is_single = np.array([len(C) == 1 for C in self.family])
        sets = [set(C) for C in self.family]
        # ancestors-or-self matrix: Anc[c, d] = 1 if support d contains support c
        self.anc = np.array([[sets[c] <= sets[d] for d in range(S)] for c in range(S)], float)
        self.children = [[] for _ in range(S)]
        for c in range(S):
            supers = [d for d in range(S) if d != c and sets[c] < sets[d]]
            if supers:
                parent = min(supers, key=lambda d: len(sets[d]))
                self.children[parent].append(c)
        self.p = 2*S
        self.key = nonsingleton(self.family)

    # ---------------------------------------------------------------- design
    def flows(self, P, Q):
        return P@self.Z.T, Q@self.Z.T

    def design(self, P, Q):
        """Phi with rows (t, i) t-major and columns (r_C..., x_C...)."""
        a, b = self.flows(P, Q)
        T = P.shape[0]
        Fr = (a[:, None, :]*self.Z.T[None, :, :]).reshape(T*self.n, self.S)
        Fx = (b[:, None, :]*self.Z.T[None, :, :]).reshape(T*self.n, self.S)
        return np.hstack((Fr, Fx))

    def linear(self, theta, P, Q):
        r, x = theta[:self.S], theta[self.S:]
        a, b = self.flows(P, Q)
        return (a*r)@self.Z+(b*x)@self.Z

    def loss_term(self, theta, P, Q, v0sq):
        """lambda_k(x; theta) as T x n (zero if theta is None)."""
        r, x = theta[:self.S], theta[self.S:]
        a, b = self.flows(P, Q)
        v_recv = np.asarray(v0sq, float).reshape(-1, 1)-(a*r+b*x)@self.anc.T
        v_recv = np.maximum(v_recv, .5)
        T = P.shape[0]
        LP = np.zeros((T, self.S)); LQ = np.zeros((T, self.S)); ell = np.zeros((T, self.S))
        for c in range(self.S):   # family is sorted by size: children come first
            for d in self.children[c]:
                LP[:, c] += LP[:, d]+.5*r[d]*ell[:, d]
                LQ[:, c] += LQ[:, d]+.5*x[d]*ell[:, d]
            ell[:, c] = ((a[:, c]+LP[:, c])**2+(b[:, c]+LQ[:, c])**2)/v_recv[:, c]
        return (r*LP+x*LQ+.25*(r**2+x**2)*ell)@self.Z

    def predict(self, theta, P, Q, v0sq, loss=True):
        y = self.linear(theta, P, Q)
        return y+self.loss_term(theta, P, Q, v0sq) if loss else y

    def sensitivity(self, theta, P, Q, v0sq, loss=True, h=1e-5):
        """Operating-point sensitivity dy/dp per sample: T x n x n (R_k + J_k)."""
        r = theta[:self.S]
        R = (self.Z.T*r)@self.Z
        T = P.shape[0]
        M = np.repeat(R[None], T, axis=0)
        if loss:
            for m in range(self.n):
                Pp = P.copy(); Pp[:, m] += h
                Pm = P.copy(); Pm[:, m] -= h
                M[:, :, m] += (self.loss_term(theta, Pp, Q, v0sq)-self.loss_term(theta, Pm, Q, v0sq))/(2*h)
        return M

    def bounds(self, r_min, upper):
        lb = np.zeros(2*self.S)
        lb[:self.S][~self.is_single] = r_min
        return lb, np.full(2*self.S, upper)


class Fit:
    """Refit state of one candidate on a growing data set."""

    def __init__(self, cand, theta0, r_min, upper):
        self.c = cand
        self.lb, self.ub = cand.bounds(r_min, upper)
        self.theta = np.clip(np.asarray(theta0, float), self.lb, self.ub)
        self.sse = np.inf
        self.free = np.ones(2*cand.S, bool)
        self.G = None
        self.offset = np.zeros(cand.n)

    def refit(self, Phi, y, lam, steps=1, P=None, Q=None, v0sq=None, use_loss=True, offsets=False, gamma=0.):
        """Bounded LS on y - lambda(theta-); `steps` fixed-point steps of the loss term.

        With `offsets`, every meter gets a free constant (fitted by demeaning);
        theta stays the physical model and `self.offset` holds the meter offsets.
        With `gamma` > 0 the rows are GLS-whitened for the shared root-meter error
        (`whiten_slots`); whitening and demeaning commute.
        """
        n = self.c.n
        A = whiten_slots(Phi, n, gamma)
        A = demean_rows(A, n) if offsets else A
        for step in range(steps):
            target = y-(lam if lam is not None else 0.)
            if not np.all(np.isfinite(target)):
                self.failures = getattr(self, "failures", 0)+1
                target = y
            b = whiten_slots(target[:, None], n, gamma).ravel()
            b = demean_rows(b[:, None], n).ravel() if offsets else b
            try:
                sol = lsq_linear(A, b, bounds=(self.lb, self.ub), method="bvls", tol=1e-12,
                                 max_iter=5000)
            except np.linalg.LinAlgError:
                self.failures = getattr(self, "failures", 0)+1
                sol = lsq_linear(A, b, bounds=(self.lb, self.ub), method="trf", tol=1e-12,
                                 max_iter=5000)
            self.theta = np.clip(sol.x, self.lb, self.ub)
            if use_loss and step < steps-1:
                lam = self.c.loss_term(self.theta, P, Q, v0sq).ravel()
        resid = b-A@self.theta
        self.sse = float(resid@resid)
        self.offset = ((target-Phi@self.theta).reshape(-1, n).mean(axis=0) if offsets else np.zeros(n))
        tol = 1e-9
        self.free = (self.theta > self.lb+tol) & (self.theta < self.ub-tol)
        self.G = A.T@A
        return self

    def frozen_sse(self, Phi, y, lam, offsets=False, gamma=0.):
        resid = y-(lam if lam is not None else 0.)-Phi@self.theta
        resid = whiten_slots(resid[:, None], self.c.n, gamma).ravel()
        if offsets:
            resid = demean_rows(resid[:, None], self.c.n).ravel()
        self.sse = float(resid@resid)
        return self

    def covariance(self, sigma2):
        """Sigma_k on free coordinates (18); bound coordinates get zero."""
        p = len(self.theta)
        cov = np.zeros((p, p))
        f = np.flatnonzero(self.free)
        if len(f):
            G = self.G[np.ix_(f, f)]
            G = G+1e-12*np.trace(G)/len(f)*np.eye(len(f))
            try:
                inv = cho_solve(cho_factor(G), np.eye(len(f)))
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(G)
            cov[np.ix_(f, f)] = sigma2*inv
        return cov
