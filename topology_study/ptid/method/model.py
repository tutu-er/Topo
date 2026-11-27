"""The model and the score of draft v16 (eqs 1-5; the load gate is eq. 3).

Data, all slots (hourly, pu): P, Q and Y = V0^2 - V^2 (T x n), V0sq (T); with the PCC power, the unmetered power
u = (uP, uQ) from the balance (eq. 2). Leaf n is then the load: its flow is u and it has no voltage row.

M (eq. 3) removes each meter's offset (the time mean) and whitens the shared PCC error with S = I - gamma 11'/n over
the meters. For a family F, RSS(F) is the residual of the bounded fit (0 <= r_C, x_C <= BOUND) under M, divided by the
noise variance sigma2, and
    crit(F) = RSS(F) / N + alpha k(F),    alpha = (p_s ln N - 2 lam) / N,    N = T n,
so the draft's score is H(F) = crit(F) N / 2 (half-BIC units). Everything here is numpy, plus Gurobi for the bounded fit.
"""
from __future__ import annotations

import copy
import math

import gurobipy as gp
import numpy as np

from .sets import basis, canonical, is_line

BOUND = 2.                    # upper bound of every r_C and x_C (pu), as in the registered code
GATE = .01                    # a load is modelled only when its mean power is at least 1% of the metered mean power
_ENV = None


def env():
    """One quiet Gurobi environment per process."""
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


# ------------------------------------------------------------------ inputs of the score
def balance(P, Q, Y, V0sq, Ppcc, Qpcc):
    """Eq. (2): PCC power minus the meters' power and the losses sum_i S_i (V0 - V_i) / V_i."""
    v0 = np.sqrt(V0sq)[:, None]
    v = np.sqrt(V0sq[:, None]-Y)
    rise = (v0-v)/v
    return Ppcc-(P*(1+rise)).sum(1), Qpcc-(Q*(1+rise)).sum(1)


def load_share(P, u):
    """The gate's statistic (Section II): the unmetered mean power as a share of the metered mean power. Below GATE no
    load is modelled (a prior setting: on the development cases no-load feeders stay within +-0.66%, loads are >= 1.53%)."""
    return float(np.mean(u[0])/np.mean(P.sum(1)))


def gls_gamma(V0sq, Y):
    """Weight of the shared PCC error: S = I - gamma 11'/n whitens the covariance I + rho 11', rho = V0^4 / V^4."""
    n = Y.shape[1]
    rho = float(np.mean(V0sq**2)/np.mean((V0sq[:, None]-Y)**2))
    return 1.-1./math.sqrt(1.+n*rho)


def parameters_per_line(P, Q, pq_sigma=.005, k=2.):
    """p_s: 2 when most meters' Q moves apart from their P beyond the meter class (x identifiable), else 1."""
    pd, qd = P-P.mean(0), Q-Q.mean(0)
    votes = 0
    for i in range(P.shape[1]):
        level = float(np.linalg.norm(Q[:, i]))
        coef = float(pd[:, i]@qd[:, i])/max(float(pd[:, i]@pd[:, i]), 1e-300)
        resid = float(np.linalg.norm(qd[:, i]-coef*pd[:, i]))
        votes += level > 0 and resid > k*math.sqrt(2)*pq_sigma*level
    return 2 if votes > P.shape[1]/2 else 1


def time_basis(T, u=None):
    """Orthonormal time directions removed at set-up: the constant; with u, also the load's profiles (D1, draft v16
    Appendix B; the score itself removes only the constant)."""
    X = np.ones((T, 1)) if u is None else np.column_stack([np.ones(T), *u])
    U, s, _ = np.linalg.svd(X/np.linalg.norm(X, axis=0), full_matrices=False)
    return U[:, s > 1e-8]


def saturated_sigma2(P, Q, Y, gamma, u=None):
    """Noise variance: residual of the saturated model (every R_ij, X_ij free, symmetric) under M; with u the load's
    terms R_is, X_is are free as well (the saturated model, draft v16 eq. 5), so the estimate does not depend on where
    the load is. Projecting out the load's profiles (D1) is the same as fitting those free terms first."""
    T, n = P.shape
    E = time_basis(T, u)
    drop = lambda A: A-E@(E.T@A)
    Pp, Qp, Yp = drop(P), drop(Q), drop(Y-gamma*Y.mean(1, keepdims=True))
    ii, jj = np.triu_indices(n)
    k = np.arange(len(ii))
    cols = []
    for F0 in (Pp, Qp):                                  # entry (i, j): meter i sees flow j, meter j sees flow i
        F = np.zeros((T, n, len(ii)))
        F[:, jj, k] = F0[:, ii]
        F[:, ii, k] = F0[:, jj]
        cols.append((F-gamma*F.mean(1, keepdims=True)).reshape(T*n, len(ii)))
    F = np.hstack(cols)
    theta, _, rank, _ = np.linalg.lstsq(F, Yp.ravel(), rcond=1e-10)
    sse = float(np.sum((Yp.ravel()-F@theta)**2))
    dof = (T-E.shape[1])*(n-int(gamma == 1.))-int(rank)
    return sse/dof


# ------------------------------------------------------------------ the model
class Model:
    """Transformed data of one feeder. `leaves` = n, or n + 1 with the load as leaf n."""

    def __init__(self, P, Q, Y, V0sq, u=None, lam=1.96, pq_sigma=.005):
        P, Q, Y, V0sq = (np.asarray(a, float) for a in (P, Q, Y, V0sq))
        T, n = P.shape
        self.T, self.n, self.N = T, n, T*n
        self.leaves = n if u is None else n+1
        self.gamma = gls_gamma(V0sq, Y)
        self.sigma2 = saturated_sigma2(P, Q, Y, self.gamma, u)
        flows_p = P if u is None else np.column_stack([P, u[0]])
        flows_q = Q if u is None else np.column_stack([Q, u[1]])
        self.Pt, self.Qt = flows_p-flows_p.mean(0), flows_q-flows_q.mean(0)      # offsets removed from the flows
        Yt = Y-self.gamma*Y.mean(1, keepdims=True)                              # S on the meters, then offsets
        self.Yt = Yt-Yt.mean(0)
        self.yy = float(np.sum(self.Yt**2))/self.sigma2
        self.p_s = parameters_per_line(P, Q, pq_sigma)
        self.lam = lam
        self.alpha = (self.p_s*math.log(self.N)-2*lam)/self.N
        if self.alpha <= 0:
            raise ValueError("the structural prior is larger than the BIC price")

    def without_load(self):
        """The same model with u = 0 (for the load's evidence)."""
        other = copy.copy(self)
        if self.leaves > self.n:
            other.Pt, other.Qt = self.Pt.copy(), self.Qt.copy()
            other.Pt[:, self.n] = 0.
            other.Qt[:, self.n] = 0.
        return other

    def flow(self, support, which="p"):
        """Summed (offset-free) flow of a support's leaves, T values."""
        F = self.Pt if which == "p" else self.Qt
        return F[:, list(support)].sum(1)

    def gram(self, family):
        """G = F'F / sigma2 and b = F'Y / sigma2 for the columns of the supports (all r columns, then all x columns).

        Column r_C, row (t, i): the summed flow of C's leaves at t, times meter i's weight W_Ci = [i in C] - gamma |C|/n
        (the meters of C after S); so F'F factorises over time and meters."""
        family = list(family)
        Z = np.zeros((len(family), self.leaves))
        for k, C in enumerate(family):
            Z[k, list(C)] = 1.
        W = Z[:, :self.n]-self.gamma*Z[:, :self.n].mean(1, keepdims=True)
        A, B = self.Pt@Z.T, self.Qt@Z.T
        Wg = W@W.T
        G = np.block([[(A.T@A)*Wg, (A.T@B)*Wg], [(B.T@A)*Wg, (B.T@B)*Wg]])/self.sigma2
        V = self.Yt@W.T
        b = np.r_[(A*V).sum(0), (B*V).sum(0)]/self.sigma2
        return G, b


def factor(G, b, yy):
    """min ||F theta - y||^2 / sigma2 as ||A theta - c||^2 + const, with A'A = G and A'c = b (eigen-decomposition)."""
    w, V = np.linalg.eigh(G)
    keep = w > max(float(w.max()), 0.)*1e-13
    A = np.sqrt(w[keep])[:, None]*V[:, keep].T
    c = (V[:, keep].T@b)/np.sqrt(w[keep])
    return A, c, yy-float(c@c)


def factor_scaled(G, b, yy):
    """factor() in scaled coefficients phi = d theta, d_k = sqrt(G_kk): returns (A, c, const, d) with
    ||F theta - y||^2 / sigma2 = ||A phi - c||^2 + const. The scaled Gram has a unit diagonal, which keeps the solver's
    numbers within a few orders of magnitude (unscaled, Gurobi declared feasible root relaxations infeasible on 47-60
    meter feeders: RESULTS.md, "Solver fix"). Zero columns (the load's own branch {s}) get d = 0: their phi is 0."""
    d = np.sqrt(np.maximum(np.diag(G), 0.))
    live = d > 1e-12*max(float(d.max()), 1e-300)
    inv = np.where(live, 1./np.where(live, d, 1.), 0.)
    A, c, const = factor(G*np.outer(inv, inv), b*inv, yy)
    return A, c, const, np.where(live, d, 0.)


def box_ls(A, c, ub=BOUND):
    """argmin ||A theta - c||^2 subject to 0 <= theta <= ub: the unconstrained solution when it is inside the box,
    otherwise Gurobi."""
    theta = np.linalg.lstsq(A, c, rcond=None)[0]
    if theta.min() >= 0. and theta.max() <= ub:
        return theta
    with gp.Model(env=env()) as m:
        for key, value in dict(Threads=1, OptimalityTol=1e-9, FeasibilityTol=1e-9, BarConvTol=1e-12).items():
            m.setParam(key, value)
        x = m.addMVar(A.shape[1], lb=0., ub=ub)
        res = m.addMVar(A.shape[0], lb=-gp.GRB.INFINITY)
        m.addConstr(res == A@x-c)
        m.setObjective(res@res)
        m.optimize()
        return np.clip(x.X, 0., ub)


class Fitter:
    """Bounded fits of sub-families of a fixed universe of supports, from one Gram matrix (cached)."""

    def __init__(self, model, universe):
        self.model = model
        self.base = basis(model.leaves)
        self.supports = canonical([*universe, *self.base])
        self.index = {C: k for k, C in enumerate(self.supports)}
        self.K = len(self.supports)
        self.G, self.b = model.gram(self.supports)
        self.cache = {}

    def rss(self, family):
        """(RSS / sigma2, theta) of the family plus the basis."""
        family = canonical([*family, *self.base])
        if family not in self.cache:
            idx = [self.index[C] for C in family]
            cols = idx+[k+self.K for k in idx]
            A, c, const = factor(self.G[np.ix_(cols, cols)], self.b[cols], self.model.yy)
            theta = box_ls(A, c)
            self.cache[family] = (const+float(np.sum((A@theta-c)**2)), theta)
        return self.cache[family]

    def crit(self, family):
        lines = sum(1 for C in canonical(family) if is_line(C, self.model.leaves))
        return self.rss(family)[0]/self.model.N+self.model.alpha*lines


def crit(model, family):
    """crit of one family (its own Fitter)."""
    return Fitter(model, family).crit(family)
