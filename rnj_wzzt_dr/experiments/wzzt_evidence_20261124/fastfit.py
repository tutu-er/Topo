"""Exact fixed-family fits on QR-compressed data, for heuristic search (Phase 1).

Every family fit solves the same bounded least squares as the package's QPFitter (normalised by the reference MSE,
coefficients in [0, bound]). Compression is exact: with F = QR, ||F theta - y||^2 = ||R theta - Q'y||^2 + c, where
c = ||(I - QQ')y||^2 is computed as a vector norm, so large levels data do not lose precision.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import lsq_linear, nnls

from rnj_wzzt_dr.identification.optimization import feature_matrices
from rnj_wzzt_dr.identification.trees import basis, canonical, children


def _bounded_ls(A, c, bound):
    """min ||A theta - c||^2, 0 <= theta <= bound (NNLS first, BVLS if the upper bound binds)."""
    if A.shape[1] == 0:
        return np.zeros(0)
    theta, _ = nnls(A, c, maxiter=50*A.shape[1])
    if theta.size and theta.max() > bound:
        theta = np.clip(lsq_linear(A, c, bounds=(0., bound), method="bvls", tol=1e-12, max_iter=5000).x, 0., bound)
    return theta


def support_columns(data, C, scale):
    z = np.zeros(data.n); z[list(C)] = 1.
    return np.column_stack((((data.p@z)[:, None]*z).ravel(), ((data.q@z)[:, None]*z).ravel()))/scale


class UniverseFitter:
    """Fits of any sub-family of a fixed universe of supports (the E pool) that contains the mandatory supports.

    Cross-products G = F'F and b = F'y are computed once. To keep precision on levels data (large y, small residual),
    objective values are evaluated around the mandatory-only fit theta0 (residual e0 = y - F theta0):
    ||F theta - y||^2 = d'Gd - 2d'F'e0 + ||e0||^2 with d = theta - theta0, which has no large cancelling terms."""

    def __init__(self, data, universe, reference_mse, alpha, bound=2., keep=None):
        self.n, self.alpha, self.bound = data.n, float(alpha), float(bound)
        self.keep = canonical(keep if keep is not None else basis(self.n))
        self.universe = canonical([*universe, *self.keep])
        self.index = {C: i for i, C in enumerate(self.universe)}
        self.K = len(self.universe)
        scale = np.sqrt(reference_mse)
        Fr, Fx = feature_matrices(data, self.universe)
        F, y = np.column_stack((Fr, Fx))/scale, data.y.ravel()/scale
        self.G, self.b = F.T@F, F.T@y
        kcols = self._cols(self.keep)
        Qk, Rk = np.linalg.qr(F[:, kcols])
        th = _bounded_ls(Rk, Qk.T@y, self.bound)
        self.theta0 = np.zeros(2*self.K); self.theta0[kcols] = th
        e0 = y-F[:, kcols]@th
        self.b0, self.r0 = F.T@e0, float(e0@e0)
        self.N = y.size
        self.cache, self.evaluations = {}, 0

    def _cols(self, family):
        idx = [self.index[C] for C in family]
        return idx+[i+self.K for i in idx]

    def internal(self, family):
        return sum(1 < len(C) < self.n for C in family)

    def fit(self, family):
        """Normalised MSE and coefficients (r then x, in canonical order of family); family must contain keep."""
        family = canonical(family)
        if family in self.cache:
            return self.cache[family]
        cols = self._cols(family)
        G = self.G[np.ix_(cols, cols)]
        w, V = np.linalg.eigh(G)
        pos = w > max(w[-1], 1e-300)*1e-13
        A = (V[:, pos]*np.sqrt(w[pos])).T
        c = (V[:, pos].T@self.b[cols])/np.sqrt(w[pos])
        theta = _bounded_ls(A, c, self.bound)
        d = theta-self.theta0[cols]
        mse = (float(d@G@d)-2.*float(d@self.b0[cols])+self.r0)/self.N
        self.cache[family] = (mse, theta)
        self.evaluations += 1
        return mse, theta

    def score(self, family):
        return self.fit(family)[0]+self.alpha*self.internal(canonical(family))


class ExtensionFitter:
    """Fits of a fixed current family plus one extra support (any compatible support, not only a pool)."""

    def __init__(self, data, family, reference_mse, alpha, bound=2.):
        self.data, self.n, self.alpha, self.bound = data, data.n, float(alpha), float(bound)
        self.family = canonical(family)
        self.scale = np.sqrt(reference_mse)
        Fr, Fx = feature_matrices(data, self.family)
        F, y = np.column_stack((Fr, Fx))/self.scale, data.y.ravel()/self.scale
        self.Qb, self.Rb = np.linalg.qr(F)
        self.qy = self.Qb.T@y
        self.y_perp = y-self.Qb@self.qy
        self.N = y.size
        self.proper = sum(1 < len(C) < self.n for C in self.family)
        theta = _bounded_ls(self.Rb, self.qy, self.bound)
        self.base_mse = (float(np.sum((self.Rb@theta-self.qy)**2))+float(self.y_perp@self.y_perp))/self.N
        self.cache, self.evaluations = {}, 0

    def objective(self, C):
        """W MIQP objective of family + C: MSE/ref + alpha*(old internal count + 1). C=None: no addition."""
        if C is None:
            return self.base_mse+self.alpha*self.proper
        C = tuple(sorted(C))
        if C in self.cache:
            return self.cache[C]
        f = support_columns(self.data, C, self.scale)
        proj = self.Qb.T@f
        f_perp = f-self.Qb@proj
        Qp, Rp = np.linalg.qr(f_perp)
        kb = self.Rb.shape[1]
        A = np.zeros((kb+2, kb+2))
        A[:kb, :kb] = self.Rb; A[:kb, kb:] = proj; A[kb:, kb:] = Rp
        rhs = np.r_[self.qy, Qp.T@self.y_perp]
        const = float(self.y_perp@self.y_perp)-float(np.sum((Qp.T@self.y_perp)**2))
        theta = _bounded_ls(A, rhs, self.bound)
        mse = (float(np.sum((A@theta-rhs)**2))+max(const, 0.))/self.N
        value = mse+self.alpha*(self.proper+1)
        self.cache[C] = value
        self.evaluations += 1
        return value


def compatible_supports(family, n, max_children=20):
    """Every support compatible with a laminar family: unions of 2..k-1 child blocks of one node.

    Returns (supports, skipped_nodes). Nodes with more than max_children children are skipped."""
    out, skipped = [], []
    for node, blocks in children(family, n).items():
        k = len(blocks)
        if k < 3:
            continue
        if k > max_children:
            skipped.append(node); continue
        for size in range(2, k):
            for S in combinations(range(k), size):
                out.append(tuple(sorted(i for b in S for i in blocks[b])))
    return out, skipped


def count_compatible(family, n):
    return sum(2**len(b)-len(b)-2 for b in children(family, n).values() if len(b) >= 3)
