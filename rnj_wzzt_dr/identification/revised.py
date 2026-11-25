"""Revised WZZT (WZZT-R; Phase 2 of WZZT_EVIDENCE_PLAN.md): identifiability-aware, BIC-consistent laminar search.

Model, for T slots and n meters (rows ordered (t, i)):

    y_t = sum_C z_C z_C'(r_C p_t + x_C q_t) + s_t 1 + o + e_t.

Nuisance terms are removed by one linear residual operator M = D (x) S_c:
- D: per-meter time demeaning (free meter offsets o), when `offsets`;
- S_c = I - c 11'/n in every slot: c = gamma, GLS for the root-meter error that every y_i = V0^2 - V_i^2 shares
  ("gls"); c = 1, a free source-bus term s_t, so no root reading is needed ("free"); or c = 0 ("none").

Score (R1): J(F) = ||M(y - F theta)||^2 / sigma2 + ln(N) p_s k(F), with sigma2 from the saturated symmetric linear
model under M, k(F) the internal supports and p_s the identifiable parameters per support (1 when Q moves with P,
2 otherwise; meter class pq_sigma). Divided by N this is objective + alpha k, with objective = SSE/(N sigma2) and
alpha = p_s ln N / N, so the original WZZT MIQP forms apply unchanged to the transformed columns.

Every transformed column is a Kronecker product a~_C (x) w_C, with a~_C the (demeaned) flow of support C and
w_C = z_C - c|C|/n 1. Gram matrices therefore come from T x K flows (sufficient statistics, S2):
<col_C, col_D> = (a~_C'a~_D)(|C n D| - (2c - c^2)|C||D|/n).

Stages: R0 guidance (P-only ordered sensitivity fit under M, RNJ trees); W (generator-first one-support additions,
with the exact unknown-support MIQP as the final no-gain check, or at every step); E (a heuristic portfolio over
the finite pool, certified by the SVD-row MIQP with NumericFocus=2 and confirmed by a cold start with another
seed); no output floor (R2: the BIC is the contraction rule). The original `algorithm.fit` is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
import time

import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint, lsq_linear, minimize, nnls

if __package__:
    from . import trees
    from .optimization import _finite, _gap, _gurobi
    from .trees import basis, canonical, children, compatible, is_laminar
else:
    import trees
    from optimization import _finite, _gap, _gurobi
    from trees import basis, canonical, children, compatible, is_laminar


@dataclass(frozen=True)
class RevisedConfig:
    guidance: str = "ponly"        # R0: "ponly" (P-only ordered fit under M) | "pq" (original 15-tree pool)
    score: str = "bic"             # R1: "bic" | "alpha" (fixed alpha per support, reference-tree MSE as unit)
    alpha: float = .003            # score="alpha" only
    source: str = "gls"            # "gls" | "free" | "none" | "estimated" (GLS strength estimated from the data)
    offsets: bool = True
    pq_sigma: float | None = .005  # meter class for the identifiable-parameter count; None counts two
    history: str = "all"           # "all": search on training + validation slots; "train": training only
    w: str = "generator"           # "generator" | "miqp" | "pool"
    w_check: bool = True           # generator mode: unknown-support MIQP as the final no-gain check
    e: str = "certified"           # "certified" | "heuristic" | "miqp"
    confirm: bool = True           # certificate confirmed by a cold start with another seed
    bound: float = 2.
    total_seconds: float = 900.
    w_mip_seconds: float = 120.
    e_search_seconds: float = 120.
    e_cert_seconds: float = 240.
    random_starts: int = 20
    beam_width: int = 20
    block_limit: int = 5
    max_frontier: int = 3000       # frontier candidates added to the E pool (largest first dropped beyond this)
    alternatives_margin: float = 6.   # BIC units: E-search families within this margin are output
    max_alternatives: int = 20
    gain_tol: float = 1e-9
    zero_tol: float = 1e-9
    seed: int = 20261111
    cert_gap: float = 1e-4
    cert_abs_gap: float = 1e-7
    qp_solver: str = "gurobi"      # ordered matrix fits (guidance, residual-RNJ generator): "gurobi" (D4) | "slsqp"

    def validate(self):
        choices = dict(guidance=("ponly", "pq"), score=("bic", "alpha"), source=("gls", "free", "none", "estimated"),
                       history=("all", "train"), w=("generator", "miqp", "pool"),
                       e=("certified", "heuristic", "miqp"), qp_solver=("slsqp", "gurobi"))
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"{key} must be one of {allowed}")


# ------------------------------------------------------------------ data and the residual operator
@dataclass
class Series:
    """Metered history, rows are time slots: p, q, y (T x n) and the root reading v0sq (T,), if any."""
    p: np.ndarray
    q: np.ndarray
    y: np.ndarray
    v0sq: np.ndarray | None = None

    def __post_init__(self):
        self.p, self.q, self.y = (np.asarray(a, float) for a in (self.p, self.q, self.y))
        if self.p.ndim != 2 or self.p.shape != self.q.shape or self.p.shape != self.y.shape:
            raise ValueError("p, q, y must have the same T x n shape")
        if self.v0sq is not None:
            self.v0sq = np.asarray(self.v0sq, float).reshape(-1)
            if len(self.v0sq) != len(self.p):
                raise ValueError("v0sq must have one value per slot")

    @property
    def n(self):
        return self.p.shape[1]

    def join(self, other):
        v = None if self.v0sq is None or other.v0sq is None else np.r_[self.v0sq, other.v0sq]
        return Series(np.r_[self.p, other.p], np.r_[self.q, other.q], np.r_[self.y, other.y], v)


def root_gamma(v0sq, y):
    """GLS strength for the shared root-meter error: S = I - gamma 11'/n whitens c(I + rho 11'), rho = V0^4/V^4."""
    n = y.shape[1]
    rho = float(np.mean(v0sq**2)/np.mean((v0sq[:, None]-y)**2))
    return 1.-1./math.sqrt(1.+n*rho)


def estimated_gamma(series, offsets=True):
    """GLS strength estimated from the data instead of assumed from the root meter's class (source="estimated").

    root_gamma assumes the root reading is as accurate as a household meter. When it is noisier, biased in time
    or absent (y built from a nominal source voltage), the per-slot common term is larger and the whitening must
    be stronger. Under the GLS model the residual of each slot has covariance s2 (I + rho 11'); with the saturated
    symmetric model fitted without any slot term (offsets kept), rho is estimated from the residual's slot means
    against its within-slot spread. Returns (gamma, info); gamma = 1 - 1/sqrt(1 + n rho), in [0, 1)."""
    T, n = series.p.shape
    design = Design(series, Transform(T, n, 0., offsets), 1.)
    F, _ = design.pair_columns()
    target = design.raw_target.ravel()
    theta, _, rank, _ = np.linalg.lstsq(F, target, rcond=1e-10)
    E = (target-F@theta).reshape(T, n)
    m = E.mean(1)
    within = float(np.sum((E-m[:, None])**2))/(T*(n-1))
    common = max(float(np.mean(m**2))-within/n, 0.)
    rho = common/max(within, 1e-300)
    return 1.-1./math.sqrt(1.+n*rho), dict(rho=rho, within=within, common=common, rank=int(rank))


class Transform:
    """M = D (x) S_c on T x n arrays; flows are demeaned in time when offsets are free."""

    def __init__(self, T, n, c, offsets):
        self.T, self.n, self.c, self.offsets = int(T), int(n), float(c), bool(offsets)
        self.kappa = 2*self.c-self.c*self.c

    def flows(self, A):
        return A-A.mean(0) if self.offsets else A

    def target(self, Y):
        Y = Y-self.c*Y.mean(1, keepdims=True)
        return Y-Y.mean(0) if self.offsets else Y

    def slot(self, F):
        """S_c along the meter axis of a T x n x m array (time demeaning is applied to flows beforehand)."""
        return F-self.c*F.mean(1, keepdims=True) if self.c else F

    @property
    def rank(self):
        return (self.T-int(self.offsets))*(self.n-int(self.c == 1.))


def indicators(family, n):
    Z = np.zeros((len(family), n))
    for k, C in enumerate(family):
        Z[k, list(C)] = 1.
    return Z


class Design:
    """The transformed, scaled least-squares problem: objective(theta) = ||y_s - F_s theta||^2 / N."""

    def __init__(self, series, transform, scale=1.):
        self.series, self.t = series, transform
        self.T, self.n = series.p.shape
        self.N = self.T*self.n
        self.Pt, self.Qt = transform.flows(series.p), transform.flows(series.q)
        self.raw_target = transform.target(series.y)
        self.rescale(scale)

    def rescale(self, scale):
        self.scale = float(scale)
        self.Yt = self.raw_target/self.scale
        self.y = self.Yt.ravel()
        self.yy = float(self.y@self.y)

    def blocks(self, family):
        """Indicators Z (K x n), meter weights W = Z - c|C|/n, and the (demeaned) flows A = P Z', B = Q Z'."""
        Z = indicators(family, self.n)
        W = Z-self.t.c*Z.mean(1, keepdims=True) if self.t.c else Z
        return Z, W, self.Pt@Z.T, self.Qt@Z.T

    def columns(self, family):
        Z, W, A, B = self.blocks(family)
        K = len(family)
        Fr = (A[:, None, :]*W.T[None, :, :]).reshape(self.N, K)
        Fx = (B[:, None, :]*W.T[None, :, :]).reshape(self.N, K)
        return np.hstack((Fr, Fx))/self.scale

    def gram(self, family, target=None):
        """G = F_s'F_s and b = F_s'target (target T x n, scaled; default y_s) from the Kronecker structure."""
        Z, W, A, B = self.blocks(family)
        Wg = W@W.T
        G = np.block([[(A.T@A)*Wg, (A.T@B)*Wg], [(B.T@A)*Wg, (B.T@B)*Wg]])/self.scale**2
        U = (self.Yt if target is None else target)@W.T
        b = np.r_[(A*U).sum(0), (B*U).sum(0)]/self.scale
        return G, b

    def residual(self, family, theta):
        """Scaled transformed residual y_s - F_s theta as T x n."""
        return self.Yt-(self.columns(family)@theta).reshape(self.T, self.n)

    def pair_columns(self):
        """Columns of every symmetric entry (i <= j) of R and X: meter i gets p_j, meter j gets p_i (under M)."""
        ii, jj = np.triu_indices(self.n)
        m = len(ii)
        out = []
        for flows in (self.Pt, self.Qt):
            F = np.zeros((self.T, self.n, m))
            F[:, jj, np.arange(m)] = flows[:, ii]
            F[:, ii, np.arange(m)] = flows[:, jj]
            out.append(self.t.slot(F).reshape(self.N, m))
        return np.hstack(out)/self.scale, (ii, jj)


def bounded_ls(A, c, bound):
    if A.shape[1] == 0:
        return np.zeros(0)
    theta, _ = nnls(A, c, maxiter=50*A.shape[1])
    if theta.size and theta.max() > bound:
        theta = np.clip(lsq_linear(A, c, bounds=(0., bound), method="bvls", tol=1e-12, max_iter=5000).x, 0., bound)
    return theta


def exact_fit(design, family, bound):
    """Bounded LS of one family on explicit transformed columns (QR-compressed; reference for tests and replay)."""
    family = canonical(family)
    F = design.columns(family)
    Qm, Rm = np.linalg.qr(F)
    qy = Qm.T@design.y
    const = float(np.sum((design.y-Qm@qy)**2))
    theta = bounded_ls(Rm, qy, bound)
    sse = float(np.sum((Rm@theta-qy)**2))+const
    return dict(family=family, theta=theta, objective=sse/design.N)


class GramFitter:
    """Exact bounded LS of any sub-family of a universe that contains `keep`, from Gram statistics.

    Objective values are evaluated around the keep-only fit theta0 (residual e0, computed explicitly), so that no
    large terms cancel: ||F theta - y||^2 = d'Gd - 2 d'F'e0 + ||e0||^2, d = theta - theta0.
    """

    def __init__(self, design, universe, keep, bound):
        self.design, self.n, self.bound = design, design.n, float(bound)
        self.keep = canonical(keep)
        self.universe = canonical([*universe, *self.keep])
        self.index = {C: i for i, C in enumerate(self.universe)}
        self.K = len(self.universe)
        self.G, self.b = design.gram(self.universe)
        base = exact_fit(design, self.keep, bound)
        cols = self._cols(self.keep)
        self.theta0 = np.zeros(2*self.K)
        self.theta0[cols] = base["theta"]
        e0 = design.residual(self.keep, base["theta"])
        _, self.b0 = design.gram(self.universe, target=e0)
        self.r0 = float(np.sum(e0**2))
        self.N = design.N
        self.cache, self.evaluations = {}, 0

    def _cols(self, family):
        idx = [self.index[C] for C in family]
        return idx+[i+self.K for i in idx]

    def fit(self, family):
        family = canonical(family)
        hit = self.cache.get(family)
        if hit is not None:
            return hit
        cols = self._cols(family)
        G = self.G[np.ix_(cols, cols)]
        w, V = np.linalg.eigh(G)
        pos = w > max(w[-1], 1e-300)*1e-13
        A = (V[:, pos]*np.sqrt(w[pos])).T
        c = (V[:, pos].T@self.b[cols])/np.sqrt(w[pos])
        theta = bounded_ls(A, c, self.bound)
        d = theta-self.theta0[cols]
        objective = (float(d@G@d)-2.*float(d@self.b0[cols])+self.r0)/self.N
        self.cache[family] = (objective, theta)
        self.evaluations += 1
        return objective, theta


def internal_count(family, n):
    return sum(1 < len(C) < n for C in family)


# ------------------------------------------------------------------ R0: P-only ordered sensitivity fit under M
def ordered_sensitivity(design, solver="slsqp"):
    """Symmetric nonnegative R with entrywise diagonal order, fitted to P only under the residual operator M.

    solver="gurobi" solves the same QP with Gurobi's barrier (trees.ordered_qp_gurobi); the default is the frozen
    SLSQP path.

    Same ordered convex QP as trees.fit_sensitivity, restricted to the P block (the identifiable combination when
    the power factor is fixed; WZZT_REVISION finding 7). With a free source term (c = 1) R is identified up to
    t 11', which leaves RNJ's ordering unchanged; it is normalised to a zero smallest off-diagonal entry.
    """
    T, n = design.T, design.n
    ii, jj = np.triu_indices(n)
    k = len(ii)
    F = np.zeros((T, n, k))
    F[:, jj, np.arange(k)] = design.Pt[:, ii]
    F[:, ii, np.arange(k)] = design.Pt[:, jj]
    feature = design.t.slot(F).reshape(T*n, k)
    target = design.raw_target.ravel()
    norm = float(np.linalg.norm(target)) or 1.
    ols = np.linalg.lstsq(feature, target, rcond=None)[0]
    R0 = np.zeros((n, n)); R0[ii, jj] = ols; R0[jj, ii] = ols
    R0 = np.maximum(R0, 0.)
    positive = np.diag(R0)[np.diag(R0) > 0.]
    margin = 1e-6*(float(np.median(positive)) if positive.size else max(float(R0.max()), 1e-12))
    scale = norm/(float(np.linalg.norm(feature)) or 1.)
    scale = max(scale, margin)
    feature_s = feature*scale/norm
    target_s = target/norm
    position = np.empty((n, n), int); position[ii, jj] = position[jj, ii] = np.arange(k)
    rows, lower = list(np.eye(k)), [0.]*k
    for i in range(n):
        for j in range(n):
            if i != j:
                row = np.zeros(k); row[position[i, i]] = 1.; row[position[i, j]] = -1.
                rows.append(row); lower.append(margin/scale)
    constraint, lower = np.asarray(rows), np.asarray(lower)
    if solver == "gurobi":
        values, qp = trees.ordered_qp_gurobi(feature_s, target_s, constraint, lower)
        info = dict(success=True, iterations=qp["iterations"], message=f"Gurobi status {qp['status']}", solver="gurobi")
    else:
        start_matrix = R0.copy()
        if n > 1:
            off = start_matrix.copy(); np.fill_diagonal(off, -np.inf)
            np.fill_diagonal(start_matrix, np.maximum(np.diag(start_matrix), off.max(axis=1)+margin))
        start = start_matrix[ii, jj]/scale
        hessian = feature_s.T@feature_s; hessian = .5*(hessian+hessian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
        shift = np.zeros(k)
        if eigenvalues[-1] > 0. and eigenvalues[0] > 1e-12*eigenvalues[-1]:
            upper = np.linalg.cholesky(hessian).T
            transform = solve_triangular(upper, np.eye(k), lower=False)
            shift = -transform.T@(feature_s.T@target_s); start = upper@start+shift
            objective = lambda u: (.5*float(u@u), u)
        else:
            roots = np.sqrt(np.maximum(eigenvalues, 1e-6*eigenvalues[-1])) if eigenvalues[-1] > 0. else np.ones(k)
            transform = eigenvectors/roots; start = roots*(eigenvectors.T@start)
            tf = feature_s@transform

            def objective(u):
                res = tf@u-target_s
                return .5*float(res@res), tf.T@res
        c = constraint@transform; d = lower+c@shift; norms = np.linalg.norm(c, axis=1)
        result = minimize(objective, start, jac=True, method="SLSQP",
                          constraints=LinearConstraint(c/norms[:, None], d/norms, np.inf),
                          options={"maxiter": 500, "ftol": 1e-12})
        values = transform@(result.x-shift)
        info = dict(success=bool(result.success), iterations=int(result.nit), message=str(result.message))
    violation = max(0., float(np.max(lower-constraint@values)))
    if not np.isfinite(values).all() or violation > 1e-6:
        raise RuntimeError(f"Ordered P-only sensitivity QP failed: {info['message']}; violation={violation:g}")
    R = np.zeros((n, n)); R[ii, jj] = values*scale; R[jj, ii] = R[ii, jj]
    R = np.maximum(R, 0.)
    if design.t.c == 1. and n > 1:
        off = R[~np.eye(n, dtype=bool)]
        R = R-off.min()
    if n > 1:
        off = R.copy(); np.fill_diagonal(off, -np.inf)
        np.fill_diagonal(R, np.maximum(np.diag(R), off.max(axis=1)+margin))
    return R, info


def guidance_ponly(design, tolerances=(0., .04, .08, .16), reference=.16, solver="slsqp"):
    R, info = ordered_sensitivity(design, solver)
    shared, depth, _ = trees.geometry(R, R, 1.)
    families = [trees.rnj(shared, depth, t) for t in tolerances]
    return dict(pool=canonical(C for f in families for C in f), families=tuple(families),
                reference=trees.rnj(shared, depth, reference), R=R, info=info)


def guidance_pq(series, transform, solver="slsqp"):
    """The original guidance (trees.rnj_guidance) on time-demeaned data when offsets are free."""
    from rnj_wzzt_dr.physical.backend import Observation
    d = (lambda a: a-a.mean(0)) if transform.offsets else (lambda a: a)
    g = trees.rnj_guidance(Observation(d(series.p), d(series.q), d(series.y), tuple(range(series.n))), solver)
    return dict(pool=canonical(g["pool"]), families=tuple(g["pool_families"]), reference=g["reference_family"],
                R=g["R"], info=g["diagnostics"])


# ------------------------------------------------------------------ noise level and identifiable parameters
def saturated_sigma2(design):
    """Residual variance of the saturated symmetric linear model (every R_ij, X_ij) under M: the noise scale."""
    F, _ = design.pair_columns()
    F = F*design.scale
    target = design.raw_target.ravel()
    theta, _, rank, _ = np.linalg.lstsq(F, target, rcond=1e-10)
    sse = float(np.sum((target-F@theta)**2))
    dof = design.t.rank-int(rank)
    if dof <= 0:
        raise ValueError("Too few rows for the saturated noise estimate")
    return sse/dof, dict(rank=int(rank), dof=int(dof), sse=sse)


def identifiable_per_support(series, pq_sigma, k=2.):
    """1 when every meter's Q moves with its P within the meter class (x unidentifiable), else 2 (majority vote).

    Per meter: the part of demeaned q not explained by demeaned p, against k sqrt(2) pq_sigma times the level of q
    (the rule of the hourly-probing study's identifiable_p)."""
    if pq_sigma is None:
        return 2, dict(rule="two per support (no meter class given)")
    p, q = series.p, series.q
    pd, qd = p-p.mean(0), q-q.mean(0)
    votes = []
    for i in range(series.n):
        level = float(np.linalg.norm(q[:, i]))
        coef = float(pd[:, i]@qd[:, i])/max(float(pd[:, i]@pd[:, i]), 1e-300)
        resid = float(np.linalg.norm(qd[:, i]-coef*pd[:, i]))
        votes.append(level > 0 and resid > k*math.sqrt(2)*pq_sigma*level)
    count = 2 if sum(votes) > series.n/2 else 1
    return count, dict(rule="majority of meters", identifiable_x=int(sum(votes)), meters=series.n)


# ------------------------------------------------------------------ W generators (on the transformed residual)
def _valid(family, n):
    fam = set(canonical(family))
    return lambda C: 1 < len(C) < n and C not in fam and all(compatible(C, D) for D in fam)


def gen_pool(family, n, pool):
    ok = _valid(family, n)
    return {C for C in canonical(pool) if ok(C)}


def gen_sibling_pairs(family, n):
    out = set()
    for node, blocks in children(family, n).items():
        if len(blocks) >= 3:
            for a, b in combinations(blocks, 2):
                out.add(tuple(sorted((*a, *b))))
    return out


def gen_residual_rnj(family, n, design, E, solver="slsqp"):
    """RNJ trees from a P-only ordered fit to the current residual (target replaced by E, scaled back)."""
    saved = design.raw_target
    try:
        design.raw_target = E*design.scale
        R, _ = ordered_sensitivity(design, solver)
    except Exception:
        return set()
    finally:
        design.raw_target = saved
    shared, depth, _ = trees.geometry(R, R, 1.)
    ok = _valid(family, n)
    out = set()
    for tol in (0., .04, .08, .16):
        out |= {C for C in canonical(trees.rnj(shared, depth, tol)) if ok(C)}
    return out


def gen_residual_clusters(family, n, E):
    out = set()
    for node, blocks in children(family, n).items():
        k = len(blocks)
        if k < 3:
            continue
        sums = np.column_stack([E[:, list(b)].sum(1) for b in blocks])
        if np.any(sums.std(0) <= 0):
            continue
        dist = np.clip(1.-np.corrcoef(sums, rowvar=False), 0., 2.)
        Z = linkage(dist[np.triu_indices(k, 1)], method="average")
        members = {i: [i] for i in range(k)}
        for step, (a, b, *_rest) in enumerate(Z):
            merged = members.pop(int(a))+members.pop(int(b))
            members[k+step] = merged
            if 2 <= len(merged) < k:
                out.add(tuple(sorted(i for j in merged for i in blocks[j])))
    return out


def gen_screen(family, n, design, E, top=50, max_children=20):
    """Rank every union of a node's child blocks by its one-support gain on E with old coefficients fixed.

    Transformed column of C: a~_C (x) w_C, w_C = z_C - c|C|/n 1. For C = union of blocks selected by s:
    e'f_r = s'sym(A'Eb)s - (c/n)(s'size)(s'A'e1), ||f_r||^2 = s'(A'A)s (|C| - kappa|C|^2/n), likewise for x.
    """
    c, kappa = design.t.c, design.t.kappa
    scored = []
    total = E.sum(1)
    for node, blocks in children(family, n).items():
        k = len(blocks)
        if k < 3 or k > max_children:
            continue
        Eb = np.column_stack([E[:, list(b)].sum(1) for b in blocks])
        Ab = np.column_stack([design.Pt[:, list(b)].sum(1) for b in blocks])/design.scale
        Bb = np.column_stack([design.Qt[:, list(b)].sum(1) for b in blocks])/design.scale
        size = np.array([len(b) for b in blocks], float)
        sym = lambda A: (A+A.T)/2
        codes = np.arange(1, 2**k-1, dtype=np.int64)
        S = ((codes[:, None] >> np.arange(k)) & 1).astype(float)
        S = S[S.sum(1) >= 2]
        csize = S@size
        quad = lambda M: np.einsum("ij,jk,ik->i", S, M, S)
        wn = csize-kappa*csize**2/n
        gr = quad(sym(Ab.T@Eb))-(c/n)*csize*(S@(Ab.T@total))
        gx = quad(sym(Bb.T@Eb))-(c/n)*csize*(S@(Bb.T@total))
        hrr, hxx, hrx = wn*quad(sym(Ab.T@Ab)), wn*quad(sym(Bb.T@Bb)), wn*quad(sym(Ab.T@Bb))
        det = hrr*hxx-hrx**2
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (hxx*gr-hrx*gx)/det
            b = (hrr*gx-hrx*gr)/det
            both = np.where((a >= 0) & (b >= 0) & (det > 1e-12*np.maximum(hrr*hxx, 1e-300)), a*gr+b*gx, -np.inf)
            only_r = np.where(gr > 0, gr**2/np.maximum(hrr, 1e-300), 0.)
            only_x = np.where(gx > 0, gx**2/np.maximum(hxx, 1e-300), 0.)
        gain = np.maximum(both, np.maximum(only_r, only_x))
        for i in np.argsort(-gain)[:top]:
            scored.append((float(gain[i]), tuple(sorted(j for t in range(k) if S[i, t] for j in blocks[t]))))
    scored.sort(reverse=True)
    out = []
    for _, C in scored:
        if C not in out:
            out.append(C)
        if len(out) >= top:
            break
    return set(out)


# ------------------------------------------------------------------ Gurobi models (SVD rows, NumericFocus)
def _env(gp):
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def _params(m, seconds, seed, numeric_focus, gap, abs_gap):
    for key, value in dict(Threads=1, OutputFlag=0, NonConvex=0, Seed=seed, TimeLimit=max(.01, seconds),
                           MIPGap=gap, MIPGapAbs=abs_gap, FeasibilityTol=1e-9, IntFeasTol=1e-9,
                           OptimalityTol=1e-9).items():
        m.setParam(key, value)
    if numeric_focus is not None:
        m.setParam("NumericFocus", numeric_focus)


def _svd_rows(F, y, rtol=1e-10):
    U, s, Vt = np.linalg.svd(F, full_matrices=False)
    if not len(s) or s[0] <= 0.:
        return np.zeros((0, F.shape[1])), np.zeros(0), float(y@y), 0
    k = int((s > s[0]*rtol).sum())
    uy = U[:, :k].T@y
    return s[:k, None]*Vt[:k], uy, float(np.sum((y-U[:, :k]@uy)**2)), k


def response_bounds(B, uy, const, family, n, N, cutoff, bound, seconds=60., margin=1.05):
    """Optimality-based bound tightening for the laminar selection MIQP.

    For every meter i, the largest diagonal responses R_ii = sum_{C contains i} r_C and X_ii of any coefficient
    vector in the box whose residual objective is at most `cutoff`: one continuous convex QCQP per meter and block
    (binaries and laminarity dropped, so this is a relaxation). The support penalties are nonnegative, so every
    family whose MIQP objective is at most the cutoff satisfies R_ii <= Rmax_i, and hence r_C <= min_{i in C}
    Rmax_i. Returns (Rmax, Xmax, info); the bounds carry a 5% safety margin; None if the cutoff is infeasible."""
    gp = _gurobi()
    K = len(family)
    member = indicators(family, n)
    rhs = N*cutoff-const
    started = time.perf_counter()
    if rhs <= 0.:
        return None
    # trivial valid bound: every coefficient is at most `bound`, and R_ii sums the supports that contain i
    out = np.tile(bound*member.sum(0), (2, 1)).astype(float)
    trivial = out.copy()
    status = []
    env = _env(gp)
    try:
        with gp.Model("WZZT_R_response_bounds", env=env) as m:
            for key, value in dict(Threads=1, OutputFlag=0, TimeLimit=max(.01, seconds), NumericFocus=2).items():
                m.setParam(key, value)
            th = m.addMVar(2*K, lb=0., ub=bound)
            res = m.addMVar(B.shape[0], lb=-float("inf"))
            m.addConstr(res == B@th-uy)
            m.addConstr(res@res <= rhs)
            for block in (0, 1):
                for i in range(n):
                    c = np.zeros(2*K)
                    c[block*K+np.flatnonzero(member[:, i])] = 1.
                    m.setObjective(c@th, gp.GRB.MAXIMIZE)
                    m.optimize()
                    status.append(int(m.Status))
                    if m.Status == gp.GRB.OPTIMAL:
                        out[block, i] = min(trivial[block, i], float(m.ObjVal)*margin+1e-9*bound)
    finally:
        env.dispose()
    return out[0], out[1], dict(seconds=time.perf_counter()-started, statuses=sorted(set(status)),
                                tightened=int(np.sum(out < trivial)))


def select_miqp(design, universe, keep, alpha, bound, seconds, *, warm=None, seed=20261111, numeric_focus=2,
                gap=1e-6, abs_gap=1e-7, cutoff=None, solver_cutoff=False):
    """E: choose a laminar subset of a finite pool (exact model; SVD-compressed transformed rows).

    With `cutoff` (an objective value known to be attained, e.g. the search result), big-M constants and
    coefficient bounds are tightened by `response_bounds` and the per-meter sums R_ii, X_ii are bounded: valid
    for every family at least as good as the cutoff, so the optimum is unchanged."""
    gp = _gurobi()
    started = time.perf_counter()
    n = design.n
    family = canonical([*universe, *keep, *basis(n)])
    fixed = set(canonical([*keep, *basis(n)]))
    proper = [C for C in family if 1 < len(C) < n]
    K = len(family)
    B, uy, const, rank = _svd_rows(design.columns(family), design.y)
    Mr, Mx, tight = np.full(K, float(bound)), np.full(K, float(bound)), None
    if cutoff is not None:
        rb = response_bounds(B, uy, const, family, n, design.N, cutoff*(1+1e-9)+1e-12, bound,
                             seconds=max(1., .25*seconds))
        if rb is not None:
            Rmax, Xmax, tight = rb
            Mr = np.array([min(float(bound), min(Rmax[list(C)])) for C in family])
            Mx = np.array([min(float(bound), min(Xmax[list(C)])) for C in family])
    env = _env(gp)
    try:
        with gp.Model("WZZT_R_selection", env=env) as m:
            _params(m, max(.01, seconds-(time.perf_counter()-started)), seed, numeric_focus, gap, abs_gap)
            if solver_cutoff and cutoff is not None:
                m.setParam("Cutoff", cutoff+1e-6*(1+abs(cutoff)))
            r = m.addMVar(K, lb=0., ub=Mr); x = m.addMVar(K, lb=0., ub=Mx)
            a = m.addMVar(len(proper), vtype="B")
            pos = {C: i for i, C in enumerate(family)}
            sel = {C: i for i, C in enumerate(proper)}
            for C, j in sel.items():
                m.addConstr(r[pos[C]] <= Mr[pos[C]]*a[j]); m.addConstr(x[pos[C]] <= Mx[pos[C]]*a[j])
                if C in fixed:
                    a[j].LB = 1.
            if tight is not None:
                member = indicators(family, n)
                for i in range(n):
                    rows = np.flatnonzero(member[:, i])
                    m.addConstr(r[rows].sum() <= Rmax[i]); m.addConstr(x[rows].sum() <= Xmax[i])
            for C, D in combinations(proper, 2):
                if not compatible(C, D):
                    m.addConstr(a[sel[C]]+a[sel[D]] <= 1.)
            for Wt in [*proper, tuple(range(n))]:
                inside = [sel[C] for C in proper if set(C) < set(Wt)]
                if inside:
                    m.addConstr(a[inside].sum() <= len(Wt)-2)
            res = m.addMVar(rank, lb=-float("inf"))
            m.addConstr(res == B[:, :K]@r+B[:, K:]@x-uy)
            m.setObjective((res@res+const)/design.N+alpha*a.sum())
            if warm is not None:
                wf = canonical([*warm, *fixed])
                fit = exact_fit(design, wf, bound)
                th = dict(zip(wf, zip(fit["theta"][:len(wf)], fit["theta"][len(wf):])))
                r0 = np.array([th.get(C, (0., 0.))[0] for C in family])
                x0 = np.array([th.get(C, (0., 0.))[1] for C in family])
                r.Start, x.Start = r0, x0
                a.Start = [float(C in th) for C in proper]
                res.Start = B@np.r_[r0, x0]-uy
            m.optimize()
            feasible = m.SolCount > 0
            try:
                objective_bound = _finite(m.ObjBound)
            except Exception:
                objective_bound = None
            out = dict(status=int(m.Status), objective=_finite(m.ObjVal) if feasible else None,
                       bound=objective_bound, gap=_finite(m.MIPGap) if feasible else None,
                       nodes=float(m.NodeCount), rank=rank, binaries=len(proper), seed=seed,
                       warm=warm is not None, seconds=time.perf_counter()-started, tightening=tight,
                       cutoff=cutoff)
            if feasible:
                out["family"] = canonical([*fixed, *(C for C in proper if a.X[sel[C]] > .5)])
    finally:
        env.dispose()
    return out


def extend_miqp(design, family, alpha, bound, seconds, *, start_support=None, seed=20261111, numeric_focus=2,
                gap=1e-4, abs_gap=1e-7):
    """W: the best single compatible support over every terminal set (original WZZT extend, transformed rows).

    z selects terminals, g = z z' and u = r_new g are linearised exactly; the prediction of the new support for
    meter i is sum_j u_ij p_j, so its columns are the symmetric pair columns under M (SVD-compressed with the old
    family's columns). The old supports stay charged; the new one costs alpha when active."""
    gp = _gurobi()
    started = time.perf_counter()
    n, family = design.n, canonical(family)
    proper = [C for C in family if 1 < len(C) < n]
    K = len(family)
    Fp, (ii, jj) = design.pair_columns()
    mpairs = len(ii)
    B, uy, const, rank = _svd_rows(np.hstack((design.columns(family), Fp)), design.y)
    flat = ii*n+jj
    env = _env(gp)
    try:
        with gp.Model("WZZT_R_unknown_support", env=env) as m:
            _params(m, seconds, seed, numeric_focus, gap, abs_gap)
            r = m.addMVar(K, lb=0., ub=bound); x = m.addMVar(K, lb=0., ub=bound)
            z = m.addMVar(n, vtype="B"); active = m.addVar(vtype="B")
            nr, nx = m.addVar(lb=0., ub=bound), m.addVar(lb=0., ub=bound)
            m.addConstr(nr <= bound*active); m.addConstr(nx <= bound*active)
            m.addConstr(z.sum() >= 2); m.addConstr(z.sum() <= n-1)
            relations = []
            for C in proper:
                inside, outside = list(C), [i for i in range(n) if i not in C]
                rel = m.addMVar(3, vtype="B")
                relations.append((C, rel))
                m.addConstr(rel.sum() == 1)
                m.addConstr(z[inside].sum() <= len(inside)*(1-rel[0]))
                m.addConstr(z[outside].sum() <= len(outside)*(1-rel[1]))
                m.addConstr(z[inside].sum() >= len(inside)*rel[2])
                m.addConstr(len(C)-z[inside].sum()+z[outside].sum() >= 1)
            g = m.addMVar((n, n), lb=0., ub=1.)
            ur = m.addMVar((n, n), lb=0., ub=bound); ux = m.addMVar((n, n), lb=0., ub=bound)
            m.addConstr(g <= z[:, None]); m.addConstr(g <= z[None, :])
            m.addConstr(g >= z[:, None]+z[None, :]-1); m.addConstr(g == g.T)
            for weights, scalar in ((ur, nr), (ux, nx)):
                m.addConstr(weights <= scalar); m.addConstr(weights <= bound*g)
                m.addConstr(weights >= scalar-bound*(1-g)); m.addConstr(weights == weights.T)
            urf, uxf = ur.reshape(-1)[flat], ux.reshape(-1)[flat]
            res = m.addMVar(rank, lb=-float("inf"))
            m.addConstr(res == B[:, :K]@r+B[:, K:2*K]@x+B[:, 2*K:2*K+mpairs]@urf+B[:, 2*K+mpairs:]@uxf-uy)
            m.setObjective((res@res+const)/design.N+alpha*(len(proper)+active))
            fit = exact_fit(design, family, bound)
            r0, x0 = fit["theta"][:K], fit["theta"][K:]
            r.Start, x.Start = r0, x0
            active.Start, nr.Start, nx.Start = 0., 0., 0.
            ur.Start, ux.Start = np.zeros((n, n)), np.zeros((n, n))
            res.Start = B@np.r_[r0, x0, np.zeros(2*mpairs)]-uy
            if start_support is not None:
                zz = np.zeros(n); zz[list(start_support)] = 1.
                z.Start, g.Start = zz, np.outer(zz, zz)
                for C, rel in relations:
                    which = 0 if not set(C) & set(start_support) else 1 if set(start_support) < set(C) else 2
                    rel.Start = np.eye(3)[which]
            m.optimize()
            feasible = m.SolCount > 0
            out = dict(status=int(m.Status), objective=_finite(m.ObjVal) if feasible else None,
                       bound=_finite(m.ObjBound), gap=_finite(m.MIPGap) if feasible else None,
                       nodes=float(m.NodeCount), rank=rank, seed=seed, seconds=time.perf_counter()-started,
                       start_objective=fit["objective"]+alpha*len(proper))
            if feasible and active.X > .5:
                C = tuple(int(i) for i in np.flatnonzero(z.X > .5))
                if C in family or not 2 <= len(C) < n or not all(compatible(C, D) for D in family):
                    raise RuntimeError("Invalid decoded unknown support")
                out["support"] = C
    finally:
        env.dispose()
    return out


# ------------------------------------------------------------------ E heuristics
class Problem:
    def __init__(self, fitter, pool, keep, alpha):
        self.f, self.n, self.alpha = fitter, fitter.n, alpha
        self.keep = canonical(keep)
        self.proper = [C for C in canonical(pool) if 1 < len(C) < self.n and C not in self.keep]
        self.crosses = {C: frozenset(D for D in self.proper if not compatible(C, D)) for C in self.proper}

    def score(self, S):
        return self.f.fit([*self.keep, *S])[0]+self.alpha*len(S)

    def addable(self, S):
        return [C for C in self.proper if C not in S and not (self.crosses[C] & S)]


def local_search(prob, start, deadline):
    S = frozenset(C for C in start if C in prob.crosses)
    best = prob.score(S)
    while time.perf_counter() < deadline:
        move, move_score = None, best-1e-12
        for C in prob.proper:
            T = S-{C} if C in S else (S-prob.crosses[C]) | {C}
            s = prob.score(T)
            if s < move_score:
                move, move_score = T, s
            if time.perf_counter() > deadline:
                break
        if move is None:
            break
        S, best = move, move_score
    return S, best


def beam(prob, width, deadline):
    level = {frozenset()}
    best_S, best = frozenset(), prob.score(frozenset())
    stale = 0
    while level and time.perf_counter() < deadline:
        expanded = {}
        for S in level:
            for C in prob.addable(S):
                T = S | {C}
                if T not in expanded:
                    expanded[T] = prob.score(T)
                if time.perf_counter() > deadline:
                    break
        if not expanded:
            break
        level = set(sorted(expanded, key=expanded.get)[:width])
        top = min(expanded.values())
        if top < best-1e-12:
            best_S, best = min(expanded, key=expanded.get), top; stale = 0
        else:
            stale += 1
            if stale >= (1 if width == 1 else 2):
                break
    return best_S, best


def random_laminar(prob, rng):
    S = set()
    for i in rng.permutation(len(prob.proper)):
        C = prob.proper[i]
        if rng.random() < .5 and not (prob.crosses[C] & S):
            S.add(C)
    return frozenset(S)


def portfolio(prob, starts, deadline, *, width=20, random_starts=20, seed=0):
    """Beam search, local search from each given start (W result, RNJ trees), local search from random starts."""
    found = {}
    budget = lambda share: min(deadline, time.perf_counter()+share*max(0., deadline-time.perf_counter()))
    S, s = beam(prob, width, budget(.34)); found["beam"] = (S, s)
    best_S, best = S, s
    for name, st in starts:
        if time.perf_counter() > deadline:
            break
        S, s = local_search(prob, st, budget(.5))
        found[f"local_{name}"] = (S, s)
        if s < best:
            best_S, best = S, s
    rng = np.random.default_rng(seed)
    for r in range(random_starts):
        if time.perf_counter() > deadline:
            break
        S, s = local_search(prob, random_laminar(prob, rng), deadline)
        if s < found.get("random", (None, np.inf))[1]:
            found["random"] = (S, s)
        if s < best:
            best_S, best = S, s
    return best_S, best, {k: float(v[1]) for k, v in found.items()}


# ------------------------------------------------------------------ the revised search
def _frontier(family, n, block_limit, cap):
    try:
        cand = trees.frontier_candidates(family, n, block_limit=block_limit)["candidates"]
    except Exception:
        return ()
    cand = [C for C in cand if 1 < len(C) < n]
    if len(cand) > cap:
        cand = sorted(cand, key=lambda C: (len(C), C))[:cap]
    return tuple(cand)


def fit(train, validation=None, config=None):
    """Run WZZT-R. train/validation are Series (or objects with p, q, y and optional v0sq).

    Returns the selected family, the W path, the E result, alternatives within the BIC margin (all as canonical
    tuples including singletons and the full set), per-stage receipts and timings."""
    config = config or RevisedConfig()
    config.validate()
    t_start = time.perf_counter()
    deadline = t_start+config.total_seconds
    as_series = lambda o: o if isinstance(o, Series) else Series(o.p, o.q, o.y, getattr(o, "v0sq", None))
    train = as_series(train)
    data = train.join(as_series(validation)) if (config.history == "all" and validation is not None) else train
    T, n = data.p.shape
    N = T*n
    if config.source == "gls":
        if data.v0sq is None:
            raise ValueError("source='gls' needs the root reading v0sq")
        c = root_gamma(data.v0sq, data.y)
    elif config.source == "estimated":
        c, _ = estimated_gamma(data, config.offsets)
    else:
        c = 1. if config.source == "free" else 0.
    transform = Transform(T, n, c, config.offsets)
    design = Design(data, transform, 1.)
    timing = {}
    t0 = time.perf_counter()
    guide = (guidance_ponly(design, solver=config.qp_solver) if config.guidance == "ponly"
             else guidance_pq(data, transform, config.qp_solver))
    timing["guidance"] = time.perf_counter()-t0
    t0 = time.perf_counter()
    p_s, p_info = identifiable_per_support(data, config.pq_sigma)
    if config.score == "bic":
        sigma2, s_info = saturated_sigma2(design)
        alpha = p_s*math.log(N)/N
    else:
        ref = exact_fit(design, canonical([*guide["reference"], *basis(n)]), config.bound)
        sigma2, s_info = ref["objective"], dict(rule="reference-tree MSE")
        alpha = config.alpha
    design.rescale(math.sqrt(sigma2))
    timing["scale"] = time.perf_counter()-t0
    bic = lambda objective, k: .5*N*(objective+alpha*k)   # BIC units (J), up to a constant
    crit = lambda family, objective: objective+alpha*internal_count(family, n)
    base = canonical(basis(n))

    # ---------------------------------------------------------- W
    t0 = time.perf_counter()
    w_deadline = t_start+.5*config.total_seconds
    current = base
    cur_obj = exact_fit(design, current, config.bound)["objective"]
    path, w_events, visited = [current], [], {current}
    while time.perf_counter() < w_deadline:
        theta = exact_fit(design, current, config.bound)["theta"]
        E = design.residual(current, theta)
        cands = gen_pool(current, n, guide["pool"])
        sources = {C: "pool" for C in cands}
        if config.w == "generator":
            valid = _valid(current, n)
            for name, gen in (("pairs", lambda: gen_sibling_pairs(current, n)),
                              ("residual_rnj", lambda: gen_residual_rnj(current, n, design, E, config.qp_solver)),
                              ("clusters", lambda: gen_residual_clusters(current, n, E)),
                              ("screen", lambda: gen_screen(current, n, design, E))):
                for C in gen():
                    if valid(C):
                        sources.setdefault(C, name); cands.add(C)
        event = dict(step=len(w_events), candidates=len(cands), sources={})
        best_family, best_score, best_C = current, crit(current, cur_obj), None
        if cands:
            stepper = GramFitter(design, [*current, *cands], current, config.bound)
            for C in cands:
                fam = canonical([*current, C])
                obj = stepper.fit(fam)[0]
                s = crit(fam, obj)
                if s < best_score-config.gain_tol:
                    best_family, best_score, best_C = fam, s, C
        event["generator_best"] = dict(support=best_C, source=sources.get(best_C), score=best_score)
        miqp_needed = (config.w == "miqp") or (config.w == "generator" and config.w_check and best_C is None)
        extendable = any(len(b) >= 3 for b in children(current, n).values())
        if miqp_needed and extendable:
            remaining = min(config.w_mip_seconds, w_deadline-time.perf_counter())
            if remaining > 1.:
                start = best_C if best_C is not None else (next(iter(cands)) if cands else None)
                rec = extend_miqp(design, current, alpha, config.bound, remaining, start_support=start,
                                  seed=config.seed)
                event["miqp"] = {k: v for k, v in rec.items() if k != "support"}
                if "support" in rec:
                    fam = canonical([*current, rec["support"]])
                    obj = exact_fit(design, fam, config.bound)["objective"]
                    event["miqp"].update(support=rec["support"], refit_score=crit(fam, obj))
                    if crit(fam, obj) < best_score-config.gain_tol:
                        best_family, best_score, best_C = fam, crit(fam, obj), rec["support"]
                        event["accepted_from"] = "miqp"
        if best_C is None:
            event["stop"] = "no_gain"
            w_events.append(event)
            break
        # zero cleanup: drop internal supports whose coefficients vanished, if the score does not get worse
        fitb = exact_fit(design, best_family, config.bound)
        k = len(best_family)
        zero = [C for i, C in enumerate(best_family) if 1 < len(C) < n
                and max(fitb["theta"][i], fitb["theta"][k+i]) <= config.zero_tol]
        if zero:
            cleaned = canonical(C for C in best_family if C not in zero)
            obj = exact_fit(design, cleaned, config.bound)["objective"]
            if crit(cleaned, obj) <= best_score+config.gain_tol:
                best_family, best_score = cleaned, crit(cleaned, obj)
                event["zero_cleanup"] = [list(C) for C in zero]
        if best_family in visited:
            event["stop"] = "repeated_family"
            w_events.append(event)
            break
        event.setdefault("accepted_from", "generator")
        event.update(added=list(best_C), score=best_score)
        w_events.append(event)
        current = best_family
        cur_obj = exact_fit(design, current, config.bound)["objective"]
        path.append(current); visited.add(current)
    timing["W"] = time.perf_counter()-t0

    # ---------------------------------------------------------- E
    t0 = time.perf_counter()
    frontier = _frontier(current, n, config.block_limit, config.max_frontier)
    universe = canonical([*guide["pool"], *(C for f in path for C in f), *frontier,
                          *combinations(range(n), 2), *base])
    efit = GramFitter(design, universe, base, config.bound)
    prob = Problem(efit, universe, base, alpha)
    internal = lambda fam: frozenset(C for C in fam if 1 < len(C) < n)
    e_record = dict(pool=len(prob.proper), frontier=len(frontier))
    w_final = internal(current)
    best_S, best = w_final, prob.score(w_final)
    if config.e in ("certified", "heuristic"):
        starts = [("W", w_final)]+[(f"rnj{i}", internal(f)) for i, f in enumerate(guide["families"])]
        e_deadline = min(deadline, time.perf_counter()+config.e_search_seconds)
        S, s, members = portfolio(prob, starts, e_deadline, width=config.beam_width,
                                  random_starts=config.random_starts, seed=config.seed)
        e_record["portfolio"] = members
        if s < best:
            best_S, best = S, s
    e_record["search_seconds"] = time.perf_counter()-t0
    certs = []
    verdict = None
    if config.e in ("certified", "miqp"):
        # e="miqp": the MIQP is the solver (warm at the W result). The certificate (Phase 2 deviation D1) is one
        # cold solve with another seed, tightened bounds and the best known value as cutoff: it must rediscover a
        # family at least that good and prove it. A false bound is then detectable (it would exclude a family we
        # hold), which a warm-started solve cannot show.
        # The certificate's verdict is in the units of its relative gap g: confirmed when the proven bound is
        # within g of the held value h; contradicted (a detected false bound) when the bound exceeds h, or the
        # solver reports nothing at or below the cutoff h + g|h| although h is attained. A contradiction is
        # escalated once with NumericFocus=3.
        runs = [dict(warm=w_final, seed=config.seed, role="solver", focus=2)] if config.e == "miqp" else []
        if config.confirm:
            runs.append(dict(warm=None, seed=7, role="certificate", focus=2))
        g = config.cert_gap
        while runs:
            run = runs.pop(0)
            remaining = min(config.e_cert_seconds, deadline-time.perf_counter())
            if remaining <= 1.:
                certs.append(dict(role=run["role"], skipped="budget")); continue
            held = best
            cut = held+g*abs(held)+1e-6*(1+abs(held)) if run["role"] != "solver" else None
            rec = select_miqp(design, universe, base, alpha, config.bound, remaining,
                              warm=None if run["warm"] is None else canonical([*base, *run["warm"]]),
                              seed=run["seed"], gap=g, abs_gap=config.cert_abs_gap, numeric_focus=run["focus"],
                              cutoff=None if cut is None else held, solver_cutoff=cut is not None)
            if cut is not None:
                rec["solver_cutoff_value"] = cut
            rec["role"] = run["role"]
            fam = rec.pop("family", None)
            if fam is not None:
                S = internal(fam)
                rec["refit_objective"] = prob.score(S)
                rec["family"] = [list(C) for C in sorted(S)]
                if rec["refit_objective"] < best-1e-12:
                    best_S, best = S, rec["refit_objective"]
                    rec["improved_search"] = True
            certs.append(rec)
            if run["role"] == "solver":
                continue
            tol = 1e-6*(1+abs(held))
            bound_ = rec.get("bound")
            contradicted = rec["status"] in (3, 6) or (bound_ is not None and bound_ > held+tol)
            confirmed = (rec["status"] == 2 and not contradicted and bound_ is not None
                         and bound_ >= best-g*abs(best)-1e-9)
            verdict = dict(contradicted=bool(contradicted), confirmed=bool(confirmed),
                           lower=None if contradicted else bound_, escalated=run["role"] == "escalation")
            if contradicted and run["role"] == "certificate":
                runs.append(dict(warm=None, seed=7, role="escalation", focus=3))
    e_record["certificates"] = certs
    if verdict is not None:
        lower = verdict["lower"]
        e_record.update(certified=verdict["confirmed"], confirmed=verdict["confirmed"],
                        contradicted=verdict["contradicted"], escalated=verdict["escalated"],
                        contradictions=sum(1 for r in certs if r.get("role") != "solver" and (
                            r.get("status") in (3, 6))),
                        lower_bound=lower, gap=None if lower is None else _gap(best, min(lower, best)))
    selected = canonical([*base, *best_S])
    e_record["seconds"] = time.perf_counter()-t0
    timing["E"] = e_record["seconds"]

    # ---------------------------------------------------------- output: path, E result, alternatives
    scored = sorted(((obj+alpha*internal_count(fam, n), fam) for fam, (obj, _) in efit.cache.items()),
                    key=lambda v: v[0])
    alternatives = []
    for s, fam in scored:
        if bic(s, 0)-bic(best, 0) > config.alternatives_margin or len(alternatives) >= config.max_alternatives:
            break
        if fam != selected:
            alternatives.append(fam)
    final = exact_fit(design, selected, config.bound)
    return dict(selected=selected, path=path, alternatives=alternatives, w_events=w_events, e=e_record,
                guidance=dict(pool=guide["pool"], families=guide["families"], reference=guide["reference"],
                              info=guide["info"]),
                score=dict(rule=config.score, alpha=alpha, sigma2=sigma2, sigma_info=s_info, p_s=p_s,
                           p_info=p_info, N=N, c=c, offsets=config.offsets,
                           selected_objective=final["objective"],
                           selected_criterion=crit(selected, final["objective"])),
                timing=dict(timing, total=time.perf_counter()-t_start), config=asdict(config))
