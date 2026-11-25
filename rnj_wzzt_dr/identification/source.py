"""WZZT-C with one unmetered load (development; METHOD_SOURCE.md).

The unmetered power is u_t = PCC - sum of the meters - losses. With h_C = 1 for the lines on the path from the PCC to
the unmetered load, every line on the path carries u_t besides its meters:

    y_t = sum_C z_C [ r_C (z_C' p_t + h_C uP_t) + x_C (z_C' q_t + h_C uQ_t) ] + root and meter terms + error.

S1 (robust). Whatever the path is, the source adds rho uP_t + chi uQ_t with free vectors rho, chi over the meters.
Projecting uP and uQ out of the time direction of flows and target removes it, as demeaning removes the meters'
constants. Transform, Gram statistics, MIQPs and certificates of WZZT-C are unchanged.

S2 (located). The source is one more leaf, number n, of a tree over n + 1 leaves: its power is known, its voltage is
not. A line is a set of leaves; it carries h_C u exactly when it contains leaf n, and the sets that contain leaf n are
nested, so they are the path. Laminarity over n + 1 leaves is all the pool MIQP needs: it chooses the lines and the
path together, with the same coefficients r_C, x_C (the path costs no parameter).
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import combo, revised as rv
from .trees import basis, canonical


# ------------------------------------------------------------------ the unmetered power
def losses(p, q, y, v0sq):
    """Line losses per slot from the meters' own voltages, without any topology.

    The exact balance is S_PCC - sum S_i = sum_i (V0 - V_i) conj(I_i) over the loads; with V0 - V_i taken as a
    magnitude difference (small angles), that is sum_i S_i (V0 - V_i) / V_i."""
    v0 = np.sqrt(v0sq)[:, None]
    v = np.sqrt(v0sq[:, None]-y)
    rise = (v0-v)/v
    return (p*rise).sum(1), (q*rise).sum(1)


def balance(ppcc, qpcc, data):
    """Unmetered P and Q per slot: PCC minus meters minus losses."""
    lp, lq = losses(data.p, data.q, data.y, data.v0sq)
    return ppcc-data.p.sum(1)-lp, qpcc-data.q.sum(1)-lq


# ------------------------------------------------------------------ S1: the source projected out
class ProjectedTransform(rv.Transform):
    """The residual operator of WZZT-R with the source's time series also removed from every meter."""

    def __init__(self, T, n, c, offsets, series):
        super().__init__(T, n, c, offsets)
        X = np.column_stack([np.asarray(s, float) for s in series])
        X = X-X.mean(0) if offsets else X
        U, s, _ = np.linalg.svd(X/np.maximum(np.linalg.norm(X, axis=0), 1e-300), full_matrices=False)
        self.U = U[:, s > 1e-8]

    def _project(self, A):
        return A-self.U@(self.U.T@A)

    def flows(self, A):
        return self._project(super().flows(A))

    def target(self, Y):
        return self._project(super().target(Y))

    @property
    def rank(self):
        return super().rank-self.U.shape[1]*(self.n-int(self.c == 1.))


def setup_projected(train, validation, up, uq, config=None):
    """combo.setup with the source projected out; pass the result to combo.fit as S."""
    config = config or rv.RevisedConfig()
    data = train.join(validation) if validation is not None else train
    T, n = data.p.shape
    N = T*n
    c = rv.root_gamma(data.v0sq, data.y)
    design = rv.Design(data, ProjectedTransform(T, n, c, config.offsets, [up, uq]), 1.)
    R, _ = rv.ordered_sensitivity(design, "gurobi")
    p_s, _ = rv.identifiable_per_support(data, config.pq_sigma)
    sigma2, _ = rv.saturated_sigma2(design)
    alpha = p_s*math.log(N)/N
    design.rescale(math.sqrt(sigma2))
    return dict(design=design, data=data, R=R, alpha=alpha, sigma2=sigma2, p_s=p_s, c=c, N=N, n=n, T=T, config=config)


# ------------------------------------------------------------------ S2: the source as a leaf without a voltage
class LeafDesign(rv.Design):
    """Least squares over families of n + 1 leaves; leaf n is the source (known power, no voltage row)."""

    def __init__(self, series, transform, scale, up, uq):
        self.meters = series.p.shape[1]
        super().__init__(series, transform, scale)
        self.n = self.meters+1                                  # leaves, for laminarity and prices
        pa, qa = np.column_stack([series.p, up]), np.column_stack([series.q, uq])
        self.Pt, self.Qt = transform.flows(pa), transform.flows(qa)

    def blocks(self, family):
        Z = rv.indicators(family, self.meters+1)
        row = Z[:, :self.meters]
        c = self.t.c
        W = row-c*row.mean(1, keepdims=True) if c else row
        return row, W, self.Pt@Z.T, self.Qt@Z.T

    def residual(self, family, theta):
        return self.Yt-(self.columns(family)@theta).reshape(self.T, self.meters)


def with_source(lines, n):
    """Every line with and without the source leaf, plus the source below single meters."""
    out = []
    for C in lines:
        out += [tuple(C), tuple(sorted((*C, n)))]
    return canonical([*out, *((i, n) for i in range(n))])


def meters_only(family, n):
    """The lines as sets of meters, and the path (the sets that contain the source leaf), largest first."""
    path = sorted((tuple(i for i in C if i < n) for C in family if n in C and len(C) > 1), key=lambda c: -len(c))
    lines = {tuple(i for i in C if i < n) for C in family}
    return canonical(C for C in lines if C), path


def attach(family, n, D):
    """A family of meter sets as a family over n + 1 leaves, with the source below the line D (None: after the trunk)."""
    out = [tuple(range(n+1))]
    for C in canonical(family):
        if len(C) == n:
            continue
        out.append(tuple(sorted((*C, n))) if D is not None and set(D) <= set(C) else C)
    return canonical([*out, *basis(n+1)])


def locate(design, family, n, bound):
    """Fit of one tree with the source at each of its nodes. Returns [(objective, D)], best first (D=None: head)."""
    nodes = [None]+[C for C in canonical(family) if 1 < len(C) < n]
    rows = [(rv.exact_fit(design, attach(family, n, D), bound)["objective"], D) for D in nodes]
    return sorted(rows, key=lambda r: r[0])


def select_with_source(design, lines, alpha, bound, seconds, n, warm=None, seed=7):
    """The exact best tree and source path over the candidate lines (n + 1 leaves)."""
    universe = [C for C in with_source(lines, n) if 1 < len(C) < n+1]
    return rv.select_miqp(design, universe, basis(n+1), alpha, bound, seconds, warm=warm, seed=seed, gap=1e-4)


# ------------------------------------------------------------------ the method
def fit(train, validation, ppcc, qpcc, config=None, *, located=True, miqp_seconds=240., settings=None):
    """Source-aware WZZT-C. ppcc, qpcc: PCC readings over training + validation slots (pu, consumption positive).

    S1: the unmetered power from the balance is projected out and combo.fit runs unchanged (tree, certificate).
    S2: the pool MIQP over n + 1 leaves on S1's candidate lines, warm-started with the source at its best place in
    S1's tree. Returns S1's tree (`selected`), S2's tree and path (`located`, `path`), the cost in half-BIC units of
    every other place of the source (`places`), and the evidence that a source exists at all (`source_evidence`)."""
    config = config or rv.RevisedConfig()
    settings = dict(settings or {})
    started = time.perf_counter()
    data = train.join(validation)
    n, bound, T = data.n, config.bound, data.p.shape[0]
    up, uq = balance(ppcc, qpcc, data)
    S = setup_projected(train, validation, up, uq, config)
    out = combo.fit(train, validation, config, S=S, **settings)
    family = out["selected"]
    result = dict(selected=family, record=out["record"], u=(up, uq), S=S, seconds_s1=time.perf_counter()-started)
    if not located:
        return result
    transform = rv.Transform(T, n, S["c"], config.offsets)
    scale = math.sqrt(S["sigma2"])
    design = LeafDesign(data, transform, scale, up, uq)
    empty = LeafDesign(data, transform, scale, 0*up, 0*uq)
    best_place = locate(design, family, n, bound)[0][1]
    start = attach(family, n, best_place)
    lines = canonical([C for name in settings.get("sources", ("rnj",)) for C in combo.source_family(S, name)
                       if 1 < len(C) < n]+[C for C in family if 1 < len(C) < n])
    rec = select_with_source(design, lines, S["alpha"], bound, miqp_seconds, n, warm=start)
    fam = rec.get("family") or start
    tree, path = meters_only(fam, n)
    score = lambda d, f: rv.exact_fit(d, f, bound)["objective"]+S["alpha"]*rv.internal_count(f, n+1)
    half = .5*S["N"]
    places = locate(design, tree, n, bound)
    result.update(located=tree, path=path, status=rec.get("status"),
                  places=[(half*(o-places[0][0]), D) for o, D in places],
                  source_evidence=half*(score(empty, attach(tree, n, None))-score(design, fam)),
                  seconds_total=time.perf_counter()-started)
    return result
