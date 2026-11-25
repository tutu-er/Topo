"""Experiment A: heuristics for WZZT's E problem (choose a laminar subset of a finite pool).

Objective (the E MIQP's own): normalised training MSE of the family (selected supports plus the mandatory basis)
+ alpha per selected internal support. All searches use fastfit.UniverseFitter; callers rescore final families.
"""
from __future__ import annotations

import time

import numpy as np

from rnj_wzzt_dr.identification.trees import basis, canonical, compatible


class Problem:
    def __init__(self, fitter, pool, keep):
        self.f, self.n = fitter, fitter.n
        self.keep = canonical([*keep, *basis(self.n)])
        self.proper = [C for C in canonical(pool) if 1 < len(C) < self.n]
        self.crosses = {C: frozenset(D for D in self.proper if not compatible(C, D)) for C in self.proper}

    def score(self, S):
        return self.f.score([*self.keep, *S])

    def addable(self, S):
        return [C for C in self.proper if C not in S and not (self.crosses[C] & S)]


def _deadline(seconds):
    return time.perf_counter()+seconds


def local_search(prob, start, deadline):
    """Best-improvement moves: add, drop, add with eviction of crossing supports."""
    S = frozenset(C for C in start if C in prob.crosses)
    best = prob.score(S)
    while time.perf_counter() < deadline:
        move, move_score = None, best-1e-12
        for C in prob.proper:
            if C in S:
                T = S-{C}
            else:
                T = (S-prob.crosses[C])|{C}
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
                T = S|{C}
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


def lasso_path(prob, deadline, n_lambda=20, iterations=400):
    """Nonnegative lasso over the pool (box [0, bound]), laminar repair by coefficient size, refit."""
    f = prob.f
    fam = canonical([*prob.keep, *prob.proper])
    cols = f._cols(fam)
    G, b = f.G[np.ix_(cols, cols)]/f.N, f.b[cols]/f.N
    m = len(fam)
    internal = np.array([1 < len(C) < prob.n for C in fam]*2)
    norms = np.sqrt(np.maximum(np.diag(G), 1e-300))
    L = 2*np.linalg.eigvalsh(G)[-1]
    theta = f.theta0[cols].copy()          # start at the mandatory-only fit
    grad = 2*(G@theta-b)
    lam_max = max(float(np.max(-grad[internal])), 1e-12)
    best_S, best = frozenset(), prob.score(frozenset())
    for lam in lam_max*np.logspace(0, -3, n_lambda):
        if time.perf_counter() > deadline:
            break
        w = np.where(internal, lam, 0.)
        x, y_, t = theta.copy(), theta.copy(), 1.
        for _ in range(iterations):
            g = 2*(G@y_-b)+w
            x_new = np.clip(y_-g/L, 0., f.bound)
            t_new = (1+np.sqrt(1+4*t*t))/2
            y_ = x_new+(t-1)/t_new*(x_new-x)
            if np.max(np.abs(x_new-x)) < 1e-10:
                x = x_new; break
            x, t = x_new, t_new
        theta = x
        weight = (theta[:m]*norms[:m]+theta[m:]*norms[m:])
        order = [i for i in np.argsort(-weight) if internal[i] and weight[i] > 1e-12]
        S = set()
        for i in order:
            C = fam[i]
            if not (prob.crosses[C] & S):
                S.add(C)
        S = frozenset(S)
        s = prob.score(S)
        if s < best:
            best_S, best = S, s
    return best_S, best


def run_all(prob, *, e_start, rnj_trees, seed=0, cap=300.):
    """H1-H9. Returns {name: dict(family, objective, seconds)}; families are sets of internal supports."""
    out = {}

    def record(name, fn):
        t = time.perf_counter()
        S, s = fn(_deadline(cap))
        out[name] = dict(family=sorted(S), objective=float(s), seconds=time.perf_counter()-t)

    record("H1_greedy", lambda d: beam(prob, 1, d))
    record("H2_local_base", lambda d: local_search(prob, frozenset(), d))
    record("H3_local_Estart", lambda d: local_search(prob, frozenset(e_start), d))
    record("H4_beam5", lambda d: beam(prob, 5, d))
    record("H5_beam20", lambda d: beam(prob, 20, d))

    def multistart(starts, d):
        best_S, best = frozenset(), np.inf
        for st in starts:
            if time.perf_counter() > d:
                break
            S, s = local_search(prob, st, d)
            if s < best:
                best_S, best = S, s
        return best_S, best
    record("H6_rnj_multistart", lambda d: multistart([frozenset(t) for t in rnj_trees], d))
    rng = np.random.default_rng(seed)
    starts = [random_laminar(prob, rng) for _ in range(20)]
    record("H7_random_multistart", lambda d: multistart(starts, d))
    record("H8_lasso", lambda d: lasso_path(prob, d))
    h8 = frozenset(map(tuple, out["H8_lasso"]["family"]))
    record("H9_lasso_local", lambda d: local_search(prob, h8, d))
    return out
