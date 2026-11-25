"""Experiment B: candidate-support generators that could replace W's unknown-support MIQP.

Each generator returns supports compatible with the current laminar family, not already in it, 1 < |C| < n.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import linkage

from rnj_wzzt_dr.identification import trees
from rnj_wzzt_dr.identification.optimization import predict
from rnj_wzzt_dr.physical.backend import Observation


def _valid(family, n):
    fam = set(trees.canonical(family))
    return lambda C: 1 < len(C) < n and C not in fam and all(trees.compatible(C, D) for D in fam)


def g1_pool(family, n, pool):
    ok = _valid(family, n)
    return sorted({tuple(sorted(C)) for C in pool if ok(tuple(sorted(C)))})


def g2_sibling_pairs(family, n):
    out = set()
    for node, blocks in trees.children(family, n).items():
        if len(blocks) >= 3:
            for a, b in combinations(blocks, 2):
                out.add(tuple(sorted((*a, *b))))
    return sorted(out)


def residual(data, point):
    return data.y-predict(data, point)


def g3_residual_rnj(family, n, data, point, weights=(1., .75, 0.), tols=(0., .04, .08, .16, .24)):
    e = residual(data, point)
    R, X, _ = trees.fit_sensitivity(Observation(data.p, data.q, e, data.labels))
    ok = _valid(family, n)
    out = set()
    for w in weights:
        shared, depth, _ = trees.geometry(R, X, w)
        for tol in tols:
            for C in trees.rnj(shared, depth, tol):
                C = tuple(sorted(C))
                if ok(C):
                    out.add(C)
    return sorted(out)


def g4_residual_clusters(family, n, data, point):
    e = residual(data, point)
    out = set()
    for node, blocks in trees.children(family, n).items():
        k = len(blocks)
        if k < 3:
            continue
        sums = np.column_stack([e[:, list(b)].sum(1) for b in blocks])
        sd = sums.std(0)
        if np.any(sd <= 0):
            continue
        corr = np.corrcoef(sums, rowvar=False)
        dist = np.clip(1.-corr, 0., 2.)
        condensed = dist[np.triu_indices(k, 1)]
        Z = linkage(condensed, method="average")
        members = {i: [i] for i in range(k)}
        for step, (a, b, *_rest) in enumerate(Z):
            merged = members.pop(int(a))+members.pop(int(b))
            members[k+step] = merged
            if 2 <= len(merged) < k:
                out.add(tuple(sorted(i for j in merged for i in blocks[j])))
    return sorted(out)


def g5_screen(family, n, data, point, scale, top=50, max_children=20):
    """Rank every compatible support by its one-support gain on the current residual (old coefficients fixed).

    For a node with child blocks B_1..B_k and C = union over s in {0,1}^k: e'f_r = s'M_r s with
    M_r = sym(E'P), ||f_r||^2 = |C| s'(P'P)s, f_r'f_x = |C| s'(P'Q)s, where E, P, Q are T x k block sums."""
    e = residual(data, point)/scale
    scored = []
    for node, blocks in trees.children(family, n).items():
        k = len(blocks)
        if k < 3 or k > max_children:
            continue
        Eb = np.column_stack([e[:, list(b)].sum(1) for b in blocks])
        Pb = np.column_stack([data.p[:, list(b)].sum(1) for b in blocks])
        Qb = np.column_stack([data.q[:, list(b)].sum(1) for b in blocks])
        size = np.array([len(b) for b in blocks], float)
        sym = lambda A: (A+A.T)/2
        Mr, Mx = sym(Eb.T@Pb), sym(Eb.T@Qb)
        Hrr, Hxx, Hrx = sym(Pb.T@Pb), sym(Qb.T@Qb), sym(Pb.T@Qb)
        codes = np.arange(1, 2**k-1, dtype=np.int64)
        S = ((codes[:, None] >> np.arange(k)) & 1).astype(float)
        count = S.sum(1)
        S, codes = S[count >= 2], codes[count >= 2]
        csize = S@size
        quad = lambda M: np.einsum("ij,jk,ik->i", S, M, S)
        gr, gx = quad(Mr), quad(Mx)
        hrr, hxx, hrx = csize*quad(Hrr), csize*quad(Hxx), csize*quad(Hrx)
        det = hrr*hxx-hrx**2
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (hxx*gr-hrx*gx)/det
            b = (hrr*gx-hrx*gr)/det
            both = np.where((a >= 0) & (b >= 0) & (det > 1e-12*np.maximum(hrr*hxx, 1e-300)),
                            a*gr+b*gx, -np.inf)
            only_r = np.where(gr > 0, gr**2/np.maximum(hrr, 1e-300), 0.)
            only_x = np.where(gx > 0, gx**2/np.maximum(hxx, 1e-300), 0.)
        gain = np.maximum(both, np.maximum(only_r, only_x))
        keep = np.argsort(-gain)[:top]
        for i in keep:
            C = tuple(sorted(j for t in range(k) if S[i, t] for j in blocks[t]))
            scored.append((float(gain[i]), C))
    scored.sort(reverse=True)
    out, seen = [], set()
    for g, C in scored:
        if C not in seen:
            seen.add(C); out.append(C)
        if len(out) >= top:
            break
    return out
