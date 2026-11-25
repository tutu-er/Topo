"""Distance-based tree builders (RNJ, NJ, RG) on exact tree metrics, and the Gurobi ordered fit against SLSQP."""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from rnj_wzzt_dr.identification import revised as rv
from rnj_wzzt_dr.identification import trees
from rnj_wzzt_dr.identification.trees import basis, canonical


def _random_tree(seed, n_target=12, internal_meters=False):
    """Random rooted tree: hidden junctions with 2-4 children, meters at the leaves (and optionally at some
    junctions). Returns parent map, edge resistance, meter of node (or None) and the meter count."""
    rng = np.random.default_rng(seed)
    parent, length, meter = {0: None}, {0: 0.}, {0: None}
    frontier, nxt, n = [0], 1, 0
    trunk = bool(rng.integers(2))
    if trunk:                                   # root -> one trunk junction
        parent[1], length[1], meter[1] = 0, rng.uniform(.2, 2.), None
        frontier, nxt = [1], 2
    while frontier:
        node = frontier.pop(0)
        k = int(rng.integers(2, 5))
        for _ in range(k):
            child = nxt; nxt += 1
            parent[child], length[child], meter[child] = node, rng.uniform(.2, 2.), None
            if n+len(frontier)+1 < n_target and rng.random() < .45:
                frontier.append(child)
                if internal_meters and rng.random() < .5:
                    meter[child] = n; n += 1
            else:
                meter[child] = n; n += 1
    return parent, length, meter, n


def _metric(parent, length, meter, n):
    path = {}
    for node in sorted(parent):
        path[node] = [] if parent[node] is None else path[parent[node]]+[node]
    where = {m: node for node, m in meter.items() if m is not None}
    depth = np.array([sum(length[v] for v in path[where[i]]) for i in range(n)])
    shared = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            common = set(path[where[i]]) & set(path[where[j]])
            shared[i, j] = sum(length[v] for v in common)
    supports = set()
    for node in parent:
        if parent[node] is not None:
            below = [meter[v] for v in parent if node in path[v] and meter[v] is not None]
            if below:
                supports.add(tuple(sorted(below)))
    return shared, depth, canonical([*basis(n), *supports])


@pytest.mark.parametrize("seed", range(12))
def test_builders_recover_exact_trees_with_meters_at_leaves(seed):
    shared, depth, truth = _metric(*_random_tree(seed))
    for build in (trees.rnj, trees.nj, trees.rg):
        assert canonical(build(shared, depth, 0.)) == truth, build.__name__


@pytest.mark.parametrize("seed", range(12))
def test_rg_recovers_meters_at_internal_nodes(seed):
    shared, depth, truth = _metric(*_random_tree(100+seed, internal_meters=True))
    assert canonical(trees.rg(shared, depth, 0.)) == truth


def test_rg_contracts_short_hidden_lines_with_tolerance():
    # root - a(0.05) - {b(1) - {m0, m1}, c(1) - {m2, m3}}: the 0.05 trunk is the whole feeder (no internal support);
    # the b and c lines are internal and long, so tolerance 0.1 keeps them and 1.5 removes them.
    parent = {0: None, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}
    length = {0: 0., 1: .05, 2: 1., 3: 1., 4: 1., 5: 1., 6: 1., 7: 1.}
    meter = {0: None, 1: None, 2: None, 3: None, 4: 0, 5: 1, 6: 2, 7: 3}
    shared, depth, truth = _metric(parent, length, meter, 4)
    assert canonical(trees.rg(shared, depth, .1)) == truth
    assert canonical(trees.rg(shared, depth, 1.5)) == basis(4)


def _series(n=6, T=80, seed=3):
    rng = np.random.default_rng(seed)
    p = rng.uniform(.2, 1., (T, n)); q = .3*p+.05*rng.normal(size=(T, n))
    fam = canonical([*basis(n), (0, 1), (0, 1, 2), (3, 4)])
    r = rng.uniform(.01, .05, len(fam)); x = rng.uniform(.005, .02, len(fam))
    Z = rv.indicators(fam, n)
    y = (p@Z.T*r)@Z+(q@Z.T*x)@Z+5e-4*rng.normal(size=(T, n))
    return dict(p=p, q=q, y=y), rv.Series(p, q, y, None), T, n


def test_gurobi_ordered_fit_matches_slsqp():
    train, _, _, _ = _series()
    R1, X1, i1 = trees.fit_sensitivity(train, "slsqp")
    R2, X2, i2 = trees.fit_sensitivity(train, "gurobi")
    assert i2["solver"].startswith("Gurobi")
    assert abs(i1["squared_residual_sum"]-i2["squared_residual_sum"]) <= 1e-7*i1["squared_residual_sum"]
    scale = max(np.abs(R1).max(), np.abs(X1).max())
    assert np.abs(R1-R2).max() <= 1e-4*scale and np.abs(X1-X2).max() <= 1e-4*scale


def test_gurobi_ponly_fit_matches_slsqp():
    _, series, T, n = _series(seed=5)
    d = rv.Design(series, rv.Transform(T, n, .7, True), 1.)
    R1, _ = rv.ordered_sensitivity(d, "slsqp")
    R2, info = rv.ordered_sensitivity(d, "gurobi")
    assert info["solver"] == "gurobi"
    assert np.abs(R1-R2).max() <= 1e-4*np.abs(R1).max()
