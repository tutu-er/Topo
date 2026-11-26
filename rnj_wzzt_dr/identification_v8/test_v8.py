"""Checks of draft v8's code.

    python -m rnj_wzzt_dr.identification_v8.test_v8          set logic, against brute force (no solver, no data)
    python -m rnj_wzzt_dr.identification_v8.test_v8 --data   also: the model and the MIQP reproduce the registered
                                                              code's numbers on one development case
"""
from __future__ import annotations

from itertools import combinations
import random
import sys

from . import anchors as an
from .sets import canonical, compatible


def random_lines(leaves, rng):
    """The lines of a random tree: merge 2 or 3 random clusters until one is left."""
    clusters = [(i,) for i in range(leaves)]
    out = []
    while len(clusters) > 1:
        k = min(len(clusters), rng.choice((2, 2, 3)))
        picked = rng.sample(range(len(clusters)), k)
        merged = tuple(sorted(i for j in picked for i in clusters[j]))
        clusters = [c for j, c in enumerate(clusters) if j not in picked]+[merged]
        if len(merged) < leaves:
            out.append(merged)
    return canonical(out)


def all_lines(leaves):
    return [C for k in range(2, leaves) for C in combinations(range(leaves), k)]


def test_universe_is_every_compatible_line():
    """Proposition 4: with K large, the anchored universe is exactly the set of lines compatible with every anchor."""
    rng = random.Random(1)
    for _ in range(300):
        n = rng.randint(3, 8)
        anchors = canonical(C for C in random_lines(n, rng) if rng.random() < .5)
        got, _ = an.anchored_universe(anchors, n, K=99)
        want = canonical(C for C in all_lines(n) if all(compatible(C, A) for A in anchors))
        assert set(got) == set(want), (n, anchors)


def test_units_partition_their_anchor():
    rng = random.Random(2)
    for _ in range(200):
        n = rng.randint(3, 9)
        anchors = canonical(C for C in random_lines(n, rng) if rng.random() < .6)
        for A, blocks in an.units(anchors, n).items():
            assert sorted(i for b in blocks for i in b) == list(A)
            assert len(blocks) >= 2


def test_big_nodes_keep_pool_lines_only():
    """A node with more than K units contributes no unions, only the compatible pool lines it encloses."""
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randint(4, 8)
        anchors = canonical(C for C in random_lines(n, rng) if rng.random() < .5)
        pool = canonical(rng.sample(all_lines(n), min(12, len(all_lines(n)))))
        got, info = an.anchored_universe(anchors, n, pool, K=2)
        sizes = {A: len(b) for A, b in an.units(anchors, n).items()}
        for C in got:
            assert all(compatible(C, A) for A in anchors)
            if C not in anchors and sizes[an.enclosing(C, anchors, n)] > 2:
                assert C in pool
        assert info["unions"] == sum(2**k-k-2 for k in sizes.values() if k <= 2)


def test_load_path_constraints():
    """Proposition 3: crossings over n + 1 leaves, leaf n the load."""
    n = 5
    meter_sets = [C for k in range(1, n+1) for C in combinations(range(n), k)]
    for C in meter_sets:
        for D in meter_sets:
            assert compatible((*C, n), (*D, n)) == (set(C) <= set(D) or set(D) <= set(C))
            crosses = bool(set(C) & set(D)) and not set(D) <= set(C)
            assert compatible((*C, n), D) == (not crosses)


def test_matches_registered_code(name="dev_small_003_head"):
    """The model reproduces the registered code's unmetered power, noise variance, price and scores, and the MIQP its
    optimum, on one development case (with and without the load)."""
    import numpy as np
    from rnj_wzzt_dr.identification import combo, final, revised as rv, source as src
    from .identify import load_dev_case, load_options
    from .miqp import tree_miqp
    from .model import Fitter, Model, balance
    rec, inp = load_dev_case(name)
    h, n = inp.hist, inp.n
    data = rv.Series(h["P"], h["Q"], h["Y"], h["V0sq"])
    config = rv.RevisedConfig()
    truth = canonical(C for C in rec["truth"] if 1 < len(C) < n)
    close = lambda a, b, tol=1e-7: abs(a-b) <= tol*max(1., abs(a), abs(b))
    # with the load
    up, uq = src.balance(h["Ppcc"], h["Qpcc"], data)
    u = balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"])
    assert np.allclose(u[0], up) and np.allclose(u[1], uq)
    S = final.priced(src.setup_projected(data, None, up, uq, config), final.LAMBDA)
    model = Model(h["P"], h["Q"], h["Y"], h["V0sq"], u, final.LAMBDA)
    assert close(model.sigma2, S["sigma2"]) and close(model.alpha, S["alpha"]) and close(model.gamma, S["c"])
    old = src.LeafDesign(data, rv.Transform(*h["P"].shape, S["c"], config.offsets), np.sqrt(S["sigma2"]), up, uq)
    for _, lines in load_options(truth, n):
        fam = canonical([*lines, *rv.basis(n+1)])
        want = rv.exact_fit(old, fam, 2.)["objective"]+S["alpha"]*rv.internal_count(fam, n+1)
        got = Fitter(model, lines).crit(lines)
        assert close(got, want, 1e-6), (got, want)
    pool = [C for C in src.with_source(truth, n) if 1 < len(C) < n+1]
    want = rv.select_miqp(old, pool, rv.basis(n+1), S["alpha"], 2., 60., gap=1e-6)["objective"]
    got = tree_miqp(model, pool, 60., gap=1e-6)["crit"]
    assert close(got, want, 1e-5), (got, want)
    # without the load
    S = final.priced(combo.setup(data, None, config), final.LAMBDA)
    model = Model(h["P"], h["Q"], h["Y"], h["V0sq"], None, final.LAMBDA)
    assert close(model.sigma2, S["sigma2"]) and close(model.alpha, S["alpha"])
    fam = canonical([*truth, *rv.basis(n)])
    want = rv.exact_fit(S["design"], fam, 2.)["objective"]+S["alpha"]*len(truth)
    assert close(Fitter(model, truth).crit(truth), want, 1e-6)


def test_gate_bounds_and_node_miqp(group="dev_small_003"):
    """On one development feeder: the gate separates its no-load twin from its head-load twin; the per-meter bounds
    leave the MIQP optimum unchanged; the node MIQP finds the same best line as enumeration."""
    from .identify import load_dev_case
    from .miqp import tree_miqp
    from .model import GATE, Fitter, Model, balance, load_share
    from .search import best_addition, node_miqp
    from .sets import basis, children
    shares = {}
    for twin in ("none", "head"):
        rec, inp = load_dev_case(f"{group}_{twin}")
        h = inp.hist
        shares[twin] = load_share(h["P"], balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"]))
    assert shares["none"] < GATE <= shares["head"], shares
    rec, inp = load_dev_case(f"{group}_none")
    h, n = inp.hist, inp.n
    model = Model(h["P"], h["Q"], h["Y"], h["V0sq"])
    truth = canonical(C for C in rec["truth"] if 1 < len(C) < n)
    pool = canonical([*truth, *random_lines(n, random.Random(4)), *random_lines(n, random.Random(5))])
    level = Fitter(model, truth).crit(truth)                       # attained, so a valid level for the bounds
    plain = tree_miqp(model, pool, 60., gap=1e-6)
    tight = tree_miqp(model, pool, 60., gap=1e-6, tighten_at=level)
    assert abs(plain["crit"]-tight["crit"]) <= 1e-6*max(1., abs(plain["crit"])), (plain["crit"], tight["crit"])
    tree = ()                                                     # every meter a child of the root: n children
    enum_line, info = best_addition(model, tree, max_children=n)
    kids = children(basis(n), n)[tuple(range(n))]
    miqp_line, _, status = node_miqp(model, tree, kids, 120.)
    assert info["enumerated"] == 1 and status == 2
    f = Fitter(model, [enum_line, miqp_line])
    assert abs(f.crit([enum_line])-f.crit([miqp_line])) <= 1e-7, (enum_line, miqp_line)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        needs_data = name in ("test_matches_registered_code", "test_gate_bounds_and_node_miqp")
        if name.startswith("test_") and (not needs_data or "--data" in sys.argv):
            fn()
            print("ok", name, flush=True)
