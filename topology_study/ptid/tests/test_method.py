"""Checks of the code (TESTS in README.md). Run from topology_study/:

    python -m ptid.tests.test_method            no case files needed: set logic against brute force, the MIQPs against
                                                enumeration, the claim check, convergence, both algorithms on a
                                                synthetic feeder
    python -m ptid.tests.test_method --data     also: tests on development cases and the real-feeder reader
    python -m ptid.tests.test_method --legacy   also: equality with the registered predecessor's code (needs the
                                                repository's rnj_wzzt_dr package on the path)
"""
from __future__ import annotations

from itertools import combinations
import math
import random
import sys

import numpy as np

from ptid.method import anchors as an
from ptid.method.sets import basis, canonical, children, compatible, crossing_map, lines_of


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


# ------------------------------------------------------------------ set logic (no solver)
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


# ------------------------------------------------------------------ a synthetic feeder
SYNTHETIC_LINES = [(0, 1), (0, 1, 2), (3, 4), (5, 6, 7), (3, 4, 5, 6, 7)]
QUICK = dict(short_seconds=10, long_seconds=10, start_seconds=2, node_seconds=10, rounds=1)


def synthetic_hour(rng, hours, n=8, load=False, block=None):
    """Readings of the linear model (eq. 1) of the tree SYNTHETIC_LINES (n = 8 meters, its trunk and service cables),
    with small meter noise; with load, an unmetered load below the line {5, 6, 7}. Fixed impedances, so that
    successive calls continue one feeder."""
    family = [*SYNTHETIC_LINES, *[(i,) for i in range(n)], tuple(range(n))]
    r = {C: .01+.04*k/len(family) for k, C in enumerate(family)}
    shape = 1+.6*np.sin(2*np.pi*(np.asarray(hours)-7)/24)
    P = .004*shape[:, None]*np.linspace(.5, 1.5, n)+.002*rng.random((len(shape), n))+(0 if block is None else block)
    Q = .3*P+.0005*rng.random(P.shape)
    up = .01*shape+.002*rng.random(len(shape)) if load else np.zeros(len(shape))
    Y = np.zeros(P.shape)
    for C in family:
        below = load and set(C) >= {5, 6, 7}
        Y[:, list(C)] += 2*r[C]*(P[:, list(C)].sum(1)+.5*Q[:, list(C)].sum(1)+(1.15*up if below else 0))[:, None]
    V0sq = 1.04+1e-4*rng.normal(size=len(shape))
    Y = Y+1e-5*rng.normal(size=Y.shape)
    rise = (np.sqrt(V0sq)[:, None]-np.sqrt(V0sq[:, None]-Y))/np.sqrt(V0sq[:, None]-Y)
    return dict(P=P, Q=Q, Y=Y, V0sq=V0sq, Ppcc=(P*(1+rise)).sum(1)+up, Qpcc=(Q*(1+rise)).sum(1)+.3*up)


def synthetic_model():
    from ptid.method.model import Model
    data = synthetic_hour(np.random.default_rng(3), np.arange(192))
    return data, Model(data["P"], data["Q"], data["Y"], data["V0sq"])


EXTRA = [(1, 2), (2, 3), (5, 6), (4, 5), (0, 1, 2, 3, 4)]


def test_tree_miqp_matches_enumeration():
    """The tree MIQP (eq. 9, scaled coefficients, default tolerances) returns the best tree over a pool, checked by
    fitting every laminar subset of the pool. Also with the per-meter bounds at an attained level, and with a required
    grouping (as in a flip test outside the tree)."""
    from ptid.method.miqp import tree_miqp
    from ptid.method.model import Fitter
    _, model = synthetic_model()
    pool = canonical([*SYNTHETIC_LINES, *EXTRA])
    fit = Fitter(model, pool)
    trees = [S for k in range(len(pool)+1) for S in combinations(pool, k)
             if all(compatible(a, b) for a, b in combinations(S, 2))]
    best = min(trees, key=fit.crit)
    for kw in (dict(), dict(tighten_at=fit.crit(best)+1e-3)):
        r = tree_miqp(model, pool, 60., gap=1e-8, **kw)
        assert set(lines_of(r["family"], model.leaves)) == set(best), (kw, r["family"], best)
    forced = min((S for S in trees if (2, 3) in S), key=fit.crit)
    r = tree_miqp(model, pool, 60., gap=1e-8, require=[[(2, 3)]])
    assert set(lines_of(r["family"], model.leaves)) == set(forced)


def test_claim_check():
    """A solve whose bound exceeds the score of a tree known to be feasible is reported as a numerical failure (status
    NUMERIC, no bound): here the 'known' score is set just below the true optimum, so a correct bound must exceed it."""
    import gurobipy as gp
    from ptid.method.miqp import tree_miqp
    _, model = synthetic_model()
    pool = canonical([*SYNTHETIC_LINES, *EXTRA])
    best = tree_miqp(model, pool, 60., gap=1e-8)
    assert best["status"] == gp.GRB.OPTIMAL
    r = tree_miqp(model, pool, 60., gap=1e-8, known=best["crit"]-1e-3)
    assert r["status"] == gp.GRB.NUMERIC and r["bound"] is None
    r = tree_miqp(model, pool, 60., gap=1e-8, known=best["crit"])
    assert r["status"] == gp.GRB.OPTIMAL and r["bound"] is not None


def test_solve_is_converged():
    """solve() returns a tree that no single removal or swap of a pool line improves (every candidate line checked)."""
    from ptid.method.identify import Settings, solve
    from ptid.method.model import Fitter
    _, model = synthetic_model()
    pool = canonical([*SYNTHETIC_LINES, *EXTRA, *random_lines(8, random.Random(7))])
    tree, info = solve(model, pool, [frozenset(random_lines(8, random.Random(8)))], 10., Settings(**QUICK))
    fit, crosses, S = Fitter(model, pool), crossing_map(pool), frozenset(tree)
    for C in crosses:
        T = S-{C} if C in S else (S-crosses[C]) | {C}
        assert fit.crit(T) >= info["crit"]-1e-12, C


def test_node_miqp_matches_enumeration():
    """Pricing (eq. 10) at a node with 8 children: the node MIQP (scaled) finds the same best line as enumeration."""
    from ptid.method.model import Fitter
    from ptid.method.search import best_addition, node_miqp
    _, model = synthetic_model()
    tree = ()                                                     # every meter a child of the root
    line, info = best_addition(model, tree, max_children=8)
    kids = children(basis(8), 8)[tuple(range(8))]
    got, _, status = node_miqp(model, tree, kids, 60.)
    assert info["enumerated"] == 1 and status == 2
    f = Fitter(model, [line, got])
    assert abs(f.crit([line])-f.crit([got])) <= 1e-7, (line, got)


def test_kappa():
    """Eq. 15: kappa(delta, q) = E min(delta, Z), Z ~ N(q/2, q): the draft's example (q = 2, delta = 1 gives 0.436) and
    a Monte Carlo estimate."""
    from ptid.method.probe import kappa
    assert abs(kappa(1., 2.)-.436) < 1e-3
    rng = np.random.default_rng(1)
    for delta, q in ((.5, .3), (3., 8.), (0., 1.), (-.2, .05)):
        z = rng.normal(q/2, math.sqrt(q), 400_000)
        assert abs(kappa(delta, q)-np.minimum(delta, z).mean()) < 5e-3, (delta, q)


def test_identify_synthetic():
    """Algorithm 1 recovers a known tree from synthetic readings, without and with an unmetered load (the load gate
    and the load's path included)."""
    from ptid.method.identify import Settings, identify
    for load in (False, True):
        data = synthetic_hour(np.random.default_rng(3), np.arange(192), load=load)
        out = identify(**data, st=Settings(**QUICK))
        assert out["gate"]["load"] == load
        assert set(out["tree"]) == set(SYNTHETIC_LINES), (load, out["tree"])
        if load:
            assert [tuple(p) for p in out["path"]] == [(3, 4, 5, 6, 7), (5, 6, 7)], out["path"]


def test_flip_tests_synthetic():
    """Eq. 6 both ways (true tree; pool with one nested and one crossing extra line): removing a true grouping and
    forcing in a crossing one are both decided; so is an outside target in probing, which needs the flip test with its
    grouping required (a local swap could only show it open, Lemma 1)."""
    from ptid.method.identify import Settings
    from ptid.method.labels import DECIDED, flip_test
    from ptid.method.model import Fitter
    from ptid.method.probe import relabel_outside
    _, model = synthetic_model()
    tree, pool = canonical(SYNTHETIC_LINES), canonical([*SYNTHETIC_LINES, (1, 2), (2, 3)])
    threshold = Fitter(model, tree).crit(tree)+4.6/(model.N/2)
    assert flip_test(model, pool, (0, 1, 2), threshold, 10.)["label"] == DECIDED
    assert flip_test(model, pool, (2, 3), threshold, 10., inside=False)["label"] == DECIDED
    row = relabel_outside(model, tree, pool, [], [(2, 3)], Settings(**QUICK))[0]
    assert row["label"] == DECIDED and not row["inside"] and row["gap"][0] > 4.6


def test_probe_synthetic():
    """Algorithm 2 runs a short day on the synthetic feeder (a linear world with fixed headroom) and keeps its caps."""
    from ptid.method.identify import Settings, identify
    from ptid.method.probe import ProbeSettings, probe_day
    rng = np.random.default_rng(4)
    data = synthetic_hour(rng, np.arange(192), load=True)

    class World:
        base_kw = 200.

        def headroom(self, hour, pending=None):
            return np.full(8, .01), np.full(8, .01)

        def run_hour(self, hour, u):
            return {k: v[0] for k, v in synthetic_hour(rng, [hour], load=True, block=u).items()}
    st = Settings(**QUICK)
    day = probe_day(data, identify(**data, st=st), World(), st, ProbeSettings(hours=3, psi=50.))
    assert day["stop"] in ("no target", "every target decided at two consecutive gates", "end of day")
    assert day["cost"] <= ProbeSettings().cap_day and len(day["gates"]) <= 4


# ------------------------------------------------------------------ on the study's cases (--data)
def data_test_cases_and_selection():
    """The development library: 288 hash-checked cases; one twin per feeder gives 72 (no load 20, head 20, middle 16,
    end 16); the real-feeder reader gives 66 inputs with readings and a truth."""
    from collections import Counter
    from ptid.cases import library, real
    rows = library.manifest()
    assert len(rows) == 288
    sel = library.one_twin_per_feeder()
    where = {r["name"]: r["source"] for r in rows}
    assert len(sel) == 72 and Counter(where[x] for x in sel) == dict(none=20, head=20, middle=16, end=16)
    rec = library.open_case(sel[0])
    h = library.readings(rec)
    assert h["P"].shape == (192, rec["n"]) and h["Ppcc"].shape == (192,)
    names = real.names()
    assert len(names) == 66
    r = real.open_real(names[0])
    assert r["hist"]["Y"].shape == (192, r["n"]) and r["truth"]
    assert library.truth(rec) <= library.exact_truth(rec) and r["truth"] <= r["exact"]     # merging only removes lines


def data_test_gate_bounds_and_node_miqp(group="dev_small_003"):
    """On one development feeder: the gate separates its no-load twin from its head-load twin; the per-meter bounds
    leave the MIQP optimum unchanged; the node MIQP finds the same best line as enumeration."""
    from ptid.cases import library
    from ptid.method.miqp import tree_miqp
    from ptid.method.model import GATE, Fitter, Model, balance, load_share
    from ptid.method.search import best_addition, node_miqp
    shares = {}
    for twin in ("none", "head"):
        h = library.readings(library.open_case(f"{group}_{twin}"))
        shares[twin] = load_share(h["P"], balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"]))
    assert shares["none"] < GATE <= shares["head"], shares
    rec = library.open_case(f"{group}_none")
    h, n = library.readings(rec), rec["n"]
    model = Model(h["P"], h["Q"], h["Y"], h["V0sq"])
    truth = canonical(library.truth(rec))
    pool = canonical([*truth, *random_lines(n, random.Random(4)), *random_lines(n, random.Random(5))])
    level = Fitter(model, truth).crit(truth)
    plain = tree_miqp(model, pool, 60., gap=1e-6)
    tight = tree_miqp(model, pool, 60., gap=1e-6, tighten_at=level)
    assert abs(plain["crit"]-tight["crit"]) <= 1e-6*max(1., abs(plain["crit"])), (plain["crit"], tight["crit"])
    line, info = best_addition(model, (), max_children=n)
    got, _, status = node_miqp(model, (), children(basis(n), n)[tuple(range(n))], 120.)
    assert info["enumerated"] == 1 and status == 2
    f = Fitter(model, [line, got])
    assert abs(f.crit([line])-f.crit([got])) <= 1e-7, (line, got)


def data_test_block_gradients(name="dev_small_003_head"):
    """probe.prediction: a block of size eps at meter i moves the S-weighted prediction by eps ds[:, i] and the
    predicted V0^2 - V^2 by eps dy[:, i] (both are linear in the meters' flows), with the load modelled."""
    from ptid.cases import library
    from ptid.method.identify import with_load
    from ptid.method.probe import forecast, prediction, setup
    rec = library.open_case(name)
    data = {k: np.asarray(v, float) for k, v in library.readings(rec).items()}
    model, fp, fq = setup(data, True, type("St", (), {"lam": 1.96})())
    n = model.n
    family = with_load(canonical(library.truth(rec)), n)[:6]
    fc = forecast(data, fp, fq, 13)
    base = prediction(model, family, fc)
    for i in (0, n-1):
        after = prediction(model, family, dict(fc, p=fc["p"]+1e-3*np.eye(model.leaves)[i]))
        assert np.allclose(after["s"]-base["s"], 1e-3*base["ds"][:, i], atol=1e-13)
        assert np.allclose(after["y"]-base["y"], 1e-3*base["dy"][:, i], atol=1e-13)


# ------------------------------------------------------------------ against the registered predecessor (--legacy)
def legacy_test_matches_registered_code(name="dev_small_003_head"):
    """The model reproduces the registered code's unmetered power, noise variance, price and scores, and the MIQP its
    optimum, on one development case (with and without the load)."""
    from rnj_wzzt_dr.identification import combo, final, revised as rv, source as src
    from ptid.cases import library
    from ptid.method.identify import load_options
    from ptid.method.miqp import tree_miqp
    from ptid.method.model import Fitter, Model, balance
    rec = library.open_case(name)
    h, n = library.readings(rec), rec["n"]
    data, config = rv.Series(h["P"], h["Q"], h["Y"], h["V0sq"]), rv.RevisedConfig()
    truth = canonical(library.truth(rec))
    close = lambda a, b, tol=1e-7: abs(a-b) <= tol*max(1., abs(a), abs(b))
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
        assert close(Fitter(model, lines).crit(lines), want, 1e-6)
    pool = [C for C in src.with_source(truth, n) if 1 < len(C) < n+1]
    want = rv.select_miqp(old, pool, rv.basis(n+1), S["alpha"], 2., 60., gap=1e-6)["objective"]
    got = tree_miqp(model, pool, 60., gap=1e-6)["crit"]
    assert close(got, want, 1e-5), (got, want)
    S = final.priced(combo.setup(data, None, config), final.LAMBDA)
    model = Model(h["P"], h["Q"], h["Y"], h["V0sq"], None, final.LAMBDA)
    assert close(model.sigma2, S["sigma2"]) and close(model.alpha, S["alpha"])
    fam = canonical([*truth, *rv.basis(n)])
    want = rv.exact_fit(S["design"], fam, 2.)["objective"]+S["alpha"]*len(truth)
    assert close(Fitter(model, truth).crit(truth), want, 1e-6)


def legacy_test_builders_match_registered(names=("soumalas11_star_l5_middle", "dev_small_003_none")):
    """builders.py gives the registered predecessor's nine trees, with and without the load (all 288 twins: RESULTS.md)."""
    from rnj_wzzt_dr.identification import combo, final, revised as rv, source as src
    from ptid.cases import library
    from ptid.method.builders import builder_trees
    from ptid.method.model import GATE, balance, load_share
    for name in names:
        rec = library.open_case(name)
        h, n = library.readings(rec), rec["n"]
        u = balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"])
        u = u if load_share(h["P"], u) >= GATE else None
        data, config = rv.Series(h["P"], h["Q"], h["Y"], h["V0sq"]), rv.RevisedConfig()
        S = final.guarded(combo.setup, data, None, config) if u is None else \
            final.guarded(src.setup_projected, data, None, u[0], u[1], config)
        old = [{C for C in combo.source_family(S, k) if 1 < len(C) < n} for k in combo.SOURCES]
        assert [set(t) for t in builder_trees(h["P"], h["Q"], h["Y"], h["V0sq"], u)] == old, name


if __name__ == "__main__":
    groups = ["test_"]
    if "--data" in sys.argv or "--legacy" in sys.argv:
        groups.append("data_test_")
    if "--legacy" in sys.argv:
        groups.append("legacy_test_")
        from pathlib import Path
        sys.path.append(str(Path(__file__).resolve().parents[3]))      # the repository root, for rnj_wzzt_dr
    for name, fn in list(globals().items()):
        if callable(fn) and any(name.startswith(g) for g in groups):
            fn()
            print("ok", name, flush=True)
