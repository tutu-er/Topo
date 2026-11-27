"""Checks of draft v12's code (identification and probing).

    python -m rnj_wzzt_dr.identification_v8.test_v8          set logic, against brute force (no solver, no data)
    python -m rnj_wzzt_dr.identification_v8.test_v8 --data   also: the model and the MIQP reproduce the registered
                                                              code's numbers on one development case
"""
from __future__ import annotations

from itertools import combinations
import random
import sys

import numpy as np

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
    from rnj_wzzt_dr.experiments.case_library_20261125.dev_cases import load_dev_case
    from .identify import load_options
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
    from rnj_wzzt_dr.experiments.case_library_20261125.dev_cases import load_dev_case
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


def test_kappa():
    """Eq. 15: kappa(delta, q) = E min(delta, Z), Z ~ N(q/2, q): the draft's example (q = 2, delta = 1 gives 0.436) and
    a Monte Carlo estimate."""
    import math
    import numpy as np
    from .probe import kappa
    assert abs(kappa(1., 2.)-.436) < 1e-3
    rng = np.random.default_rng(1)
    for delta, q in ((.5, .3), (3., 8.), (0., 1.), (-.2, .05)):
        z = rng.normal(q/2, math.sqrt(q), 400_000)
        assert abs(kappa(delta, q)-np.minimum(delta, z).mean()) < 5e-3, (delta, q)


def test_block_gradients(name="dev_small_003_head"):
    """probe.prediction: a block of size eps at meter i moves the S-weighted prediction by eps ds[:, i] and the
    predicted V0^2 - V^2 by eps dy[:, i] (both are linear in the meters' flows), with the load modelled."""
    import numpy as np
    from rnj_wzzt_dr.experiments.case_library_20261125.dev_cases import load_dev_case
    from .identify import with_load
    from .probe import forecast, prediction, setup
    rec, inp = load_dev_case(name)
    data = {k: np.asarray(inp.hist[k], float) for k in ("P", "Q", "Y", "V0sq", "Ppcc", "Qpcc")}
    model, fp, fq = setup(data, True, type("St", (), {"lam": 1.96})())
    n = model.n
    family = with_load(canonical(C for C in rec["truth"] if 1 < len(C) < n), n)[:6]
    fc = forecast(data, fp, fq, 13)
    base = prediction(model, family, fc)
    for i in (0, n-1):
        moved = dict(fc, p=fc["p"]+1e-3*np.eye(model.leaves)[i])
        after = prediction(model, family, moved)
        assert np.allclose(after["s"]-base["s"], 1e-3*base["ds"][:, i], atol=1e-13)
        assert np.allclose(after["y"]-base["y"], 1e-3*base["dy"][:, i], atol=1e-13)


SYNTHETIC_LINES = [(0, 1), (0, 1, 2), (3, 4), (5, 6, 7), (3, 4, 5, 6, 7)]


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


QUICK = dict(short_seconds=10, long_seconds=10, start_seconds=2, polish_seconds=2, node_seconds=10, rounds=1)


def test_identify_synthetic():
    """Algorithm 1 recovers a known tree from synthetic readings, without and with an unmetered load (the load gate
    and the load's path included). Needs no case library: it runs with the folder alone."""
    from .identify import Settings, identify
    for load in (False, True):
        data = synthetic_hour(np.random.default_rng(3), np.arange(192), load=load)
        out = identify(**data, st=Settings(**QUICK))
        assert out["gate"]["load"] == load
        assert set(out["tree"]) == set(SYNTHETIC_LINES), (load, out["tree"])
        if load:
            assert [tuple(p) for p in out["path"]] == [(3, 4, 5, 6, 7), (5, 6, 7)], out["path"]


def test_flip_tests_synthetic():
    """Eq. 6 both ways on the synthetic feeder (true tree, pool with one nested and one crossing extra line): removing
    a true grouping and forcing in a crossing one are both decided; so is an outside target in probing, which needs
    the flip test with its grouping required (a local swap could only show it open, Lemma 1)."""
    from .identify import Settings
    from .labels import DECIDED, flip_test
    from .model import Fitter, Model
    from .probe import relabel_outside
    data = synthetic_hour(np.random.default_rng(3), np.arange(192))
    model = Model(data["P"], data["Q"], data["Y"], data["V0sq"])
    tree, pool = canonical(SYNTHETIC_LINES), canonical([*SYNTHETIC_LINES, (1, 2), (2, 3)])
    threshold = Fitter(model, tree).crit(tree)+4.6/(model.N/2)
    assert flip_test(model, pool, (0, 1, 2), threshold, 10.)["label"] == DECIDED
    assert flip_test(model, pool, (2, 3), threshold, 10., inside=False)["label"] == DECIDED
    row = relabel_outside(model, tree, pool, [], [(2, 3)], Settings(**QUICK))[0]
    assert row["label"] == DECIDED and not row["inside"] and row["gap"][0] > 4.6


def test_probe_synthetic():
    """Algorithm 2 runs a short day on the synthetic feeder (a linear world with fixed headroom) and keeps its caps."""
    from .identify import Settings, identify
    from .probe import ProbeSettings, probe_day
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


def test_builders_match_registered(names=("soumalas11_star_l5_middle", "dev_small_003_none")):
    """builders.py gives the registered predecessor's nine trees (identification/), with and without the load. The full
    check over the development twins is in paper/DEV_RESULTS_V12.md."""
    from rnj_wzzt_dr.experiments.case_library_20261125.dev_cases import load_dev_case
    from rnj_wzzt_dr.identification import combo, final, revised as rv, source as src
    from .builders import builder_trees
    from .model import GATE, balance, load_share
    for name in names:
        rec, inp = load_dev_case(name)
        h, n = inp.hist, inp.n
        u = balance(h["P"], h["Q"], h["Y"], h["V0sq"], h["Ppcc"], h["Qpcc"])
        u = u if load_share(h["P"], u) >= GATE else None
        data, config = rv.Series(h["P"], h["Q"], h["Y"], h["V0sq"]), rv.RevisedConfig()
        S = final.guarded(combo.setup, data, None, config) if u is None else \
            final.guarded(src.setup_projected, data, None, u[0], u[1], config)
        old = [{C for C in combo.source_family(S, k) if 1 < len(C) < n} for k in combo.SOURCES]
        assert [set(t) for t in builder_trees(h["P"], h["Q"], h["Y"], h["V0sq"], u)] == old, name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        needs_data = name in ("test_matches_registered_code", "test_gate_bounds_and_node_miqp", "test_block_gradients",
                              "test_builders_match_registered")
        if name.startswith("test_") and (not needs_data or "--data" in sys.argv):
            fn()
            print("ok", name, flush=True)
