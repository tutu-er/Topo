"""Tests of the case library (no identifier, no solver)."""
import numpy as np
import pytest

from rnj_wzzt_dr.physical import backend
from rnj_wzzt_dr.experiments.hourly_probing_20261118 import scenario as sc
from rnj_wzzt_dr.experiments.case_library_20261125 import build, feeders, loads, physics

needs_data = pytest.mark.skipif(not loads.PROFILES.exists() and not loads.CACHE.exists(),
                                reason="the customers' records are not on this machine")


def inner(network):
    labels = feeders.labels_of(network)
    n = len(labels)
    return {tuple(sorted(C)) for C in sc.truth_family(network, labels) if 1 < len(C) < n}


@pytest.mark.parametrize("case", feeders.OLD_CASES)
def test_star_is_the_old_case(case):
    """The star plan has the old case's meters, lines and topology."""
    old = backend.load_case(case)
    new = feeders.to_network(feeders.star_plan(case))
    assert feeders.labels_of(new) == tuple(b["bus_id"] for b in old["buses"] if b["is_terminal"])
    z = {(b["from_bus"], b["to_bus"]): (b["r_ohm"], b["x_ohm"]) for b in old["branches"]}
    assert {(b["from_bus"], b["to_bus"]): (b["r_ohm"], b["x_ohm"]) for b in new["branches"]} == z
    labels = feeders.labels_of(new)
    old_lines = {tuple(sorted(C)) for C in backend.case_truth(case, labels)}
    assert inner(new) == {C for C in old_lines if 1 < len(C) < len(labels)}


@pytest.mark.parametrize("case", feeders.OLD_CASES)
def test_street_keeps_meters_services_and_groups(case):
    """The street version keeps every meter, customer count and service line; every node has at most two lines out;
    all meters hang on one trunk; the old laterals' groups of meters are still groups."""
    star, street = feeders.star_plan(case), feeders.street_plan(case)
    meters = star.below(feeders.ROOT)
    assert sorted(street.below(feeders.ROOT)) == sorted(meters)
    for m in meters:
        assert street.nodes[m]["customers"] == star.nodes[m]["customers"]
        assert street.nodes[m]["r"] >= star.nodes[m]["r"]-1e-12          # the service line (plus merged street, if any)
    assert len(street.children(feeders.ROOT)) == 1
    assert all(len(street.children(b)) in (0, 2) for b in street.nodes if b != feeders.ROOT)
    laterals = {frozenset(star.below(b)) for b in star.nodes if b != feeders.ROOT and not star.nodes[b]["customers"]
                and star.nodes[b]["parent"] != feeders.ROOT}
    assert laterals <= {frozenset(street.below(b)) for b in street.nodes}


def test_random_plan_is_reproducible_and_reduced():
    """Same seed, same feeder; streams differ; the meter count is in range; no node that no meter can place."""
    for size, (low, high) in feeders.SIZES.items():
        for seed in range(5):
            a, b = feeders.random_plan(seed, size), feeders.random_plan(seed, size)
            assert a.nodes == b.nodes
            n = len(a.below(feeders.ROOT))
            assert low <= n <= high
            for node, v in a.nodes.items():
                if node != feeders.ROOT and v["customers"] == 0:
                    assert len(a.children(node)) >= 2
            assert len(a.children(feeders.ROOT)) == 1
    assert feeders.random_plan(1, "small").nodes != feeders.random_plan(1, "small", stream="test").nodes


def test_power_flow_matches_the_backend():
    net = feeders.to_network(feeders.random_plan(3, "medium"))
    labels = feeders.labels_of(net)
    rng = np.random.default_rng(0)
    p = rng.uniform(0, .02, (8, len(labels))); q = .3*p; v0 = np.full(8, 1.02)
    order, v, s, _ = physics.power_flow(net, labels, p, q, v0)
    ours = v[:, [order.index(l) for l in labels]]
    theirs, _ = backend._ac_voltages(net, labels, p, q, v0)
    np.testing.assert_allclose(ours, theirs, atol=1e-9)
    assert np.all(s.real > p.sum(axis=1))                                # the PCC supplies loads and losses


def test_unmetered_load_enters_the_balance_and_lowers_voltages():
    net = feeders.to_network(feeders.street_plan("paper15"))
    labels = feeders.labels_of(net)
    p = np.full((4, len(labels)), .01); q = .3*p; v0 = np.full(4, 1.02)
    bus = build.source_bus(net, "end")
    order, v, s, _ = physics.power_flow(net, labels, p, q, v0)
    _, v2, s2, _ = physics.power_flow(net, labels, p, q, v0, (bus, np.full(4, .02), np.full(4, .005)))
    extra = (s2-s).real
    assert np.all(extra > .02) and np.all(extra < .022)                   # the load itself plus a little loss
    path = build.source_path(net, bus)
    drop = (v-v2)[:, [order.index(l) for l in labels]].mean(axis=0)
    inside = set(path[-1])
    assert min(drop[i] for i in inside) > max(drop[i] for i in set(range(len(labels)))-inside)
    assert all(set(b) < set(a) for a, b in zip(path, path[1:])) and len(path[0]) == len(labels)


def test_quarter_hours_keep_the_hour():
    rng = np.random.default_rng(1)
    hourly = rng.uniform(0, 2, (30, 5))
    fine = loads.quarter_hours(hourly, rng)
    np.testing.assert_allclose(physics.hourly(fine), hourly)
    assert fine.min() >= 0


def test_sources_are_balanced():
    """Over nine feeders every location meets every share with every kind once."""
    for where in build.SOURCES[1:]:
        got = {(build.source_of(i, where)["share"], build.source_of(i, where)["kind"]) for i in range(9)}
        assert len(got) == 9
    assert build.source_of(0, "none") is None


@needs_data
def test_pools_share_no_customer_and_profiles_have_mean_one():
    assert not set(loads.pool("dev")) & set(loads.pool("test"))
    for kind in build.KINDS:
        p, _ = loads.source_profile(kind, 9*96, "dev", 0, [1, 2])
        assert abs(p.mean()-1) < 1e-12 and p.min() >= 0


@needs_data
def test_twins_differ_only_by_the_unmetered_load():
    plan = feeders.street_plan("soumalas11")
    a = build.simulate(plan, "dev", 1)
    b = build.simulate(plan, "dev", 1, dict(where="end", kind="block", share=.05))
    for k in ("P", "Q", "V0"):
        np.testing.assert_array_equal(a["true"][k], b["true"][k])
    assert a["truth"] == b["truth"]
    on = b["source"]["P"] > 0
    assert np.all(b["true"]["V"][on] < a["true"]["V"][on]) and np.allclose(b["true"]["V"][~on], a["true"]["V"][~on], atol=1e-9)
    share = b["source"]["P"].mean()/a["true"]["P"].sum(axis=1).mean()
    assert abs(share-.05) < 1e-12
    # star and street twins share their loads
    np.testing.assert_array_equal(a["true"]["P"], build.simulate(feeders.star_plan("soumalas11"), "dev", 1)["true"]["P"])


@needs_data
def test_input_has_the_fields_of_the_studies():
    rec = build.simulate(feeders.street_plan("paper15"), "dev", 2)
    rec.update(name="x", group="x", set="named", rejected=0)
    inp = build.to_input(rec)
    assert inp.hist["P"].shape == (192, 15) and inp.hist["Ppcc"].shape == (192,)
    assert inp.day["P"].shape == (24, 15)
    fine = build.to_input(rec, per_hour=4)
    assert fine.hist["P"].shape == (192, 15) and fine.day["P"].shape == (96, 15)
    again = build.to_input(rec)
    np.testing.assert_array_equal(inp.hist["Y"], again.hist["Y"])        # readings are reproducible
    err = inp.hist["P"]/physics.hourly(rec["true"]["P"])[:192]-1
    assert .003 < err.std() < .009
    assert {tuple(sorted(C)) for C in inp.truth_key if 1 < len(C) < 15} == {
        tuple(sorted(C)) for C in rec["truth"] if 1 < len(C) < 15}
