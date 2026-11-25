"""Tests of the source-aware method (identification/source.py)."""
import numpy as np
import pytest

from rnj_wzzt_dr.identification import revised as rv, source as src
from rnj_wzzt_dr.identification.trees import basis, canonical

N = 6
TREE = [(0, 1), (0, 1, 2), (3, 4, 5), (4, 5)]


def make(path, seed=0, noise=1e-5, T=192):
    """Linear data of a known tree with the source below the last line of `path` (a set of meters, or None)."""
    rng = np.random.default_rng(seed)
    p = np.abs(rng.normal(.02, .01, (T, N)))
    q = .3*p*np.exp(.3*rng.normal(size=p.shape))
    up = .02*(rng.random(T) > .5)*np.abs(rng.normal(1, .2, T))
    uq = .2*up
    fam = canonical([*TREE, *basis(N)])
    y = np.zeros((T, N))
    for k, C in enumerate(fam):
        r, x = .05+.01*k, .02+.004*k
        carries = len(C) == N or (path is not None and set(path) <= set(C))
        flow_p, flow_q = p[:, list(C)].sum(1)+carries*up, q[:, list(C)].sum(1)+carries*uq
        y[:, list(C)] += (r*flow_p+x*flow_q)[:, None]
    v0sq = np.full(T, 1.04)
    y = y+noise*rng.normal(size=y.shape)
    return rv.Series(p, q, y, v0sq), up, uq


def test_projection_equals_free_coefficients():
    """Projecting the source out gives the residual of a fit with free source coefficients for every meter."""
    data, up, uq = make((0, 1, 2), noise=1e-3)
    T = len(up)
    d = rv.Design(data, src.ProjectedTransform(T, N, 0., True, [up, uq]), 1.)
    F = d.columns(canonical([*TREE, *basis(N)]))
    sse = np.sum((d.y-F@np.linalg.lstsq(F, d.y, rcond=None)[0])**2)
    plain = rv.Design(data, rv.Transform(T, N, 0., True), 1.)
    G = plain.columns(canonical([*TREE, *basis(N)]))
    extra = np.zeros((T, N, 2*N))
    for i in range(N):
        extra[:, i, i], extra[:, i, N+i] = up-up.mean(), uq-uq.mean()
    H = np.hstack((G, extra.reshape(T*N, -1)))
    sse2 = np.sum((plain.y-H@np.linalg.lstsq(H, plain.y, rcond=None)[0])**2)
    assert sse == pytest.approx(sse2, rel=1e-8)


def test_leaf_columns_are_the_physics():
    """A line with the source leaf carries the source; its rows are its meters only."""
    data, up, uq = make((4, 5))
    T = len(up)
    d = src.LeafDesign(data, rv.Transform(T, N, 0., False), 1., up, uq)
    cols = d.columns([(4, 5, N)])
    expect = np.zeros((T, N))
    expect[:, [4, 5]] = (data.p[:, 4]+data.p[:, 5]+up)[:, None]
    np.testing.assert_allclose(cols[:, 0], expect.ravel())
    G, b = d.gram([(4, 5, N), (0, 1)])
    F = d.columns([(4, 5, N), (0, 1)])
    np.testing.assert_allclose(G, F.T@F, rtol=1e-10)
    np.testing.assert_allclose(b, F.T@d.y, rtol=1e-10)


@pytest.mark.parametrize("place", [None, (0, 1, 2), (0, 1), (4, 5)])
def test_the_tree_and_the_place_are_found_together(place):
    data, up, uq = make(place, seed=3)
    T = len(up)
    d = src.LeafDesign(data, rv.Transform(T, N, 0., True), 1e-5, up, uq)
    lines = [*TREE, (0, 2), (3, 4), (1, 2, 3), (2, 3)]
    rec = src.select_with_source(d, lines, 2*np.log(T*N)/(T*N), 2., 60., N)
    tree, path = src.meters_only(rec["family"], N)
    assert {C for C in tree if 1 < len(C) < N} == set(TREE)
    expected = [C for C in sorted(TREE, key=lambda c: -len(c)) if place is not None and set(place) <= set(C)]
    assert path == [tuple(range(N)), *expected]
    assert src.locate(d, tree, N, 2.)[0][1] == place


def test_losses_match_the_power_flow():
    from rnj_wzzt_dr.experiments.case_library_20261125 import feeders, physics
    net = feeders.to_network(feeders.street_plan("paper15"))
    labels = feeders.labels_of(net)
    rng = np.random.default_rng(5)
    p = rng.uniform(.002, .02, (20, len(labels)))
    q = p*rng.uniform(.15, .4, p.shape)
    v0 = np.full(20, 1.02)
    order, v, s, _ = physics.power_flow(net, labels, p, q, v0)
    vm = v[:, [order.index(l) for l in labels]]
    lp, lq = src.losses(p, q, v0[:, None]**2-vm**2, v0**2)
    true = s.real-p.sum(1)
    assert np.max(np.abs(lp/true-1)) < .06                       # topology-free estimate within 6% of the loss


def test_completion_finds_a_missing_line_and_certifies():
    """A tree with a line missing from the pool is completed by the enumeration certificate."""
    from rnj_wzzt_dr.identification import source_cert as sc2
    data, up, uq = make((0, 1, 2), seed=4)
    T = len(up)
    alpha = 2*np.log(T*N)/(T*N)
    d = src.LeafDesign(data, rv.Transform(T, N, 0., True), 1e-5, up, uq)
    start = [C for C in src.attach([C for C in TREE if C != (4, 5)], N, (0, 1, 2)) if 1 < len(C) < N+1]
    fam = canonical([*start, *basis(N+1)])
    before = sc2.certify(d, fam, alpha, 2., N+1)
    assert before["best"] is not None and not before["certified"]
    done, rec = sc2.complete(d, fam, [*start], alpha, 2., N)
    tree, path = src.meters_only(done, N)
    assert {C for C in tree if 1 < len(C) < N} == set(TREE) and rec["certified"]
    assert path == [tuple(range(N)), (0, 1, 2)]
