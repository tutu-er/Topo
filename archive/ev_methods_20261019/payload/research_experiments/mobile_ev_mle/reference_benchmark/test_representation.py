"""Independent physical equivalence tests for exact zero virtual meter leaves."""
import numpy as np
import pytest

from .. import likelihood as lk
from .representation import (encode_physical_tree, impedance_base_ohm,
                              physical_tree_ac, public_fit_options)


def branch(u, v, r=.04, x=.02, **extra):
    return dict(from_bus=u, to_bus=v, r_ohm=r, x_ohm=x, **extra)


def compare(branches, measured, *, root=0, source=None, base_kw=100., base_kv=.4):
    e = encode_physical_tree(branches, measured, root, base_kv, base_kw, source_bus=source)
    rng = np.random.default_rng(601)
    p = rng.uniform(.2, 1.2, (5, len(measured)))
    q = p*rng.uniform(.05, .5, p.shape)
    sp = np.array([-.5, .1, -.2, .0, .4]); sq = -.3*sp+.07
    root_v = np.linspace(.99, 1.02, len(p))
    physical = physical_tree_ac(branches, measured, root, base_kv, base_kw,
        p, q, e.source_bus, sp, sq, root_v)
    reduced = lk.ac_predict(e.candidate, e.r, e.x, p/base_kw, q/base_kw,
                            sp/base_kw, sq/base_kw, root_v)
    assert np.max(abs(physical['v']-reduced['v'])) < 2e-11
    ids = list(e.candidate.terminal_edges)
    assert np.max(abs(physical['measured_voltage']-reduced['voltage'][:, ids])) < 2e-11
    for key in ('master_p', 'master_q'):
        assert np.max(abs(physical[key]-reduced[key])) < 2e-11
    assert physical['ac_equation_residual'] < 2e-11
    assert physical['power_balance_residual'] < 2e-11
    return e, physical, reduced


def test_internal_loads_are_exact_zero_virtual_leaves():
    edges = [branch(0, 1), branch(1, 2), branch(2, 3), branch(2, 4)]
    e, _, _ = compare(edges, [1, 2, 3, 4], source=1)
    singleton = {tuple(e.candidate.supports[k]): k for k in e.metadata['virtual_zero_edges']}
    assert (0,) in singleton and (1,) in singleton
    assert e.r[singleton[(0,)]] == e.x[singleton[(0,)]] == 0.
    assert e.r[singleton[(1,)]] == e.x[singleton[(1,)]] == 0.


def test_root_multiple_outlets_has_zero_full_support():
    e, _, _ = compare([branch(0, 1), branch(0, 2), branch(2, 3)], [1, 2, 3])
    assert e.source_bus == 0
    assert e.r[0] == e.x[0] == 0.
    assert 0 in e.metadata['virtual_zero_edges']


def test_series_aggregation_and_unobserved_empty_branch():
    edges = [branch(0, 1, .01, .02), branch(1, 2, .04, .03),
             branch(2, 3), branch(2, 4), branch(1, 9)]
    e, _, _ = compare(edges, [3, 4], source=2)
    assert e.r[0] == pytest.approx(.05/e.zbase_ohm)
    assert e.x[0] == pytest.approx(.05/e.zbase_ohm)
    assert e.metadata['edge_physical_indices'][0] == [0, 1]
    assert e.metadata['removed_zero_injection_empty_support_edges'] == [4]
    assert e.physical_edge_to_candidate_edge[4] is None


def test_observed_order_and_closed_status():
    edges = [branch(0, 1), branch(1, 2), branch(1, 3), branch(2, 3, status=False)]
    e, _, _ = compare(edges, [3, 1, 2])
    assert e.observation_bus_ids == (3, 1, 2)
    assert e.source_bus == 1


def test_zero_physical_edges_and_nonzero_source_q():
    e, _, _ = compare([branch(0, 1, 0., 0.), branch(1, 2), branch(1, 3)], [1, 2, 3])
    assert e.r[0] == e.x[0] == 0.


def test_source_cannot_be_inside_an_aggregated_full_support_chain():
    edges = [branch(0, 1), branch(1, 2), branch(2, 3), branch(2, 4)]
    with pytest.raises(ValueError, match='LCA'):
        encode_physical_tree(edges, [3, 4], 0, .4, 100., source_bus=1)


@pytest.mark.parametrize('edges', [
    [branch(0, 1), branch(1, 2), branch(2, 0)],
    [branch(0, 1), branch(0, 1)],
    [branch(0, 1), branch(2, 3)],
])
def test_invalid_physical_graph_rejected(edges):
    with pytest.raises(ValueError):
        encode_physical_tree(edges, [1, 2], 0, .4, 100.)


def test_source_sign_and_base_change_do_not_change_physics():
    edges = [branch(0, 1), branch(1, 2), branch(1, 3)]
    _, a, _ = compare(edges, [1, 2, 3], base_kw=100.)
    _, b, _ = compare(edges, [1, 2, 3], base_kw=1000.)
    assert np.max(abs(a['v']-b['v'])) < 2e-12
    assert np.max(abs(a['master_p']*100.-b['master_p']*1000.)) < 2e-10
    assert np.max(abs(a['master_q']*100.-b['master_q']*1000.)) < 2e-10
    assert impedance_base_ohm(12.66, 10000.) == pytest.approx(16.02756)


def test_exact_power_derivatives_with_internal_measurement():
    e, _, _ = compare([branch(0, 1), branch(1, 2), branch(1, 3)], [1, 2, 3])
    p = np.array([[.01, .02, .015]]); q = .3*p
    phy = lk.ac_predict(e.candidate, e.r, e.x, p, q, -.002, .001)
    jac = lk._response_derivatives(e.candidate, phy, True)
    h = 1e-6
    for j in range(3):
        delta = np.zeros_like(p); delta[0, j] = h
        high = lk.ac_predict(e.candidate, e.r, e.x, p+delta, q, -.002, .001)
        low = lk.ac_predict(e.candidate, e.r, e.x, p-delta, q, -.002, .001)
        assert np.max(abs((high['v']-low['v'])/(2*h)-jac['v'][:, :, 2*e.candidate.m+j])) < 2e-8
        assert np.max(abs((high['master_q']-low['master_q'])/(2*h)-jac['master_q'][:, 2*e.candidate.m+j])) < 2e-8


def test_public_domain_has_zero_lower_bound_and_no_truth_argument():
    options = public_fit_options(base_kv=.4, base_kw=200., r_ohm_upper=1., x_ohm_upper=.5,
        ratio_bounds=(.01, 5.), source_p_kw_bound=30., source_q_kvar_bound=10., max_series_edges=4)
    assert options['r_bounds'] == pytest.approx((0., 5.))
    assert options['x_bounds'] == pytest.approx((0., 2.5))
    assert options['source_p_bounds'] == (-.15, .15)
    assert options['source_q_bounds'] == (-.05, .05)


@pytest.mark.parametrize('name', ['case33', 'cigre_residential', 'paper11', 'paper15'])
def test_exported_reference_networks_preserve_full_ac(name):
    from .networks import build_network
    net = build_network(name)
    ids = net['observation_bus_ids']; buses = {b['bus_id']: b for b in net['buses']}
    e = encode_physical_tree(net['branches'], ids, net['root_bus'], net['base_kv'],
                             net['base_kw'], net['source_bus'], name=name)
    p0 = np.array([buses[b]['pd_kw'] for b in ids])
    q0 = np.array([buses[b]['qd_kvar'] for b in ids])
    p = np.array([.4, .55, .7])[:, None]*p0
    q = np.array([.45, .65, .6])[:, None]*q0
    sp = np.array([-.01, .0, .01])*p0.sum()
    sq = np.array([.003, -.002, .004])*p0.sum()
    original = physical_tree_ac(net['branches'], ids, net['root_bus'], net['base_kv'],
        net['base_kw'], p, q, e.source_bus, sp, sq)
    mapped = lk.ac_predict(e.candidate, e.r, e.x, p/net['base_kw'], q/net['base_kw'],
                           sp/net['base_kw'], sq/net['base_kw'])
    assert np.max(abs(original['v']-mapped['v'])) < 2e-10
    assert np.max(abs(original['master_p']-mapped['master_p'])) < 2e-10
    assert np.max(abs(original['master_q']-mapped['master_q'])) < 2e-10
