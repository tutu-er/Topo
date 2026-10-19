"""Full legacy oracle, finite differences, zero edges, and fit equivalence."""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from .. import likelihood as lk
from ..ev_rx_fit import _RxInputProblem
from . import fitting
from .derivatives import fit_response_derivatives, optimizer_problem_jacobian
from .networks import build_network
from .representation import encode_physical_tree
from .test_fitting import fixture as small_fixture


def native(name, H):
    net = build_network(name); ids = net['observation_bus_ids']
    encoded = encode_physical_tree(net['branches'], ids, net['root_bus'], net['base_kv'],
                                   net['base_kw'], net['source_bus'])
    buses = {row['bus_id']:row for row in net['buses']}
    rng = np.random.default_rng(2301)
    p = np.array([buses[i]['pd_kw'] for i in ids])[None, :]*rng.uniform(.3, .6, (H, len(ids)))/net['base_kw']
    q = np.array([buses[i]['qd_kvar'] for i in ids])[None, :]*rng.uniform(.3, .6, (H, len(ids)))/net['base_kw']
    sp, sq = np.full(H, -.0002), np.full(H, .0001)
    phy = lk.ac_predict(encoded.candidate, encoded.r, encoded.x, p, q, sp, sq)
    D = lk.Observations(p, q, phy['v'], master_p=phy['master_p'], master_q=phy['master_q'])
    meter = lk.MeasurementModel(.0001, .0001, .0005, master_p_sigma=.0005, master_q_sigma=.0005)
    options = dict(r_bounds=(0., max(float(encoded.r.sum()), .05)),
                   x_bounds=(0., max(float(encoded.x.sum()), .03)), ratio_bounds=(.05, 5.),
                   source_p_bounds=(-.1, .1), source_q_bounds=(-.1, .1), voltage_bounds=(.8, 1.2))
    state = dict(r=encoded.r, x=encoded.x, source_p=sp, source_q=sq)
    return encoded.candidate, D, meter, options, state, phy


@pytest.mark.parametrize('name', ['case33', 'cigre_residential', 'paper11', 'paper15'])
@pytest.mark.parametrize('H', [3, 12])
def test_selected_rhs_all_elements_equal_legacy_full_jacobian(name, H):
    c, D, meter, options, state, phy = native(name, H)
    old = lk._response_derivatives(c, phy, True)
    for lines in (True, False):
        indices = np.r_[np.arange(2*c.m), np.arange(2*c.m+2*c.n, 2*c.m+2*c.n+2)] if lines else np.arange(2*c.m+2*c.n, 2*c.m+2*c.n+2)
        selected = fit_response_derivatives(c, phy, include_lines=lines)
        for key in selected:
            assert np.max(abs(selected[key]-old[key][..., indices])) < 1e-12
    free = _RxInputProblem(c, D, meter, initial=state, **options)
    assert np.max(abs(optimizer_problem_jacobian(free, free.start)-free.jacobian(free.start))) < 1e-10
    fixed = fitting._SourceInputProblem(c, D, meter, initial=None,
        fixed_r=state['r'], fixed_x=state['x'], call_counter=None, **options)
    assert np.max(abs(optimizer_problem_jacobian(fixed, fixed.start)-fixed.jacobian(fixed.start))) < 1e-10


@pytest.mark.parametrize('name', ['case33', 'cigre_residential'])
def test_native_source_pq_finite_difference_and_zero_edge_r(name):
    c, D, meter, options, state, phy = native(name, 3)
    jac = fit_response_derivatives(c, phy, include_lines=True)
    h = 1e-7
    for label, column in (('source_p', -2), ('source_q', -1)):
        high, low = dict(state), dict(state)
        high[label] = state[label]+h; low[label] = state[label]-h
        a = lk.ac_predict(c, high['r'], high['x'], D.p, D.q, high['source_p'], high['source_q'])
        b = lk.ac_predict(c, low['r'], low['x'], D.p, D.q, low['source_p'], low['source_q'])
        for key in ('v', 'master_p', 'master_q'):
            assert np.max(abs((a[key]-b[key])/(2*h)-jac[key][..., column])) < 2e-7
    zero = np.flatnonzero((state['r'] == 0)&(state['x'] == 0))
    assert len(zero)
    edge = int(zero[0]); one = state['r'].copy(); two = state['r'].copy()
    one[edge] += h; two[edge] += 2*h
    a = lk.ac_predict(c, one, state['x'], D.p, D.q, state['source_p'], state['source_q'])
    b = lk.ac_predict(c, two, state['x'], D.p, D.q, state['source_p'], state['source_q'])
    for key in ('v', 'master_p', 'master_q'):
        numeric = (-3*phy[key]+4*a[key]-b[key])/(2*h)
        assert np.max(abs(numeric-jac[key][..., edge])) < 3e-7


@pytest.mark.parametrize('fixed', [False, True])
def test_complete_small_fit_same_loss_status_and_iterations_as_full_rhs(monkeypatch, fixed):
    c, D, meter, opts, state = small_fixture(zero_stub=True)
    kwargs = dict(opts, initial=state)
    if fixed:
        kwargs.update(fixed_r=state['r'], fixed_x=state['x'])
    fast = fitting.fit_rx_inputs(c, D, meter, **kwargs)
    def full(problem, vector):
        return csr_matrix(problem.jacobian(vector))
    monkeypatch.setattr(fitting, '_sparse_jacobian', full)
    original = fitting.fit_rx_inputs(c, D, meter, **kwargs)
    assert fast.success == original.success
    assert fast.status == original.status
    assert fast.nfev == original.nfev
    assert abs(fast.loss-original.loss) < 1e-10
    assert abs(fast.optimality-original.optimality) < 1e-8
    assert np.max(abs(fast.predicted['v']-original.predicted['v'])) < 1e-12


def test_collapsed_box_cone_corner_free_mask_matches_full_jacobian():
    c, D, meter, _, _ = small_fixture()
    # Both original boxes are strict, while their cone intersection fixes
    # R=X=.02. Only the per-time source coordinates remain free.
    problem = _RxInputProblem(c, D, meter,
        r_bounds=(.01, .02), x_bounds=(.02, .03), ratio_bounds=(.5, 1.),
        source_p_bounds=(-.1, .1), source_q_bounds=(-.1, .1), voltage_bounds=(.8, 1.2))
    assert not np.any(problem.free[:2*c.m])
    assert problem.width == 2*D.t
    actual = optimizer_problem_jacobian(problem, problem.start)
    expected = problem.jacobian(problem.start)
    assert np.max(abs(actual-expected)) < 1e-10
