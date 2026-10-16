"""Analytic and independent finite-difference audits of bounded QP intervals."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import LinearConstraint

from . import ev_mobile_v4_bounded as b


def _pair(a, B, J, domain, **kwargs):
    B = np.asarray(B, float)
    return b.build_pair('a', 'b', a, B, J, np.zeros((1, B.shape[1])), domain, **kwargs)


def test_scalar_active_box_exact_distance_and_price():
    pair = _pair([0.], [[1.]], [[-1.]], b.make_domain([-.5], [.5]))
    for u in [-2., -.5, -.1, 0., .4, .5, 1.7]:
        result = b.solve_pair(pair, [[u]])
        truth = .5*max(abs(u)-.5, 0.)**2
        assert result['explanation_audit']['feasible']
        assert result['dual_lower'] <= truth+1e-12 <= result['primal_upper']+1e-12
        assert result['primal'] == pytest.approx(truth, abs=1e-11)
        assert result['completed']
        assert result['price'][0] == pytest.approx(np.sign(u)*max(abs(u)-.5, 0.), abs=1e-8)


def test_rank_deficient_parameters_have_unique_residual_and_price():
    domain = b.make_domain([-2., -2.], [2., 2.])
    pair = _pair([0., 0.], np.eye(2), -np.ones((2, 2)), domain)
    result = b.solve_pair(pair, [[1., -1.]])
    assert result['primal'] == pytest.approx(1.)
    np.testing.assert_allclose(result['residual'], [1., -1.], atol=1e-10)
    np.testing.assert_allclose(result['price'], [1., -1.], atol=1e-10)
    assert result['completed']


def test_shared_two_time_parameter_retains_balanced_excitation():
    domain = b.make_domain([-.2], [.2])
    pair = _pair([0., 0.], np.eye(2), -np.ones((2, 1)), domain)
    assert b.solve_pair(pair, [[1., 0.]])['primal'] == pytest.approx(.34)
    assert b.solve_pair(pair, [[.5, .5]])['primal'] == pytest.approx(.09)
    # Independent sources/lines at both times would be a different model.
    free_time = _pair([0., 0.], np.eye(2), -np.eye(2), b.make_domain([-2., -2.], [2., 2.]))
    assert b.solve_pair(free_time, [[1., 0.]])['primal'] < 1e-20


def test_original_ratio_boundary_is_active_not_ignored():
    domain = b.physical_delta_domain([1.], [.4], np.zeros((0, 2)),
        r_bounds=(.1, 2.), x_bounds=(.05, 2.), source_bounds=(-.5, .5), ratio_bounds=(.25, .5))
    pair = _pair([0., -.6], np.zeros((2, 1)), np.eye(2), domain)
    result = b.solve_pair(pair, [[0.]])
    absolute = np.array([1., .4])+result['z']
    np.testing.assert_allclose(absolute, [1.2, .6], atol=2e-7)
    assert absolute[1] == pytest.approx(.5*absolute[0], abs=1e-8)
    assert result['primal'] == pytest.approx(.1, abs=1e-9)
    assert result['primal_upper']-result['dual_lower'] < 1e-7


def test_source_coordinate_order_and_original_bounds():
    center = np.array([[.1, -.1], [.2, -.2]])
    domain = b.physical_delta_domain([.2], [.15], center,
        r_bounds=(.05, .5), x_bounds=(.02, .5), source_bounds=(-.3, .3), ratio_bounds=(.1, 2.))
    np.testing.assert_allclose(domain.lower[2:], (-.3-center).ravel())
    np.testing.assert_allclose(domain.upper[2:], (.3-center).ravel())
    assert domain.audit(np.zeros(6))['feasible']
    assert not domain.audit([0., 0., .3, 0., 0., 0.])['feasible']


def test_support_lagrange_remainder_does_not_use_LP_primal(monkeypatch):
    domain = b.make_domain([-1., -1.], [1., 1.], LinearConstraint([[1., 1.]], -np.inf, .2))
    monkeypatch.setattr(b, 'linprog', lambda *args, **kwargs: SimpleNamespace(
        status=1, success=False, fun=-1e20, ineqlin=SimpleNamespace(marginals=np.array([-3.]))))
    result = b.support_lower(domain, [-1., -2.])
    # Independent vertex/edge enumeration is only an audit of the bound.
    values = [np.array([-1., -1.]), np.array([-1., 1.]), np.array([1., -1.]),
              np.array([-.8, 1.]), np.array([1., -.8])]
    truth = min(np.array([-1., -2.])@z for z in values)
    assert result['lower'] <= truth
    assert result['LP_primal'] == -2e20
    assert result['lower'] > -100.


def test_arbitrary_residual_support_is_global_with_nonzero_QP_gap():
    domain = b.make_domain([0.], [1.])
    pair = _pair([.3], [[1.]], [[1.]], domain, self_interval=(.1, .2),
                 affine_offset=.7, affine_control=[-.4], affine_reference=[[.5]])
    result = b.solve_pair(pair, [[-.7]])
    line = result['support_affine']
    for x in np.linspace(-2., 2., 41):
        target = .3+x
        raw = .5*(target+np.clip(-target, 0., 1.))**2
        corrected_lower = raw-.2+.7-.4*(x-.5)
        assert line['intercept']+line['coefficient'][0]*x <= corrected_lower+2e-10


def test_affine_parent_value_and_gradient_alignment():
    pair = _pair([.4], [[2.]], [[-1.]], b.make_domain([-.1], [.1]),
                 self_interval=(.2, .2), affine_offset=4.3, affine_control=[-1.2],
                 affine_reference=[[.2]])
    result = b.solve_pair(pair, [[.2]])
    assert result['score_lower'] == pytest.approx(result['dual_lower']-.2+4.3)
    assert result['score_upper'] == pytest.approx(result['primal_upper']-.2+4.3)
    eps = 1e-5
    up = b.solve_pair(pair, [[.2+eps]])['score_upper']
    down = b.solve_pair(pair, [[.2-eps]])['score_upper']
    assert result['price'][0] == pytest.approx((up-down)/(2*eps), rel=1e-5, abs=1e-7)
    np.testing.assert_allclose(result['price'], result['raw_price']-1.2)


def test_self_interval_is_reference_a_and_aggregation_preserves_bounds():
    domain = b.make_domain([], [])
    pairs = [b.build_pair(a, c, [1.], [[1.]], np.zeros((1, 0)), [[0.]], domain,
                         self_interval=(2., 3.) if a == 'a' else (5., 6.))
             for a, c in [('a', 'b'), ('b', 'a')]]
    result = b.evaluate_pairs(pairs, [[0.]])
    assert result['lower'] == pytest.approx(-4.)
    assert result['upper'] == pytest.approx(-3.)
    assert result['solutions'][0]['reference_a_self_interval'] == (2., 3.)


def test_empty_infeasible_or_invalid_domains_fail_explicitly():
    with pytest.raises(ValueError, match='empty'):
        b.make_domain([0.], [1.], LinearConstraint([[1.]], 2., np.inf))
    with pytest.raises(ValueError, match='anchor'):
        b.make_domain([0.], [1.], anchor=[2.])
    with pytest.raises(ValueError, match='matching ordered'):
        b.make_domain([1.], [0.])
    with pytest.raises(ValueError, match='positive absolute R'):
        b.physical_delta_domain([.1], [.1], [0.], r_bounds=(0., 1.),
                               x_bounds=(0., 1.), source_bounds=(-1., 1.), ratio_bounds=(.1, 2.))


def test_fixed_parameter_coordinates_are_removed_without_altering_fit():
    pair = _pair([2., 1.], np.eye(2), np.eye(2), b.make_domain([-.5, .2], [-.5, .2]))
    result = b.solve_pair(pair, [[0., 0.]])
    assert result['primal'] == pytest.approx(.5*(1.5**2+1.2**2))
    assert result['completed']
    assert result['solver']['backend_used'] == 'none'


def test_gurobi_and_scipy_agree_on_ratio_active_QP():
    from .ev_original_mip_backend import gurobi_available
    if not gurobi_available()['available']: pytest.skip('Gurobi unavailable')
    domain = b.make_domain([-.4, -.4], [.4, .4], LinearConstraint([[1., -1.]], -.1, .1))
    pair = _pair([-.8, .2], [[1.], [.3]], np.eye(2), domain)
    scipy = b.solve_pair(pair, [[.2]], backend='scipy')
    gurobi = b.solve_pair(pair, [[.2]], backend='gurobi')
    assert scipy['primal'] == pytest.approx(gurobi['primal'], abs=1e-8)
    assert gurobi['completed']
    assert gurobi['explanation_audit']['feasible']
