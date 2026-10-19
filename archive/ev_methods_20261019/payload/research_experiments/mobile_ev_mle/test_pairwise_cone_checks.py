"""Gate-1 independent checks; no AC or operational action selection."""
from fractions import Fraction
import numpy as np
import pytest
from scipy.optimize import nnls
from .pairwise_cone_checks import pairwise_cone_distance, check_dual_candidate, envelope_price, _smooth_case, run_analytic_checks


def test_positive_other_cherry_is_a_legal_zero_distance_explanation():
    fit = pairwise_cone_distance([[1., 0.]], [[1., 0.]], np.empty((1, 0)), [1., 0.])
    assert fit['current_free_projection_distance_squared'] == 1.
    assert fit['distance_squared'] == 0.
    assert fit['eta'] == [1., 0.] and fit['residual'] == [0.]
    assert fit['primal_upper_bound'] == fit['dual_lower_bound'] == 0.
    assert fit['dual']['candidate_feasible_exact']


def test_nonnegative_eta_prevents_wrong_sign_free_compensation():
    fit = pairwise_cone_distance([[1., 0.], [1., 0.]], [[-1., 0.], [0., 0.]], np.empty((2, 0)), [1., 0.])
    assert fit['distance_squared'] == 2.
    assert fit['unconstrained_competitor_distance_squared'] == 1.
    assert fit['eta'] == [0., 0.]
    assert fit['dual']['competition_dot_exact'] == ['-1', '0']
    assert fit['dual_lower_bound'] == fit['primal_upper_bound'] == 2.


def test_rank_deficient_common_nuisance_is_projected_once():
    fit = pairwise_cone_distance(np.eye(2), [[1., 0.], [0., 0.]], [[0., 0., 0.], [1., 2., 0.]], [1., 1.])
    assert fit['nuisance_rank'] == 1 and fit['projected_competitor_rank'] == 1
    assert fit['current_free_projection_distance_squared'] == 1.
    assert fit['distance_squared'] == 0.
    np.testing.assert_array_equal(fit['delta_nu'], [-1., 0., 0.])
    assert fit['dual']['nuisance_dot_exact'] == ['0', '0', '0']


@pytest.mark.parametrize('ar,expected_rank,expected_eta', [
    ([[1., 2.], [0., 0.]], 1, [3., 0.]),
    ([[1., -1.], [0., 0.]], 1, [3., 0.]),
    ([[0., 0.], [0., 0.]], 0, [0., 0.])])
def test_dependent_opposite_and_zero_competitor_columns(ar, expected_rank, expected_eta):
    fit = pairwise_cone_distance(np.eye(2), ar, np.empty((2, 0)), [3., 1.])
    assert fit['projected_competitor_rank'] == expected_rank
    np.testing.assert_array_equal(fit['eta'], expected_eta)
    assert fit['distance_squared'] == (10. if expected_rank == 0 else 1.)
    assert fit['dual']['candidate_feasible_exact']


def test_negative_target_uses_opposite_positive_eta_not_negative_eta():
    fit = pairwise_cone_distance([[-1., 0.], [1., 0.]], [[1., -1.], [0., 0.]], np.empty((2, 0)), [1., 0.])
    assert fit['eta'] == [0., 1.] and fit['distance_squared'] == 1.
    assert fit['dual']['competition_dot_exact'] == ['0', '0']


def test_no_rank_threshold_drops_a_tiny_nonzero_nuisance_column():
    fit = pairwise_cone_distance(np.eye(2), np.zeros((2, 2)), [[1e-14, 0.], [0., 1.]], [1., 2.])
    assert fit['nuisance_rank'] == 2 and fit['distance_squared'] == 0.
    assert abs(fit['delta_nu'][0]) >= 1e14
    assert fit['primal_upper_bound'] >= fit['dual_lower_bound'] == 0.
    assert Fraction.from_float(fit['primal_dual_gap']) >= Fraction.from_float(fit['primal_upper_bound'])-Fraction.from_float(fit['dual_lower_bound'])



def test_strictly_positive_competing_witness_also_obeys_finite_lower_bound():
    fit = pairwise_cone_distance(np.eye(2), np.eye(2), np.empty((2, 0)), [.003, .002], eta_lower=[1e-6, 1e-6])
    assert fit['current_free_projection_distance_squared'] == pytest.approx(1.3e-5)
    assert fit['distance_squared'] == 0.
    assert fit['eta'] == [.003, .002]
    assert all(1e-6 <= v <= .1 for v in fit['eta'])
    assert fit['dual']['candidate_feasible_exact']


def test_finite_positive_lower_bound_is_not_the_closure():
    args = (np.eye(2), np.eye(2), np.empty((2, 0)), [0., 1.])
    closure = pairwise_cone_distance(*args)
    bounded = pairwise_cone_distance(*args, eta_lower=[.2, .3])
    assert closure['distance_squared'] == 0.
    assert bounded['distance_squared'] == pytest.approx(.04)
    assert bounded['eta'] == [.2, 1.]
    assert bounded['domain'] == 'finite_lower_bound_translated_cone'
    assert Fraction(bounded['primal_upper_bound_exact']) >= Fraction(bounded['dual']['candidate_value_exact'])


def test_independent_scipy_nnls_oracle_on_integer_and_rank_deficient_matrices():
    rng = np.random.default_rng(71041)
    for index in range(24):
        a_s = rng.integers(-3, 4, (5, 2)).astype(float)
        a_r = rng.integers(-3, 4, (5, 2)).astype(float)
        b = rng.integers(-2, 3, (5, 2)).astype(float)
        if index % 3 == 0:
            b[:, 1] = 2*b[:, 0]
        if index % 4 == 0:
            a_r[:, 1] = -a_r[:, 0]
        zeta = np.array([1., .5]);lower = np.array([.125, .25])
        fit = pairwise_cone_distance(a_s, a_r, b, zeta, eta_lower=lower)
        # Independent floating projection and NNLS, rather than face enumeration.
        projected_ar = a_r-b@np.linalg.lstsq(b, a_r, rcond=None)[0]
        raw = a_s@zeta-a_r@lower
        projected_raw = raw-b@np.linalg.lstsq(b, raw, rcond=None)[0]
        eta_excess, norm = nnls(projected_ar, projected_raw, maxiter=100)
        assert fit['distance_squared'] == pytest.approx(norm**2, rel=2e-11, abs=2e-11)
        eta, delta = np.array(fit['eta']), np.array(fit['delta_nu'])
        assert np.all(eta >= lower)
        np.testing.assert_allclose(a_s@zeta-a_r@eta+b@delta, fit['residual'], atol=2e-12)
        assert fit['dual']['candidate_feasible_exact']
        exact_dual = Fraction(fit['dual']['candidate_value_exact'])
        assert Fraction.from_float(fit['dual_lower_bound']) <= exact_dual
        assert Fraction.from_float(fit['primal_upper_bound']) >= Fraction(fit['primal_upper_bound_exact']) >= exact_dual
        assert Fraction(fit['exact_primal_dual_gap']) == Fraction(fit['exact_gap_identity']) >= 0


def test_tolerance_positive_dual_dot_is_rejected_with_zero_fallback():
    candidate = check_dual_candidate([[1., 0.]], np.empty((1, 0)), [1.], [1e-12])
    assert 0 < candidate['competition_dot'][0] < 1e-10
    assert candidate['candidate_value'] > 0.
    assert not candidate['candidate_feasible_exact']
    assert candidate['valid_lower_bound'] == 0.
    assert candidate['valid_lower_bound_source'] == 'unconditional_zero_dual'
    assert candidate['dual_vector_for_valid_bound_exact'] == ['0']


def test_tiny_nonzero_nuisance_dual_dot_is_also_rejected():
    candidate = check_dual_candidate(np.zeros((1, 2)), [[1.]], [1.], [1e-12])
    assert candidate['nuisance_dot'][0] > 0.
    assert not candidate['candidate_feasible_exact'] and candidate['valid_lower_bound'] == 0.


def test_envelope_includes_competitor_and_rotating_nuisance_terms():
    a_s, a_r, b = _smooth_case(0.)
    price = envelope_price(a_s, a_r, b, [1., 0.], [[.2, 0.], [.1, 0.], [.3, 0.]], [[0., 0.], [.1, 0.], [0., 0.]], [[0.], [.2], [0.]])
    assert price['fit']['strict_complementarity_exact']
    assert price['fit']['binding_coordinates'] == [1]
    assert price['source_term'] == pytest.approx(.2)
    assert price['competitor_term'] == pytest.approx(-.2)
    assert price['nuisance_term'] == pytest.approx(-.8)
    assert price['price'] == pytest.approx(-.8)
    for h in (1e-3, 5e-4, 1e-4):
        minus = pairwise_cone_distance(*_smooth_case(-h), [1., 0.])
        plus = pairwise_cone_distance(*_smooth_case(h), [1., 0.])
        fd = (plus['distance_squared']-minus['distance_squared'])/(2*h)
        assert abs(fd-price['price']) < 2e-7
        assert minus['binding_coordinates'] == plus['binding_coordinates'] == [1]
        assert minus['nuisance_rank'] == plus['nuisance_rank'] == 1
        assert abs(fd-(price['price']-price['nuisance_term'])) > .7
        assert abs(fd-(price['price']-price['competitor_term'])) > .19



@pytest.mark.parametrize('free_count', [0, 2])
def test_envelope_on_zero_and_two_free_coordinate_stable_faces(free_count):
    def matrices(u):
        if free_count == 0:
            return np.array([[1+.2*u, 0.], [1+.1*u, 0.]]), -np.eye(2), np.empty((2, 0))
        return np.array([[1+.1*u, 0.], [1+.2*u, 0.], [1+.3*u, 0.]]), np.array([[1., 0.], [0., 1.], [.1*u, 0.]]), np.empty((3, 0))
    a_s, a_r, b = matrices(0.)
    da = np.array([[.2, 0.], [.1, 0.]]) if free_count == 0 else np.array([[.1, 0.], [.2, 0.], [.3, 0.]])
    dr = np.zeros((2, 2)) if free_count == 0 else np.array([[0., 0.], [0., 0.], [.1, 0.]])
    price = envelope_price(a_s, a_r, b, [1., 0.], da, dr, b)
    assert len(price['fit']['nonbinding_coordinates']) == free_count
    assert price['fit']['strict_complementarity_exact']
    assert price['price'] == pytest.approx(.6 if free_count == 0 else .4)
    for h in (1e-3, 5e-4, 1e-4):
        minus = pairwise_cone_distance(*matrices(-h), [1., 0.])
        plus = pairwise_cone_distance(*matrices(h), [1., 0.])
        assert minus['binding_coordinates'] == plus['binding_coordinates'] == price['fit']['binding_coordinates']
        fd = (plus['distance_squared']-minus['distance_squared'])/(2*h)
        assert abs(fd-price['price']) < 2e-7


def test_binding_face_switch_is_not_a_stable_face_derivative_check():
    outcomes = []
    for u in (-1e-3, 0., 1e-3):
        fit = pairwise_cone_distance([[u, 0.], [1., 0.]], np.eye(2), np.empty((2, 0)), [1., 0.])
        outcomes.append(fit)
    assert outcomes[0]['binding_coordinates'] == [0]
    assert outcomes[2]['binding_coordinates'] == []
    assert not outcomes[1]['strict_complementarity_exact']
    assert outcomes[0]['distance_squared'] == pytest.approx(1e-6)
    assert outcomes[2]['distance_squared'] == 0.


def test_rank_change_and_constant_rank_cone_change_can_be_discontinuous():
    nuisance_values = []
    cone_values = []
    for u in (-.001, 0., .001):
        nuisance_values.append(pairwise_cone_distance([[1., 0.]], [[0., 0.]], [[u]], [1., 0.])['distance_squared'])
        fit = pairwise_cone_distance([[-1., 0.]], [[1., u]], np.empty((1, 0)), [1., 0.])
        assert fit['projected_competitor_rank'] == 1
        cone_values.append(fit['distance_squared'])
    assert nuisance_values == [0., 1., 0.]
    assert cone_values == [0., 1., 1.]


@pytest.mark.parametrize('which,bad', [
    ('a_s', np.ones((2, 1))), ('a_r', np.ones((3, 2))),
    ('b', np.ones((3, 1))), ('zeta', [1.]),
    ('zeta', [-1., 0.]), ('eta_lower', [-1., 0.]),
    ('a_r', [[np.nan, 0.], [1., 0.]]),
    ('b', [[np.inf], [0.]]), ('a_s', [[1j, 0.], [0., 1.]])])
def test_invalid_dimensions_nonfinite_and_domains_are_rejected(which, bad):
    args = dict(a_s=np.eye(2), a_r=np.eye(2), b=np.empty((2, 0)), zeta=[1., 1.], eta_lower=[0., 0.])
    args[which] = bad
    with pytest.raises(ValueError):
        pairwise_cone_distance(**args)


def test_analytic_evidence_has_exact_certificates_and_no_ac_or_EV_actions():
    record = run_analytic_checks()
    assert record['ac_calls'] == record['actual_EV_actions_selected'] == 0
    assert record['stage'] == 'gate_1_analytic_only'
    assert all(case['dual']['candidate_feasible_exact'] for case in record['cases'].values())
    assert all(case['primal_upper_bound'] >= case['dual_lower_bound'] for case in record['cases'].values())
    assert len(record['source_sha256']) == 2
