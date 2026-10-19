"""Independent algebra, QP and rejection-direction checks."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2

from research_experiments.mobile_ev_mle.local_response_score import rejection, scoring
from research_experiments.mobile_ev_mle.local_rx_identifiability.cases import (
    candidate_bank, make_case,
)


def sample_data(count=24):
    case = make_case('paper15', 'ordinary', 99173, total=count, boundary='U')
    return case['observed'], case['meter'], case['domain']


def test_metric_projection_matches_independent_box_solver():
    rng = np.random.default_rng(731)
    residual = rng.normal(size=(40, 2))
    lower = rng.uniform(-1., -.1, size=(40, 2))
    upper = rng.uniform(.1, 1., size=(40, 2))
    factors = rng.normal(size=(40, 2, 2))
    covariance = factors @ np.swapaxes(factors, 1, 2)+.2*np.eye(2)
    projected = rejection.project_box_metric(residual, covariance, lower, upper)
    for t in range(len(residual)):
        precision = np.linalg.inv(covariance[t])
        def fun(delta):
            e = residual[t]-delta
            return e @ precision @ e, -2*precision @ e
        ref = minimize(fun, np.clip(residual[t], lower[t], upper[t]), jac=True,
                       bounds=list(zip(lower[t], upper[t])), method='L-BFGS-B',
                       options=dict(ftol=1e-14, gtol=1e-10, maxiter=500))
        got = projected['residual'][t] @ precision @ projected['residual'][t]
        assert got == pytest.approx(ref.fun, rel=1e-8, abs=1e-10)
    assert np.all(projected['delta'] >= lower-1e-14)
    assert np.all(projected['delta'] <= upper+1e-14)


def test_reduced_qp_gradient_has_correct_sign_and_active_box_contribution():
    rng = np.random.default_rng(85)
    n, width = 11, 4
    design = rng.normal(size=(n, 2, width))
    y = rng.normal(size=(n, 2))
    covariance = np.broadcast_to([[2., .6], [.6, 1.]], (n, 2, 2))
    point = rng.normal(size=width)
    lo, hi = np.full((n, 2), -.3), np.full((n, 2), .2)
    value, gradient = rejection._qp_value_gradient(point, y, design, covariance, lo, hi)
    for j in range(width):
        step = np.zeros(width); step[j] = 1e-6
        p = rejection._qp_value_gradient(point+step, y, design, covariance, lo, hi)[0]['loss']
        m = rejection._qp_value_gradient(point-step, y, design, covariance, lo, hi)[0]['loss']
        assert gradient[j] == pytest.approx((p-m)/2e-6, rel=1e-7, abs=1e-7)
    assert value['loss'] > 0


@pytest.mark.parametrize('index', [0, 1])
def test_zero_remainder_qp_agrees_with_original_fixed_weight_qp(index):
    obs, sigma, domain = sample_data()
    candidate = candidate_bank()[index]
    fitted = rejection.fit_delta_qp(candidate, obs, sigma, domain, 0., 0.)
    reference = scoring.fit_qp(candidate, obs, sigma, domain, weight='upper')
    assert fitted['qualified'], fitted
    assert reference['qualified'], reference
    assert fitted['loss'] == pytest.approx(reference['loss'], rel=1e-7, abs=2e-6)
    assert fitted['lower_bound'] <= fitted['loss']+1e-10
    assert fitted['absolute_gap'] < 1e-5


def test_remainder_relaxation_and_tangent_lower_bound_direction():
    obs, sigma, domain = sample_data(12)
    candidate = candidate_bank()[2]
    narrow = rejection.fit_delta_qp(candidate, obs, sigma, domain, -1e-5, 1e-5)
    wider = rejection.fit_delta_qp(candidate, obs, sigma, domain, -1e-4, 1e-4)
    assert narrow['qualified'], narrow
    assert wider['qualified'], wider
    assert wider['loss'] <= narrow['loss']+1e-6
    assert wider['lower_bound'] <= wider['loss']+1e-10
    # Independent feasible points upper-bound the optimum; every tangent
    # certificate must lie below their exact delta-profiled losses.
    data = scoring.prepare(obs, sigma)
    b, active, design, lower, upper, a = scoring._qp_geometry(candidate, data, domain)
    covariance = np.asarray(wider['covariance'])
    rng = np.random.default_rng(21)
    for _ in range(20):
        radii = rng.uniform(b['zlo'], b['zhi'], len(active))
        angles = rng.uniform(*b['angles'], len(active))
        point = np.r_[radii*np.cos(angles), radii*np.sin(angles)]
        feasible_loss = rejection._qp_value_gradient(point, data['y'], design, covariance,
                                                     -1e-4, 1e-4)[0]['loss']
        assert wider['lower_bound'] <= feasible_loss+1e-9


def test_square_noise_vertex_bound_covers_every_point_in_the_declared_event():
    obs, sigma, domain = sample_data(17)
    data = scoring.prepare(obs, sigma)
    covariance = scoring.fixed_covariance(candidate_bank()[0], data, domain, weight='upper')
    result = rejection.square_noise_bound(data, covariance, alpha_square=.01)
    cutoff = result['raw_standard_normal_cutoff']
    precision = np.linalg.inv(covariance)
    rng = np.random.default_rng(45)
    for _ in range(100):
        error = rng.uniform(-cutoff, cutoff, data['sigma_v'].shape)*data['sigma_v']
        square = (error**2-data['sigma_v']**2) @ data['D'].T
        value = np.einsum('ta,tab,tb->', square, precision, square)
        assert value <= result['bound']**2+1e-12


@pytest.mark.parametrize('alpha', [.05, .01])
def test_alpha_split_and_degrees_of_freedom_are_predeclared(alpha):
    meter = rejection.alpha_split(alpha, source='meter')
    public = rejection.alpha_split(alpha, source='public')
    assert meter['alpha_pq'] == pytest.approx(.2*alpha)
    assert meter['alpha_square'] == pytest.approx(.2*alpha)
    assert meter['alpha_gaussian'] == pytest.approx(.6*alpha)
    assert public['alpha_pq'] == 0
    assert public['alpha_square']+public['alpha_gaussian'] == pytest.approx(alpha)
    obs, sigma, domain = sample_data(9)
    data = scoring.prepare(obs, sigma)
    covariance = scoring.fixed_covariance(candidate_bank()[0], data, domain, weight='upper')
    fake = dict(qualified=True, covariance=covariance, lower_bound=1e8)
    audited = rejection.rejection_envelope(fake, data, alpha)
    assert audited['df'] == 18
    assert audited['gaussian_radius'] == pytest.approx(np.sqrt(chi2.ppf(1.-.6*alpha, 18)))
    assert audited['envelope_exceeded']
    assert not audited['reject']
    activated = rejection.rejection_envelope(fake, data, alpha, gate_passed=True)
    assert activated['reject']


def test_optimization_failure_never_rejects_and_invalid_inputs_are_saved():
    obs, sigma, domain = sample_data(8)
    data = scoring.prepare(obs, sigma)
    failed = rejection.fit_delta_qp(candidate_bank()[0], obs, sigma, domain, 1., -1.)
    assert failed['status'] == 'failed'
    assert not failed['qualified']
    assert not rejection.rejection_envelope(failed, data, .05, gate_passed=True)['reject']
    with pytest.raises(ValueError, match='predeclared'):
        rejection.alpha_split(.1)
    with pytest.raises(ValueError, match='source'):
        rejection.alpha_split(.05, source='fitted')


def test_adding_huge_covariance_is_conservative_for_fixed_point():
    residual = np.array([[.8, -.7], [1., .3]])
    lower, upper = -.1, .1
    original = rejection.project_box_metric(residual, np.eye(2), lower, upper)
    loose = rejection.project_box_metric(residual, 3*np.eye(2), lower, upper)
    assert loose['loss'] <= original['loss']
