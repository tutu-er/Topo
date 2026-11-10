"""Independent elementary checks of the Fourier evaluator, not topology tests."""
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from research_experiments.wzzt_dr_selection.dr_continuous_independent.continuous import (
    IndependentResiduals, _truncated_fourier, exact_single_probability,
)


def test_scaled_interval_transform_against_real_quadrature():
    for lo, hi in [(-2., -.3), (.2, 2.5), (-1.5, 2.), (0., np.inf)]:
        frequencies = np.array([[.01, .5, 4., 20.]])
        actual = _truncated_fourier(np.array([lo]), np.array([hi]), frequencies)[0]
        for frequency, expected in zip(frequencies[0], actual):
            real = quad(lambda z: math.exp(-z*z/2) / math.sqrt(2*math.pi) * math.cos(frequency*z),
                        lo, hi, epsabs=1e-10, limit=200)[0]
            imag = quad(lambda z: math.exp(-z*z/2) / math.sqrt(2*math.pi) * math.sin(frequency*z),
                        lo, hi, epsabs=1e-10, limit=200)[0]
            assert abs(expected - complex(real, imag)) < 2e-10


def test_exact_single_probability_and_half_normal_distribution():
    # Threshold roots from a direct three-piece hand calculation.
    assert abs(exact_single_probability(1., 0., 1., 0.)
               - (ndtr(1/.03)-ndtr(-1/2.03))) < 1e-14
    # Equal candidate residuals give -.03 |Z|. This distribution has known CDF.
    threshold = -.02
    expected = 2 * ndtr(abs(threshold)/.03) - 1
    assert abs(exact_single_probability(0., 0., 1., threshold)-expected) < 1e-14
    assert exact_single_probability(0., 0., 1., 0.) == 0.
    assert exact_single_probability(1., 1., 0., -.03) == 0.


def test_characteristic_normalization_moments_and_symmetry():
    model = IndependentResiduals([1., .3, -.2], [0., .1, .7], [1., .5, .8])
    t = np.array([-1., -1e-4, 0., 1e-4, 1.])
    cf = model.characteristic(t)
    assert abs(cf[2]-1) < 1e-14
    assert np.max(np.abs(cf)-1) < 1e-13
    assert np.max(np.abs(cf-np.conj(cf[::-1]))) < 1e-13
    assert abs((1-cf[3].real)/(1e-8/2)-1) < 2e-6


def test_two_cell_inversion_against_nested_elementary_cdf():
    c, f, sd = [.5, .8], [0., .2], [1., .7]
    threshold = 0.
    def integrand(z):
        first = abs(c[0]+sd[0]*z)-1.03*abs(f[0]+sd[0]*z)
        remaining = exact_single_probability(c[1], f[1], sd[1], 2*threshold-first)
        return math.exp(-z*z/2) / math.sqrt(2*math.pi) * remaining
    reference = quad(integrand, -9., 9., points=[-.5, 0.], epsabs=1e-9, limit=500)[0]
    result = IndependentResiduals(c, f, sd).probability(threshold, cutoff=1024., step=.05, refinements=1)
    assert abs(result['probability']-reference) < 1e-5
    assert result['certified_error'] is None
    # The refinement difference is not incorrectly relabelled a complete bound.
    assert result['analytic_tail_bound'] > result['refinement_difference']


def test_directional_gradient_matches_fixed_quadrature_with_nonzero_center_offset():
    coarse, fine, sigma = np.array([.5, .8]), [0., .2], [1., .7]
    model = IndependentResiduals(coarse, fine, sigma)
    normalization = model.sum_mean + .12, model.sum_sd
    options = dict(cutoff=256., step=.05, refinements=0, normalization=normalization)
    result = model.probability(0., daC=np.eye(2), daF=np.zeros((2, 2)), **options)
    for j in range(2):
        direction = np.eye(2)[j] * 1e-6
        positive = IndependentResiduals(coarse+direction, fine, sigma).probability(0., **options)['probability']
        negative = IndependentResiduals(coarse-direction, fine, sigma).probability(0., **options)['probability']
        assert abs((positive-negative)/2e-6-result['gradient'][j]) < 1e-6


def test_deterministic_strict_event_and_zero_sigma_cells():
    assert IndependentResiduals([1.], [0.], [0.]).probability(1.)['probability'] == 0.
    assert IndependentResiduals([1.], [0.], [0.]).probability(.9)['probability'] == 1.
    mixed = IndependentResiduals([1., 1.], [0., 0.], [0., 1.])
    result = mixed.probability(.4, cutoff=2048., step=.1, refinements=1)
    expected = exact_single_probability(1., 0., 1., -.2)
    assert abs(result['probability']-expected) < 1e-5
