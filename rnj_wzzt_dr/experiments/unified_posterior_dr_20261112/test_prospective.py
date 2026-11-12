"""Closed normal tails and independent dense Gaussian moment calculations."""
import unittest
import numpy as np
from scipy.stats import norm

from .prospective import ForecastLaw, active_allocation, failure_bound, spectral_moment


class MomentTests(unittest.TestCase):
    def test_fixed_power_matches_gaussian_log_odds_mgf(self):
        law = ForecastLaw(np.array([[1., 0.], [2., 0.]]), relative_sd=0.)
        difference = np.array([[.2, 0.]])
        value = spectral_moment(difference, law, .04)
        self.assertAlmostEqual(value["T"], 5/8, places=13)
        self.assertEqual(value["half_logdet"], 0.)

    def test_spectral_matches_independent_dense_integral(self):
        law = ForecastLaw(np.array([[.7, .9, .2, .1], [.8, .4, .3, .2], [.6, .5, .1, .3]]), .12, .4)
        dh = np.array([[.2, .1, -.1, .3], [.1, -.1, .2, .1]])
        sigma2 = .03; got = spectral_moment(dh, law, sigma2)
        factor = (dh[:, :2]*law.sd[:2]+dh[:, 2:]*law.sd[2:])/np.sqrt(sigma2)
        corr = law.time_ar**np.abs(np.arange(3)[:, None]-np.arange(3)[None, :])
        covariance = np.kron(corr, factor@factor.T)
        mean = (law.mean@dh.T/np.sqrt(sigma2)).ravel()
        G = np.eye(6)+covariance/4
        expected = mean@np.linalg.solve(G, mean)/8+np.linalg.slogdet(G)[1]/2
        self.assertAlmostEqual(got["T"], expected, places=12)

    def test_exponential_tail_bound_is_conservative_for_exact_normal(self):
        for strength in (.2, 1., 3., 6.):
            history = np.array([0., -.4]); beta = .9
            T = {1: dict(T=strength**2/8)}
            out = failure_bound(history, T, 0, beta)
            exact = norm.cdf((.4+strength**2/2-np.log(9))/strength)
            self.assertLessEqual(out["success_lower"], exact+1e-12)

    def test_same_response_cannot_pass_large_posterior_gate(self):
        out = failure_bound(np.log([.5, .5]), {1: dict(T=0.)}, 0, .9)
        self.assertEqual(out["success_lower"], 0.)

    def test_full_bound_covers_two_correlated_opposite_competitors(self):
        beta = .9; B = (1-beta)/beta
        for strength in (1., 3., 6.):
            moments = {1: dict(T=strength**2/8), 2: dict(T=strength**2/8)}
            bound = failure_bound(np.log([1/3]*3), moments, 0, beta)
            argument = B*np.exp(strength**2/2)/2
            exact = 0. if argument < 1 else 2*norm.cdf(np.arccosh(argument)/strength)-1
            self.assertLessEqual(bound["success_lower"], exact+1e-12)

    def test_allocated_active_and_uniform_tail_cover_full_bound(self):
        pairs = [dict(competitor=1, log_c=0., T=7.),
                 dict(competitor=2, log_c=-7., T=1.)]
        allocation = active_allocation(pairs)
        self.assertEqual(allocation["active"], [1])
        tail = np.exp(allocation["log_tail"])
        full = sum(np.exp(p["log_c"]-p["T"]) for p in pairs)
        conservative = np.exp(-7.)+tail
        self.assertLessEqual(full, conservative)
        self.assertLessEqual(conservative, .1)


if __name__ == "__main__":
    unittest.main()
