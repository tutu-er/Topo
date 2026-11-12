"""Control mapping and necessary probability upper bounds on full device boxes."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy.stats import norm

from .pricing import GaussianGate
from .prospective import ForecastLaw, spectral_moment


class PricingTests(unittest.TestCase):
    def test_layered_factor_matches_fixed_covariance_controlled_moment(self):
        law = ForecastLaw(np.array([[.7, .9, .2, .1], [.8, .4, .3, .2], [.6, .5, .1, .3]]), .12, .4)
        dh = np.array([[.2, .1, -.1, .3], [.1, -.1, .2, .1]])
        node = np.array([0, 1]); ratio = np.array([.2, .3]); sigma2=.03
        gate = GaussianGate(dh, law, sigma2, node, ratio, 10., competitor=1, h=1.)
        delivery = np.array([[.2, -.1, .3], [-.1, .2, .1]])
        power = np.zeros_like(law.mean)
        power[:, :2] = delivery.T/10
        power[:, 2:] = delivery.T*ratio/10
        controlled = SimpleNamespace(mean=law.mean+power, sd=law.sd, U=law.U, values=law.values)
        expected = spectral_moment(dh, controlled, sigma2)["T"]
        self.assertAlmostEqual(gate.value(delivery), expected, places=12)

    def test_capacity_upper_covers_exact_fixed_power_success(self):
        law = ForecastLaw(np.array([[.1, 0.]]), relative_sd=0.)
        gate = GaussianGate(np.array([[.1, 0.]]), law, 1., np.array([0]), np.array([0.]), 1., competitor=1, h=4.)
        low, high = np.array([[-.2]]), np.array([[.2]])
        bounds = gate.capacity_bounds(low, high, .1, .9)
        for action in np.linspace(-.2, .2, 51):
            a = abs(.1*(.1+action))
            actual = norm.cdf((.1+a*a/2-np.log(9))/a) if a else 0.
            self.assertLessEqual(actual, bounds["pair_success_upper"]+1e-12)
            self.assertLessEqual(gate.value([[action]]), bounds["moment_upper"]+1e-12)
        self.assertTrue(bounds["original_chance_infeasible"])


if __name__ == "__main__":
    unittest.main()
