"""Independent analytic, quadrature, Monte Carlo and derivative kernel tests."""

import math
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from conditional_integral import conditional_probability


class ConditionalIntegralTests(unittest.TestCase):
    def test_abs_tail(self):
        r = conditional_probability([0.0], [1.0], [1.0], 1.0)
        self.assertAlmostEqual(r.probability, 2.0 * ndtr(-1.0), places=14)
        self.assertTrue(r.gradient_valid)
        np.testing.assert_allclose(r.coeff_grad_a, [0.0], atol=1e-15)
        np.testing.assert_allclose(r.coeff_grad_b, [2.0 * math.exp(-0.5) / math.sqrt(2 * math.pi)])
        self.assertEqual(r.threshold_roots, (-1.0, 1.0))

    def test_negative_weight_interval(self):
        r = conditional_probability([0.0], [1.0], [-1.0], -1.0)
        self.assertAlmostEqual(r.probability, 1.0 - 2.0 * ndtr(-1.0), places=14)
        self.assertEqual(r.intervals, ((-1.0, 1.0),))

    def test_shifted_abs_coeff_derivatives(self):
        a, b, h = 0.4, 0.8, 1.2
        r = conditional_probability([a], [b], [1.0], h)
        lower, upper = (-h-a)/b, (h-a)/b
        phi = lambda z: math.exp(-z*z/2) / math.sqrt(2*math.pi)
        self.assertAlmostEqual(r.probability, ndtr(lower) + ndtr(-upper), places=14)
        self.assertAlmostEqual(r.coeff_grad_a[0], (phi(upper)-phi(lower))/b, places=13)
        self.assertAlmostEqual(r.coeff_grad_b[0], (upper*phi(upper)-lower*phi(lower))/b, places=13)

    def test_strict_threshold_platform(self):
        r = conditional_probability([1.0, 0.0], [1.0, 1.0], [1.0, -1.0], 1.0)
        self.assertEqual(r.probability, 0.0)
        self.assertFalse(r.gradient_valid)
        self.assertIn("threshold_platform", r.diagnostics["gradient_invalid_reasons"])
        self.assertGreaterEqual(r.diagnostics["threshold_platforms"], 1)

    def test_platform_below_threshold(self):
        r = conditional_probability([1.0, 0.0], [1.0, 1.0], [1.0, -1.0], 0.0)
        self.assertAlmostEqual(r.probability, ndtr(0.5), places=14)
        self.assertTrue(r.gradient_valid)

    def test_identical_terms_cancel(self):
        r = conditional_probability([1.0, 1.0], [2.0, 2.0], [1.0, -1.0], 0.0)
        self.assertEqual(r.probability, 0.0)
        self.assertFalse(r.gradient_valid)
        self.assertEqual(r.diagnostics["unique_knots"], 1)

    def test_duplicate_knots(self):
        r = conditional_probability([0, 0, 0], [1, 2, -3], [1, 1, 1], 3.0)
        self.assertEqual(r.diagnostics["unique_knots"], 1)
        self.assertAlmostEqual(r.probability, 2 * ndtr(-0.5), places=14)
        self.assertTrue(r.gradient_valid)

    def test_zero_b_constant_terms(self):
        r = conditional_probability([0.4, 0.0], [0.0, 1.0], [1.0, 1.0], 1.4)
        self.assertAlmostEqual(r.probability, 2 * ndtr(-1), places=14)
        self.assertTrue(r.gradient_valid)
        self.assertGreater(r.coeff_grad_a[0], 0)
        self.assertAlmostEqual(r.coeff_grad_b[0], 0, places=14)

    def test_pure_constant_and_empty(self):
        self.assertEqual(conditional_probability([2], [0], [1], 1).probability, 1)
        self.assertEqual(conditional_probability([2], [0], [1], 2).probability, 0)
        self.assertEqual(conditional_probability([], [], [], -1).probability, 1)
        self.assertEqual(conditional_probability([], [], [], 0).probability, 0)

    def test_zero_weight_does_not_add_kink(self):
        r = conditional_probability([0, 1], [0, 1], [0, 1], 2)
        self.assertTrue(r.gradient_valid)
        self.assertEqual(r.coeff_grad_a[0], 0)
        self.assertEqual(r.coeff_grad_b[0], 0)

    def test_moving_identically_zero_term_invalid(self):
        r = conditional_probability([0, 0], [0, 1], [1, 1], 1)
        self.assertFalse(r.gradient_valid)
        self.assertIsNone(r.coeff_grad_a)
        self.assertIn("identically_zero_residual", r.diagnostics["gradient_invalid_reasons"])

    def test_restricted_chain_fixed_zero_row_is_safe(self):
        r = conditional_probability([0, 0], [0, 1], [1, 1], 1,
                                    da=[[0], [1]], coefficient_gradients=False)
        self.assertTrue(r.gradient_valid)
        np.testing.assert_allclose(r.gradient, [0.0], atol=1e-14)
        self.assertIsNone(r.coeff_grad_a)

    def test_kink_at_threshold_is_invalid(self):
        r = conditional_probability([0], [1], [1], 0)
        self.assertEqual(r.probability, 1.0)
        self.assertFalse(r.gradient_valid)
        self.assertIn("threshold_at_or_near_kink", r.diagnostics["gradient_invalid_reasons"])

    def test_small_slopes_not_zeroed(self):
        r = conditional_probability([0], [1e-12], [1], 1e-12)
        self.assertAlmostEqual(r.probability, 2 * ndtr(-1), places=14)
        self.assertTrue(r.gradient_valid)
        self.assertGreater(r.coeff_grad_b[0], 1e11)

    def test_scale_invariance_extreme(self):
        a = np.array([0.3, -0.7, 1.1, -0.2])
        b = np.array([1.2, 0.4, -0.6, 1.3])
        w = np.array([1.0, -1.1, 0.8, -0.4])
        base = conditional_probability(a, b, w, 0.2)
        self.assertTrue(base.gradient_valid)
        for scale in (1e-250, 1e-12, 1e12, 1e250):
            with self.subTest(scale=scale):
                r = conditional_probability(a*scale, b*scale, w, 0.2*scale)
                self.assertAlmostEqual(r.probability, base.probability, places=13)
                self.assertTrue(r.gradient_valid)
                np.testing.assert_allclose(r.coeff_grad_a*scale, base.coeff_grad_a, rtol=1e-11, atol=1e-12)
                np.testing.assert_allclose(r.coeff_grad_b*scale, base.coeff_grad_b, rtol=1e-11, atol=1e-12)

    def test_far_tail_mass(self):
        r = conditional_probability([0], [1], [-1], -9)
        self.assertAlmostEqual(r.probability, 1 - 2*ndtr(-9), places=15)
        r = conditional_probability([0], [1], [1], 9)
        self.assertAlmostEqual(r.probability / (2*ndtr(-9)), 1, places=12)

    def test_very_small_interval(self):
        r = conditional_probability([0], [1], [-1], -1e-20)
        self.assertAlmostEqual(r.probability / (math.sqrt(2 / math.pi) * 1e-20), 1, places=13)

    def test_random_quadrature_reference(self):
        rng = np.random.default_rng(610203)
        for case in range(10):
            a, b = rng.normal(size=(2, 8))
            w = np.r_[np.ones(4), -1.02*np.ones(4)]
            h = float(rng.normal())
            result = conditional_probability(a, b, w, h)
            knots = sorted(float(t) for t in -a/b if -12 < t < 12)
            def integrand(z):
                return math.exp(-z*z/2) / math.sqrt(2*math.pi) * (np.dot(w, np.abs(a+b*z)) > h)
            reference, error = quad(integrand, -12, 12, points=knots, epsabs=2e-9, limit=1000)
            self.assertLess(abs(result.probability-reference), 1e-7, msg=f"case {case}; quad error={error}")

    def test_random_monte_carlo_reference(self):
        rng = np.random.default_rng(610204)
        z = rng.normal(size=250_000)
        for case in range(5):
            a, b = rng.normal(size=(2, 10))
            w = rng.normal(size=10)
            h = float(rng.normal())
            r = conditional_probability(a, b, w, h)
            indicator = (w @ np.abs(a[:, None]+b[:, None]*z)) > h
            empirical = indicator.mean()
            bound = 5 * math.sqrt(r.probability*(1-r.probability)/z.size) + 2/z.size
            self.assertLessEqual(abs(empirical-r.probability), bound, msg=f"case {case}")

    def test_random_coefficient_gradient_finite_difference(self):
        rng = np.random.default_rng(610205)
        for case in range(10):
            a, b = rng.normal(size=(2, 7))
            w = rng.normal(size=7)
            h = float(rng.normal())
            r = conditional_probability(a, b, w, h)
            self.assertTrue(r.gradient_valid)
            for target, analytic in (("a", r.coeff_grad_a), ("b", r.coeff_grad_b)):
                numeric = []
                for i in range(7):
                    step = np.eye(7)[i]*1e-6
                    if target == "a":
                        plus = conditional_probability(a+step, b, w, h, coefficient_gradients=False).probability
                        minus = conditional_probability(a-step, b, w, h, coefficient_gradients=False).probability
                    else:
                        plus = conditional_probability(a, b+step, w, h, coefficient_gradients=False).probability
                        minus = conditional_probability(a, b-step, w, h, coefficient_gradients=False).probability
                    numeric.append((plus-minus)/2e-6)
                np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=2e-7, err_msg=f"{case}/{target}")

    def test_full_chain_rule(self):
        rng = np.random.default_rng(610206)
        a, b, w = rng.normal(size=(3, 9))
        da, db = rng.normal(size=(2, 9, 3))
        h = 0.7
        r = conditional_probability(a, b, w, h, da=da, db=db)
        self.assertTrue(r.gradient_valid)
        numeric = []
        for j in range(3):
            eps = 1e-6
            plus = conditional_probability(a+eps*da[:,j], b+eps*db[:,j], w, h).probability
            minus = conditional_probability(a-eps*da[:,j], b-eps*db[:,j], w, h).probability
            numeric.append((plus-minus)/(2*eps))
        np.testing.assert_allclose(r.gradient, numeric, rtol=1e-5, atol=1e-7)

    def test_probability_only_and_mapping_interface(self):
        r = conditional_probability([0], [1], [1], 1, coefficient_gradients=False)
        self.assertFalse(r.gradient_valid)
        self.assertIsNone(r.coeff_grad_a)
        self.assertEqual(r["probability"], r.probability)
        self.assertEqual(r.get("missing", 7), 7)
        with self.assertRaises(KeyError):
            _ = r["missing"]

    def test_invalid_input(self):
        invalid = [([1], [1,2], [1], 0), ([float("nan")], [1], [1], 0), ([1], [1], [1], math.inf)]
        for args in invalid:
            with self.assertRaises(ValueError):
                conditional_probability(*args)
        with self.assertRaises(ValueError):
            conditional_probability([1], [1], [1], 0, da=[[1,2]], db=[[1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
