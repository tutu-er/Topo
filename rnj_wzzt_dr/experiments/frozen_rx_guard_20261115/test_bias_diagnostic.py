import unittest
import numpy as np
from .bias_diagnostic import decompose


class BiasIdentityTests(unittest.TestCase):
    def test_constant_bias_can_harm_with_zero_incremental_error(self):
        d = decompose([[1.]], [[-9.]], [[1.1]], [[-8.9]], [[1.]], [[0.]], [[1.]], 1.)
        self.assertAlmostEqual(d['A_background_bias_rotated'], -1.)
        self.assertAlmostEqual(d['B1_response_error_against_old_difference'], 0.)
        self.assertAlmostEqual(d['B2_response_error_against_added_difference'], 0.)
        self.assertAlmostEqual(d['K_increment'], .105)
        self.assertAlmostEqual(d['clean_increment'], -.895)

    def test_random_direct_sse_difference(self):
        rng = np.random.default_rng(2026111523)
        for _ in range(20):
            X0, X = rng.normal(size=(2, 7, 6))
            Y0, Y = rng.normal(size=(2, 7, 3))
            Hi, Hj, K = rng.normal(size=(3, 3, 6))
            d = decompose(X0, Y0, X, Y, Hi, Hj, K, .7, 2.)
            self.assertLess(d['identity_error'], 1e-10)

    def test_pure_quadratic_error_and_zero_action(self):
        d = decompose([[0.]], [[0.]], [[2.]], [[1.]], [[1.]], [[0.]], [[1.]], 1.)
        self.assertEqual(d['A_background_bias_rotated'], 0.)
        self.assertEqual(d['B1_response_error_against_old_difference'], 0.)
        self.assertEqual(d['B2_response_error_against_added_difference'], -2.)
        z = decompose([[1.]], [[.8]], [[1.]], [[.8]], [[1.]], [[0.]], [[1.]], .01)
        self.assertEqual(z['sum_model_gap_increment'], 0.)
        self.assertEqual(z['incremental_K_error_rms_over_sigma'], 0.)


if __name__ == '__main__':
    unittest.main()
