"""Pure deterministic checks: no physical data, fitting, or large Monte Carlo."""
import unittest

import numpy as np

from response_law import (BASE_RELATIVE_SD, MAIN_LAW, MISSPEC_LAW, ResponseLaw,
                          analytic_covariance, draw, draw_base_normals,
                          innovation_dimension, linear_factor,
                          multipliers_from_base, resolve_law,
                          standardized_from_base)


class ResponseLawTests(unittest.TestCase):
    def test_exact_factor_covariance_main_and_misspec(self):
        for law in (MAIN_LAW, MISSPEC_LAW):
            with self.subTest(law=law.name):
                factor = linear_factor(3, 5, law)
                covariance = analytic_covariance(3, 5, law)
                np.testing.assert_allclose(factor @ factor.T, covariance, rtol=2e-15, atol=5e-16)
                np.testing.assert_allclose(np.diag(covariance), 1., rtol=0, atol=2e-16)

    def test_covariance_entries(self):
        law = MAIN_LAW
        covariance = analytic_covariance(2, 4, law)
        self.assertAlmostEqual(covariance[0, 4], .15 + .20)
        self.assertAlmostEqual(covariance[0, 5], .15 + .20*.8)
        self.assertAlmostEqual(covariance[0, 3], .15+.25+(.20+.40)*.8**3)

    def test_stationary_initial_state_and_ar_recurrence(self):
        law = ResponseLaw("common_ar_only", 1., (0., 0., 1., 0.), .8)
        base = np.zeros((1, innovation_dimension(2, 3)))
        base[0, 3:6] = [2., 3., 4.]
        z = standardized_from_base(base, 2, 3, law)
        expected = [2., .8*2.+.6*3., .8*(.8*2.+.6*3.)+.6*4.]
        np.testing.assert_allclose(z[0], [expected, expected])

    def test_day_and_node_effects_are_not_redrawn_over_time(self):
        base = np.zeros((1, innovation_dimension(2, 3)))
        base[0, :3] = [2., 3., 4.]
        day = standardized_from_base(base, 2, 3, ResponseLaw("day", 1., (1.,0.,0.,0.), .8))
        node = standardized_from_base(base, 2, 3, ResponseLaw("node", 1., (0.,1.,0.,0.), .8))
        np.testing.assert_array_equal(day, np.full((1,2,3), 2.))
        np.testing.assert_array_equal(node[0], [[3.,3.,3.], [4.,4.,4.]])

    def test_seed_repeatability_full_node_stream(self):
        first = draw(4242, 3, 15, 96)
        second = draw(4242, 3, 15, 96)
        np.testing.assert_array_equal(first, second)
        base = draw_base_normals(4242, 3, 15, 96)
        np.testing.assert_array_equal(first, multipliers_from_base(base, 15, 96))
        np.testing.assert_array_equal(first[:, [2,0]], second[:, [2,0]])
        self.assertFalse(np.array_equal(first, draw(4243, 3, 15, 96)))

    def test_law_change_reuses_underlying_innovations(self):
        base = draw_base_normals(73, 2, 3, 4)
        for law in (MAIN_LAW, MISSPEC_LAW):
            expected = law.mean + law.relative_sd * standardized_from_base(base, 3, 4, law)
            np.testing.assert_array_equal(draw(73, 2, 3, 4, law), expected)

    def test_mean_shift_keeps_absolute_multiplier_sd(self):
        np.testing.assert_array_equal(multipliers_from_base(np.zeros((2, 12)), 2, 3, MAIN_LAW), np.ones((2,2,3)))
        np.testing.assert_array_equal(multipliers_from_base(np.zeros((2, 12)), 2, 3, MISSPEC_LAW), np.full((2,2,3), .9))
        for law in (MAIN_LAW, MISSPEC_LAW):
            np.testing.assert_allclose(np.diag(analytic_covariance(2, 3, law, multiplier=True)), BASE_RELATIVE_SD**2)
        self.assertAlmostEqual(MISSPEC_LAW.coefficient_of_variation, BASE_RELATIVE_SD/.9)

    def test_transform_is_linear(self):
        first = draw_base_normals(10, 2, 2, 3)
        second = draw_base_normals(11, 2, 2, 3)
        np.testing.assert_allclose(standardized_from_base(2*first-3*second, 2, 3),
            2*standardized_from_base(first, 2, 3)-3*standardized_from_base(second, 2, 3), atol=1e-15)

    def test_single_node_single_slot(self):
        factor = linear_factor(1, 1)
        self.assertEqual(factor.shape, (1,4))
        np.testing.assert_allclose(factor @ factor.T, [[1.]])

    def test_protocol_mapping_round_trip(self):
        self.assertEqual(resolve_law(MAIN_LAW.to_dict()), MAIN_LAW)
        self.assertEqual(resolve_law("misspec"), MISSPEC_LAW)

    def test_no_clipping(self):
        law = ResponseLaw("day_only", 1., (1.,0.,0.,0.), 0.)
        base = np.array([[-100.,0.,0.,0.], [100.,0.,0.,0.]])
        multipliers = multipliers_from_base(base, 1, 1, law)
        self.assertLess(multipliers[0,0,0], 0.)
        self.assertGreater(multipliers[1,0,0], 2.)

    def test_invalid_input_rejected(self):
        for kwargs in ({"weights":(.1,.2,.3,.3)}, {"weights":(-.1,.3,.4,.4)},
                       {"phi":1.}, {"mean":0.}, {"relative_sd":-1.}):
            params = dict(name="bad", mean=1., weights=MAIN_LAW.weights, phi=.8)
            params.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ResponseLaw(**params)
        with self.assertRaises(ValueError):
            draw(1, 1, 0, 3)
        with self.assertRaises(ValueError):
            standardized_from_base(np.zeros((2, 3)), 2, 3)
        with self.assertRaises(ValueError):
            standardized_from_base(np.full((2,12), np.nan), 2, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
