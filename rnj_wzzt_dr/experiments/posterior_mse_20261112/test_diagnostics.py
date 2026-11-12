"""Finite-library decision-set checks, independent of case truth."""
import unittest
import numpy as np
from .diagnose import credible_set


def models(probabilities):
    return [dict(index=i, topology_id=str(i), probability=p,
                 log_probability=float(np.log(p)), validation_mse=i+1.)
            for i, p in enumerate(probabilities)]


class DecisionSetTests(unittest.TestCase):
    def test_singleton_above_gate(self):
        result = credible_set(models([.95, .04, .01]), .9)
        self.assertEqual(result["indices"], [0])

    def test_two_required_for_close_pair(self):
        result = credible_set(models([.5744, .4255, .0001]), .9)
        self.assertEqual(result["indices"], [0, 1])
        self.assertAlmostEqual(result["mass"], .9999)

    def test_boundary_equal_weights_are_all_retained(self):
        result = credible_set(models([.86, .07, .07]), .9)
        self.assertEqual(result["indices"], [0, 1, 2])

    def test_pairwise_90_is_insufficient_for_whole_library(self):
        probabilities = np.array([10., 1., 1., 1.])/13
        self.assertGreater(probabilities[0]/(probabilities[0]+probabilities[1]), .9)
        self.assertLess(probabilities[0], .9)
        self.assertGreater(len(credible_set(models(probabilities), .9)["indices"]), 1)


if __name__ == "__main__":
    unittest.main()
