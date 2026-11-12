"""Evidence preservation and identical-response counterexamples."""
import unittest
import numpy as np

from .posterior import add_observations, log_weights, retained


class PosteriorTests(unittest.TestCase):
    def test_frozen_variance_does_not_change_with_new_losses(self):
        H = np.array([[[1.]], [[2.]]])
        result = add_observations([1., 2.], H, [[1.]], [[1.]], .25)
        self.assertEqual(result["sigma2"], .25)
        np.testing.assert_allclose(result["sse"], [1., 3.])
        self.assertAlmostEqual(result["log_probability"][0]-result["log_probability"][1], 4.)

    def test_sequential_and_combined_evidence_agree(self):
        H = np.array([[[1.]], [[2.]]]); p = np.array([[1.], [2.]])
        y = np.array([[1.1], [1.9]])
        first = add_observations([1., 2.], H, p[:1], y[:1], .25)
        after = add_observations(first["sse"], H, p[1:], y[1:], .25)
        combined = add_observations([1., 2.], H, p, y, .25)
        np.testing.assert_allclose(after["log_probability"], combined["log_probability"])

    def test_identical_predictions_do_not_win_by_tie_priority(self):
        H = np.ones((2, 1, 1))
        result = add_observations([1., 1.], H, [[1.]], [[1.1]], .25)
        np.testing.assert_allclose(np.exp(result["log_probability"]), [.5, .5])
        self.assertEqual(retained(result["log_probability"], .9), [0, 1])

    def test_two_pairwise_90_percent_scores_are_not_full_90_percent(self):
        logp = log_weights([0., 2*np.log(9), 2*np.log(9)], 1.)
        self.assertAlmostEqual(np.exp(logp[0]), 9/11)
        self.assertEqual(len(retained(logp, .9)), 3)

    def test_boundary_ties_all_retained(self):
        self.assertEqual(retained(np.log([.8, .1, .1]), .9), [0, 1, 2])

    def test_underflow_keeps_finite_log_evidence(self):
        result = log_weights([1., 10000.], 1.)
        self.assertTrue(np.isfinite(result).all())
        self.assertEqual(np.exp(result)[1], 0.)


if __name__ == "__main__":
    unittest.main()
