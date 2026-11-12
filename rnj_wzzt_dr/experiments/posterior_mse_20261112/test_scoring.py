"""Analytic checks, scale invariance, and finite-library failure boundaries."""
import unittest
import copy
import numpy as np
from .scoring import weights, score, independent_check


class WeightTests(unittest.TestCase):
    def test_two_model_analytic_logistic(self):
        out = weights([1., 1.01], 1000)
        self.assertAlmostEqual(out["probability"][0], 1/(1+np.exp(-5.)), places=13)
        self.assertEqual(out["sigma2"], 1.)
        self.assertEqual(out["s"], .002)

    def test_equal_losses_give_uniform_weights(self):
        np.testing.assert_allclose(weights([3., 3., 3.], 100)["probability"], [1/3]*3)

    def test_measurement_unit_changes_cancel(self):
        expected = weights([1., 1.02, 2.], 100)["probability"]
        for factor in (1e-12, 1e12):
            np.testing.assert_allclose(weights(np.array([1., 1.02, 2.])*factor, 100)["probability"],
                                       expected, atol=1e-13, rtol=1e-12)

    def test_more_independent_cells_sharpen_evidence(self):
        self.assertGreater(weights([1., 1.01], 1000)["probability"][0],
                           weights([1., 1.01], 10)["probability"][0])

    def test_order_changes_only_permute_results(self):
        np.testing.assert_allclose(weights([4., 1., 2.], 20)["probability"],
                                   weights([1., 2., 4.], 20)["probability"][[2, 0, 1]])

    def test_single_candidate_has_unit_weight_without_truth_claim(self):
        self.assertEqual(weights([17.], 100)["probability"][0], 1.)

    def test_extreme_losses_retain_finite_log_probability(self):
        out = weights([1e-9, 1.], 1000)
        self.assertEqual(out["probability"][1], 0.)
        self.assertTrue(np.isfinite(out["log_probability"]).all())

    def test_zero_variance_and_invalid_inputs_are_not_repaired(self):
        for losses, cells in (([0., 1.], 10), ([-1., 1.], 10), ([np.nan], 10), ([1.], 0)):
            with self.assertRaises(ValueError): weights(losses, cells)

    def test_duplicate_points_do_not_increase_topology_prior(self):
        from ...tests.test_pipeline import fixture
        from ...identification import algorithm
        observed, result = fixture()
        expected = score(result, observed, observed)
        points = [c["point"] for c in result["frozen_candidates"]]
        selected, _, bank = algorithm._output([*points, copy.deepcopy(points[0])], observed, observed,
                                               result["reference_mse"], algorithm.SearchConfig())
        result = dict(result, selected=selected, frozen_candidates=bank)
        actual = score(result, observed, observed)
        self.assertEqual(actual["raw_points"], expected["raw_points"]+1)
        self.assertEqual(actual["distinct_topologies"], expected["distinct_topologies"])
        np.testing.assert_allclose([m["probability"] for m in expected["models"]],
                                   [m["probability"] for m in actual["models"]])

    def test_representative_is_selected_on_training_not_validation(self):
        from ...observations import Observation
        from ...identification import algorithm
        from ...identification.optimization import predict
        family = [[0], [1], [2], [0, 1]]
        a = dict(supports=family, r=[.1, .1, .1, .2], x=[0.]*4)
        b = dict(supports=family, r=[.1, .1, .1, .3], x=[0.]*4)
        empty = Observation(np.eye(3), np.zeros((3, 3)), np.zeros((3, 3)), (1, 2, 3))
        train = Observation(empty.p, empty.q, predict(empty, a)+.001, empty.labels)
        val = Observation(empty.p, empty.q, predict(empty, b)+.001, empty.labels)
        selected, _, bank = algorithm._output([a, b], train, val, 1., algorithm.SearchConfig())
        self.assertEqual(selected["index"], 1)
        result = dict(method="111", selected=selected, frozen_candidates=bank, reference_mse=1.,
                      config=vars(algorithm.SearchConfig()), events=[])
        out = score(result, train, val)
        self.assertEqual(out["models"][0]["index"], 0)
        self.assertEqual(out["models"][0]["member_indices"], [0, 1])

    def test_independent_support_prediction_and_likelihood_agree(self):
        from ...tests.test_pipeline import fixture
        observed, result = fixture()
        out = score(result, observed, observed)
        audit = independent_check(result, observed, observed, out)
        self.assertEqual(audit["status"], "passed")


if __name__ == "__main__":
    unittest.main()
