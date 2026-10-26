"""Independent mathematical checks for the small-system reference, not production WZZT."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import linprog

from research_experiments.wzzt_dr_selection.core import (
    all_blocks, block_features, compatible_next_candidates, effective_blocks,
    enumerate_laminar_families, fit_l1, is_laminar, predict, select_next_block,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.P = np.array([[1., 0.], [1., 1.], [1., 3.]])
        self.Q = np.zeros_like(self.P)
        self.R = np.array([[3., 2., 1.], [2., 3., 1.], [1., 1., 2.]])
        self.Y = self.R @ self.P
        self.B = ((0,), (1,), (2,))

    def test_features_and_prediction_against_direct_matrices(self):
        P = np.array([[1., 2.], [3., 4.], [5., 6.]])
        Q = P[:, ::-1] / 2
        blocks = ((0,), (0, 2), (0, 1, 2))
        beta = np.array([1., 2., 3., 4., 5., 6.])
        R, X = np.zeros((3, 3)), np.zeros((3, 3))
        for k, block in enumerate(blocks):
            z = np.array([float(i in block) for i in range(3)])
            R += beta[2*k] * np.outer(z, z)
            X += beta[2*k+1] * np.outer(z, z)
        assert_allclose(predict(P, Q, blocks, beta), R @ P + X @ Q)
        assert_allclose(block_features(P, Q, blocks)[:, 0],
                        np.array([1., 2., 0., 0., 0., 0.]))

    def test_hand_calculated_omitted_line_counterexample(self):
        selection = select_next_block(self.P, self.Q, self.Y, self.B)
        self.assertEqual(selection.status, "selected")
        self.assertFalse(selection.stopped)
        self.assertEqual(selection.best_candidates, ((0, 2),))
        self.assertAlmostEqual(selection.base_fit.loss, 20/3)
        self.assertAlmostEqual(selection.fits[(0, 2)].loss, 0.)
        self.assertAlmostEqual(selection.fits[(0, 1, 2)].loss, 5/3)
        self.assertAlmostEqual(selection.fits[(0, 1)].loss, 20/3)
        self.assertAlmostEqual(selection.improvement, 20/3)
        for fit in [selection.base_fit, *selection.fits.values()]:
            self.assertTrue(fit.success, fit.message)
            self.assertLess(fit.constraint_violation, 1e-8)
            self.assertLess(fit.dual_violation, 1e-8)
            self.assertLess(abs(fit.gap), 1e-8)

    def test_all_old_weights_are_jointly_refitted(self):
        base = fit_l1(self.P, self.Q, self.Y, self.B)
        addition = fit_l1(self.P, self.Q, self.Y, self.B + ((0, 2),))
        assert_allclose(base.beta[::2], [6., 6., 7/3])
        assert_allclose(addition.beta[::2], [8/3, 6., 2/3, 5/3])
        # With the old weights frozen, the first observation cannot be fitted:
        # its old prediction is already exactly 6 and the new column is positive.
        old_prediction = base.predict(self.P, self.Q)
        self.assertAlmostEqual(old_prediction[0, 0], self.Y[0, 0])
        self.assertGreater(self.Y[0, 1] - old_prediction[0, 1], 0)
        self.assertAlmostEqual(addition.loss, 0.)

    def test_primal_and_dual_independently_recomputed(self):
        rng = np.random.default_rng(617)
        P, Q = rng.normal(size=(3, 7)), rng.normal(size=(3, 7))
        blocks = self.B + ((0, 1), (0, 1, 2))
        weights = rng.uniform(0, 2, 2 * len(blocks))
        Y = predict(P, Q, blocks, weights) + rng.normal(0, .1, (3, 7))
        fit = fit_l1(P, Q, Y, blocks)
        self.assertTrue(fit.success, fit.message)
        F, y = block_features(P, Q, blocks), Y.ravel()
        primal = np.abs(y - F @ fit.beta).sum()
        independent_dual = linprog(-y, A_ub=F.T, b_ub=np.zeros(F.shape[1]),
                                   bounds=[(-1., 1.)] * y.size, method="highs")
        self.assertTrue(independent_dual.success)
        assert_allclose(fit.loss, primal, atol=1e-8)
        assert_allclose(fit.dual_bound, y @ fit.dual, atol=1e-8)
        assert_allclose(fit.dual_bound, -independent_dual.fun, atol=1e-8)
        assert_allclose(fit.gap, primal - y @ fit.dual, atol=1e-10)
        self.assertLessEqual(np.max(F.T @ fit.dual), 1e-8)
        self.assertLessEqual(np.max(np.abs(fit.dual)), 1. + 1e-8)
        assert_allclose(fit.beta * (F.T @ fit.dual), 0., atol=1e-8)

    def test_nonnegative_weights_and_zero_are_allowed(self):
        P = np.ones((1, 3))
        fit = fit_l1(P, np.zeros_like(P), -P, ((0,),))
        self.assertTrue(fit.success)
        assert_allclose(fit.beta, 0.)
        self.assertEqual(fit.loss, 3.)
        self.assertEqual(fit.effective_blocks(), ())
        self.assertEqual(effective_blocks(((0,), (1,)), [0., 0., 0., 2.]), ((1,),))

    def test_exact_true_full_family_fits(self):
        blocks = self.B + ((0, 1), (0, 1, 2))
        fit = fit_l1(self.P, self.Q, self.Y, blocks)
        self.assertTrue(fit.success)
        self.assertAlmostEqual(fit.loss, 0.)
        assert_allclose(fit.predict(self.P, self.Q), self.Y)

    def test_all_best_ties_and_stop_are_preserved(self):
        zeros = np.zeros((3, 2))
        selection = select_next_block(self.P, self.Q, zeros, self.B)
        self.assertTrue(selection.stopped)
        self.assertEqual(selection.status, "stopped")
        self.assertEqual(selection.best_candidates,
                         ((0, 1), (0, 2), (1, 2), (0, 1, 2)))
        self.assertEqual(selection.improvement, 0.)

    def test_laminar_exhaustion_and_optional_root(self):
        for n, expected in [(1, 1), (2, 2), (3, 8), (4, 52)]:
            families = enumerate_laminar_families(n)
            self.assertEqual(len(families), expected)
            self.assertEqual(len(set(families)), expected)
            for family in families:
                self.assertTrue(is_laminar(family))
                self.assertTrue(all((i,) in family for i in range(n)))
            if n > 1:
                root = tuple(range(n))
                self.assertTrue(any(root not in f for f in families))
                self.assertTrue(any(root in f for f in families))
        self.assertFalse(is_laminar(((0, 1), (1, 2))))
        with self.assertRaises(ValueError):
            enumerate_laminar_families(5)

    def test_compatible_candidates_obey_existing_blocks(self):
        selected = self.B + ((0, 1),)
        self.assertEqual(compatible_next_candidates(3, selected), ((0, 1, 2),))
        self.assertEqual(all_blocks(1), ((0,),))
        with self.assertRaises(ValueError):
            fit_l1(self.P, self.Q, self.Y, self.B + ((0, 1), (1, 2)))
        with self.assertRaises(ValueError):
            select_next_block(self.P, self.Q, self.Y, selected, ((1, 2),))

    def test_empty_support_and_no_candidates(self):
        fit = fit_l1(self.P, self.Q, self.Y, ())
        self.assertTrue(fit.success)
        self.assertEqual(fit.beta.shape, (0,))
        self.assertAlmostEqual(fit.loss, np.abs(self.Y).sum())
        assert_allclose(fit.predict(self.P, self.Q), np.zeros_like(self.Y))
        selection = select_next_block(self.P, self.Q, self.Y, self.B, ())
        self.assertTrue(selection.stopped)
        self.assertEqual(selection.status, "no_candidates")

    def test_failed_candidate_is_not_reported_as_a_stop(self):
        base = fit_l1(self.P, self.Q, self.Y, self.B)
        failed = replace(base, status="iteration_limit", solver_status=1,
                         message="deliberately simulated unfinished LP")
        with patch("research_experiments.wzzt_dr_selection.core.fit_l1",
                   side_effect=[base, failed]):
            selection = select_next_block(self.P, self.Q, self.Y, self.B,
                                          ((0, 2),))
        self.assertEqual(selection.status, "incomplete")
        self.assertIsNone(selection.stopped)
        self.assertEqual(selection.failed_candidates, ((0, 2),))
        self.assertEqual(selection.best_candidates, ())
        self.assertTrue(np.isnan(selection.improvement))

    def test_invalid_shapes_duplicates_and_nonfinite_values(self):
        with self.assertRaises(ValueError):
            block_features(self.P, self.Q[:, :1], self.B)
        with self.assertRaises(ValueError):
            block_features(self.P, self.Q, ((0,), (0,)))
        with self.assertRaises(ValueError):
            block_features(self.P, self.Q, ((3,),))
        with self.assertRaises(ValueError):
            fit_l1(self.P, self.Q, np.full_like(self.Y, np.nan), self.B)


if __name__ == "__main__":
    unittest.main()
