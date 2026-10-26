import json
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import linprog

from research_experiments.wzzt_dr_selection.core import block_features
from research_experiments.wzzt_dr_selection.optimal_face import (
    common_mode_counterexample, fit_future_on_history_optimal_face,
    fixed_support_profile_diagnostic,
)


class OptimalFaceTests(unittest.TestCase):
    def test_common_history_freezing_an_arbitrary_optimum_fails(self):
        report = common_mode_counterexample()
        face = report["optimal_face"]
        self.assertTrue(face["success"], face)
        self.assertEqual(face["coefficient_bounds"], "nonnegative_only")
        self.assertAlmostEqual(face["history_fit"]["loss"], 0.)
        self.assertAlmostEqual(face["frozen_history_loss"], 0.)
        self.assertTrue(face["frozen_history_compatible"])
        self.assertAlmostEqual(face["frozen_future_loss"], 2.)
        self.assertAlmostEqual(face["history_loss"], 0.)
        self.assertAlmostEqual(face["future_loss"], 0.)
        self.assertAlmostEqual(face["frozen_minus_face_future_loss"], 2.)
        self.assertFalse(face["exact_arithmetic_derivative_certificate"])
        json.dumps(report, allow_nan=False)

    def test_finite_lambda_secant_is_not_the_right_derivative(self):
        # Lh(beta)=beta, Lf(beta)=|2-2beta|; Q columns remain unbounded.
        report = fixed_support_profile_diagnostic(
            [[1.]], [[0.]], [[0.]], [[2.]], [[0.]], [[2.]], ((0,),),
            lambdas=(1e-4, .1, 1.))
        face = report["optimal_face"]
        self.assertTrue(face["success"], face)
        self.assertAlmostEqual(face["future_loss"], 2.)
        for item in report["finite_weights"][:2]:
            self.assertEqual(item["status"], "optimal")
            self.assertAlmostEqual(item["secant_slope"], 2.)
        last = report["finite_weights"][-1]
        self.assertAlmostEqual(last["value"], 1.)
        self.assertAlmostEqual(last["secant_slope"], 1.)
        self.assertAlmostEqual(last["face_solution_upper_bound"], 2.)
        self.assertAlmostEqual(last["upper_bound_minus_value"], 1.)

    def test_tolerance_is_reported_and_changes_the_face_objective(self):
        result = fit_future_on_history_optimal_face(
            [[1.]], [[0.]], [[0.]], [[2.]], [[0.]], [[2.]], ((0,),),
            history_atol=.1)
        self.assertTrue(result["success"], result)
        self.assertAlmostEqual(result["history_cap"], .1)
        self.assertAlmostEqual(result["added_history_slack"], .1)
        self.assertAlmostEqual(result["history_loss"], .1)
        self.assertAlmostEqual(result["future_loss"], 1.8)
        self.assertAlmostEqual(result["history_excess_over_reference"], .1)

    def test_noncompact_history_face_can_fit_a_newly_excited_coefficient(self):
        result = fit_future_on_history_optimal_face(
            [[0.]], [[0.]], [[0.]], [[1.]], [[0.]], [[9.]], ((0,),))
        self.assertTrue(result["success"], result)
        self.assertAlmostEqual(result["future_loss"], 0.)
        self.assertAlmostEqual(result["beta"][0], 9.)
        self.assertAlmostEqual(result["history_loss"], 0.)

    def test_independent_dual_and_original_losses(self):
        Ph, Qh, Yh = np.array([[1.]]), np.array([[0.]]), np.array([[0.]])
        Pf, Qf, Yf = np.array([[2.]]), np.array([[0.]]), np.array([[2.]])
        blocks = ((0,),)
        result = fit_future_on_history_optimal_face(Ph, Qh, Yh, Pf, Qf, Yf, blocks,
                                                   history_atol=.1)
        self.assertTrue(result["success"], result)
        Fh, Ff = block_features(Ph, Qh, blocks), block_features(Pf, Qf, blocks)
        yh, yf, cap = Yh.ravel(), Yf.ravel(), result["history_cap"]
        mh, mf = yh.size, yf.size
        # Independently solve the dual in qh, qf, alpha coordinates.
        A = np.block([[Fh.T, Ff.T, np.zeros((Fh.shape[1], 1))],
                      [np.eye(mh), np.zeros((mh, mf)), -np.ones((mh, 1))],
                      [-np.eye(mh), np.zeros((mh, mf)), -np.ones((mh, 1))]])
        objective = np.concatenate([-yh, -yf, [cap]])
        dual = linprog(objective, A_ub=A, b_ub=np.zeros(A.shape[0]),
                       bounds=[(None, None)]*mh + [(-1, 1)]*mf + [(0, None)],
                       method="highs")
        self.assertTrue(dual.success)
        assert_allclose(result["dual_bound"], -dual.fun, atol=1e-8)
        beta = np.array(result["beta"])
        assert_allclose(result["future_loss"], np.abs(yf-Ff@beta).sum(), atol=1e-8)
        assert_allclose(result["history_loss"], np.abs(yh-Fh@beta).sum(), atol=1e-8)
        self.assertLessEqual(result["history_loss"], cap+1e-8)
        self.assertLessEqual(result["constraint_violation"], 1e-8)
        self.assertLessEqual(result["dual_violation"], 1e-8)
        self.assertAlmostEqual(result["gap"], 0.)

    def test_incompatible_frozen_solution_is_not_a_valid_comparison(self):
        result = fit_future_on_history_optimal_face(
            [[1.]], [[0.]], [[0.]], [[2.]], [[0.]], [[2.]], ((0,),),
            frozen_beta=[1., 0.])
        self.assertTrue(result["success"])
        self.assertFalse(result["frozen_history_compatible"])
        self.assertIsNone(result["frozen_minus_face_future_loss"])

    def test_empty_support_and_json_serialization(self):
        result = fixed_support_profile_diagnostic(
            [[1.]], [[0.]], [[3.]], [[2.]], [[0.]], [[4.]], (), lambdas=(.1, 1.))
        self.assertTrue(result["optimal_face"]["success"])
        self.assertAlmostEqual(result["optimal_face"]["future_loss"], 4.)
        self.assertEqual(result["optimal_face"]["beta"], [])
        self.assertAlmostEqual(result["finite_weights"][-1]["value"], 7.)
        json.dumps(result, allow_nan=False)

    def test_invalid_data_and_tolerances(self):
        args = ([[1.]], [[0.]], [[0.]], [[2.]], [[0.]], [[2.]], ((0,),))
        with self.assertRaises(ValueError):
            fit_future_on_history_optimal_face(*args, history_atol=-1.)
        with self.assertRaises(ValueError):
            fixed_support_profile_diagnostic(*args, lambdas=(0.,))
        with self.assertRaises(ValueError):
            fit_future_on_history_optimal_face(*args, frozen_beta=[1.])
        with self.assertRaises(ValueError):
            fit_future_on_history_optimal_face([[1.]], [[0.]], [[0.]],
                                               [[1.], [2.]], [[0.], [0.]],
                                               [[1.], [2.]], ((0,),))


if __name__ == "__main__":
    unittest.main()
