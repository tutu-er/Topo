"""Independent small-system checks of numerical matrix-relaxation bounds."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import linprog

from research_experiments.wzzt_dr_selection import completion_bounds as bounds
from research_experiments.wzzt_dr_selection.core import enumerate_laminar_families, fit_l1


class CompletionBoundsTests(unittest.TestCase):
    def setUp(self):
        self.P = np.eye(3)
        self.Q = np.zeros((3, 3))
        self.R = np.array([[3., 2., 1.], [2., 3., 1.], [1., 1., 2.]])

    def test_fixed_symmetric_matrices_and_combined_P_Q(self):
        R = np.array([[3., 1.], [1., 2.]])
        X = np.array([[.5, .2], [.2, .7]])
        P = np.column_stack([np.eye(2), np.zeros((2, 2))])
        Q = np.column_stack([np.zeros((2, 2)), np.eye(2)])
        Y = R@P + X@Q
        fit = bounds.matrix_relaxation_fit(P, Q, Y, order=(0, 1, 1),
                                           coefficient=(1., 2.))
        self.assertTrue(fit["success"], fit)
        assert_allclose(fit["R"], R, atol=1e-8)
        assert_allclose(fit["X"], X, atol=1e-8)
        self.assertAlmostEqual(fit["loss"], 0.)
        self.assertFalse(fit["legal_tree_model"])
        self.assertFalse(fit["strict_arithmetic_certificate"])
        self.assertLessEqual(fit["constraint_violation"], 1e-8)
        self.assertLessEqual(fit["dual_violation"], 1e-8)
        json.dumps(fit, allow_nan=False)

    def test_true_tree_satisfies_relaxed_constraints(self):
        fit = bounds.matrix_relaxation_fit(self.P, self.Q, self.R,
                                           confirmed=((0, 1), (0, 1, 2)))
        self.assertTrue(fit["success"], fit)
        assert_allclose(fit["R"], self.R)
        R, X = np.array(fit["R"]), np.array(fit["X"])
        for M in (R, X):
            assert_allclose(M, M.T)
            self.assertGreaterEqual(M.min(), -1e-8)
            for i in range(3):
                self.assertGreaterEqual(M[i, i]+1e-8, M[i].max())
        self.assertTrue(fit["confirmed_conditional"])
        self.assertEqual(fit["comparison_scope"], "same_confirmed_equalities_completion_class")

    def test_relaxation_deliberately_accepts_a_non_tree_matrix(self):
        # The off-diagonal triple (2,1,2) has a unique minimum, which is not
        # a common-path tree similarity; nevertheless it meets this relaxation.
        M = np.array([[3., 2., 1.], [2., 3., 2.], [1., 2., 3.]])
        result = bounds.matrix_relaxation_fit(self.P, self.Q, M)
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["loss"], 0.)
        assert_allclose(result["R"], M)

    def test_order_and_observed_root_change_the_fit(self):
        Y = np.array([[3., 1.], [1., 2.]])
        P, Q = np.eye(2), np.zeros((2, 2))
        order = bounds.matrix_relaxation_fit(P, Q, Y, order=(0, 0, 1))
        root = bounds.matrix_relaxation_fit(P, Q, Y, order=(0, 1, -1))
        multiple = bounds.matrix_relaxation_fit(
            P, Q, Y, order=((0, 1, -1), (1, 0, -1)))
        for fit in (order, root, multiple):
            self.assertTrue(fit["success"], fit)
            self.assertAlmostEqual(fit["loss"], 2.)
        self.assertAlmostEqual(order["R"][0][0], order["R"][0][1])
        self.assertAlmostEqual(root["R"][0][1], 0.)
        self.assertAlmostEqual(multiple["loss"], root["loss"])

    def test_independent_reduced_dual_lp_checks_sign_and_bound(self):
        Y = np.array([[3., 1.], [1., 2.]])
        fit = bounds.matrix_relaxation_fit(np.eye(2), np.zeros((2, 2)), Y,
                                           order=(0, 0, 1))
        # The order and diagonal inequalities imply R=[[s,s],[s,s+t]],
        # with s,t>=0. This is an independently derived equivalent model.
        F = np.array([[1., 0.], [1., 0.], [1., 0.], [1., 1.]])
        y = Y.ravel()
        independent = linprog(-y, A_ub=F.T, b_ub=np.zeros(2),
                              bounds=[(-1., 1.)]*4, method="highs")
        self.assertTrue(independent.success)
        self.assertTrue(fit["success"])
        self.assertAlmostEqual(fit["dual_bound"], -independent.fun)
        self.assertAlmostEqual(fit["loss"], 2.)
        self.assertAlmostEqual(fit["gap"], 0.)
        assert_allclose(y @ np.array(fit["dual"]["observation_vector"]),
                        fit["dual_bound"], atol=1e-8)
        self.assertLessEqual(max(fit["dual"]["inequality_multipliers"]), 1e-8)
        self.assertGreaterEqual(min(fit["dual"]["reduced_cost"]), -1e-8)

    def test_omission_bounds_against_complete_tree_enumeration(self):
        families = enumerate_laminar_families(3)
        for C, expected_exact, expected_relaxed in [((0, 1), 2., 2.),
                                                   ((0, 1, 2), 4., 2.)]:
            exact_values = []
            for family in families:
                if C not in family:
                    fitted = fit_l1(self.P, self.Q, self.R, family)
                    self.assertTrue(fitted.success)
                    exact_values.append(fitted.loss)
            exact = min(exact_values)
            result = bounds.block_omission_lower_bound(self.P, self.Q, self.R, C)
            self.assertTrue(result["success"], result)
            self.assertAlmostEqual(exact, expected_exact)
            self.assertAlmostEqual(result["lower_bound"], expected_relaxed)
            self.assertLessEqual(result["lower_bound"], exact+1e-8)
        # The root example has a genuine matrix-relaxation gap of 2.
        self.assertGreater(expected_exact, expected_relaxed)

    def test_wrong_block_has_zero_omission_bound(self):
        result = bounds.block_omission_lower_bound(self.P, self.Q, self.R, (0, 2))
        self.assertTrue(result["success"], result)
        self.assertAlmostEqual(result["lower_bound"], 0.)
        self.assertEqual(result["branch_count"], 4)
        self.assertEqual(result["eligible_j"], [0, 2])
        for row in result["per_j"]:
            self.assertEqual(row["branch_count"], 2)
            self.assertTrue(row["all_branches_optimal"])

    def test_zero_weights_and_zero_direction_are_retained(self):
        without_inner = np.eye(3)+np.ones((3, 3))
        result = bounds.block_omission_lower_bound(
            self.P, self.Q, without_inner, (0, 1))
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["lower_bound"], 0.)
        without_root = np.eye(3)
        without_root[:2, :2] += 1.
        root = bounds.block_omission_lower_bound(self.P, self.Q, without_root, (0, 1, 2))
        self.assertTrue(root["success"])
        self.assertAlmostEqual(root["lower_bound"], 0.)
        zero = bounds.block_omission_lower_bound(self.P, self.Q, np.zeros((3, 3)), (0, 1))
        self.assertTrue(zero["success"])
        self.assertAlmostEqual(zero["lower_bound"], 0.)
        direction = bounds.matrix_relaxation_fit(self.P, self.Q, self.R,
                                                 order=(0, 0, -1), coefficient=(0., 0.))
        self.assertTrue(direction["success"])
        self.assertAlmostEqual(direction["loss"], 0.)

    def test_singleton_root_and_nonnegative_constraint(self):
        result = bounds.block_omission_lower_bound([[1.]], [[0.]], [[2.]], (0,))
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["lower_bound"], 2.)
        self.assertEqual(result["per_j"][0]["branches"][0]["k"], -1)
        negative = bounds.matrix_relaxation_fit([[1.]], [[0.]], [[-2.]])
        self.assertTrue(negative["success"])
        self.assertAlmostEqual(negative["R"][0][0], 0.)
        self.assertAlmostEqual(negative["loss"], 2.)

    def test_confirmed_external_equalities_only_and_invalid_inputs(self):
        fit = bounds.matrix_relaxation_fit(self.P, self.Q, self.R, confirmed=((0, 2),))
        self.assertTrue(fit["success"], fit)
        self.assertAlmostEqual(fit["R"][0][1], fit["R"][2][1])
        self.assertGreater(fit["loss"], 0.)  # A wrong confirmation is not silently discarded.
        self.assertTrue(fit["confirmed_conditional"])
        with self.assertRaises(ValueError):
            bounds.matrix_relaxation_fit(self.P, self.Q, self.R,
                                         confirmed=((0, 1), (1, 2)))
        with self.assertRaises(ValueError):
            bounds.block_omission_lower_bound(self.P, self.Q, self.R, (1, 0),
                                               confirmed=((0, 1),))
        with self.assertRaises(ValueError):
            bounds.matrix_relaxation_fit(self.P, self.Q, self.R, confirmed=((3,),))
        with self.assertRaises(ValueError):
            bounds.matrix_relaxation_fit(self.P, self.Q, self.R, order=(-1, 0, 1))
        with self.assertRaises(ValueError):
            bounds.matrix_relaxation_fit(self.P, self.Q, self.R, coefficient=(-1., 1.))
        with self.assertRaises(ValueError):
            bounds.matrix_relaxation_fit(self.P, self.Q[:, :1], self.R)

    def test_branch_min_max_uses_dual_values_not_primal_values(self):
        def fake(*args, order=None, **kwargs):
            i, j, k = order
            value = {(0, 0): 1., (1, 0): 4., (0, 1): 3., (1, 1): 2.}[i, j]
            return {"status": "optimal", "success": True, "dual_bound": value,
                    "loss": 90.+value}
        with patch.object(bounds, "matrix_relaxation_fit", side_effect=fake):
            result = bounds.block_omission_lower_bound(self.P, self.Q, self.R, (0, 1))
        self.assertTrue(result["success"])
        self.assertEqual([j["lower_bound"] for j in result["per_j"]], [1., 2.])
        self.assertEqual(result["lower_bound"], 2.)
        self.assertEqual(result["maximizing_j"], [1])

    def test_failed_branch_invalidates_its_j_and_is_preserved(self):
        def fake(*args, order=None, **kwargs):
            i, j, k = order
            failed = (i, j) == (0, 0)
            return {"status": "iteration_limit" if failed else "optimal",
                    "success": not failed, "dual_bound": 100. if j == 0 else 2.,
                    "loss": 120. if j == 0 else 3.}
        with patch.object(bounds, "matrix_relaxation_fit", side_effect=fake):
            result = bounds.block_omission_lower_bound(self.P, self.Q, self.R, (0, 1))
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["eligible_j"], [1])
        self.assertEqual(result["incomplete_j"], [0])
        self.assertIsNone(result["per_j"][0]["lower_bound"])
        self.assertEqual(result["lower_bound"], 2.)
        self.assertEqual(result["branch_count"], 4)
        self.assertEqual(result["per_j"][0]["branches"][0]["fit"]["status"], "iteration_limit")
        failure = {"status": "iteration_limit", "success": False,
                   "dual_bound": None, "loss": None}
        with patch.object(bounds, "matrix_relaxation_fit", return_value=failure):
            result = bounds.block_omission_lower_bound(self.P, self.Q, self.R, (0, 1))
        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["lower_bound"])
        self.assertEqual(result["trivial_bound"], 0.)
        json.dumps(result, allow_nan=False)

    def test_solver_failure_retained_without_fabricated_arrays(self):
        unfinished = SimpleNamespace(status=1, success=False, x=None, nit=1,
                                     message="unfinished test fixture",
                                     ineqlin=SimpleNamespace(marginals=None),
                                     eqlin=SimpleNamespace(marginals=None))
        with patch.object(bounds, "linprog", return_value=unfinished):
            result = bounds.matrix_relaxation_fit(self.P, self.Q, self.R)
        self.assertFalse(result["success"])
        self.assertEqual(result["status"], "iteration_limit")
        self.assertIsNone(result["R"])
        self.assertIsNone(result["loss"])
        self.assertIsNone(result["dual_bound"])
        self.assertEqual(result["solver_status"], 1)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
