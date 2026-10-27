"""Independent small LP checks for history-compatible linear functionals."""
from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from research_experiments.wzzt_dr_selection import compatible_functionals as cf


class CompatibleFunctionalTests(unittest.TestCase):
    def setUp(self):
        self.blocks = ((0,), (1,), (0, 1))
        self.P = np.ones((2, 1))
        self.Q = np.zeros_like(self.P)
        self.Y = 2 * np.ones_like(self.P)
        self.hint = np.array([2., 0., 2., 0., 0., 0.])

    def check_json(self, result):
        json.dumps(result, allow_nan=False)

    def test_generic_dual_sign_and_objective(self):
        c = np.array([3., 2.])
        A = np.array([[1., 1.]])
        b = np.array([4.])
        result = cf.maximize_nonnegative_lp(c, A_ub=A, b_ub=b)
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["value"], 12.)
        self.assertAlmostEqual(result["upper_bound"], 12.)
        x = np.asarray(result["x"])
        w = np.asarray(result["dual"]["inequality"])
        self.assertGreaterEqual(w.min(), 0.)
        np.testing.assert_array_less(c - 1e-8, A.T @ w)
        self.assertLessEqual(float((A @ x - b).max()), 1e-8)
        self.assertAlmostEqual(float(b @ w - c @ x), result["gap"])
        self.check_json(result)

    def test_equality_dual_negative_value(self):
        result = cf.maximize_nonnegative_lp([-1.], A_eq=[[1.]], b_eq=[2.])
        self.assertEqual(result["status"], "optimal")
        self.assertAlmostEqual(result["value"], -2.)
        self.assertAlmostEqual(result["upper_bound"], -2.)
        self.assertAlmostEqual(result["dual"]["equality"][0], -1.)

    def test_nonunique_history_root_interval(self):
        c = np.array([0., 0., 0., 0., 1., 0.])
        maximum = cf.maximize_history_functional(self.P, self.Q, self.Y, self.blocks, c)
        minimum = cf.maximize_history_functional(self.P, self.Q, self.Y, self.blocks, -c)
        self.assertEqual(maximum["status"], "optimal")
        self.assertEqual(minimum["status"], "optimal")
        self.assertAlmostEqual(maximum["value"], 1.)
        self.assertAlmostEqual(minimum["value"], 0.)
        for result in (maximum, minimum):
            r0, _, r1, _, shared, _ = result["beta"]
            self.assertAlmostEqual(r0 + 2 * shared, 2.)
            self.assertAlmostEqual(r1 + 2 * shared, 2.)
            self.assertAlmostEqual(result["history_loss"], 0.)
            self.check_json(result)

    def test_p_only_future_effect_finite_despite_unidentified_x(self):
        # At future p=(1,0), first output is r0+shared=2-shared, hence [1,2].
        c = np.array([1., 0., 0., 0., 1., 0.])
        upper = cf.maximize_history_functional(self.P, self.Q, self.Y, self.blocks, c)
        lower = cf.maximize_history_functional(self.P, self.Q, self.Y, self.blocks, -c)
        self.assertEqual(upper["status"], "optimal")
        self.assertEqual(lower["status"], "optimal")
        self.assertAlmostEqual(upper["value"], 2.)
        self.assertAlmostEqual(lower["value"], -1.)

    def test_p_only_x_unbounded_verified_with_hint(self):
        c = np.array([0., 1., 0., 0., 0., 0.])
        result = cf.maximize_history_functional(
            self.P, self.Q, self.Y, self.blocks, c, feasible_beta=self.hint)
        self.assertEqual(result["status"], "unbounded_verified")
        self.assertTrue(result["compatible"])
        self.assertFalse(result["bounded"])
        self.assertEqual(result["lp_calls"], 2)
        self.assertEqual(result["LPcalls"], 2)
        self.assertIsNone(result["upper_bound"])
        self.assertEqual(result["supremum"], "positive_infinity")
        self.assertFalse(result["strict_arithmetic_certificate"])
        d = np.asarray(result["direction_beta"])
        self.assertTrue(np.all(d >= -1e-8))
        self.assertGreaterEqual(float(c @ d), 1. - 1e-8)
        # Direct independent action of the three z z.T blocks on common history.
        r0, _, r1, _, shared, _ = d
        np.testing.assert_allclose([r0 + 2 * shared, r1 + 2 * shared], [0., 0.], atol=1e-8)
        self.check_json(result)

    def test_unbounded_without_hint_has_feasibility_phase(self):
        c = np.array([0., 1., 0., 0., 0., 0.])
        result = cf.maximize_history_functional(self.P, self.Q, self.Y, self.blocks, c)
        self.assertEqual(result["status"], "unbounded_verified")
        self.assertEqual(result["lp_calls"], 3)
        self.assertEqual([p["phase"] for p in result["phase_results"]],
                         ["primary", "feasible_point", "recession_direction"])

    def test_zero_feature_and_zero_or_negative_objective(self):
        P = np.zeros((1, 1))
        for c, expected_status, expected_value in (
                ([1., 0.], "unbounded_verified", 0.),
                ([-1., 0.], "optimal", 0.),
                ([0., 0.], "optimal", 0.)):
            with self.subTest(c=c):
                result = cf.maximize_history_functional(P, P, P, ((0,),), c)
                self.assertEqual(result["status"], expected_status)
                self.assertAlmostEqual(result["value"], expected_value)
                self.check_json(result)

    def test_epsilon_positive_l1_budget(self):
        P, Q, Y = np.array([[1.]]), np.array([[0.]]), np.array([[2.]])
        upper = cf.maximize_history_functional(P, Q, Y, ((0,),), [1., 0.], epsilon=.5)
        lower = cf.maximize_history_functional(P, Q, Y, ((0,),), [-1., 0.], epsilon=.5)
        self.assertEqual(upper["status"], "optimal")
        self.assertEqual(lower["status"], "optimal")
        self.assertAlmostEqual(upper["value"], 2.5)
        self.assertAlmostEqual(lower["value"], -1.5)
        self.assertAlmostEqual(upper["history_loss"], .5)
        self.assertAlmostEqual(lower["history_loss"], .5)
        self.assertAlmostEqual(upper["upper_bound"], 2.5)
        self.check_json(upper)

    def test_epsilon_is_sum_not_mean(self):
        P, Q, Y = np.ones((1, 2)), np.zeros((1, 2)), 2 * np.ones((1, 2))
        result = cf.maximize_history_functional(P, Q, Y, ((0,),), [1., 0.], epsilon=1.)
        self.assertAlmostEqual(result["value"], 2.5)
        self.assertAlmostEqual(result["history_loss"], 1.)

    def test_infeasible_preserved_without_positive_claim(self):
        result = cf.maximize_history_functional([[1.]], [[0.]], [[-1.]], ((0,),), [1., 0.])
        self.assertEqual(result["status"], "infeasible_unverified")
        self.assertFalse(result["success"])
        self.assertIsNone(result["compatible"])
        self.assertIsNone(result["value"])
        self.assertIsNone(result["beta"])
        self.check_json(result)

    def test_generic_unbounded_inequality_ray(self):
        result = cf.maximize_nonnegative_lp(
            [1., 0.], A_eq=[[0., 1.]], b_eq=[1.],
            A_ub=[[-1., 1.]], b_ub=[0.], feasible_x=[1., 1.])
        self.assertEqual(result["status"], "unbounded_verified")
        d = np.asarray(result["direction"])
        self.assertAlmostEqual(d[1], 0.)
        self.assertLessEqual(-d[0] + d[1], 1e-8)
        self.assertGreaterEqual(d[0], 1. - 1e-8)
        self.check_json(result)

    def test_invalid_hint_is_rejected_and_not_used_as_witness(self):
        result = cf.maximize_nonnegative_lp(
            [1., 0.], A_eq=[[0., 1.]], b_eq=[1.], feasible_x=[0., 0.])
        self.assertEqual(result["status"], "unbounded_verified")
        self.assertFalse(result["feasible_hint_accepted"])
        self.assertEqual(result["lp_calls"], 3)
        self.assertAlmostEqual(result["x"][1], 1.)

    def test_unbounded_solver_status_alone_insufficient(self):
        unbounded = SimpleNamespace(status=3, success=False, message="unbounded", x=None)
        failed = SimpleNamespace(status=4, success=False, message="numerical failure", x=None)
        with patch.object(cf, "linprog", side_effect=[unbounded, failed]):
            result = cf.maximize_nonnegative_lp([1.])
        self.assertEqual(result["status"], "unbounded_unverified")
        self.assertFalse(result["success"])
        self.assertIsNone(result["direction"])
        self.assertIsNone(result["supremum"])
        self.assertEqual(result["lp_calls"], 2)

    def test_bad_ray_is_not_verified(self):
        unbounded = SimpleNamespace(status=3, success=False, message="unbounded", x=None)
        invalid_ray = SimpleNamespace(status=0, success=True, message="claimed", x=np.array([0.]))
        with patch.object(cf, "linprog", side_effect=[unbounded, invalid_ray]):
            result = cf.maximize_nonnegative_lp([1.], feasible_x=[0.])
        self.assertEqual(result["status"], "unbounded_unverified")
        self.assertIsNone(result["direction"])
        self.assertGreater(result["ray_diagnostics"]["violation"], .9)

    def test_solver_exception_is_retained(self):
        with patch.object(cf, "linprog", side_effect=RuntimeError("test failure")):
            result = cf.maximize_nonnegative_lp([-1.], feasible_x=[0.])
        self.assertEqual(result["status"], "solver_exception")
        self.assertFalse(result["success"])
        self.assertTrue(result["compatible"])
        self.assertEqual(result["value"], 0.)
        self.assertIn("RuntimeError", result["phase_results"][0]["message"])
        self.assertEqual(result["lp_calls"], 1)

    def test_wrong_dual_sign_does_not_pass_optimal(self):
        claimed = SimpleNamespace(
            status=0, success=True, message="claimed", x=np.array([1.]),
            ineqlin=SimpleNamespace(marginals=np.array([1.])))
        with patch.object(cf, "linprog", return_value=claimed):
            result = cf.maximize_nonnegative_lp([1.], A_ub=[[1.]], b_ub=[1.])
        self.assertEqual(result["status"], "optimal_unverified")
        self.assertEqual(result["value"], 1.)
        self.assertIsNone(result["upper_bound"])
        self.assertGreater(result["dual_violation"], .9)

    def test_empty_variables_and_support(self):
        feasible = cf.maximize_nonnegative_lp([], A_eq=np.empty((1, 0)), b_eq=[0.])
        impossible = cf.maximize_nonnegative_lp([], A_eq=np.empty((1, 0)), b_eq=[1.])
        self.assertEqual(feasible["status"], "optimal")
        self.assertEqual(feasible["value"], 0.)
        self.assertEqual(feasible["lp_calls"], 0)
        self.assertEqual(impossible["status"], "infeasible_verified")
        self.assertFalse(impossible["compatible"])
        wrapped = cf.maximize_history_functional([[1.]], [[0.]], [[0.]], (), [])
        self.assertEqual(wrapped["beta"], [])
        self.assertEqual(wrapped["status"], "optimal")

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            cf.maximize_nonnegative_lp([1.], A_eq=[[1.]])
        with self.assertRaises(ValueError):
            cf.maximize_nonnegative_lp([1.], A_ub=[[1., 2.]], b_ub=[0.])
        with self.assertRaises(ValueError):
            cf.maximize_nonnegative_lp([float("nan")])
        with self.assertRaises(ValueError):
            cf.maximize_nonnegative_lp([1.], tol=0)
        with self.assertRaises(ValueError):
            cf.maximize_nonnegative_lp([1.], feasible_x=[0., 1.])
        with self.assertRaises(ValueError):
            cf.maximize_history_functional(self.P, self.Q, self.Y, self.blocks, [1.], epsilon=-1)
        with self.assertRaises(ValueError):
            cf.maximize_history_functional(self.P, self.Q, self.Y, self.blocks, np.ones(6), epsilon=-1)
        for blocks in (((0, 1), (1, 2)), ((0,), (0,)), ((0, 0),), ((3,),)):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                cf.maximize_history_functional(np.ones((3, 1)), np.zeros((3, 1)),
                                               np.ones((3, 1)), blocks,
                                               np.zeros(2 * len(blocks)))


if __name__ == "__main__":
    unittest.main()
