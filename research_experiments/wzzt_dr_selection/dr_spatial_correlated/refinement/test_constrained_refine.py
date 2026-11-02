"""Small analytic objectives only; no study case, AC or new random path."""
import unittest
import numpy as np
from constrained_refine import refine, _step


class LinearProbability:
    def evaluate(self, u):
        return dict(probability=.5+.4*float(np.asarray(u)[0, 0]),
                    gradient=np.array([.4]+[0.]*(np.asarray(u).size-1)))


class ConcaveProbability:
    def evaluate(self, u):
        x = float(np.asarray(u)[0, 0])
        return dict(probability=.4+.55*x-.2*x*x, gradient=np.array([.55-.4*x]))


class RestorationTests(unittest.TestCase):
    def test_phaseI_minimum_slack_then_movement(self):
        step, info = _step(np.zeros((1, 1)), np.ones(1), .5, np.array([[.4]]),
                           np.ones(1), np.ones(1), .7, .8, "restore", 10)
        self.assertIsNotNone(step)
        np.testing.assert_allclose(step[0], [[.5]], atol=2e-7)
        self.assertLess(info["slack"], 1e-9)

    def test_restoration_and_hard_cost_acceptance(self):
        answer = refine(LinearProbability(), np.ones(1), np.zeros((1, 1)), np.ones(1),
                        fixed_capacity=np.ones(1), target=.7, maxiter=8, seconds=10)
        self.assertTrue(answer["planning_feasible"])
        self.assertAlmostEqual(answer["best"]["planning_probability"], .7, delta=2e-7)
        self.assertAlmostEqual(answer["best"]["cost"], .125, delta=2e-6)
        for row in answer["trace"]:
            for trial in row["trials"]:
                if trial["accepted"] and row["phase"] == "restore":
                    self.assertGreater(trial["probability"], row["probability"])
                if trial["accepted"] and row["phase"] == "reduce_cost":
                    self.assertGreaterEqual(trial["probability"], .7-1e-7)

    def test_concave_actual_probability_is_rechecked(self):
        answer = refine(ConcaveProbability(), np.ones(1), np.zeros((1, 1)), np.ones(1),
                        fixed_capacity=np.ones(1), target=.7, maxiter=12, seconds=10)
        self.assertTrue(answer["planning_feasible"])
        x = answer["best"]["mean_u"][0][0]
        self.assertGreaterEqual(.4+.55*x-.2*x*x, .7-1e-7)
        self.assertLess(answer["best"]["cost"], .4)

    def test_initial_feasible_point_is_not_lost(self):
        answer = refine(LinearProbability(), np.ones(1), np.array([[.9]]), np.ones(1),
                        fixed_capacity=np.ones(1), target=.7, maxiter=8, seconds=10)
        self.assertTrue(answer["planning_feasible"])
        self.assertLess(answer["best"]["cost"], .405)
        self.assertGreaterEqual(answer["best"]["planning_probability"], .7-1e-7)

    def test_zero_capacity_is_retained(self):
        answer = refine(LinearProbability(), np.array([1., 0.]), np.zeros((2, 1)), np.array([1., 0.]),
                        fixed_capacity=np.array([1., 0.]), target=.7, maxiter=8, seconds=10)
        self.assertTrue(answer["planning_feasible"])
        self.assertEqual(answer["best"]["capacity_kw"][1], 0.)
        self.assertEqual(answer["best"]["mean_u"][1][0], 0.)

    def test_no_gradient_or_feasible_proof_is_invented(self):
        class Flat:
            def evaluate(self, u):
                return dict(probability=.5, gradient=np.zeros(np.asarray(u).size))
        answer = refine(Flat(), np.ones(1), np.zeros((1, 1)), np.ones(1), maxiter=4, seconds=10)
        self.assertFalse(answer["planning_feasible"])
        self.assertFalse(answer["infeasibility_proven"])
        self.assertEqual(answer["status"], "linearized_restoration_stalled")


if __name__ == "__main__":
    unittest.main(verbosity=2)
