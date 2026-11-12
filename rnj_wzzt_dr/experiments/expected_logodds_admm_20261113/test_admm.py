"""Analytic solutions, failure starts, exact sharing projection and device slices."""
import unittest
import numpy as np

from ...dr.pipeline import load_resources
from .admm import project_outside, sharing_projection, run, ScalarConvexDevice
from .evidence import Evidence, build, direct_gains
from .device_steps import resource_slice, initial_requests, OriginalDeviceStep


class Tests(unittest.TestCase):
    def test_outside_projection_including_zero(self):
        np.testing.assert_allclose(project_outside(np.array([.3, .4]), 2.), [1.2, 1.6])
        np.testing.assert_array_equal(project_outside(np.array([3., 4.]), 2.), [3., 4.])
        self.assertAlmostEqual(np.linalg.norm(project_outside(np.zeros(2), 2.)), 2.)

    def test_sharing_projection_has_minimum_correction(self):
        e = Evidence(np.array([[[.5]]]), np.array([[[1., 1.]]]), np.ones(1), .7, buffer=0.)
        z = sharing_projection(e, np.zeros((1, 2, 1, 1)))
        required = np.sqrt(.25+.7)-.5
        np.testing.assert_allclose(z.ravel(), [required/2]*2)
        for split in np.linspace(0, required, 41):
            self.assertLessEqual(float(np.sum(z*z)), split**2+(required-split)**2+1e-12)

    def test_one_dimensional_exact_optimum_from_infeasible_zero_start(self):
        e = Evidence(np.array([[[.5]]]), np.ones((1, 1, 1)), np.ones(1), .7)
        d = ScalarConvexDevice(-5., 5.)
        result = run(e, [d], np.zeros((1, 1)), lambda u:float(np.sum(u*u)), iterations=500, rho=2.)
        exact = (np.sqrt(.25+.7)-.5)**2
        self.assertTrue(result["feasible"])
        self.assertLess(abs(result["best"]["fee"]-exact), 1e-4)
        self.assertGreaterEqual(result["best"]["gains"][0], .7-1e-8)

    def test_symmetric_zero_start_can_fail_despite_feasibility(self):
        e = Evidence(np.array([[[.5]], [[-.5]]]), np.ones((2, 1, 1)), np.ones(2), .7)
        result = run(e, [ScalarConvexDevice(-5., 5.)], np.zeros((1, 1)), lambda u:float(np.sum(u*u)), iterations=80, rho=2.)
        self.assertFalse(result["feasible"])
        self.assertTrue(e.feasible(np.array([[2.]])))

    def test_symmetric_case_nonzero_start_matches_known_branch(self):
        e = Evidence(np.array([[[.5]], [[-.5]]]), np.ones((2, 1, 1)), np.ones(2), .7)
        result = run(e, [ScalarConvexDevice(-5., 5., initial=2.)], np.array([[2.]]), lambda u:float(np.sum(u*u)), iterations=500, rho=10.)
        exact = (.5+np.sqrt(.25+.7))**2
        self.assertTrue(result["feasible"])
        self.assertLess(abs(result["best"]["fee"]-exact), 1e-3)

    def test_unsuitable_penalty_can_only_retain_initial_plan(self):
        e = Evidence(np.array([[[.5]], [[-.5]]]), np.ones((2, 1, 1)), np.ones(2), .7)
        result = run(e, [ScalarConvexDevice(-5., 5., initial=2.)], np.array([[2.]]), lambda u:float(np.sum(u*u)), iterations=100, rho=2.)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["best"]["iteration"], -1)
        self.assertAlmostEqual(result["best"]["fee"], 4.)
        self.assertFalse(result["global_optimality_proved"])

    def test_resource_slice_preserves_original_controller_and_fees(self):
        original, problem, _ = load_resources("paper15", labels=tuple(range(101, 116)))
        x = initial_requests(problem, original, "rotation25")
        expected = original.nominal(x); restored = []
        for r in range(original.R):
            part = resource_slice(problem, r)
            point = part.nominal(part.requests_to_values(x[r:r+1]))
            restored.append(point["mean_u"][0])
        np.testing.assert_allclose(restored, expected, atol=1e-14)
        self.assertTrue(original.audit(np.asarray(restored))["passed"])

    def test_gain_effects_against_independent_full_H_prediction(self):
        original, problem, base = load_resources("paper15", labels=tuple(range(101, 116)))
        rng = np.random.default_rng(17)
        H = rng.normal(0, .01, (3, 15, 30)); forecast=rng.normal(.01, .001, (96, 30))
        x = initial_requests(problem, original, "rotation25"); delivery = original.nominal(x)
        evidence = build(H, 0, [1, 2], forecast, 1e-6, original, base, .7)
        independent = direct_gains(H, 0, [1, 2], forecast, 1e-6, original, base, delivery)
        np.testing.assert_allclose(evidence.gains(delivery), independent, atol=1e-11)

    def test_native_device_update_replays_original_model(self):
        import gurobipy as gp
        from time import perf_counter
        original, problem, _ = load_resources("paper15", labels=tuple(range(101, 116)))
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0); env.start()
            step = OriginalDeviceStep(problem, 1, np.zeros((1, 24)), 1., env)
            try:
                delivery, report = step.solve(1., -np.ones(96), 10., perf_counter()+1.)
                self.assertFalse(report["no_incumbent"])
                assembled = np.zeros((original.R, 96)); assembled[1] = delivery
                self.assertTrue(original.audit(assembled)["passed"])
                part = step.problem.nominal(step.problem.requests_to_values(step.requests))
                self.assertAlmostEqual(part["fee"], original.fees(assembled), places=10)
                self.assertLess(report["replay_difference"], 1e-7)
            finally:
                step.close()


if __name__ == "__main__":
    unittest.main()
